#!/usr/bin/env python3
"""
Phase 1: Teacher Vector Extraction

This script implements Phase 1 of the V2 focused experiment:
- Compute whitening matrix from unembedding
- Estimate parent vectors and child deltas using LDA
- Validate geometric claims (angles, interventions, controls)

Estimated runtime: 2-3 hours
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from polytope_hsae.activations import ActivationCapture, ActivationConfig
from polytope_hsae.concepts import (ConceptCurator, ConceptSplitter,
                                    PromptGenerator)
from polytope_hsae.estimators import (ConceptVectorEstimator, LDAEstimator,
                                      validate_orthogonality)
# Project imports
from polytope_hsae.geometry import (CausalGeometry, compute_polytope_angles,
                                    intervention_test)
from polytope_hsae.validation import GeometryValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_experiment(config):
    """Set up experiment directories and logging."""
    exp_dir = Path(config["logging"]["save_dir"]) / config["logging"]["phase_1_log"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = exp_dir / f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info(f"Phase 1 experiment directory: {exp_dir}")
    return exp_dir


def _set_all_seeds(seed: int = 123):
    """Set all random seeds for reproducibility."""
    import random

    import numpy as np

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_create_concepts(config, exp_dir):
    """Load existing concepts or create new ones."""
    concepts_file = exp_dir / "concept_hierarchies.json"

    if concepts_file.exists():
        logger.info("Loading existing concept hierarchies")
        from polytope_hsae.concepts import load_concept_hierarchies

        hierarchies = load_concept_hierarchies(str(concepts_file))
    else:
        logger.info("Creating new concept hierarchies")

        # Create concept curator
        curator = ConceptCurator(
            max_parents=config["concepts"]["parents"],
            max_children_per_parent=config["concepts"]["max_children_per_parent"],
            min_children_per_parent=2,
        )

        # Curate hierarchies
        hierarchies = curator.curate_concept_hierarchies()

        # Generate prompts
        prompt_generator = PromptGenerator(
            prompts_per_concept=config["concepts"]["prompts_per_concept"]
        )

        hierarchies = [
            prompt_generator.populate_hierarchy_prompts(h) for h in hierarchies
        ]

        # Save concepts
        from polytope_hsae.concepts import save_concept_hierarchies

        save_concept_hierarchies(hierarchies, str(concepts_file))

    logger.info(f"Using {len(hierarchies)} concept hierarchies")
    return hierarchies


def capture_activations(config, hierarchies, exp_dir):
    """Capture activations for all concepts."""
    activations_file = exp_dir / "activations.h5"

    if activations_file.exists():
        logger.info("Loading existing activations")
        activation_capture = ActivationCapture(
            ActivationConfig(
                model_name=config["model"]["name"],
                layer_name=config["model"]["activation_hook"],
                batch_size=32,
                device=config["run"]["device"],
            )
        )
        return activation_capture.load_hierarchical_activations(str(activations_file))

    logger.info("Capturing new activations")

    # Set up activation capture
    activation_config = ActivationConfig(
        model_name=config["model"]["name"],
        layer_name=config["model"]["activation_hook"],
        batch_size=32,
        device=config["run"]["device"],
    )

    activation_capture = ActivationCapture(activation_config)

    # Capture hierarchical activations
    activations = activation_capture.capture_hierarchical_activations(
        hierarchies, negative_sampling_ratio=1.0, save_path=str(activations_file)
    )

    return activations


def extract_teacher_vectors(config, hierarchies, activations, exp_dir, unembedding_matrix):
    """Extract parent vectors and child deltas using LDA."""
    logger.info("Extracting teacher vectors")

    logger.info(f"Unembedding matrix shape: {unembedding_matrix.shape}")

    # Create causal geometry (standard approach)
    geometry = CausalGeometry.from_unembedding(
        unembedding_matrix, shrinkage=config["geometry"]["shrinkage"]
    )
    # Save precomputed geometry for later phases
    geometry_file = exp_dir / "geometry.pt"
    geometry.save(geometry_file)
    logger.info(f"Saved geometry to {geometry_file}")

    # Create LDA estimator
    lda_estimator = LDAEstimator(
        shrinkage=config["geometry"]["estimator"]["lda_shrinkage"],
        min_vocab_count=config["geometry"]["estimator"]["min_vocab_count"],
        class_balance=True,
    )

    # Create concept vector estimator
    estimator = ConceptVectorEstimator(lda_estimator, geometry)

    # Separate parent and child activations
    parent_activations = {}
    child_activations = {}

    for concept_id, pos_neg_dict in activations.items():
        if any(concept_id.startswith(h.parent.synset_id) for h in hierarchies):
            parent_activations[concept_id] = pos_neg_dict
        else:
            # Find parent for this child
            parent_id = None
            for h in hierarchies:
                if any(child.synset_id == concept_id for child in h.children):
                    parent_id = h.parent.synset_id
                    break

            if parent_id:
                if parent_id not in child_activations:
                    child_activations[parent_id] = {}
                child_activations[parent_id][concept_id] = pos_neg_dict

    # Estimate parent vectors
    parent_vectors = estimator.estimate_parent_vectors(parent_activations)

    # Estimate child vectors and deltas in causal space
    child_vectors, child_deltas = estimator.estimate_child_deltas(
        parent_vectors, child_activations
    )

    # Estimate projectors for H-SAE initialization with energy-based dimension selection
    projectors = estimator.estimate_child_subspace_projectors(
        parent_vectors, child_deltas, 
        subspace_dim=config["hsae"]["subspace_dim"],
        energy_threshold=config["geometry"]["estimator"].get("energy_threshold", 0.85)
    )

    # Save results
    results = {
        "parent_vectors": {k: v.tolist() for k, v in parent_vectors.items()},
        "child_vectors": {
            pid: {cid: vec.tolist() for cid, vec in cdict.items()}
            for pid, cdict in child_vectors.items()
        },
        "child_deltas": {
            parent_id: {child_id: delta.tolist() for child_id, delta in deltas.items()}
            for parent_id, deltas in child_deltas.items()
        },
        "projectors": {
            parent_id: {"down": down.tolist(), "up": up.tolist()}
            for parent_id, (down, up) in projectors.items()
        },
        "config": {
            "m_top": len(parent_vectors),
            "m_low": max(len(deltas) for deltas in child_deltas.values()) if child_deltas else 0,
            "subspace_dim": config["hsae"]["subspace_dim"],
            "model_dim": parent_vectors[list(parent_vectors.keys())[0]].shape[0] if parent_vectors else 0,
        },
    }

    with open(exp_dir / "teacher_vectors.json", "w") as f:
        json.dump(results, f, indent=2)

    return parent_vectors, child_vectors, child_deltas, projectors, geometry


def validate_geometry(
    config,
    parent_vectors,
    child_vectors,
    child_deltas,
    geometry,
    exp_dir,
    model,
    tokenizer,
    unembedding_matrix,
    activations,
    hierarchies,
):
    """Validate geometric claims with metric triangulation."""
    logger.info("Validating geometric claims with metric triangulation")

    # Create validator
    validator = GeometryValidator(geometry)

    # Reconstruct activation splits for alternative geometries
    parent_activations = {}
    child_activations = {}
    for concept_id, pos_neg_dict in activations.items():
        if any(concept_id.startswith(h.parent.synset_id) for h in hierarchies):
            parent_activations[concept_id] = pos_neg_dict
        else:
            parent_id = None
            for h in hierarchies:
                if any(child.synset_id == concept_id for child in h.children):
                    parent_id = h.parent.synset_id
                    break
            if parent_id:
                child_activations.setdefault(parent_id, {})[concept_id] = pos_neg_dict

    # Metric triangulation: test multiple geometry sources
    geometry_sources = ["from_unembedding"]  # Always include the main one
    triangulation_results = {}

    # Try alternative geometry sources if model and tokenizer are provided
    # Skip for very large vocabularies to avoid OOM issues
    vocab_size = model.config.vocab_size if model is not None else 0
    skip_alt_geometries = vocab_size > 100000  # Skip for vocabs > 100K
    
    if model is not None and tokenizer is not None and not skip_alt_geometries:
        # Create dummy prompts for geometry validation
        test_prompts = [
            "The capital of France is Paris.",
            "Water boils at 100 degrees Celsius.",
            "The sun is a star in our solar system.",
            "Dogs are domestic animals.",
            "Mathematics is the study of numbers.",
        ]

        from polytope_hsae.geometry import CausalGeometry
        from polytope_hsae.estimators import ConceptVectorEstimator, LDAEstimator

        # Geometry from residual activations
        try:
            acts_list = []
            for data in parent_activations.values():
                acts_list.append(data["pos"])
                acts_list.append(data["neg"])
            for pdata in child_activations.values():
                for data in pdata.values():
                    acts_list.append(data["pos"])
                    acts_list.append(data["neg"])
            stacked = torch.cat(acts_list, dim=0)
            dummy_U = torch.zeros(1, stacked.shape[1])
            act_geometry = CausalGeometry(
                dummy_U,
                whitening="residual_acts",
                residual_acts=stacked,
                shrinkage=config["geometry"]["shrinkage"],
            )
            geometry_sources.append("from_activations")
            lda = LDAEstimator(
                shrinkage=config["geometry"]["estimator"]["lda_shrinkage"],
                min_vocab_count=config["geometry"]["estimator"]["min_vocab_count"],
                class_balance=True,
            )
            estimator_act = ConceptVectorEstimator(lda, act_geometry)
            pv_act = estimator_act.estimate_parent_vectors(parent_activations)
            cv_act, cd_act = estimator_act.estimate_child_deltas(pv_act, child_activations)
            act_validator = GeometryValidator(act_geometry)
            act_ortho = act_validator.test_hierarchical_orthogonality(
                pv_act,
                cv_act,
                cd_act,
                threshold_degrees=config["eval"]["targets"]["median_angle_deg"],
            )
            triangulation_results["from_activations"] = {
                "median_angle_deg": act_ortho["median_angle_deg"],
                "fraction_above_80deg": act_ortho["fraction_above_threshold"],
                "passes_threshold": act_ortho["median_angle_deg"]
                >= config["eval"]["targets"]["median_angle_deg"],
            }
            logger.info(
                f"Activations geometry: median angle = {act_ortho['median_angle_deg']:.1f}°"
            )
        except Exception as e:
            logger.warning(f"Could not test activations geometry: {e}")

        # Test from_logit_cov geometry
        try:
            logit_geometry = CausalGeometry.from_logit_cov(
                model, tokenizer, test_prompts, shrinkage=config["geometry"]["shrinkage"]
            )
            geometry_sources.append("from_logit_cov")

            logit_validator = GeometryValidator(logit_geometry)
            logit_ortho = logit_validator.test_hierarchical_orthogonality(
                parent_vectors,
                child_vectors,
                child_deltas,
                threshold_degrees=config["eval"]["targets"]["median_angle_deg"],
            )
            triangulation_results["from_logit_cov"] = {
                "median_angle_deg": logit_ortho["median_angle_deg"],
                "fraction_above_80deg": logit_ortho["fraction_above_threshold"],
                "passes_threshold": logit_ortho["median_angle_deg"]
                >= config["eval"]["targets"]["median_angle_deg"],
            }
            logger.info(
                f"Logit-cov geometry: median angle = {logit_ortho['median_angle_deg']:.1f}°"
            )
        except Exception as e:
            logger.warning(f"Could not test logit-cov geometry: {e}")

        # Test fisher_like geometry
        try:
            fisher_geometry = CausalGeometry.fisher_like(
                model, tokenizer, test_prompts, shrinkage=config["geometry"]["shrinkage"]
            )
            geometry_sources.append("fisher_like")
            fisher_validator = GeometryValidator(fisher_geometry)
            fisher_ortho = fisher_validator.test_hierarchical_orthogonality(
                parent_vectors,
                child_vectors,
                child_deltas,
                threshold_degrees=config["eval"]["targets"]["median_angle_deg"],
            )
            triangulation_results["fisher_like"] = {
                "median_angle_deg": fisher_ortho["median_angle_deg"],
                "fraction_above_80deg": fisher_ortho["fraction_above_threshold"],
                "passes_threshold": fisher_ortho["median_angle_deg"]
                >= config["eval"]["targets"]["median_angle_deg"],
            }
            logger.info(
                f"Fisher-like geometry: median angle = {fisher_ortho['median_angle_deg']:.1f}°"
            )
        except Exception as e:
            logger.warning(f"Could not test Fisher-like geometry: {e}")
    elif skip_alt_geometries:
        logger.warning(
            f"Skipping alternative geometries due to large vocabulary (size={vocab_size:,}) to avoid OOM"
        )
    else:
        logger.warning("Model or tokenizer not provided; skipping alternative geometries")

    # Test hierarchical orthogonality
    orthogonality_results = validator.test_hierarchical_orthogonality(
        parent_vectors,
        child_vectors,
        child_deltas,
        threshold_degrees=config["eval"]["targets"]["median_angle_deg"],
    )

    # Identity and whitening diagnostics
    identity_test = {}
    try:
        # Build a small fp32 batch from captured activations
        sample_list = []
        for d in parent_activations.values():
            sample_list.append(d["pos"][:16])
            sample_list.append(d["neg"][:16])
            if len(sample_list) >= 8:
                break
        for pdata in child_activations.values():
            for d in pdata.values():
                sample_list.append(d["pos"][:16])
                sample_list.append(d["neg"][:16])
                if len(sample_list) >= 16:
                    break
            if len(sample_list) >= 16:
                break
        if sample_list:
            X = torch.cat(sample_list, dim=0).to(torch.float32)
            identity_test = geometry.test_linear_identity(X)
    except Exception as e:
        logger.warning(f"Identity test failed: {e}")

    try:
        whiten_stats = geometry.whitening_invariant_stats()
    except Exception as e:
        logger.warning(f"Whitening invariant stats failed: {e}")
        whiten_stats = {}

    # Handle ragged children per parent: compute per-parent, then aggregate
    all_angles = []
    for pid, pvec in parent_vectors.items():
        deltas = child_deltas.get(pid, {})
        for delt in deltas.values():
            if torch.norm(delt) > 1e-6:
                ang = geometry.causal_angle(pvec, delt)
                all_angles.append(torch.rad2deg(ang).item())

    angle_stats = {
        "median_angle_deg": torch.tensor(np.median(all_angles))
        if all_angles
        else torch.tensor(float("nan")),
        "mean_angle_deg": torch.tensor(np.mean(all_angles))
        if all_angles
        else torch.tensor(float("nan")),
        "q25_angle_deg": torch.tensor(np.percentile(all_angles, 25))
        if all_angles
        else torch.tensor(float("nan")),
        "q75_angle_deg": torch.tensor(np.percentile(all_angles, 75))
        if all_angles
        else torch.tensor(float("nan")),
    }

    # Run control experiments using preloaded unembedding matrix
    # Reduce shuffles for large vocabularies to save memory
    n_shuffles = 10 if vocab_size > 100000 else 50
    control_results = validator.run_control_experiments(
        parent_vectors,
        child_vectors,
        child_deltas,
        unembedding_matrix,
        n_shuffles=n_shuffles,
    )

    # Validate orthogonality with estimator function
    orthogonality_validation = validate_orthogonality(
        parent_vectors, child_deltas, geometry
    )

    # Add standard geometry results to triangulation
    triangulation_results["from_unembedding"] = {
        "median_angle_deg": orthogonality_results["median_angle_deg"],
        "fraction_above_80deg": orthogonality_results["fraction_above_threshold"],
        "passes_threshold": orthogonality_results["median_angle_deg"] >= config["eval"]["targets"]["median_angle_deg"]
    }

    # Quick sanity plot of angle distributions
    try:
        import matplotlib.pyplot as plt

        pair_parents = []
        pair_deltas = []
        for pid, deltas in child_deltas.items():
            if pid not in parent_vectors:
                continue
            for delt in deltas.values():
                if torch.norm(delt) > 1e-6:
                    pair_parents.append(parent_vectors[pid])
                    pair_deltas.append(delt)
        if pair_parents:
            parent_stack = torch.stack(pair_parents)
            delta_stack = torch.stack(pair_deltas)
            rand_parents = parent_stack[torch.randperm(len(parent_stack))]
            rand_deltas = delta_stack[torch.randperm(len(delta_stack))]
            label_perm_angles = torch.rad2deg(
                geometry.causal_angle(rand_parents, rand_deltas)
            ).cpu().numpy()
            identity_geometry = geometry.__class__(
                unembedding_matrix, whitening="identity", shrinkage=geometry.shrinkage
            )
            euclid_angles = torch.rad2deg(
                identity_geometry.causal_angle(parent_stack, delta_stack)
            ).cpu().numpy()
            plt.figure()
            plt.hist(
                orthogonality_results["angles_deg"], bins=30, density=True, alpha=0.5, label="true"
            )
            plt.hist(
                label_perm_angles, bins=30, density=True, alpha=0.5, label="label-perm"
            )
            plt.hist(
                euclid_angles, bins=30, density=True, alpha=0.5, label="euclid"
            )
            plt.legend()
            plt.xlabel("θ (deg)")
            plt.ylabel("pdf")
            plt.savefig(exp_dir / "angle_sanity_plot.png")
            plt.close()
    except Exception as e:
        logger.warning(f"Could not create sanity plot: {e}")
    
    # Compute triangulation consensus: need ≥2 geometries passing and medians close
    passing_medians = [
        res["median_angle_deg"]
        for res in triangulation_results.values()
        if res["median_angle_deg"] >= config["eval"]["targets"]["median_angle_deg"]
    ]
    passing_geometries = len(passing_medians)
    median_spread = max(passing_medians) - min(passing_medians) if passing_medians else float("inf")
    triangulation_passes = passing_geometries >= 2 and median_spread <= 15.0

    logger.info(
        f"Metric triangulation: {passing_geometries}/{len(triangulation_results)} geometries pass"
    )
    logger.info(
        f"Median spread among passing geometries: {median_spread:.1f}°"
    )
    logger.info(f"Triangulation consensus: {'PASS' if triangulation_passes else 'FAIL'}")

    # Compile results
    validation_results = {
        "orthogonality_test": {
            "median_angle_deg": orthogonality_results["median_angle_deg"],
            "fraction_above_80deg": orthogonality_results["fraction_above_threshold"],
            "fraction_above_85deg": orthogonality_results["fraction_above_85deg"],
            "passes_threshold": orthogonality_results["median_angle_deg"]
            >= config["eval"]["targets"]["median_angle_deg"],
        },
        "identity_test": identity_test,
        "whitening_diagnostics": whiten_stats,
        "angle_statistics": {
            "median_angle_deg": angle_stats["median_angle_deg"].item(),
            "mean_angle_deg": angle_stats["mean_angle_deg"].item(),
            "q25_angle_deg": angle_stats["q25_angle_deg"].item(),
            "q75_angle_deg": angle_stats["q75_angle_deg"].item(),
        },
        "metric_triangulation": {
            "geometry_sources": geometry_sources,
            "results_per_geometry": triangulation_results,
            "passing_geometries": passing_geometries,
            "total_geometries": len(triangulation_results),
            "triangulation_passes": triangulation_passes
        },
        "orthogonality_validation": orthogonality_validation,
        "control_experiments": control_results,
        "passes_validation": (
            triangulation_passes and
            orthogonality_results["fraction_above_threshold"] >= 0.5
        ),
    }

    # Save validation results
    with open(exp_dir / "validation_results.json", "w") as f:
        json.dump(validation_results, f, indent=2, default=str)

    # Log key results
    logger.info(
        f"Median angle: {validation_results['orthogonality_test']['median_angle_deg']:.1f}°"
    )
    # Check what keys are actually available
    ortho_test = validation_results.get("orthogonality_test", {})
    if "fraction_above_threshold" in ortho_test:
        fraction_key = "fraction_above_threshold"
    elif "fraction_above_80deg" in ortho_test:
        fraction_key = "fraction_above_80deg"
    else:
        fraction_key = list(ortho_test.keys())[0] if ortho_test else "unknown"

    logger.info(
        f"Fraction above threshold: {ortho_test.get(fraction_key, 'N/A'):.3f}"
        if isinstance(ortho_test.get(fraction_key), (int, float))
        else f"Available keys: {list(ortho_test.keys())}"
    )
    logger.info(f"Validation passed: {validation_results['passes_validation']}")

    return validation_results


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Teacher Vector Extraction")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run with minimal data for testing"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set all seeds for reproducibility
    _set_all_seeds(config.get("run", {}).get("seed", 123))

    # Override device if specified
    if args.device:
        config["run"]["device"] = args.device

    # Handle dry run
    if args.dry_run:
        logger.info("Running in DRY RUN mode with minimal data")
        config["concepts"]["parents"] = 5  # Minimal for testing
        config["concepts"]["max_children_per_parent"] = 2
        config["concepts"]["prompts_per_concept"] = 4
        config["data"]["token_budget"] = 1000  # Very small for testing
        config["model"][
            "name"
        ] = "distilgpt2"  # Use a small, available model for testing

    # Set up experiment
    exp_dir = setup_experiment(config)

    # Initialize W&B
    wandb.init(
        project=config.get("wandb_project", "polytope-hsae"),
        name=f"phase1_teacher_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
        tags=["phase1", "teacher-extraction", "geometric-validation"],
        notes=f"Teacher vector extraction and geometric validation: {config.get('run', {}).get('notes', 'Phase 1 extraction')}",
    )

    # Load or create concepts
    hierarchies = load_or_create_concepts(config, exp_dir)

    # Capture activations
    activations = capture_activations(config, hierarchies, exp_dir)

    # Load transformer model once and derive unembedding matrix
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": config["run"]["device"]},
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if hasattr(model, "lm_head") and isinstance(model.lm_head, torch.nn.Linear):
        unembedding_matrix = model.lm_head.weight.data
    else:
        try:
            unembedding_matrix = model.get_output_embeddings().weight.data
        except Exception:
            unembedding_matrix = model.get_input_embeddings().weight.data

    # Extract teacher vectors
    parent_vectors, child_vectors, child_deltas, projectors, geometry = extract_teacher_vectors(
        config, hierarchies, activations, exp_dir, unembedding_matrix
    )

    # Log teacher extraction results to W&B
    teacher_stats = {
        "teacher/n_parents": len(parent_vectors),
        "teacher/n_children_total": sum(len(deltas) for deltas in child_deltas.values()),
        "teacher/avg_children_per_parent": sum(len(deltas) for deltas in child_deltas.values()) / len(parent_vectors) if parent_vectors else 0,
        "teacher/model_dim": parent_vectors[list(parent_vectors.keys())[0]].shape[0] if parent_vectors else 0,
        "teacher/subspace_dim": config["hsae"]["subspace_dim"],
    }
    wandb.log(teacher_stats)

    # Validate geometry
    validation_results = validate_geometry(
        config,
        parent_vectors,
        child_vectors,
        child_deltas,
        geometry,
        exp_dir,
        model,
        tokenizer,
        unembedding_matrix,
        activations,
        hierarchies,
    )

    # Log comprehensive validation results to W&B
    validation_metrics = {
        # Orthogonality test results
        "validation/median_angle_deg": validation_results["orthogonality_test"]["median_angle_deg"],
        "validation/fraction_above_80deg": validation_results["orthogonality_test"]["fraction_above_80deg"],
        "validation/fraction_above_85deg": validation_results["orthogonality_test"]["fraction_above_85deg"],
        "validation/passes_orthogonality_threshold": validation_results["orthogonality_test"]["passes_threshold"],
        
        # Angle statistics
        "validation/mean_angle_deg": validation_results["angle_statistics"]["mean_angle_deg"],
        "validation/q25_angle_deg": validation_results["angle_statistics"]["q25_angle_deg"],
        "validation/q75_angle_deg": validation_results["angle_statistics"]["q75_angle_deg"],
        
        # Metric triangulation
        "validation/triangulation_passes": validation_results["metric_triangulation"]["triangulation_passes"],
        "validation/passing_geometries": validation_results["metric_triangulation"]["passing_geometries"],
        "validation/total_geometries": validation_results["metric_triangulation"]["total_geometries"],
        
        # Overall validation
        "validation/passes_validation": validation_results["passes_validation"],
    }
    
    # Log individual geometry results if available
    for geometry_name, results in validation_results["metric_triangulation"]["results_per_geometry"].items():
        validation_metrics[f"triangulation/{geometry_name}/median_angle_deg"] = results["median_angle_deg"]
        validation_metrics[f"triangulation/{geometry_name}/fraction_above_80deg"] = results["fraction_above_80deg"]
        validation_metrics[f"triangulation/{geometry_name}/passes_threshold"] = results["passes_threshold"]
    
    # Log control experiment results if available
    if "control_experiments" in validation_results and "summary" in validation_results["control_experiments"]:
        controls = validation_results["control_experiments"]["summary"]["controls"]
        for control_name, control_data in controls.items():
            validation_metrics[f"control/{control_name}/median_angle"] = control_data["median_angle"]
            validation_metrics[f"control/{control_name}/effect_size"] = control_data["effect_size_vs_original"]
    
    wandb.log(validation_metrics)

    # Final report
    logger.info("Phase 1 completed!")
    logger.info(f"Results saved to: {exp_dir}")

    if validation_results["passes_validation"]:
        logger.info("✅ Geometric validation PASSED - ready for Phase 2")
    else:
        logger.warning(
            "❌ Geometric validation FAILED - check results before proceeding"
        )

    # Cleanup
    del model
    if tokenizer is not None:
        del tokenizer
    torch.cuda.empty_cache()

    # Finish W&B run
    wandb.finish()

    return validation_results["passes_validation"]


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
