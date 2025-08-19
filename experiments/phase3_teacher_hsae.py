#!/usr/bin/env python3
"""
Phase 3: Teacher-Initialized H-SAE Training

This script implements Phase 3 of the V2 focused experiment:
- Initialize H-SAE with teacher vectors from Phase 1
- Two-stage training: freeze decoder (1,500 steps) + adapt (8,500 steps)
- Compare against Phase 2 baseline on purity, leakage, and steering metrics

Estimated runtime: 12-14 hours
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

from polytope_hsae.activations import ActivationCapture
from polytope_hsae.estimators import ConceptVectorEstimator
from polytope_hsae.geometry import CausalGeometry
from polytope_hsae.metrics import (compute_comprehensive_metrics,
                                   log_metrics_summary)
# Project imports
from polytope_hsae.models import (HierarchicalSAE, HSAEConfig,
                                  create_teacher_initialized_hsae)
from polytope_hsae.training import HSAETrainer, create_data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_experiment(config):
    """Set up experiment directories and logging."""
    exp_dir = Path(config["logging"]["save_dir"]) / config["logging"]["phase_3_log"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = exp_dir / f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info(f"Phase 3 experiment directory: {exp_dir}")
    return exp_dir


def load_teacher_vectors(config):
    """Load teacher vectors from Phase 1."""
    phase1_dir = Path(config["logging"]["save_dir"]) / config["logging"]["phase_1_log"]
    teacher_file = phase1_dir / "teacher_vectors.json"

    if not teacher_file.exists():
        raise FileNotFoundError(f"Teacher vectors not found: {teacher_file}")

    logger.info(f"Loading teacher vectors from {teacher_file}")

    with open(teacher_file, "r") as f:
        teacher_data = json.load(f)

    # Convert back to tensors
    parent_vectors = {
        k: torch.tensor(v) for k, v in teacher_data["parent_vectors"].items()
    }
    child_deltas = {
        parent_id: {child_id: torch.tensor(delta) for child_id, delta in deltas.items()}
        for parent_id, deltas in teacher_data["child_deltas"].items()
    }
    projectors = {
        parent_id: (torch.tensor(proj["down"]), torch.tensor(proj["up"]))
        for parent_id, proj in teacher_data["projectors"].items()
    }

    logger.info(
        f"Loaded {len(parent_vectors)} parent vectors and {sum(len(deltas) for deltas in child_deltas.values())} child deltas"
    )

    return parent_vectors, child_deltas, projectors


def load_activations(config, exp_dir):
    """Load activations from Phase 1."""
    activations_file = (
        Path(config["logging"]["save_dir"])
        / config["logging"]["phase_1_log"]
        / "activations.h5"
    )

    if not activations_file.exists():
        raise FileNotFoundError(f"Activations not found: {activations_file}")

    logger.info(f"Loading activations from {activations_file}")

    activation_capture = ActivationCapture(None)
    activations = activation_capture.load_hierarchical_activations(
        str(activations_file)
    )

    # Convert to training format
    all_activations = []
    for concept_id, pos_neg_dict in activations.items():
        all_activations.append(pos_neg_dict["pos"])
        all_activations.append(pos_neg_dict["neg"])

    combined_activations = torch.cat(all_activations, dim=0)
    logger.info(f"Loaded {combined_activations.shape[0]} activation samples")

    return combined_activations


def create_teacher_hsae(config, input_dim: int, parent_vectors, projectors, geometry):
    """Create teacher-initialized H-SAE."""
    hsae_config = HSAEConfig(
        input_dim=input_dim,
        n_parents=config["hsae"]["m_top"],
        topk_parent=config["hsae"]["topk_parent"],
        subspace_dim=config["hsae"]["subspace_dim"],
        n_children_per_parent=config["hsae"]["m_low"],
        topk_child=config["hsae"]["topk_child"],
        l1_parent=config["hsae"]["l1_parent"],
        l1_child=config["hsae"]["l1_child"],
        biorth_lambda=config["hsae"]["biorth_lambda"],
        causal_ortho_lambda=config["hsae"]["causal_ortho_lambda"],
        router_temp_start=config["hsae"]["router_temp"]["start"],
        router_temp_end=config["hsae"]["router_temp"]["end"],
        top_level_beta=config["hsae"]["top_level_beta"],
        # Use config settings or defaults
        use_tied_decoders_parent=config["hsae"].get("use_tied_decoders_parent", False),
        use_tied_decoders_child=config["hsae"].get("use_tied_decoders_child", False),
        tie_projectors=config["hsae"].get("tie_projectors", False),
        use_decoder_bias=config["hsae"].get("use_decoder_bias", True),
        use_offdiag_biorth=config["hsae"].get("use_offdiag_biorth", False),
    )

    # Convert parent vectors to tensor (take first N parents)
    parent_vector_list = list(parent_vectors.values())[: hsae_config.n_parents]
    parent_tensor = torch.stack(parent_vector_list)

    # Convert projectors to list (take first N)
    projector_list = list(projectors.values())[: hsae_config.n_parents]

    model = create_teacher_initialized_hsae(
        config=hsae_config,
        parent_vectors=parent_tensor,
        child_projectors=projector_list,
        geometry=geometry,
    )

    logger.info(
        f"Created teacher-initialized H-SAE with {sum(p.numel() for p in model.parameters())} parameters"
    )

    return model


def load_geometry(config):
    """Load geometry from Phase 1 for causal orthogonality."""
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(config["model"]["name"])
        if hasattr(model, "lm_head"):
            unembedding_matrix = model.lm_head.weight.data
        else:
            unembedding_matrix = model.get_output_embeddings().weight.data
    except Exception as e:
        logger.warning(f"Could not load geometry: {e}")
        return None

    geometry = CausalGeometry(
        unembedding_matrix, shrinkage=config["geometry"]["shrinkage"]
    )

    return geometry


def train_teacher_hsae(model, activations, config, exp_dir, geometry):
    """Train teacher-initialized H-SAE with two-stage schedule."""
    logger.info("Starting teacher-initialized H-SAE training")

    # Create data loaders
    device = torch.device(config["run"]["device"])
    model = model.to(device)

    # Unit normalize if specified
    if config.get("data", {}).get("unit_norm", False):
        activations = torch.nn.functional.normalize(activations, dim=-1)

    # Split data
    n_samples = activations.shape[0]
    n_train = int(0.8 * n_samples)
    n_val = n_samples - n_train

    train_activations = activations[:n_train]
    val_activations = activations[n_train:]

    train_loader = create_data_loader(
        train_activations,
        batch_size=config["training"]["batch_size_acts"],
        shuffle=True,
    )
    val_loader = create_data_loader(
        val_activations, batch_size=config["training"]["batch_size_acts"], shuffle=False
    )

    # Initialize trainer
    trainer = HSAETrainer(
        model=model,
        config=config,
        geometry=geometry,  # Provide geometry for causal orthogonality
        use_wandb=True,
    )

    # Initialize W&B
    if trainer.use_wandb:
        wandb.init(
            project=config.get("wandb_project", "polytope-hsae"),
            name=f"phase3_teacher_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            tags=["phase3", "teacher-init", "hsae"],
        )

        # Log model architecture
        wandb.log(
            {
                "model/n_parameters": sum(p.numel() for p in model.parameters()),
                "model/n_parents": model.config.n_parents,
                "model/subspace_dim": model.config.subspace_dim,
                "model/topk_parent": model.config.topk_parent,
                "model/teacher_init": True,
            }
        )

    # Two-stage training
    stabilize_steps = config["training"]["teacher_init"]["stage_A"]["steps"]
    adapt_steps = config["training"]["teacher_init"]["stage_B"]["steps"]

    history = trainer.train_teacher_init(
        train_loader=train_loader,
        val_loader=val_loader,
        stabilize_steps=stabilize_steps,
        adapt_steps=adapt_steps,
    )

    # Final evaluation
    logger.info("Running final evaluation")
    final_metrics = compute_comprehensive_metrics(
        model=model, data_loader=val_loader, device=str(device)
    )

    log_metrics_summary(final_metrics, logger)

    # Save results
    results = {
        "training_history": history,
        "final_metrics": final_metrics,
        "config": config,
        "model_info": {
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "architecture": "teacher_init_hsae",
            "stabilize_steps": stabilize_steps,
            "adapt_steps": adapt_steps,
        },
    }

    results_file = exp_dir / "teacher_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save model
    model_file = exp_dir / "teacher_hsae_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config,
            "training_step": stabilize_steps + adapt_steps,
            "teacher_initialized": True,
        },
        model_file,
    )

    if trainer.use_wandb:
        wandb.log({"final_metrics": final_metrics})
        wandb.finish()

    logger.info(f"Teacher-initialized training completed. Results saved to {exp_dir}")
    return results


def compare_with_baseline(teacher_results, config):
    """Compare teacher-initialized results with baseline from Phase 2."""
    baseline_file = (
        Path(config["logging"]["save_dir"])
        / config["logging"]["phase_2_log"]
        / "baseline_results.json"
    )

    if not baseline_file.exists():
        logger.warning("Baseline results not found - cannot compare")
        return {}

    with open(baseline_file, "r") as f:
        baseline_results = json.load(f)

    # Compare key metrics
    teacher_metrics = teacher_results["final_metrics"]
    baseline_metrics = baseline_results["final_metrics"]

    comparison = {}

    # Reconstruction metrics (should be similar)
    comparison["reconstruction"] = {
        "teacher_1_EV": teacher_metrics.get("1-EV", 1.0),
        "baseline_1_EV": baseline_metrics.get("1-EV", 1.0),
        "ev_difference": baseline_metrics.get("1-EV", 1.0)
        - teacher_metrics.get("1-EV", 1.0),
        "teacher_1_CE": teacher_metrics.get("1-CE", 10.0),
        "baseline_1_CE": baseline_metrics.get("1-CE", 10.0),
        "ce_difference": baseline_metrics.get("1-CE", 10.0)
        - teacher_metrics.get("1-CE", 10.0),
    }

    # Purity metrics (teacher should be better)
    comparison["purity"] = {
        "teacher_parent_purity": teacher_metrics.get("parent_purity_mean", 0.0),
        "baseline_parent_purity": baseline_metrics.get("parent_purity_mean", 0.0),
        "purity_improvement": teacher_metrics.get("parent_purity_mean", 0.0)
        - baseline_metrics.get("parent_purity_mean", 0.0),
        "teacher_parent_auc": teacher_metrics.get("parent_auc_mean", 0.5),
        "baseline_parent_auc": baseline_metrics.get("parent_auc_mean", 0.5),
        "auc_improvement": teacher_metrics.get("parent_auc_mean", 0.5)
        - baseline_metrics.get("parent_auc_mean", 0.5),
    }

    # Leakage metrics (teacher should have less leakage)
    comparison["leakage"] = {
        "teacher_parent_leakage": teacher_metrics.get("parent_leakage_mean", 10.0),
        "baseline_parent_leakage": baseline_metrics.get("parent_leakage_mean", 10.0),
        "leakage_reduction": (
            baseline_metrics.get("parent_leakage_mean", 10.0)
            - teacher_metrics.get("parent_leakage_mean", 10.0)
        )
        / baseline_metrics.get("parent_leakage_mean", 10.0),
        "teacher_child_leakage": teacher_metrics.get("child_leakage_mean", 10.0),
        "baseline_child_leakage": baseline_metrics.get("child_leakage_mean", 10.0),
        "child_leakage_reduction": (
            baseline_metrics.get("child_leakage_mean", 10.0)
            - teacher_metrics.get("child_leakage_mean", 10.0)
        )
        / baseline_metrics.get("child_leakage_mean", 10.0),
    }

    # Check if targets are met
    targets_met = {
        "purity_improvement_10pp": comparison["purity"]["purity_improvement"] >= 0.10,
        "leakage_reduction_20pct": comparison["leakage"]["leakage_reduction"] >= 0.20,
        "reconstruction_parity": abs(comparison["reconstruction"]["ev_difference"])
        <= 0.05,
    }

    comparison["targets_met"] = targets_met
    comparison["overall_success"] = all(targets_met.values())

    logger.info("=" * 50)
    logger.info("TEACHER vs BASELINE COMPARISON")
    logger.info("=" * 50)
    logger.info(
        f"Purity improvement: {comparison['purity']['purity_improvement']:.4f} (target: ≥0.10)"
    )
    logger.info(
        f"Leakage reduction: {comparison['leakage']['leakage_reduction']:.4f} (target: ≥0.20)"
    )
    logger.info(
        f"EV difference: {comparison['reconstruction']['ev_difference']:.4f} (target: ≤0.05)"
    )
    logger.info(f"Overall success: {comparison['overall_success']}")
    logger.info("=" * 50)

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Teacher-Initialized H-SAE Training"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override device if specified
    if args.device:
        config["run"]["device"] = args.device

    # Set up experiment
    exp_dir = setup_experiment(config)

    # Load teacher vectors from Phase 1
    parent_vectors, child_deltas, projectors = load_teacher_vectors(config)

    # Load geometry for causal orthogonality
    geometry = load_geometry(config)

    # Load activations from Phase 1
    activations = load_activations(config, exp_dir)

    # Create teacher-initialized H-SAE
    model = create_teacher_hsae(
        config, activations.shape[1], parent_vectors, projectors, geometry
    )

    # Train teacher-initialized H-SAE
    results = train_teacher_hsae(model, activations, config, exp_dir, geometry)

    # Compare with baseline
    comparison = compare_with_baseline(results, config)

    # Save comparison
    if comparison:
        comparison_file = exp_dir / "baseline_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)

    # Final report
    logger.info("Phase 3 completed!")
    logger.info(f"Results saved to: {exp_dir}")

    success = True
    if comparison:
        success = comparison["overall_success"]
        if success:
            logger.info("✅ Teacher initialization SUCCESSFUL - targets met!")
        else:
            logger.warning(
                "⚠️  Teacher initialization shows mixed results - check targets"
            )

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
