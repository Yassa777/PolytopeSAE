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
                                   compute_explained_variance,
                                   log_metrics_summary)
# Project imports
from polytope_hsae.models import (HierarchicalSAE, HSAEConfig,
                                  create_teacher_initialized_hsae)
from polytope_hsae.sanity_checker import TrainingSanityChecker
from polytope_hsae.training import HSAETrainer, TrainingRestartException, create_data_loader

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

    activations = ActivationCapture.load_hierarchical_activations(str(activations_file))

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


def train_teacher_hsae_with_sanity_checks(model, activations, config, exp_dir, geometry):
    """Train teacher-initialized H-SAE with automated sanity checking."""
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        logger.info(f"🚀 Starting teacher-initialized H-SAE training (attempt {attempt}/{max_attempts})")
        
        try:
            results = _train_teacher_hsae_single_attempt(model, activations, config, exp_dir, geometry, attempt)
            logger.info("✅ Training completed successfully!")
            return results
            
        except TrainingRestartException as e:
            logger.info(f"🔄 Training restart requested: {e}")
            if attempt >= max_attempts:
                logger.error(f"❌ Max attempts ({max_attempts}) reached - stopping")
                raise
            
            # Recreate model with updated config and teacher initialization
            logger.info("🔧 Recreating model with updated configuration...")
            parent_vectors, child_deltas, projectors = load_teacher_vectors(config)
            model = create_teacher_hsae(config, activations.shape[1], parent_vectors, projectors, geometry)
            continue
            
        except Exception as e:
            logger.error(f"❌ Training failed with error: {e}")
            raise
    
    raise RuntimeError("Training failed after all attempts")


def _train_teacher_hsae_single_attempt(model, activations, config, exp_dir, geometry, attempt_num):
    """Single training attempt with sanity checking."""
    logger.info(f"Starting teacher-initialized H-SAE training (attempt {attempt_num})")

    # Create data loaders
    device = torch.device(config["run"]["device"])
    model = model.to(device)

    # Unit normalize if specified
    if config.get("data", {}).get("unit_norm", False):
        activations = torch.nn.functional.normalize(activations, dim=-1)

    # Split data (70/15/15 as per config spec)
    n_samples = activations.shape[0]
    n_train = int(0.70 * n_samples)
    n_val = int(0.15 * n_samples)
    n_test = n_samples - n_train - n_val

    train_activations = activations[:n_train]
    val_activations = activations[n_train:n_train + n_val]
    test_activations = activations[n_train + n_val:]
    
    logger.info(f"Data split: train={n_train}, val={n_val}, test={n_test} samples")

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
    
    # Initialize sanity checker
    sanity_checker = TrainingSanityChecker(config, model, trainer, exp_dir)
    logger.info("🔍 Sanity checker initialized")

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

    # Two-stage training with sanity checking
    stabilize_steps = config["training"]["teacher_init"]["stage_A"]["steps"]
    adapt_steps = config["training"]["teacher_init"]["stage_B"]["steps"]

    history = trainer.train_teacher_init(
        train_loader=train_loader,
        val_loader=val_loader,
        stabilize_steps=stabilize_steps,
        adapt_steps=adapt_steps,
        sanity_checker=sanity_checker,
    )

    # Final evaluation
    logger.info("Running final evaluation")
    
    # Run identity test with geometry (critical validation)
    logger.info("Running linear identity test with causal geometry")
    val_sample = next(iter(val_loader))
    if isinstance(val_sample, (list, tuple)):
        val_sample = val_sample[0]
    val_sample = val_sample[:100].to(device)  # Use first 100 samples
    
    # Test whiten → unwhiten identity
    identity_results = geometry.test_linear_identity(val_sample)
    logger.info(f"🔍 Identity Test Results:")
    for key, value in identity_results.items():
        logger.info(f"  {key}: {value}")
    
    if not identity_results["identity_ok"]:
        logger.warning("⚠️  Linear identity test FAILED - geometry pipeline may have issues!")
    else:
        logger.info("✅ Linear identity test PASSED - geometry pipeline working correctly")
    
    try:
        final_metrics = compute_comprehensive_metrics(
            model=model, data_loader=val_loader, device=str(device)
        )
        logger.info("✅ Comprehensive metrics computed successfully")
        
        # Log key metrics with detailed EV breakdown
        final_ev = 1.0 - final_metrics.get('1-EV', 1.0)
        logger.info(f"📊 FINAL METRICS BREAKDOWN:")
        logger.info(f"  EV (Explained Variance): {final_ev:.4f}")
        logger.info(f"  CE proxy: {final_metrics.get('1-CE', 0.0):.4f}")
        logger.info(f"  Parent sparsity: {final_metrics.get('parent_sparsity', 0.0):.4f}")
        logger.info(f"  Child sparsity: {final_metrics.get('child_sparsity', 0.0):.4f}")
        
        # Compute both standard and causal EV for comprehensive analysis
        logger.info("🔍 Computing dual EV breakdown (standard + causal):")
        model.eval()
        with torch.no_grad():
            val_sample = next(iter(val_loader))
            if isinstance(val_sample, (list, tuple)):
                val_sample = val_sample[0]
            val_sample = val_sample.to(device)
            x_hat, _, _ = model(val_sample)
            
            # Use dual EV computation with geometry for both metrics
            from polytope_hsae.metrics import compute_dual_explained_variance
            ev_results = compute_dual_explained_variance(
                val_sample, x_hat, geometry=geometry, print_components=True
            )
            logger.info(f"🎯 PRIMARY - Standard EV (flat SAE comparable): {ev_results['standard_ev']:.6f}")
            logger.info(f"📐 SECONDARY - Causal EV (Mahalanobis): {ev_results['causal_ev']:.6f}")
        
        log_metrics_summary(final_metrics, logger)
    except Exception as e:
        logger.error(f"❌ Failed to compute comprehensive metrics: {e}")
        logger.error("Computing basic metrics instead...")
        
        # Fallback: compute basic metrics manually
        model.eval()
        total_ev = 0
        total_ce = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    batch_data = batch[0].to(device)
                else:
                    batch_data = batch.to(device)
                
                x_hat, (parent_codes, child_codes), model_metrics = model(batch_data)
                
                from polytope_hsae.metrics import compute_explained_variance, compute_cross_entropy_proxy
                ev = compute_explained_variance(batch_data, x_hat)
                ce = compute_cross_entropy_proxy(batch_data, x_hat)
                
                total_ev += ev
                total_ce += ce
                n_batches += 1
        
        final_metrics = {
            "1-EV": 1.0 - (total_ev / n_batches),
            "1-CE": total_ce / n_batches,
            "parent_sparsity": torch.mean((parent_codes > 0).float()).item(),
            "child_sparsity": torch.mean((child_codes > 0).float()).item(),
        }
        
        logger.info(f"Fallback EV: {total_ev / n_batches:.4f}")
        logger.info(f"Fallback CE: {total_ce / n_batches:.4f}")

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
        # Log comprehensive final metrics to W&B
        wandb_metrics = {}
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                wandb_metrics[f"final/{key}"] = value
        
        # Also log the key summary metrics
        ev = 1.0 - final_metrics.get("1-EV", 1.0)
        wandb_metrics.update({
            "final/explained_variance": ev,
            "final/cross_entropy_proxy": final_metrics.get("1-CE", 0.0),
            "final/success": 1.0 if ev > 0.5 else 0.0,  # Success flag
            "final/teacher_initialized": 1.0,  # Flag for teacher init
        })
        
        wandb.log(wandb_metrics)
        logger.info(f"📊 Logged {len(wandb_metrics)} final metrics to W&B")
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

    # Set seeds for reproducibility
    seed = config.get("run", {}).get("seed", 123)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seeds to {seed}")

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
    results = train_teacher_hsae_with_sanity_checks(model, activations, config, exp_dir, geometry)

    # Compare with baseline
    comparison = compare_with_baseline(results, config)

    # Save comparison
    if comparison:
        comparison_file = exp_dir / "baseline_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)

    # Final report (no hard success gate based on EV threshold)
    final_ev = 1.0 - results["final_metrics"].get("1-EV", 1.0)
    logger.info(f"📊 PHASE 3 FINAL PERFORMANCE:")
    logger.info(f"  Standard EV: {final_ev:.4f}")
    logger.info(f"  Total Parameters: {results['model_info']['n_parameters']:,}")
    
    logger.info("✅ Phase 3 teacher-init training COMPLETED")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info("🔄 Ready for Phase 4 evaluation and comparison")

    success = True  # Always successful if training completes

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
