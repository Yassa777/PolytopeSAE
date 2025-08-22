#!/usr/bin/env python3
"""
Phase 2: Baseline H-SAE Training

This script implements Phase 2 of the V2 focused experiment:
- Train randomly initialized H-SAE for 7,000 steps
- Log reconstruction, purity, leakage, and usage statistics
- Establish baseline performance metrics

Estimated runtime: 8-10 hours
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
from torch.utils.data import Subset

from polytope_hsae.activations import HDF5ActivationsDataset
from polytope_hsae.metrics import (
    compute_comprehensive_metrics,
    compute_explained_variance,
    log_metrics_summary,
)
# Project imports
from polytope_hsae.models import HierarchicalSAE, HSAEConfig
from polytope_hsae.sanity_checker import TrainingSanityChecker
from polytope_hsae.training import (
    HSAETrainer,
    TrainingRestartException,
    create_data_loader,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_experiment(config):
    """Set up experiment directories and logging."""
    exp_dir = Path(config["logging"]["save_dir"]) / config["logging"]["phase_2_log"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = exp_dir / f"phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info(f"Phase 2 experiment directory: {exp_dir}")
    return exp_dir


def load_activations(config, exp_dir):
    """Create a streaming dataset of Phase 1 activations."""
    activations_file = (
        Path(config["logging"]["save_dir"])
        / config["logging"]["phase_1_log"]
        / "activations.h5"
    )

    if not activations_file.exists():
        raise FileNotFoundError(f"Activations not found: {activations_file}")

    logger.info(f"Using activation dataset {activations_file}")

    dataset = HDF5ActivationsDataset(
        str(activations_file),
        unit_norm=config.get("data", {}).get("unit_norm", False),
    )
    logger.info(f"Activation dataset provides {len(dataset)} samples")

    return dataset


def create_baseline_hsae(config, input_dim: int):
    """Create baseline H-SAE with random initialization."""
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
        causal_ortho_lambda=0.0,  # No causal ortho for baseline
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

    model = HierarchicalSAE(hsae_config)
    logger.info(
        f"Created baseline H-SAE with {sum(p.numel() for p in model.parameters())} parameters"
    )

    return model


def train_baseline_hsae_with_sanity_checks(model, dataset, config, exp_dir):
    """Train baseline H-SAE with automated sanity checking and restart capability."""
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        logger.info(f"🚀 Starting baseline H-SAE training (attempt {attempt}/{max_attempts})")
        
        try:
            results = _train_baseline_hsae_single_attempt(model, dataset, config, exp_dir, attempt)
            logger.info("✅ Training completed successfully!")
            return results
            
        except TrainingRestartException as e:
            logger.info(f"🔄 Training restart requested: {e}")
            if attempt >= max_attempts:
                logger.error(f"❌ Max attempts ({max_attempts}) reached - stopping")
                raise
            
            # Recreate model with updated config
            logger.info("🔧 Recreating model with updated configuration...")
            model = create_baseline_hsae(config, dataset.feature_dim)
            continue
            
        except Exception as e:
            logger.error(f"❌ Training failed with error: {e}")
            raise
    
    raise RuntimeError("Training failed after all attempts")

def _train_baseline_hsae_single_attempt(model, dataset, config, exp_dir, attempt_num):
    """Single training attempt with sanity checking."""
    logger.info(f"Starting baseline H-SAE training (attempt {attempt_num})")

    # Create data loaders
    device = torch.device(config["run"]["device"])
    model = model.to(device)

    # Split data (70/15/15 as per config spec)
    n_samples = len(dataset)
    n_train = int(0.70 * n_samples)
    n_val = int(0.15 * n_samples)
    n_test = n_samples - n_train - n_val

    train_ds = Subset(dataset, range(n_train))
    val_ds = Subset(dataset, range(n_train, n_train + n_val))
    test_ds = Subset(dataset, range(n_train + n_val, n_samples))

    logger.info(f"Data split: train={n_train}, val={n_val}, test={n_test} samples")

    # Auto-adjust batch size to ensure training can proceed
    original_batch_size = config["training"]["batch_size_acts"]
    max_safe_batch_size = n_train  # Maximum possible batch size
    
    # Choose a reasonable batch size that's safe
    batch_size = min(original_batch_size, max_safe_batch_size)
    
    # Prefer common batch sizes for better performance
    common_batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    for size in reversed(common_batch_sizes):
        if size <= max_safe_batch_size:
            batch_size = size
            break
    
    if batch_size != original_batch_size:
        logger.info(f"🔧 Auto-adjusted batch size: {original_batch_size} → {batch_size}")
        logger.info(f"   Reason: Training dataset has {n_train} samples")
        logger.info(f"   This ensures at least {n_train // batch_size} batches per epoch")
    
    # Ensure validation batch size doesn't exceed validation dataset size
    val_batch_size = min(batch_size, n_val) if n_val > 0 else batch_size
    if val_batch_size != batch_size:
        logger.info(f"🔧 Validation batch size adjusted: {batch_size} → {val_batch_size}")
        logger.info(f"   Reason: Validation dataset has {n_val} samples")
    
    # Get dataloader config
    dl_config = config.get("dataloader", {})
    
    train_loader = create_data_loader(
        train_ds,
        batch_size=batch_size,
        device=config["run"]["device"],
        shuffle=True,
        num_workers=dl_config.get("num_workers", None),
        pin_memory=dl_config.get("pin_memory", None),
    )
    val_loader = create_data_loader(
        val_ds,
        batch_size=val_batch_size,
        device=config["run"]["device"],
        shuffle=False,
        drop_last=False,  # Always keep validation samples
        num_workers=dl_config.get("num_workers", None),
        pin_memory=dl_config.get("pin_memory", None),
    )

    # Initialize trainer
    trainer = HSAETrainer(
        model=model,
        config=config,
        geometry=None,  # No geometry for baseline
        use_wandb=True,
        use_compile=False,  # Disable torch.compile for faster short runs
    )
    
    # Verify GPU usage
    logger.info(f"🔧 Device verification: {trainer.device}")
    logger.info(f"🔧 Model on device: {next(model.parameters()).device}")
    if torch.cuda.is_available():
        logger.info(f"🔧 CUDA available: {torch.cuda.device_count()} devices")
        logger.info(f"🔧 Current CUDA device: {torch.cuda.current_device()}")
    else:
        logger.warning("⚠️  CUDA not available - running on CPU")
    
    # Initialize sanity checker
    sanity_checker = TrainingSanityChecker(config, model, trainer, exp_dir)
    logger.info("🔍 Sanity checker initialized")

    # Initialize W&B
    if trainer.use_wandb:
        wandb.init(
            project=config.get("wandb_project", "polytope-hsae"),
            name=f"phase2_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            tags=["phase2", "baseline", "hsae"],
        )

        # Log model architecture
        wandb.log(
            {
                "model/n_parameters": sum(p.numel() for p in model.parameters()),
                "model/n_parents": model.config.n_parents,
                "model/subspace_dim": model.config.subspace_dim,
                "model/topk_parent": model.config.topk_parent,
            },
            step=0,
        )

    # Training with sanity checking - support schedule-only freeze control
    total_steps = config["training"]["baseline"]["total_steps"]
    
    # Check if we should use schedule-only freeze (same freeze→adapt schedule but with random init)
    if config["training"]["baseline"].get("schedule_only_freeze", False):
        logger.info("🔄 Running baseline with schedule-only freeze control (random init + freeze→adapt)")
        # Use the same two-stage freeze→adapt schedule as teacher-init but with random init and no causal-ortho
        stabilize_steps = config["training"]["teacher_init"]["stage_A"]["steps"]
        adapt_steps = config["training"]["teacher_init"]["stage_B"]["steps"]
        
        history = trainer.train_teacher_init(
            train_loader, val_loader,
            stabilize_steps=stabilize_steps,
            adapt_steps=adapt_steps,
            sanity_checker=sanity_checker,
        )
    else:
        history = trainer.train_baseline(
            train_loader=train_loader, 
            val_loader=val_loader, 
            total_steps=total_steps,
            sanity_checker=sanity_checker
        )

    # Final evaluation
    logger.info("Running final evaluation")
    
    # Run identity test on a sample of validation data (sanity check)
    logger.info("Running linear identity test on validation data")
    val_sample = next(iter(val_loader))
    if isinstance(val_sample, (list, tuple)):
        val_sample = val_sample[0]
    val_sample = val_sample[:100].to(device)  # Use first 100 samples
    
    # Since baseline has no geometry, this should show EV ~1.0 (identity)
    identity_ev = compute_explained_variance(val_sample, val_sample)
    logger.info(f"Identity test EV (should be ~1.0): {identity_ev:.6f}")
    
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
        
        # Compute detailed EV breakdown (standard + causal + logit for comprehensive check)
        logger.info("🔍 Computing detailed EV breakdown:")
        model.eval()
        with torch.no_grad():
            val_sample = next(iter(val_loader))
            if isinstance(val_sample, (list, tuple)):
                val_sample = val_sample[0]
            val_sample = val_sample.to(device)
            x_hat, _, _ = model(val_sample)
            
            # Standard EV
            from polytope_hsae.metrics import compute_dual_explained_variance, compute_logit_ev
            ev_results = compute_dual_explained_variance(
                val_sample, x_hat, geometry=None, print_components=True
            )
            logger.info(f"✅ Baseline H-SAE Standard EV: {ev_results['standard_ev']:.6f}")
            
            # For fair comparison, also compute causal-EV by creating geometry just for eval
            try:
                from polytope_hsae.geometry import CausalGeometry
                # Load unembedding matrix for causal geometry construction
                import transformers
                model_name = config["model"]["name"]
                hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map=device
                )
                unembedding_matrix = hf_model.lm_head.weight.detach()
                
                # Create geometry for causal-EV evaluation only
                eval_geometry = CausalGeometry.from_unembedding(
                    unembedding_matrix, shrinkage=config["geometry"].get("shrinkage", 0.05)
                )
                eval_geometry.to(str(device))
                
                causal_ev_results = compute_dual_explained_variance(
                    val_sample, x_hat, geometry=eval_geometry, print_components=False
                )
                logger.info(f"✅ Baseline H-SAE Causal EV: {causal_ev_results['causal_ev']:.6f}")
                
                # Logit-EV for comprehensive check
                logit_ev = compute_logit_ev(val_sample, x_hat, unembedding_matrix)
                logger.info(f"✅ Baseline H-SAE Logit EV: {logit_ev:.6f}")
                
                # Clean up
                del hf_model
                
            except Exception as e:
                logger.warning(f"Could not compute causal/logit EV: {e}")
                logger.info("Continuing with standard EV only")
        
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
            "architecture": "baseline_hsae",
        },
    }

    results_file = exp_dir / "baseline_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save model
    model_file = exp_dir / "baseline_hsae_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config,
            "training_step": total_steps,
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
        })
        
        wandb.log(wandb_metrics)
        logger.info(f"📊 Logged {len(wandb_metrics)} final metrics to W&B")
        wandb.finish()

    logger.info(f"Baseline training completed. Results saved to {exp_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Baseline H-SAE Training")
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

    # Load activations from Phase 1
    activation_ds = load_activations(config, exp_dir)

    # Create baseline H-SAE
    model = create_baseline_hsae(config, input_dim=activation_ds.feature_dim)

    # Train baseline H-SAE with sanity checking
    results = train_baseline_hsae_with_sanity_checks(model, activation_ds, config, exp_dir)

    # Report final performance (no hard success gate)
    final_ev = 1.0 - results["final_metrics"].get("1-EV", 1.0)
    logger.info(f"📊 PHASE 2 FINAL PERFORMANCE:")
    logger.info(f"  Standard EV: {final_ev:.4f}")
    logger.info(f"  Reconstruction Loss: {results['final_metrics'].get('recon_loss', 'N/A')}")
    logger.info(f"  Total Parameters: {results['model_info']['n_parameters']:,}")

    # Success based on completion, not absolute EV threshold
    logger.info("✅ Phase 2 baseline training COMPLETED")
    logger.info("🔄 Ready for Phase 3 comparison")

    return True  # Always successful if training completes


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
