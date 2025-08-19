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

from polytope_hsae.activations import ActivationCapture
from polytope_hsae.metrics import (compute_comprehensive_metrics,
                                   log_metrics_summary)
# Project imports
from polytope_hsae.models import HierarchicalSAE, HSAEConfig
from polytope_hsae.training import HSAETrainer, create_data_loader

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
        # Use both positive and negative examples for training
        all_activations.append(pos_neg_dict["pos"])
        all_activations.append(pos_neg_dict["neg"])

    combined_activations = torch.cat(all_activations, dim=0)
    logger.info(f"Loaded {combined_activations.shape[0]} activation samples")

    return combined_activations


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


def train_baseline_hsae(model, activations, config, exp_dir):
    """Train baseline H-SAE."""
    logger.info("Starting baseline H-SAE training")

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
        geometry=None,  # No geometry for baseline
        use_wandb=True,
    )

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
            }
        )

    # Training
    total_steps = config["training"]["baseline"]["total_steps"]
    history = trainer.train_baseline(
        train_loader=train_loader, val_loader=val_loader, total_steps=total_steps
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
        wandb.log({"final_metrics": final_metrics})
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

    # Override device if specified
    if args.device:
        config["run"]["device"] = args.device

    # Set up experiment
    exp_dir = setup_experiment(config)

    # Load activations from Phase 1
    activations = load_activations(config, exp_dir)

    # Create baseline H-SAE
    model = create_baseline_hsae(config, input_dim=activations.shape[1])

    # Train baseline H-SAE
    results = train_baseline_hsae(model, activations, config, exp_dir)

    # Check if training was successful
    final_ev = 1.0 - results["final_metrics"].get("1-EV", 1.0)
    logger.info(f"Final explained variance: {final_ev:.4f}")

    if final_ev > 0.5:  # Reasonable threshold
        logger.info("✅ Baseline training SUCCESSFUL - ready for Phase 3")
        success = True
    else:
        logger.warning("⚠️  Baseline training shows low EV - check hyperparameters")
        success = False

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
