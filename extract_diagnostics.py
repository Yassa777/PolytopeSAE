#!/usr/bin/env python3
"""
Extract all diagnostic metrics from experiment results and log to W&B.

This script extracts metrics for the diagnostic tables from all completed phases
and logs them as structured JSON to W&B for easy access and reporting.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml

# Handle wandb import gracefully
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("W&B not available - install with: pip install wandb")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_get(data: dict, *keys, default="N/A"):
    """Safely get nested dictionary values."""
    try:
        result = data
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError):
        return default


def extract_angle_metrics(base_path: Path) -> Dict[str, Any]:
    """Extract angle validation metrics from Phase 1."""
    metrics = {}
    try:
        val_file = base_path / "teacher_extraction" / "validation_results.json"
        with open(val_file) as f:
            val_results = json.load(f)
        
        ortho = val_results.get("orthogonality_test", {})
        angle_stats = val_results.get("angle_statistics", {})
        
        metrics.update({
            "angles_median_deg": safe_get(ortho, "median_angle_deg", default=0),
            "angles_frac_above_80": safe_get(ortho, "fraction_above_80deg", default=0),
            "angles_frac_above_85": safe_get(ortho, "fraction_above_85deg", default=0),
            "angles_mean_deg": safe_get(angle_stats, "mean_angle_deg", default=0),
            "angles_q25_deg": safe_get(angle_stats, "q25_angle_deg", default=0),
            "angles_q75_deg": safe_get(angle_stats, "q75_angle_deg", default=0),
            "angles_passes_threshold": safe_get(ortho, "passes_threshold", default=False),
        })
        
        logger.info(f"✅ Extracted angle metrics: median={metrics['angles_median_deg']:.2f}°")
        
    except Exception as e:
        logger.warning(f"❌ Could not extract angle metrics: {e}")
        
    return metrics


def extract_whitening_metrics(base_path: Path) -> Dict[str, Any]:
    """Extract whitening matrix diagnostics from geometry."""
    metrics = {}
    try:
        geom_file = base_path / "teacher_extraction" / "geometry.pt"
        geom_data = torch.load(geom_file, map_location="cpu")
        
        # Handle different possible formats
        W = None
        if hasattr(geom_data, 'W'):
            W = geom_data.W.numpy()
        elif isinstance(geom_data, dict) and 'W' in geom_data:
            W = geom_data['W'].numpy()
        elif isinstance(geom_data, dict) and 'whitening_matrix' in geom_data:
            W = geom_data['whitening_matrix'].numpy()
        else:
            # Try to find any matrix-like object
            for key, value in geom_data.items() if isinstance(geom_data, dict) else []:
                if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                    W = value.numpy()
                    logger.info(f"Using matrix from key: {key}")
                    break
        
        if W is not None:
            # Whitening matrix should satisfy W @ W.T ≈ I
            gram = W @ W.T
            diag_vals = np.diag(gram)
            
            # Off-diagonal elements (set diagonal to 0)
            offdiag = gram.copy()
            np.fill_diagonal(offdiag, 0)
            
            metrics.update({
                "whitening_diag_mean": float(np.mean(diag_vals)),
                "whitening_diag_std": float(np.std(diag_vals)),
                "whitening_offdiag_max": float(np.max(np.abs(offdiag))),
                "whitening_offdiag_rms": float(np.sqrt(np.mean(offdiag**2))),
                "whitening_matrix_shape": list(W.shape),
            })
            
            logger.info(f"✅ Extracted whitening metrics: diag_mean={metrics['whitening_diag_mean']:.3f}")
        else:
            logger.warning(f"Could not find whitening matrix in geometry file. Available keys: {list(geom_data.keys()) if isinstance(geom_data, dict) else 'N/A'}")
        
    except Exception as e:
        logger.warning(f"❌ Could not extract whitening metrics: {e}")
        
    return metrics


def extract_identity_metrics(base_path: Path) -> Dict[str, Any]:
    """Extract identity test metrics (geometry validation)."""
    metrics = {}
    try:
        val_file = base_path / "teacher_extraction" / "validation_results.json"
        with open(val_file) as f:
            val_results = json.load(f)
        
        # Look for identity/geometry validation results
        identity = val_results.get("identity_test", {})
        validation = val_results.get("orthogonality_validation", {})
        
        metrics.update({
            "identity_ev": safe_get(identity, "explained_variance", default=-999),
            "identity_mse": safe_get(identity, "mse", default=9999),
            "identity_max_error": safe_get(identity, "max_error", default=9999),
            "geometry_validation_passes": safe_get(val_results, "passes_validation", default=False),
        })
        
        logger.info(f"✅ Extracted identity metrics: EV={metrics['identity_ev']}")
        
    except Exception as e:
        logger.warning(f"❌ Could not extract identity metrics: {e}")
        
    return metrics


def extract_training_metrics(base_path: Path, phase_name: str, file_prefix: str) -> Dict[str, Any]:
    """Extract training metrics from a specific phase."""
    metrics = {}
    try:
        results_file = base_path / phase_name / f"{file_prefix}_results.json"
        with open(results_file) as f:
            results = json.load(f)
        
        final = results.get("final_metrics", {})
        config = results.get("config", {})
        training_history = results.get("training_history", {})
        
        # Core reconstruction metrics
        metrics.update({
            f"{file_prefix}_recon_ev": 1.0 - safe_get(final, "1-EV", default=1.0),
            f"{file_prefix}_recon_ce": safe_get(final, "1-CE", default=999),
            f"{file_prefix}_purity": safe_get(final, "purity", default=0),
            f"{file_prefix}_leakage": safe_get(final, "leakage", default=1),
            f"{file_prefix}_steering_leakage": safe_get(final, "steering_leakage", default=1),
        })
        
        # Sparsity metrics
        metrics.update({
            f"{file_prefix}_parent_sparsity": safe_get(final, "parent_sparsity", default=0),
            f"{file_prefix}_child_sparsity": safe_get(final, "child_sparsity", default=0),
            f"{file_prefix}_parent_usage": safe_get(final, "parent_usage", default=0),
            f"{file_prefix}_child_usage": safe_get(final, "child_usage", default=0),
        })
        
        # Training configuration
        hsae_config = config.get("hsae", {})
        training_config = config.get("training", {})
        
        metrics.update({
            f"{file_prefix}_l1_parent": safe_get(hsae_config, "l1_parent", default=0),
            f"{file_prefix}_l1_child": safe_get(hsae_config, "l1_child", default=0),
            f"{file_prefix}_lr": safe_get(training_config, "lr", default=0),
            f"{file_prefix}_batch_size": safe_get(training_config, "batch_size_acts", default=0),
            f"{file_prefix}_total_steps": safe_get(training_config, "baseline", "total_steps", default=0) if file_prefix == "baseline" else safe_get(training_config, "teacher_init", "total_steps", default=0),
        })
        
        # Model architecture
        metrics.update({
            f"{file_prefix}_n_parents": safe_get(hsae_config, "n_parents", default=0),
            f"{file_prefix}_topk_parent": safe_get(hsae_config, "topk_parent", default=0),
            f"{file_prefix}_n_children_per_parent": safe_get(hsae_config, "n_children_per_parent", default=0),
            f"{file_prefix}_subspace_dim": safe_get(hsae_config, "subspace_dim", default=0),
        })
        
        logger.info(f"✅ Extracted {file_prefix} metrics: EV={metrics[f'{file_prefix}_recon_ev']:.3f}")
        
    except Exception as e:
        logger.warning(f"❌ Could not extract {file_prefix} metrics: {e}")
        
    return metrics


def extract_config_metrics(config_path: str) -> Dict[str, Any]:
    """Extract global configuration metrics."""
    metrics = {}
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        metrics.update({
            "seed": safe_get(config, "run", "seed", default=123),
            "model_name": safe_get(config, "model", "name", default="unknown"),
            "dtype": safe_get(config, "model", "dtype", default="unknown"),
            "device": safe_get(config, "run", "device", default="unknown"),
            "token_budget": safe_get(config, "data", "token_budget", default=0),
            "batch_size_acts": safe_get(config, "training", "batch_size_acts", default=0),
            "whitening_method": safe_get(config, "geometry", "whitening", default="unknown"),
            "shrinkage": safe_get(config, "geometry", "shrinkage", default=0),
        })
        
        logger.info(f"✅ Extracted config metrics: model={metrics['model_name']}")
        
    except Exception as e:
        logger.warning(f"❌ Could not extract config metrics: {e}")
        
    return metrics


def extract_steering_metrics(base_path: Path) -> Dict[str, Any]:
    """Extract steering experiment metrics from Phase 4."""
    metrics = {}
    try:
        # Steering results
        steering_file = base_path / "evaluation_steering" / "steering_results.json"
        if steering_file.exists():
            with open(steering_file) as f:
                steering = json.load(f)
            
            # Extract success rates and effect sizes
            for model_name, results in steering.items():
                if isinstance(results, dict) and "analysis" in results:
                    analysis = results["analysis"]
                    metrics.update({
                        f"steering_{model_name}_success_rate": safe_get(analysis, "success_rate", default=0),
                        f"steering_{model_name}_effect_size_mean": safe_get(analysis, "effect_size_mean", default=0),
                        f"steering_{model_name}_leakage_mean": safe_get(analysis, "leakage_mean", default=0),
                    })
        
        # Ablation results
        ablation_file = base_path / "evaluation_steering" / "ablation_results.json"
        if ablation_file.exists():
            with open(ablation_file) as f:
                ablation = json.load(f)
            
            for model_name, results in ablation.items():
                if isinstance(results, dict):
                    metrics.update({
                        f"ablation_{model_name}_1_ev": safe_get(results, "1-EV", default=1),
                        f"ablation_{model_name}_1_ce": safe_get(results, "1-CE", default=999),
                        f"ablation_{model_name}_purity": safe_get(results, "purity", default=0),
                        f"ablation_{model_name}_leakage": safe_get(results, "leakage", default=1),
                    })
        
        logger.info(f"✅ Extracted steering/ablation metrics")
        
    except Exception as e:
        logger.warning(f"❌ Could not extract steering metrics: {e}")
        
    return metrics


def create_diagnostic_verdicts(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Create pass/fail verdicts for diagnostic checks."""
    verdicts = {}
    
    # Angle checks
    verdicts["angles_median"] = "PASS" if metrics.get("angles_median_deg", 0) >= 80 else "FAIL"
    verdicts["angles_frac_80"] = "PASS" if metrics.get("angles_frac_above_80", 0) >= 0.9 else "FAIL"
    verdicts["angles_frac_85"] = "PASS" if metrics.get("angles_frac_above_85", 0) >= 0.7 else "FAIL"
    
    # Whitening checks
    diag_mean = metrics.get("whitening_diag_mean", 0)
    verdicts["whitening_diag"] = "PASS" if 0.95 <= diag_mean <= 1.05 else "FAIL"
    
    offdiag_max = metrics.get("whitening_offdiag_max", 1)
    verdicts["whitening_offdiag"] = "PASS" if offdiag_max <= 0.05 else "FAIL"
    
    # Identity checks
    identity_ev = metrics.get("identity_ev", -999)
    verdicts["identity_ev"] = "PASS" if identity_ev >= 0.95 else "FAIL"
    
    # Reconstruction checks
    baseline_ev = metrics.get("baseline_recon_ev", 0)
    teacher_ev = metrics.get("teacher_recon_ev", 0)
    verdicts["baseline_recon"] = "PASS" if baseline_ev >= 0.8 else "SUSPECT" if baseline_ev >= 0.5 else "FAIL"
    verdicts["teacher_recon"] = "PASS" if teacher_ev >= 0.8 else "SUSPECT" if teacher_ev >= 0.5 else "FAIL"
    
    return verdicts


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract diagnostics and log to W&B")
    parser.add_argument("--base", type=str, default="runs/v2_focused", help="Base results directory")
    parser.add_argument("--config", type=str, default="configs/v2-focused.yaml", help="Config file")
    parser.add_argument("--project", type=str, default="polytope-hsae", help="W&B project name")
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    
    args = parser.parse_args()
    
    base_path = Path(args.base)
    
    logger.info(f"🔍 Extracting diagnostics from {base_path}")
    
    # Collect all metrics
    all_metrics = {}
    
    # Phase 1: Angles, whitening, identity
    all_metrics.update(extract_angle_metrics(base_path))
    all_metrics.update(extract_whitening_metrics(base_path))
    all_metrics.update(extract_identity_metrics(base_path))
    
    # Phase 2: Baseline training
    all_metrics.update(extract_training_metrics(base_path, "baseline_hsae", "baseline"))
    
    # Phase 3: Teacher-init training
    all_metrics.update(extract_training_metrics(base_path, "teacher_init_hsae", "teacher"))
    
    # Phase 4: Steering and ablations
    all_metrics.update(extract_steering_metrics(base_path))
    
    # Global config
    all_metrics.update(extract_config_metrics(args.config))
    
    # Create verdicts
    verdicts = create_diagnostic_verdicts(all_metrics)
    all_metrics["verdicts"] = verdicts
    
    # Add summary stats
    summary = {
        "total_checks": len(verdicts),
        "passed_checks": sum(1 for v in verdicts.values() if v == "PASS"),
        "failed_checks": sum(1 for v in verdicts.values() if v == "FAIL"),
        "suspect_checks": sum(1 for v in verdicts.values() if v == "SUSPECT"),
    }
    all_metrics["summary"] = summary
    
    logger.info(f"📊 Summary: {summary['passed_checks']}/{summary['total_checks']} checks passed")
    
    # Save locally
    output_file = base_path / "diagnostics_complete.json"
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    logger.info(f"💾 Saved diagnostics to {output_file}")
    
    # Log to W&B
    if not args.no_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=args.project,
                name="diagnostics_extraction",
                tags=["diagnostics", "metrics", "validation"],
                notes="Complete diagnostic metrics from all experimental phases"
            )
            
            # Log the full metrics as JSON
            wandb.log({
                "diagnostics": all_metrics,
                "summary_stats": summary,
                **{f"metric_{k}": v for k, v in all_metrics.items() if isinstance(v, (int, float, bool))},
                **{f"verdict_{k}": v for k, v in verdicts.items()},
            })
            
            # Upload the JSON file as artifact
            artifact = wandb.Artifact("diagnostics", type="metrics")
            artifact.add_file(str(output_file))
            wandb.log_artifact(artifact)
            
            wandb.finish()
            logger.info("✅ Logged diagnostics to W&B")
            
        except Exception as e:
            logger.error(f"❌ Failed to log to W&B: {e}")
    elif not args.no_wandb:
        logger.warning("⚠️ W&B not available - skipping W&B logging")
    
    # Print key results
    print("\n" + "="*60)
    print("KEY DIAGNOSTIC RESULTS")
    print("="*60)
    print(f"Angles median: {all_metrics.get('angles_median_deg', 'N/A')} ({verdicts.get('angles_median', 'N/A')})")
    print(f"Identity EV: {all_metrics.get('identity_ev', 'N/A')} ({verdicts.get('identity_ev', 'N/A')})")
    print(f"Baseline recon EV: {all_metrics.get('baseline_recon_ev', 'N/A')} ({verdicts.get('baseline_recon', 'N/A')})")
    print(f"Teacher recon EV: {all_metrics.get('teacher_recon_ev', 'N/A')} ({verdicts.get('teacher_recon', 'N/A')})")
    print(f"\nOverall: {summary['passed_checks']}/{summary['total_checks']} checks passed")
    print("="*60)
    
    return all_metrics


if __name__ == "__main__":
    main()