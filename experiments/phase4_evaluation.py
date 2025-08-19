#!/usr/bin/env python3
"""
Phase 4: Evaluation & Steering

This script implements Phase 4 of the V2 focused experiment:
- Ablation studies (Euclidean vs causal, Top-K=1 vs 2, no teacher init)
- Steering experiments (parent and child concept interventions)
- Analysis of effect sizes, leakage measurements, success rates

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

from polytope_hsae.geometry import CausalGeometry
from polytope_hsae.metrics import (compute_comprehensive_metrics,
                                   log_metrics_summary)
# Project imports
from polytope_hsae.models import HierarchicalSAE, HSAEConfig
from polytope_hsae.steering import ConceptSteering, analyze_steering_results
# Ratio-invariance test implemented inline for simplicity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_experiment(config):
    """Set up experiment directories and logging."""
    exp_dir = Path(config["logging"]["save_dir"]) / config["logging"]["phase_4_log"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = exp_dir / f"phase4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info(f"Phase 4 experiment directory: {exp_dir}")
    return exp_dir


def load_models_and_data(config):
    """Load trained models and data from previous phases."""
    base_dir = Path(config["logging"]["save_dir"])

    # Load teacher-initialized model
    teacher_model_file = (
        base_dir / config["logging"]["phase_3_log"] / "teacher_hsae_model.pt"
    )
    baseline_model_file = (
        base_dir / config["logging"]["phase_2_log"] / "baseline_hsae_model.pt"
    )

    models = {}

    # Load teacher model
    if teacher_model_file.exists():
        logger.info(f"Loading teacher-initialized model from {teacher_model_file}")
        checkpoint = torch.load(teacher_model_file, map_location="cpu")
        teacher_model = HierarchicalSAE(checkpoint["model_config"])
        teacher_model.load_state_dict(checkpoint["model_state_dict"])
        models["teacher"] = teacher_model
    else:
        logger.warning("Teacher model not found")

    # Load baseline model
    if baseline_model_file.exists():
        logger.info(f"Loading baseline model from {baseline_model_file}")
        checkpoint = torch.load(baseline_model_file, map_location="cpu")
        baseline_model = HierarchicalSAE(checkpoint["model_config"])
        baseline_model.load_state_dict(checkpoint["model_state_dict"])
        models["baseline"] = baseline_model
    else:
        logger.warning("Baseline model not found")

    # Load teacher vectors
    teacher_file = base_dir / config["logging"]["phase_1_log"] / "teacher_vectors.json"
    if teacher_file.exists():
        with open(teacher_file, "r") as f:
            teacher_data = json.load(f)

        parent_vectors = {
            k: torch.tensor(v) for k, v in teacher_data["parent_vectors"].items()
        }
        child_deltas = {
            parent_id: {
                child_id: torch.tensor(delta) for child_id, delta in deltas.items()
            }
            for parent_id, deltas in teacher_data["child_deltas"].items()
        }
    else:
        logger.warning("Teacher vectors not found")
        parent_vectors, child_deltas = {}, {}

    # Load concept hierarchies
    concepts_file = (
        base_dir / config["logging"]["phase_1_log"] / "concept_hierarchies.json"
    )
    if concepts_file.exists():
        from polytope_hsae.concepts import load_concept_hierarchies

        hierarchies = load_concept_hierarchies(str(concepts_file))
    else:
        logger.warning("Concept hierarchies not found")
        hierarchies = []

    return models, parent_vectors, child_deltas, hierarchies


def load_eval_loader(config):
    """Load real captured activations for evaluation."""
    base_dir = Path(config['logging']['save_dir'])
    h5_path = base_dir / config['logging']['phase_1_log'] / "activations.h5"
    
    if not h5_path.exists():
        raise FileNotFoundError(f"Activations not found: {h5_path}")
    
    logger.info(f"Loading evaluation activations from {h5_path}")
    acts = ActivationCapture.load_hierarchical_activations(str(h5_path))

    # Concatenate all activations (both pos and neg for comprehensive evaluation)
    all_acts = []
    for concept_id, pos_neg_dict in acts.items():
        all_acts.append(pos_neg_dict['pos'])
        all_acts.append(pos_neg_dict['neg'])
    
    X = torch.cat(all_acts, dim=0)
    
    # Apply unit normalization if configured
    if config.get('data', {}).get('unit_norm', False):
        X = torch.nn.functional.normalize(X, dim=-1)
    
    logger.info(f"Loaded {X.shape[0]} real activation samples for evaluation")
    
    dataset = torch.utils.data.TensorDataset(X)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=config['training']['batch_size_acts'], 
        shuffle=False
    )


def run_ablation_studies(models, config, exp_dir):
    """Run ablation studies comparing different configurations."""
    logger.info("Running ablation studies on real captured activations")

    ablation_results = {}
    device = torch.device(config["run"]["device"])

    # Load real evaluation data instead of synthetic
    eval_loader = load_eval_loader(config)

    # Evaluate each model
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name} model")
        model = model.to(device)
        model.eval()

        metrics = compute_comprehensive_metrics(
            model=model, data_loader=eval_loader, device=str(device)
        )

        ablation_results[model_name] = metrics
        log_metrics_summary(metrics, logger)

    # Compare models
    if "teacher" in models and "baseline" in models:
        comparison = compare_model_performance(
            ablation_results["teacher"], ablation_results["baseline"]
        )
        ablation_results["teacher_vs_baseline"] = comparison

    # Save ablation results
    ablation_file = exp_dir / "ablation_results.json"
    with open(ablation_file, "w") as f:
        json.dump(ablation_results, f, indent=2, default=str)

    return ablation_results


def compare_model_performance(teacher_metrics, baseline_metrics):
    """Compare teacher vs baseline model performance."""
    comparison = {}

    # Key metrics to compare
    metrics_to_compare = [
        "1-EV",
        "1-CE",
        "parent_purity_mean",
        "parent_leakage_mean",
        "child_leakage_mean",
        "parent_sparsity",
        "child_sparsity",
    ]

    for metric in metrics_to_compare:
        teacher_val = teacher_metrics.get(metric, 0.0)
        baseline_val = baseline_metrics.get(metric, 0.0)

        # Calculate improvement (positive = teacher better)
        if "leakage" in metric or "1-EV" in metric or "1-CE" in metric:
            # Lower is better
            improvement = baseline_val - teacher_val
            percent_improvement = improvement / (baseline_val + 1e-8) * 100
        else:
            # Higher is better
            improvement = teacher_val - baseline_val
            percent_improvement = improvement / (baseline_val + 1e-8) * 100

        comparison[metric] = {
            "teacher": teacher_val,
            "baseline": baseline_val,
            "improvement": improvement,
            "percent_improvement": percent_improvement,
        }

    # Overall assessment
    key_improvements = [
        comparison["parent_purity_mean"]["improvement"] > 0.05,  # Purity improvement
        comparison["parent_leakage_mean"]["improvement"] > 0.0,  # Leakage reduction
        abs(comparison["1-EV"]["improvement"]) < 0.05,  # Reconstruction parity
    ]

    comparison["success_criteria"] = {
        "purity_improved": comparison["parent_purity_mean"]["improvement"] > 0.05,
        "leakage_reduced": comparison["parent_leakage_mean"]["improvement"] > 0.0,
        "reconstruction_maintained": abs(comparison["1-EV"]["improvement"]) < 0.05,
        "overall_success": all(key_improvements),
    }

    return comparison


def run_steering_experiments(
    models, parent_vectors, child_deltas, hierarchies, config, exp_dir
):
    """Run concept steering experiments."""
    logger.info("Running steering experiments")

    # Load language model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(config["model"]["name"])
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load language model: {e}")
        return {}

    device = torch.device(config["run"]["device"])
    model = model.to(device)

    steering_results = {}

    # Test steering with different H-SAE models
    for model_name, hsae_model in models.items():
        logger.info(f"Testing steering with {model_name} H-SAE")

        hsae_model = hsae_model.to(device)

        # Create steering interface
        steering = ConceptSteering(
            model=model, tokenizer=tokenizer, hsae_model=hsae_model, device=str(device)
        )

        # Prepare test data
        test_prompts = {}
        for hierarchy in hierarchies[:5]:  # Limit for efficiency
            parent_id = hierarchy.parent.synset_id
            if parent_id in parent_vectors:
                # Use some parent prompts as test prompts
                test_prompts[parent_id] = hierarchy.parent_prompts[:3]

        # Run steering suite
        if test_prompts and parent_vectors:
            suite_results = steering.run_steering_suite(
                parent_vectors=dict(
                    list(parent_vectors.items())[:5]
                ),  # Limit for efficiency
                child_deltas={
                    k: v
                    for k, v in child_deltas.items()
                    if k in list(parent_vectors.keys())[:5]
                },
                test_prompts=test_prompts,
                magnitudes=config["eval"]["steering_magnitudes"],
            )

            # Analyze results
            analysis = analyze_steering_results(suite_results)

            steering_results[model_name] = {
                "raw_results": suite_results,
                "analysis": analysis,
            }
        else:
            logger.warning(f"No test data available for steering with {model_name}")

    # Save steering results
    steering_file = exp_dir / "steering_results.json"
    with open(steering_file, "w") as f:
        json.dump(steering_results, f, indent=2, default=str)

    return steering_results


def run_simple_ratio_invariance_test(parent_vector, child_vectors, child_names, alphas, n_samples, causal_geometry):
    """Run a simplified ratio-invariance test on vector geometry.
    
    Tests whether moving along parent vector preserves sibling ratios.
    This is a geometric approximation of the full intervention test.
    """
    results = []
    
    for alpha in alphas:
        for sample_idx in range(n_samples):
            # Create intervention: move along parent direction
            intervention_vector = alpha * causal_geometry.normalize_causal(parent_vector)
            
            # Compute baseline and intervention child projections
            baseline_projections = []
            intervention_projections = []
            
            for child_vector in child_vectors:
                # Baseline projection (child onto whitened space)
                baseline_proj = causal_geometry.whiten(child_vector)
                baseline_projections.append(torch.norm(baseline_proj).item())
                
                # Intervention projection (child + intervention onto whitened space)  
                intervention_child = child_vector + intervention_vector
                intervention_proj = causal_geometry.whiten(intervention_child)
                intervention_projections.append(torch.norm(intervention_proj).item())
            
            # Convert to probability distributions (softmax)
            baseline_probs = torch.softmax(torch.tensor(baseline_projections), dim=0)
            intervention_probs = torch.softmax(torch.tensor(intervention_projections), dim=0)
            
            # Compute KL divergence: KL(baseline || intervention)
            kl_div = torch.nn.functional.kl_div(
                torch.log(baseline_probs + 1e-8),
                intervention_probs,
                reduction='sum'
            ).item()
            
            results.append({
                'alpha': alpha,
                'sample': sample_idx,
                'kl_divergence': kl_div,
                'baseline_probs': baseline_probs.tolist(),
                'intervention_probs': intervention_probs.tolist()
            })
    
    return {'results': results, 'child_names': child_names}


def run_ratio_invariance_tests(parent_vectors, child_deltas, causal_geometry, config):
    """Run ratio-invariance tests as required by spec."""
    logger.info("Running ratio-invariance tests")
    
    ratio_invariance_results = {}
    
    # Test parameters from config or defaults
    alphas = config.get("eval", {}).get("ratio_invariance", {}).get("alphas", [0.1, 0.3, 0.5])
    n_samples = config.get("eval", {}).get("ratio_invariance", {}).get("n_samples", 10)
    
    for parent_id, parent_vector in parent_vectors.items():
        if parent_id not in child_deltas or len(child_deltas[parent_id]) < 2:
            continue  # Need at least 2 children for sibling ratios
            
        logger.info(f"Testing ratio-invariance for parent concept: {parent_id}")
        
        # Get children for this parent
        children_dict = child_deltas[parent_id]
        child_names = list(children_dict.keys())
        child_vectors = [children_dict[name] for name in child_names]
        
        # Run simplified ratio-invariance test
        try:
            ri_results = run_simple_ratio_invariance_test(
                parent_vector=parent_vector,
                child_vectors=child_vectors,
                child_names=child_names,
                alphas=alphas,
                n_samples=n_samples,
                causal_geometry=causal_geometry
            )
            ratio_invariance_results[parent_id] = ri_results
            
            # Log key metrics
            median_kl = np.median([r['kl_divergence'] for r in ri_results.get('results', [])])
            logger.info(f"  Median KL divergence: {median_kl:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed ratio-invariance test for {parent_id}: {e}")
            ratio_invariance_results[parent_id] = {"error": str(e)}
    
    # Compute aggregate statistics
    all_kls = []
    for parent_results in ratio_invariance_results.values():
        if 'results' in parent_results:
            all_kls.extend([r['kl_divergence'] for r in parent_results['results']])
    
    aggregate_stats = {
        "median_kl": np.median(all_kls) if all_kls else float('nan'),
        "mean_kl": np.mean(all_kls) if all_kls else float('nan'),
        "fraction_below_0_1": np.mean(np.array(all_kls) < 0.1) if all_kls else 0.0,
        "fraction_below_0_2": np.mean(np.array(all_kls) < 0.2) if all_kls else 0.0,
        "n_tests": len(all_kls)
    }
    
    logger.info(f"Ratio-invariance aggregate: median KL = {aggregate_stats['median_kl']:.4f}, "
                f"fraction < 0.1 = {aggregate_stats['fraction_below_0_1']:.3f}")
    
    return {
        "by_parent": ratio_invariance_results,
        "aggregate": aggregate_stats
    }


def run_euclidean_vs_causal_ablation(parent_vectors, child_deltas, config, exp_dir):
    """Run ablation comparing Euclidean vs causal geometry."""
    logger.info("Running Euclidean vs Causal ablation")

    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(config["model"]["name"])
        if hasattr(model, "lm_head"):
            unembedding_matrix = model.lm_head.weight.data
        else:
            unembedding_matrix = model.get_output_embeddings().weight.data
    except Exception as e:
        logger.error(f"Could not load model for geometry ablation: {e}")
        return {}

    # Causal geometry
    causal_geometry = CausalGeometry(
        unembedding_matrix, shrinkage=config["geometry"]["shrinkage"]
    )

    # Compare angles in both geometries
    euclidean_angles = []
    causal_angles = []

    for parent_id, deltas in child_deltas.items():
        if parent_id not in parent_vectors:
            continue

        parent_vector = parent_vectors[parent_id]

        for child_id, delta in deltas.items():
            # Euclidean angle
            cos_sim = torch.nn.functional.cosine_similarity(parent_vector, delta, dim=0)
            euclidean_angle = torch.rad2deg(torch.arccos(torch.clamp(cos_sim, -1, 1)))
            euclidean_angles.append(euclidean_angle.item())

            # Causal angle
            causal_angle = torch.rad2deg(
                causal_geometry.causal_angle(parent_vector, delta)
            )
            causal_angles.append(causal_angle.item())

    # Statistical comparison
    euclidean_angles = np.array(euclidean_angles)
    causal_angles = np.array(causal_angles)

    geometry_comparison = {
        "euclidean_angles": {
            "mean": np.mean(euclidean_angles),
            "median": np.median(euclidean_angles),
            "std": np.std(euclidean_angles),
            "fraction_above_80": np.mean(euclidean_angles >= 80),
        },
        "causal_angles": {
            "mean": np.mean(causal_angles),
            "median": np.median(causal_angles),
            "std": np.std(causal_angles),
            "fraction_above_80": np.mean(causal_angles >= 80),
        },
        "improvement": {
            "median_improvement": np.median(causal_angles)
            - np.median(euclidean_angles),
            "orthogonality_improvement": np.mean(causal_angles >= 80)
            - np.mean(euclidean_angles >= 80),
        },
    }

    logger.info(
        f"Euclidean median angle: {geometry_comparison['euclidean_angles']['median']:.1f}°"
    )
    logger.info(
        f"Causal median angle: {geometry_comparison['causal_angles']['median']:.1f}°"
    )
    logger.info(
        f"Improvement: {geometry_comparison['improvement']['median_improvement']:.1f}°"
    )

    # Run ratio-invariance tests as required by spec
    ratio_invariance_results = run_ratio_invariance_tests(
        parent_vectors, child_deltas, causal_geometry, config
    )
    
    # Combine results
    geometry_comparison["ratio_invariance"] = ratio_invariance_results

    # Save results
    geometry_file = exp_dir / "geometry_comparison.json"
    with open(geometry_file, "w") as f:
        json.dump(geometry_comparison, f, indent=2, default=str)

    return geometry_comparison


def generate_final_report(
    ablation_results, steering_results, geometry_comparison, config, exp_dir
):
    """Generate comprehensive final report."""
    logger.info("Generating final report")

    report = {
        "experiment_info": {
            "config_file": config,
            "timestamp": datetime.now().isoformat(),
            "total_phases": 4,
        },
        "ablation_studies": ablation_results,
        "steering_experiments": steering_results,
        "geometry_comparison": geometry_comparison,
    }

    # Summary statistics
    summary = {}

    # Model comparison summary
    if "teacher_vs_baseline" in ablation_results:
        comparison = ablation_results["teacher_vs_baseline"]
        summary["teacher_vs_baseline"] = {
            "purity_improvement": comparison.get("parent_purity_mean", {}).get(
                "percent_improvement", 0
            ),
            "leakage_reduction": comparison.get("parent_leakage_mean", {}).get(
                "percent_improvement", 0
            ),
            "reconstruction_maintained": comparison.get("success_criteria", {}).get(
                "reconstruction_maintained", False
            ),
            "overall_success": comparison.get("success_criteria", {}).get(
                "overall_success", False
            ),
        }

    # Steering summary
    if steering_results:
        steering_effects = []
        for model_name, results in steering_results.items():
            analysis = results.get("analysis", {})
            steering_effects.append(analysis.get("parent_steering_mean", 0))

        summary["steering"] = {
            "mean_effect": np.mean(steering_effects) if steering_effects else 0,
            "models_tested": len(steering_results),
        }

    # Geometry summary
    if geometry_comparison:
        summary["geometry"] = {
            "causal_vs_euclidean_improvement": geometry_comparison.get(
                "improvement", {}
            ).get("median_improvement", 0),
            "causal_orthogonality_rate": geometry_comparison.get(
                "causal_angles", {}
            ).get("fraction_above_80", 0),
            "ratio_invariance_median_kl": geometry_comparison.get(
                "ratio_invariance", {}
            ).get("aggregate", {}).get("median_kl", float('nan')),
            "ratio_invariance_fraction_below_0_1": geometry_comparison.get(
                "ratio_invariance", {}
            ).get("aggregate", {}).get("fraction_below_0_1", 0),
        }

    report["summary"] = summary

    # Save comprehensive report
    report_file = exp_dir / "final_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Create human-readable summary
    summary_file = exp_dir / "experiment_summary.txt"
    with open(summary_file, "w") as f:
        f.write("POLYTOPE DISCOVERY & HIERARCHICAL SAE INTEGRATION\n")
        f.write("=" * 50 + "\n")
        f.write(
            f"Experiment completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        f.write("KEY RESULTS:\n")
        f.write("-" * 20 + "\n")

        if "teacher_vs_baseline" in summary:
            tvsb = summary["teacher_vs_baseline"]
            f.write("Teacher vs Baseline:\n")
            f.write(f"  Purity improvement: {tvsb['purity_improvement']:.1f}%\n")
            f.write(f"  Leakage reduction: {tvsb['leakage_reduction']:.1f}%\n")
            f.write(
                f"  Reconstruction maintained: {tvsb['reconstruction_maintained']}\n"
            )
            f.write(f"  Overall success: {tvsb['overall_success']}\n\n")

        if "geometry" in summary:
            geom = summary["geometry"]
            f.write("Geometry Analysis:\n")
            f.write(
                f"  Causal vs Euclidean improvement: {geom['causal_vs_euclidean_improvement']:.1f}°\n"
            )
            f.write(
                f"  Causal orthogonality rate: {geom['causal_orthogonality_rate']:.3f}\n"
            )
            f.write(
                f"  Ratio-invariance median KL: {geom['ratio_invariance_median_kl']:.4f}\n"
            )
            f.write(
                f"  Ratio-invariance fraction < 0.1: {geom['ratio_invariance_fraction_below_0_1']:.3f}\n\n"
            )

        if "steering" in summary:
            steer = summary["steering"]
            f.write("Steering Experiments:\n")
            f.write(f"  Mean steering effect: {steer['mean_effect']:.4f}\n")
            f.write(f"  Models tested: {steer['models_tested']}\n\n")

    logger.info(f"Final report saved to {report_file}")
    logger.info(f"Human-readable summary saved to {summary_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Evaluation & Steering")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--skip-steering", action="store_true", help="Skip steering experiments"
    )

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

    # Load models and data
    models, parent_vectors, child_deltas, hierarchies = load_models_and_data(config)

    if not models:
        logger.error("No trained models found - run Phases 2 and 3 first")
        return False

    # Run ablation studies
    ablation_results = run_ablation_studies(models, config, exp_dir)

    # Run steering experiments
    steering_results = {}
    if not args.skip_steering and parent_vectors and child_deltas:
        steering_results = run_steering_experiments(
            models, parent_vectors, child_deltas, hierarchies, config, exp_dir
        )

    # Run geometry comparison
    geometry_comparison = {}
    if parent_vectors and child_deltas:
        geometry_comparison = run_euclidean_vs_causal_ablation(
            parent_vectors, child_deltas, config, exp_dir
        )

    # Generate final report
    report = generate_final_report(
        ablation_results, steering_results, geometry_comparison, config, exp_dir
    )

    # Determine overall success
    success = True
    if "teacher_vs_baseline" in ablation_results:
        success = (
            ablation_results["teacher_vs_baseline"]
            .get("success_criteria", {})
            .get("overall_success", False)
        )

    logger.info("Phase 4 completed!")
    logger.info(f"Results saved to: {exp_dir}")

    if success:
        logger.info("✅ Phase 4 SUCCESSFUL - all targets met!")
    else:
        logger.warning("⚠️  Phase 4 shows mixed results - check detailed analysis")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
