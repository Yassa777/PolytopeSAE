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
import logging
import yaml
from pathlib import Path
import torch
from tqdm import tqdm
import json
import numpy as np
from datetime import datetime

# Project imports
from polytope_hsae.geometry import CausalGeometry, compute_polytope_angles, intervention_test
from polytope_hsae.estimators import ConceptVectorEstimator, LDAEstimator, validate_orthogonality
from polytope_hsae.validation import GeometryValidator
from polytope_hsae.concepts import ConceptCurator, PromptGenerator, ConceptSplitter
from polytope_hsae.activations import ActivationCapture, ActivationConfig
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_experiment(config):
    """Set up experiment directories and logging."""
    exp_dir = Path(config['logging']['save_dir']) / config['logging']['phase_1_log']
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = exp_dir / f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Phase 1 experiment directory: {exp_dir}")
    return exp_dir


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
            max_parents=config['concepts']['parents'],
            max_children_per_parent=config['concepts']['max_children_per_parent'],
            min_children_per_parent=2
        )
        
        # Curate hierarchies
        hierarchies = curator.curate_concept_hierarchies()
        
        # Generate prompts
        prompt_generator = PromptGenerator(
            prompts_per_concept=config['concepts']['prompts_per_concept']
        )
        
        hierarchies = [prompt_generator.populate_hierarchy_prompts(h) for h in hierarchies]
        
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
        activation_capture = ActivationCapture(ActivationConfig(
            model_name=config['model']['name'],
            layer_name=config['model']['activation_hook'],
            batch_size=32,
            device=config['run']['device']
        ))
        return activation_capture.load_hierarchical_activations(str(activations_file))
    
    logger.info("Capturing new activations")
    
    # Set up activation capture
    activation_config = ActivationConfig(
        model_name=config['model']['name'],
        layer_name=config['model']['activation_hook'],
        batch_size=32,
        device=config['run']['device']
    )
    
    activation_capture = ActivationCapture(activation_config)
    
    # Capture hierarchical activations
    activations = activation_capture.capture_hierarchical_activations(
        hierarchies,
        negative_sampling_ratio=1.0,
        save_path=str(activations_file)
    )
    
    return activations


def extract_teacher_vectors(config, activations, exp_dir):
    """Extract parent vectors and child deltas using LDA."""
    logger.info("Extracting teacher vectors")
    
    # Load model to get unembedding matrix
    model = AutoModel.from_pretrained(config['model']['name'])
    
    # Get unembedding matrix (assuming it's the final linear layer)
    unembedding_matrix = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' in name.lower():
            unembedding_matrix = module.weight.data
            break
        elif isinstance(module, torch.nn.Linear) and hasattr(module, 'weight') and module.weight.shape[0] > 10000:
            # Likely the unembedding layer (large vocab size)
            unembedding_matrix = module.weight.data
            break
    
    if unembedding_matrix is None:
        # Fallback: use embed_tokens transposed
        if hasattr(model, 'embed_tokens'):
            unembedding_matrix = model.embed_tokens.weight.data.T
        else:
            raise ValueError("Could not find unembedding matrix")
    
    logger.info(f"Unembedding matrix shape: {unembedding_matrix.shape}")
    
    # Create causal geometry
    geometry = CausalGeometry(
        unembedding_matrix,
        shrinkage=config['geometry']['shrinkage']
    )
    
    # Create LDA estimator
    lda_estimator = LDAEstimator(
        shrinkage=config['geometry']['estimator']['lda_shrinkage'],
        min_vocab_count=config['geometry']['estimator']['min_vocab_count'],
        class_balance=True
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
    
    # Estimate child deltas
    child_deltas = estimator.estimate_child_deltas(parent_vectors, child_activations)
    
    # Estimate projectors for H-SAE initialization
    projectors = estimator.estimate_child_subspace_projectors(
        parent_vectors, child_deltas, config['hsae']['subspace_dim']
    )
    
    # Save results
    results = {
        'parent_vectors': {k: v.tolist() for k, v in parent_vectors.items()},
        'child_deltas': {
            parent_id: {child_id: delta.tolist() for child_id, delta in deltas.items()}
            for parent_id, deltas in child_deltas.items()
        },
        'projectors': {
            parent_id: {
                'down': down.tolist(),
                'up': up.tolist()
            }
            for parent_id, (down, up) in projectors.items()
        }
    }
    
    with open(exp_dir / "teacher_vectors.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return parent_vectors, child_deltas, projectors, geometry


def validate_geometry(config, parent_vectors, child_deltas, geometry, exp_dir):
    """Validate geometric claims."""
    logger.info("Validating geometric claims")
    
    # Create validator
    validator = GeometryValidator(geometry)
    
    # Test hierarchical orthogonality
    orthogonality_results = validator.test_hierarchical_orthogonality(
        parent_vectors, child_deltas, 
        threshold_degrees=config['eval']['targets']['median_angle_deg']
    )
    
    # Compute angle statistics
    angle_stats = compute_polytope_angles(
        torch.stack(list(parent_vectors.values())),
        torch.stack([
            torch.stack(list(deltas.values())) 
            for deltas in child_deltas.values()
        ]),
        geometry
    )
    
    # Run control experiments
    # Load unembedding matrix again for controls
    model = AutoModel.from_pretrained(config['model']['name'])
    unembedding_matrix = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' in name.lower():
            unembedding_matrix = module.weight.data
            break
    
    if unembedding_matrix is not None:
        control_results = validator.run_control_experiments(
            parent_vectors, child_deltas, unembedding_matrix, n_shuffles=50
        )
    else:
        control_results = {"error": "Could not run controls - no unembedding matrix"}
    
    # Validate orthogonality with estimator function
    orthogonality_validation = validate_orthogonality(parent_vectors, child_deltas, geometry)
    
    # Compile results
    validation_results = {
        'orthogonality_test': {
            'median_angle_deg': orthogonality_results['median_angle_deg'],
            'fraction_above_80deg': orthogonality_results['fraction_above_threshold'],
            'fraction_above_85deg': orthogonality_results['fraction_above_85deg'],
            'passes_threshold': orthogonality_results['median_angle_deg'] >= config['eval']['targets']['median_angle_deg']
        },
        'angle_statistics': {
            'median_angle_deg': angle_stats['median_angle_deg'].item(),
            'mean_angle_deg': angle_stats['mean_angle_deg'].item(),
            'q25_angle_deg': angle_stats['q25_angle_deg'].item(),
            'q75_angle_deg': angle_stats['q75_angle_deg'].item(),
        },
        'orthogonality_validation': orthogonality_validation,
        'control_experiments': control_results,
        'passes_validation': (
            orthogonality_results['median_angle_deg'] >= config['eval']['targets']['median_angle_deg'] and
            orthogonality_results['fraction_above_threshold'] >= 0.5
        )
    }
    
    # Save validation results
    with open(exp_dir / "validation_results.json", 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    # Log key results
    logger.info(f"Median angle: {validation_results['orthogonality_test']['median_angle_deg']:.1f}°")
    logger.info(f"Fraction above 80°: {validation_results['orthogonality_test']['fraction_above_80deg']:.3f}")
    logger.info(f"Validation passed: {validation_results['passes_validation']}")
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Teacher Vector Extraction")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if specified
    if args.device:
        config['run']['device'] = args.device
    
    # Set up experiment
    exp_dir = setup_experiment(config)
    
    # Load or create concepts
    hierarchies = load_or_create_concepts(config, exp_dir)
    
    # Capture activations
    activations = capture_activations(config, hierarchies, exp_dir)
    
    # Extract teacher vectors
    parent_vectors, child_deltas, projectors, geometry = extract_teacher_vectors(
        config, activations, exp_dir
    )
    
    # Validate geometry
    validation_results = validate_geometry(
        config, parent_vectors, child_deltas, geometry, exp_dir
    )
    
    # Final report
    logger.info("Phase 1 completed!")
    logger.info(f"Results saved to: {exp_dir}")
    
    if validation_results['passes_validation']:
        logger.info("✅ Geometric validation PASSED - ready for Phase 2")
    else:
        logger.warning("❌ Geometric validation FAILED - check results before proceeding")
    
    return validation_results['passes_validation']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)