"""
Concept steering experiments for H-SAE models.

This module implements steering interventions using parent and child concept vectors,
with support for measuring steering precision and leakage.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging

from .metrics import compute_steering_leakage

logger = logging.getLogger(__name__)


class ConceptSteering:
    """Handles concept steering interventions using H-SAE representations."""
    
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 hsae_model,
                 device: str = "cuda"):
        """
        Initialize concept steering.
        
        Args:
            model: Pre-trained language model
            tokenizer: Tokenizer for the model
            hsae_model: Trained H-SAE model
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.hsae_model = hsae_model
        self.device = device
        
        self.model.eval()
        self.hsae_model.eval()
    
    def steer_with_parent_vector(self,
                               prompts: List[str],
                               parent_vector: torch.Tensor,
                               magnitude: float = 1.0,
                               layer_name: str = "last") -> Dict[str, Any]:
        """
        Steer generation using a parent concept vector.
        
        Args:
            prompts: List of input prompts
            parent_vector: Parent concept vector to add [input_dim]
            magnitude: Steering magnitude
            layer_name: Layer to apply steering to
            
        Returns:
            Dictionary with steering results
        """
        logger.info(f"Steering with parent vector (magnitude={magnitude})")
        
        # Tokenize prompts
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        parent_vector = parent_vector.to(self.device)
        
        # Get baseline logits
        with torch.no_grad():
            baseline_outputs = self.model(**inputs, output_hidden_states=True)
            baseline_logits = baseline_outputs.logits
        
        # Apply steering intervention
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Add steering vector to last token
            hidden_states[:, -1] += magnitude * parent_vector
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        
        # Find target layer and register hook
        hook_handle = None
        if layer_name == "last":
            # Apply to final hidden states before unembedding
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight') and module.weight.shape[0] > 1000:  # Likely unembedding
                    hook_handle = module.register_forward_hook(
                        lambda m, i, o: self.model.lm_head(i[0] + magnitude * parent_vector.unsqueeze(0).unsqueeze(0))
                    )
                    break
        
        if hook_handle is None:
            # Fallback: modify hidden states directly
            with torch.no_grad():
                steered_outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = steered_outputs.hidden_states[-1]
                hidden_states[:, -1] += magnitude * parent_vector
                steered_logits = self.model.lm_head(hidden_states)
        else:
            with torch.no_grad():
                steered_outputs = self.model(**inputs, output_hidden_states=True)
                steered_logits = steered_outputs.logits
            hook_handle.remove()
        
        # Compute steering effects
        logit_deltas = steered_logits - baseline_logits
        
        return {
            'baseline_logits': baseline_logits,
            'steered_logits': steered_logits,
            'logit_deltas': logit_deltas,
            'magnitude': magnitude,
            'prompts': prompts,
        }
    
    def steer_with_child_delta(self,
                             prompts: List[str],
                             parent_vector: torch.Tensor,
                             child_delta: torch.Tensor,
                             parent_magnitude: float = 1.0,
                             child_magnitude: float = 1.0,
                             layer_name: str = "last") -> Dict[str, Any]:
        """
        Steer generation using parent + child delta vectors.
        
        Args:
            prompts: List of input prompts
            parent_vector: Parent concept vector [input_dim]
            child_delta: Child delta vector δ_{c|p} [input_dim]
            parent_magnitude: Magnitude for parent vector
            child_magnitude: Magnitude for child delta
            layer_name: Layer to apply steering to
            
        Returns:
            Dictionary with steering results
        """
        logger.info(f"Steering with parent + child delta (α={parent_magnitude}, β={child_magnitude})")
        
        # Combined steering vector
        steering_vector = parent_magnitude * parent_vector + child_magnitude * child_delta
        
        return self.steer_with_parent_vector(prompts, steering_vector, magnitude=1.0, layer_name=layer_name)
    
    def steer_with_hsae_activation(self,
                                 prompts: List[str],
                                 parent_idx: int,
                                 child_idx: Optional[int] = None,
                                 magnitude: float = 1.0) -> Dict[str, Any]:
        """
        Steer by directly activating H-SAE latents.
        
        Args:
            prompts: List of input prompts
            parent_idx: Parent latent to activate
            child_idx: Child latent to activate (optional)
            magnitude: Activation magnitude
            
        Returns:
            Dictionary with steering results
        """
        logger.info(f"Steering with H-SAE activation: parent={parent_idx}, child={child_idx}")
        
        # Tokenize and get activations
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get original activations (assuming we have them)
        # This is a simplified version - in practice you'd need to capture activations from the model
        with torch.no_grad():
            # Mock activation capture - replace with actual activation capture
            batch_size = inputs['input_ids'].shape[0]
            input_dim = self.hsae_model.config.input_dim
            original_activations = torch.randn(batch_size, input_dim).to(self.device)
            
            # Get H-SAE reconstruction
            x_hat, (parent_codes, child_codes), metrics = self.hsae_model(original_activations)
            
            # Modify codes
            modified_parent_codes = parent_codes.clone()
            modified_child_codes = child_codes.clone()
            
            # Activate specific parent
            modified_parent_codes[:, parent_idx] = magnitude
            
            # Activate specific child if specified
            if child_idx is not None:
                modified_child_codes[:, parent_idx, child_idx] = magnitude
            
            # Reconstruct with modified codes
            steered_activations = self.hsae_model.decode(modified_parent_codes, modified_child_codes)
            
            # Compute difference
            activation_delta = steered_activations - original_activations
        
        # Apply activation delta as steering (this would need proper integration with model)
        return self.steer_with_parent_vector(prompts, activation_delta.mean(dim=0), magnitude=1.0)
    
    def evaluate_steering_precision(self,
                                  steering_results: Dict[str, Any],
                                  target_concept: str,
                                  sibling_concepts: List[str]) -> Dict[str, float]:
        """
        Evaluate steering precision by measuring target vs sibling effects.
        
        Args:
            steering_results: Results from steering experiment
            target_concept: Target concept name
            sibling_concepts: List of sibling concept names
            
        Returns:
            Dictionary with precision metrics
        """
        # Get token IDs for concepts
        target_tokens = self._get_concept_tokens(target_concept)
        sibling_tokens = []
        for sibling in sibling_concepts:
            sibling_tokens.extend(self._get_concept_tokens(sibling))
        
        # Compute steering leakage
        leakage_metrics = compute_steering_leakage(
            steering_results['baseline_logits'],
            steering_results['steered_logits'],
            target_tokens,
            sibling_tokens
        )
        
        return leakage_metrics
    
    def _get_concept_tokens(self, concept: str) -> List[int]:
        """Get token IDs associated with a concept."""
        # Simple tokenization - in practice you'd want more sophisticated concept-to-token mapping
        tokens = self.tokenizer.encode(concept, add_special_tokens=False)
        return tokens
    
    def run_steering_suite(self,
                          parent_vectors: Dict[str, torch.Tensor],
                          child_deltas: Dict[str, Dict[str, torch.Tensor]],
                          test_prompts: Dict[str, List[str]],
                          magnitudes: List[float] = [0.5, 1.0, 2.0]) -> Dict[str, Any]:
        """
        Run comprehensive steering evaluation suite.
        
        Args:
            parent_vectors: Parent concept vectors
            child_deltas: Child delta vectors
            test_prompts: Test prompts for each concept
            magnitudes: Steering magnitudes to test
            
        Returns:
            Dictionary with comprehensive steering results
        """
        logger.info("Running comprehensive steering evaluation")
        
        results = {
            'parent_steering': {},
            'child_steering': {},
            'precision_metrics': {},
        }
        
        for parent_id, parent_vector in parent_vectors.items():
            if parent_id not in test_prompts:
                continue
                
            prompts = test_prompts[parent_id][:5]  # Limit for efficiency
            
            # Test parent steering at different magnitudes
            parent_results = []
            for magnitude in magnitudes:
                steering_result = self.steer_with_parent_vector(
                    prompts, parent_vector, magnitude=magnitude
                )
                parent_results.append({
                    'magnitude': magnitude,
                    'mean_effect': torch.mean(torch.abs(steering_result['logit_deltas'])).item(),
                    'max_effect': torch.max(torch.abs(steering_result['logit_deltas'])).item(),
                })
            
            results['parent_steering'][parent_id] = parent_results
            
            # Test child steering if available
            if parent_id in child_deltas:
                child_results = {}
                for child_id, child_delta in child_deltas[parent_id].items():
                    child_magnitude_results = []
                    for magnitude in magnitudes:
                        steering_result = self.steer_with_child_delta(
                            prompts, parent_vector, child_delta, 
                            parent_magnitude=1.0, child_magnitude=magnitude
                        )
                        child_magnitude_results.append({
                            'magnitude': magnitude,
                            'mean_effect': torch.mean(torch.abs(steering_result['logit_deltas'])).item(),
                            'max_effect': torch.max(torch.abs(steering_result['logit_deltas'])).item(),
                        })
                    child_results[child_id] = child_magnitude_results
                
                results['child_steering'][parent_id] = child_results
        
        logger.info("Steering evaluation completed")
        return results


def analyze_steering_results(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyze steering results and compute summary statistics.
    
    Args:
        results: Results from run_steering_suite
        
    Returns:
        Dictionary with summary statistics
    """
    # Analyze parent steering
    parent_effects = []
    for parent_id, parent_results in results['parent_steering'].items():
        for result in parent_results:
            parent_effects.append(result['mean_effect'])
    
    # Analyze child steering
    child_effects = []
    for parent_id, child_results in results['child_steering'].items():
        for child_id, child_magnitude_results in child_results.items():
            for result in child_magnitude_results:
                child_effects.append(result['mean_effect'])
    
    summary = {
        'parent_steering_mean': np.mean(parent_effects) if parent_effects else 0.0,
        'parent_steering_std': np.std(parent_effects) if parent_effects else 0.0,
        'child_steering_mean': np.mean(child_effects) if child_effects else 0.0,
        'child_steering_std': np.std(child_effects) if child_effects else 0.0,
        'n_parent_tests': len(parent_effects),
        'n_child_tests': len(child_effects),
    }
    
    return summary