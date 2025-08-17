"""
Validation and control experiments for geometric claims.

This module implements orthogonality tests, ratio-invariance checks, and control
experiments to validate the geometric structure of concept hierarchies.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class GeometryValidator:
    """Validates geometric claims about concept hierarchies."""
    
    def __init__(self, geometry, significance_level: float = 0.05):
        """
        Initialize geometry validator.
        
        Args:
            geometry: CausalGeometry instance
            significance_level: Alpha level for statistical tests
        """
        self.geometry = geometry
        self.alpha = significance_level
    
    def test_hierarchical_orthogonality(self,
                                      parent_vectors: Dict[str, torch.Tensor],
                                      child_deltas: Dict[str, Dict[str, torch.Tensor]],
                                      threshold_degrees: float = 80.0) -> Dict[str, Any]:
        """
        Test the claim that ⟨ℓ_p, δ_{c|p}⟩_c ≈ 0.
        
        Args:
            parent_vectors: Parent concept vectors
            child_deltas: Child delta vectors  
            threshold_degrees: Angle threshold for "orthogonal"
            
        Returns:
            Dictionary with test results
        """
        logger.info("Testing hierarchical orthogonality")
        
        angles_deg = []
        inner_products = []
        parent_child_pairs = []
        
        for parent_id, deltas in child_deltas.items():
            if parent_id not in parent_vectors:
                continue
                
            parent_vector = parent_vectors[parent_id]
            
            for child_id, delta in deltas.items():
                if torch.norm(delta) < 1e-6:  # Skip zero deltas
                    continue
                    
                # Compute angle and inner product
                angle_rad = self.geometry.causal_angle(parent_vector, delta)
                angle_deg = torch.rad2deg(angle_rad).item()
                inner_prod = self.geometry.causal_inner_product(parent_vector, delta).item()
                
                angles_deg.append(angle_deg)
                inner_products.append(inner_prod)
                parent_child_pairs.append((parent_id, child_id))
        
        angles_deg = np.array(angles_deg)
        inner_products = np.array(inner_products)
        
        # Compute statistics
        results = {
            'n_pairs': len(angles_deg),
            'angles_deg': angles_deg,
            'inner_products': inner_products,
            'parent_child_pairs': parent_child_pairs,
            
            # Angle statistics
            'median_angle_deg': np.median(angles_deg),
            'mean_angle_deg': np.mean(angles_deg),
            'std_angle_deg': np.std(angles_deg),
            'q25_angle_deg': np.percentile(angles_deg, 25),
            'q75_angle_deg': np.percentile(angles_deg, 75),
            
            # Orthogonality metrics
            'fraction_above_threshold': np.mean(angles_deg >= threshold_degrees),
            'fraction_above_85deg': np.mean(angles_deg >= 85.0),
            'fraction_above_87deg': np.mean(angles_deg >= 87.0),
            
            # Inner product statistics (should be near zero)
            'mean_abs_inner_product': np.mean(np.abs(inner_products)),
            'median_abs_inner_product': np.median(np.abs(inner_products)),
            'max_abs_inner_product': np.max(np.abs(inner_products)),
        }
        
        # Statistical test: are angles significantly > 80 degrees?
        t_stat, p_value = stats.ttest_1samp(angles_deg, threshold_degrees)
        results['ttest_vs_threshold'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha and t_stat > 0
        }
        
        return results
    
    def test_ratio_invariance(self,
                            model, tokenizer,
                            parent_vectors: Dict[str, torch.Tensor],
                            test_contexts: Dict[str, List[str]],
                            sibling_groups: Dict[str, List[str]],
                            alpha_values: List[float] = [0.5, 1.0, 2.0],
                            kl_threshold: float = 0.10) -> Dict[str, Any]:
        """
        Test ratio-invariance: moving along ℓ_p should preserve sibling ratios.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            parent_vectors: Parent concept vectors
            test_contexts: Dict mapping parent_id -> list of test contexts
            sibling_groups: Dict mapping parent_id -> list of sibling token strings
            alpha_values: Intervention magnitudes
            kl_threshold: KL divergence threshold for invariance
            
        Returns:
            Dictionary with invariance test results
        """
        logger.info("Testing ratio-invariance")
        
        device = next(model.parameters()).device
        results = {
            'alpha_values': alpha_values,
            'parent_results': {},
            'overall_stats': {}
        }
        
        all_kl_divergences = []
        
        for parent_id, contexts in test_contexts.items():
            if parent_id not in parent_vectors or parent_id not in sibling_groups:
                continue
                
            logger.info(f"Testing invariance for parent {parent_id}")
            parent_vector = parent_vectors[parent_id].to(device)
            sibling_tokens = sibling_groups[parent_id]
            
            # Get sibling token IDs
            sibling_ids = []
            for token_str in sibling_tokens:
                token_id = tokenizer.encode(token_str, add_special_tokens=False)
                if len(token_id) == 1:  # Single token only
                    sibling_ids.append(token_id[0])
            
            if len(sibling_ids) < 2:
                logger.warning(f"Insufficient sibling tokens for {parent_id}")
                continue
            
            parent_kls = []
            
            for context in contexts[:10]:  # Limit contexts for compute
                # Get baseline sibling distribution
                inputs = tokenizer(context, return_tensors='pt').to(device)
                with torch.no_grad():
                    baseline_outputs = model(**inputs)
                    baseline_logits = baseline_outputs.logits[0, -1, sibling_ids]
                    baseline_probs = torch.softmax(baseline_logits, dim=0)
                
                context_kls = []
                
                for alpha in alpha_values:
                    # Define intervention hook
                    def intervention_hook(module, input, output):
                        output[0][:, -1] += alpha * parent_vector
                        return output
                    
                    # Find and register hook
                    hook_handle = None
                    for name, module in model.named_modules():
                        if 'final' in name.lower() or 'last' in name.lower():
                            hook_handle = module.register_forward_hook(intervention_hook)
                            break
                    
                    if hook_handle is None:
                        continue
                    
                    # Run with intervention
                    with torch.no_grad():
                        intervention_outputs = model(**inputs)
                        intervention_logits = intervention_outputs.logits[0, -1, sibling_ids]
                        intervention_probs = torch.softmax(intervention_logits, dim=0)
                    
                    hook_handle.remove()
                    
                    # Compute KL divergence
                    kl_div = torch.nn.functional.kl_div(
                        torch.log(intervention_probs + 1e-8),
                        baseline_probs,
                        reduction='sum'
                    ).item()
                    
                    context_kls.append(kl_div)
                
                parent_kls.extend(context_kls)
            
            # Store results for this parent
            results['parent_results'][parent_id] = {
                'kl_divergences': parent_kls,
                'median_kl': np.median(parent_kls),
                'mean_kl': np.mean(parent_kls),
                'fraction_below_threshold': np.mean(np.array(parent_kls) < kl_threshold)
            }
            
            all_kl_divergences.extend(parent_kls)
        
        # Overall statistics
        all_kl_divergences = np.array(all_kl_divergences)
        results['overall_stats'] = {
            'median_kl': np.median(all_kl_divergences),
            'mean_kl': np.mean(all_kl_divergences),
            'std_kl': np.std(all_kl_divergences),
            'q90_kl': np.percentile(all_kl_divergences, 90),
            'fraction_below_threshold': np.mean(all_kl_divergences < kl_threshold),
            'fraction_below_020': np.mean(all_kl_divergences < 0.20),
        }
        
        return results
    
    def run_control_experiments(self,
                              parent_vectors: Dict[str, torch.Tensor],
                              child_deltas: Dict[str, Dict[str, torch.Tensor]],
                              unembedding_matrix: torch.Tensor,
                              n_shuffles: int = 100) -> Dict[str, Any]:
        """
        Run control experiments to validate geometric claims.
        
        Args:
            parent_vectors: Original parent vectors
            child_deltas: Original child deltas
            unembedding_matrix: Original unembedding matrix
            n_shuffles: Number of shuffle iterations
            
        Returns:
            Dictionary with control experiment results
        """
        logger.info("Running control experiments")
        
        results = {
            'shuffled_unembedding': [],
            'random_parent_replacement': [],
            'label_permutation': []
        }
        
        # Original orthogonality scores
        original_orthogonality = self.test_hierarchical_orthogonality(parent_vectors, child_deltas)
        original_median_angle = original_orthogonality['median_angle_deg']
        
        for shuffle_idx in range(n_shuffles):
            if shuffle_idx % 20 == 0:
                logger.info(f"Control shuffle {shuffle_idx}/{n_shuffles}")
            
            # Control 1: Shuffled unembedding rows
            shuffled_U = unembedding_matrix[torch.randperm(unembedding_matrix.shape[0])]
            shuffled_geometry = self.geometry.__class__(shuffled_U, self.geometry.shrinkage)
            
            # Re-compute parent vectors with shuffled geometry (using same activations)
            # Note: This is a simplified version - in practice you'd re-run the full estimation
            shuffled_angles = []
            for parent_id, deltas in child_deltas.items():
                if parent_id not in parent_vectors:
                    continue
                parent_vector = parent_vectors[parent_id]
                for child_id, delta in deltas.items():
                    if torch.norm(delta) > 1e-6:
                        angle = shuffled_geometry.causal_angle(parent_vector, delta)
                        shuffled_angles.append(torch.rad2deg(angle).item())
            
            if shuffled_angles:
                results['shuffled_unembedding'].append(np.median(shuffled_angles))
            
            # Control 2: Random parent replacement
            parent_ids = list(parent_vectors.keys())
            if len(parent_ids) >= 2:
                # Randomly reassign parent vectors
                shuffled_parent_ids = parent_ids.copy()
                np.random.shuffle(shuffled_parent_ids)
                
                random_angles = []
                for i, parent_id in enumerate(parent_ids):
                    if parent_id in child_deltas:
                        # Use shuffled parent vector
                        random_parent = parent_vectors[shuffled_parent_ids[i]]
                        for child_id, delta in child_deltas[parent_id].items():
                            if torch.norm(delta) > 1e-6:
                                angle = self.geometry.causal_angle(random_parent, delta)
                                random_angles.append(torch.rad2deg(angle).item())
                
                if random_angles:
                    results['random_parent_replacement'].append(np.median(random_angles))
            
            # Control 3: Label permutation (simplified)
            # This would require re-running estimation with permuted labels
            # For now, we'll use a proxy by randomly pairing parent vectors with child deltas
            all_deltas = []
            for deltas in child_deltas.values():
                all_deltas.extend(list(deltas.values()))
            
            if len(all_deltas) > 0 and len(parent_vectors) > 0:
                parent_list = list(parent_vectors.values())
                permuted_angles = []
                
                for _ in range(min(20, len(all_deltas))):  # Sample some pairs
                    random_parent = parent_list[np.random.randint(len(parent_list))]
                    random_delta = all_deltas[np.random.randint(len(all_deltas))]
                    if torch.norm(random_delta) > 1e-6:
                        angle = self.geometry.causal_angle(random_parent, random_delta)
                        permuted_angles.append(torch.rad2deg(angle).item())
                
                if permuted_angles:
                    results['label_permutation'].append(np.median(permuted_angles))
        
        # Compute summary statistics for controls
        summary = {
            'original_median_angle': original_median_angle,
            'controls': {}
        }
        
        for control_name, angles in results.items():
            if angles:
                summary['controls'][control_name] = {
                    'median_angle': np.median(angles),
                    'mean_angle': np.mean(angles),
                    'std_angle': np.std(angles),
                    'effect_size_vs_original': (original_median_angle - np.mean(angles)) / np.std(angles) if np.std(angles) > 0 else 0
                }
        
        return {
            'raw_results': results,
            'summary': summary
        }


def compute_intervention_effect_sizes(intervention_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute Cohen's d effect sizes for intervention experiments.
    
    Args:
        intervention_results: Results from intervention_test function
        
    Returns:
        Dictionary with effect size statistics
    """
    target_deltas = np.array(intervention_results['target_deltas'])
    sibling_deltas = np.array(intervention_results['sibling_deltas'])
    
    # Cohen's d for each alpha
    effect_sizes = []
    for i in range(len(target_deltas)):
        target_mean = target_deltas[i]
        sibling_mean = sibling_deltas[i]
        
        # Approximate pooled standard deviation (simplified)
        pooled_std = np.sqrt((np.var(target_deltas) + np.var(sibling_deltas)) / 2)
        
        if pooled_std > 0:
            cohens_d = (target_mean - sibling_mean) / pooled_std
            effect_sizes.append(cohens_d)
    
    return {
        'effect_sizes': effect_sizes,
        'mean_effect_size': np.mean(effect_sizes) if effect_sizes else 0,
        'max_effect_size': np.max(effect_sizes) if effect_sizes else 0,
        'target_vs_sibling_ratio': np.mean(np.abs(target_deltas) / (np.abs(sibling_deltas) + 1e-8))
    }