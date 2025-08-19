"""
Statistical estimators for concept vectors using LDA and related methods.

This module implements LDA-based estimators for finding parent vectors and child
deltas in the whitened causal space.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import LedoitWolf
from typing import Tuple, Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class LDAEstimator:
    """LDA-based estimator for concept directions in causal space."""
    
    def __init__(self, 
                 shrinkage: float = 0.1,
                 min_vocab_count: int = 50,
                 class_balance: bool = True):
        """
        Initialize LDA estimator.
        
        Args:
            shrinkage: Shrinkage parameter for within-class covariance
            min_vocab_count: Minimum number of samples per class
            class_balance: Whether to balance classes during estimation
        """
        self.shrinkage = shrinkage
        self.min_vocab_count = min_vocab_count
        self.class_balance = class_balance
        
    def estimate_binary_direction(self, 
                                 X_pos: torch.Tensor, 
                                 X_neg: torch.Tensor,
                                 geometry) -> torch.Tensor:
        """
        Estimate LDA direction for binary classification in causal space.
        
        Args:
            X_pos: Positive examples [N_pos, d]
            X_neg: Negative examples [N_neg, d] 
            geometry: CausalGeometry instance for whitening
            
        Returns:
            LDA direction vector normalized under causal norm
        """
        # Whiten the data
        X_pos_whitened = geometry.whiten(X_pos)
        X_neg_whitened = geometry.whiten(X_neg)
        
        # Check minimum sample requirements
        if len(X_pos_whitened) < self.min_vocab_count or len(X_neg_whitened) < self.min_vocab_count:
            logger.warning(f"Insufficient samples: pos={len(X_pos_whitened)}, neg={len(X_neg_whitened)}")
            
        # Balance classes if requested
        if self.class_balance:
            min_samples = min(len(X_pos_whitened), len(X_neg_whitened))
            X_pos_whitened = X_pos_whitened[:min_samples]
            X_neg_whitened = X_neg_whitened[:min_samples]
        
        # Compute class means
        mu_pos = X_pos_whitened.mean(dim=0)
        mu_neg = X_neg_whitened.mean(dim=0)
        
        # Compute within-class scatter matrix with shrinkage
        X_combined = torch.cat([X_pos_whitened, X_neg_whitened], dim=0)
        
        # Center within each class
        X_pos_centered = X_pos_whitened - mu_pos
        X_neg_centered = X_neg_whitened - mu_neg
        X_centered = torch.cat([X_pos_centered, X_neg_centered], dim=0)
        
        # Within-class scatter with shrinkage
        S_w = torch.cov(X_centered.T) + self.shrinkage * torch.eye(X_centered.shape[1])
        
        try:
            # LDA direction: S_w^{-1} (μ_1 - μ_0)
            direction = torch.linalg.solve(S_w, mu_pos - mu_neg)
        except torch.linalg.LinAlgError:
            logger.warning("Singular within-class scatter matrix, using pseudoinverse")
            direction = torch.linalg.pinv(S_w) @ (mu_pos - mu_neg)
            
        # Transform back to original space and normalize
        direction_original = direction @ geometry.W.T  # W^T maps whitened back to original
        return geometry.normalize_causal(direction_original)
    
    def estimate_multiclass_directions(self,
                                     X_list: List[torch.Tensor],
                                     geometry,
                                     method: str = 'one_vs_rest') -> torch.Tensor:
        """
        Estimate directions for multiclass classification.
        
        Args:
            X_list: List of tensors, one per class [N_i, d]
            geometry: CausalGeometry instance
            method: 'one_vs_rest' or 'pairwise'
            
        Returns:
            Matrix of class directions [n_classes, d]
        """
        n_classes = len(X_list)
        d = X_list[0].shape[1]
        directions = torch.zeros(n_classes, d)
        
        if method == 'one_vs_rest':
            for i, X_pos in enumerate(X_list):
                # Combine all other classes as negative
                X_neg = torch.cat([X_list[j] for j in range(n_classes) if j != i], dim=0)
                directions[i] = self.estimate_binary_direction(X_pos, X_neg, geometry)
                
        elif method == 'pairwise':
            # Use sklearn's LDA for multiclass
            X_all = torch.cat(X_list, dim=0)
            y_all = torch.cat([torch.full((len(X_list[i]),), i) for i in range(n_classes)])
            
            # Whiten data
            X_whitened = geometry.whiten(X_all)
            
            # Fit LDA with proper solver for shrinkage
            lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=self.shrinkage)
            lda.fit(X_whitened.to(dtype=torch.float32).cpu().numpy(),
                    y_all.cpu().numpy())
            
            # Extract directions and transform back
            for i in range(n_classes):
                direction_whitened = torch.from_numpy(lda.coef_[i]).float()
                direction_original = direction_whitened @ geometry.W.T
                directions[i] = geometry.normalize_causal(direction_original)
        
        return directions


class ConceptVectorEstimator:
    """High-level estimator for parent vectors and child deltas."""
    
    def __init__(self, 
                 lda_estimator: Optional[LDAEstimator] = None,
                 geometry = None):
        """
        Initialize concept vector estimator.
        
        Args:
            lda_estimator: LDA estimator instance
            geometry: CausalGeometry instance
        """
        self.lda_estimator = lda_estimator or LDAEstimator()
        self.geometry = geometry
        
    def estimate_parent_vectors(self,
                              parent_activations: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Estimate parent concept vectors.
        
        Args:
            parent_activations: Dict mapping parent_id -> {'pos': tensor, 'neg': tensor}
            
        Returns:
            Dict mapping parent_id -> parent_vector
        """
        parent_vectors = {}
        
        for parent_id, data in parent_activations.items():
            logger.info(f"Estimating parent vector for {parent_id}")
            
            X_pos = data['pos']  # Activations where parent concept is present
            X_neg = data['neg']  # Activations where parent concept is absent
            
            parent_vector = self.lda_estimator.estimate_binary_direction(
                X_pos, X_neg, self.geometry
            )
            parent_vectors[parent_id] = parent_vector
            
        return parent_vectors
    
    def estimate_child_deltas(self,
                            parent_vectors: Dict[str, torch.Tensor],
                            child_activations: Dict[str, Dict[str, Dict[str, torch.Tensor]]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Estimate child delta vectors δ_{c|p} = ℓ_c - ℓ_p.
        
        Args:
            parent_vectors: Dict mapping parent_id -> parent_vector
            child_activations: Dict mapping parent_id -> child_id -> {'pos': tensor, 'neg': tensor}
            
        Returns:
            Dict mapping parent_id -> child_id -> delta_vector
        """
        child_deltas = {}
        
        for parent_id, children_data in child_activations.items():
            if parent_id not in parent_vectors:
                logger.warning(f"No parent vector for {parent_id}, skipping children")
                continue
                
            parent_vector = parent_vectors[parent_id]
            child_deltas[parent_id] = {}
            
            for child_id, data in children_data.items():
                logger.info(f"Estimating child delta for {parent_id}/{child_id}")
                
                X_pos = data['pos']  # Activations where child concept is present
                X_neg = data['neg']  # Activations where child concept is absent
                
                # Estimate child vector
                child_vector = self.lda_estimator.estimate_binary_direction(
                    X_pos, X_neg, self.geometry
                )
                
                # Compute delta: δ_{c|p} = ℓ_c - ℓ_p
                delta_vector = child_vector - parent_vector
                child_deltas[parent_id][child_id] = delta_vector
                
        return child_deltas
    
    def estimate_child_subspace_projectors(self,
                                         parent_vectors: Dict[str, torch.Tensor],
                                         child_deltas: Dict[str, Dict[str, torch.Tensor]],
                                         subspace_dim: int = 96) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Estimate down/up projectors for child subspaces using SVD.
        
        Args:
            parent_vectors: Parent concept vectors
            child_deltas: Child delta vectors
            subspace_dim: Dimension of child subspace
            
        Returns:
            Dict mapping parent_id -> (down_projector, up_projector)
        """
        projectors = {}
        
        for parent_id, deltas in child_deltas.items():
            if len(deltas) == 0:
                continue
                
            logger.info(f"Computing projectors for parent {parent_id}")
            
            # Stack child deltas
            delta_matrix = torch.stack(list(deltas.values()))  # [n_children, d]
            
            # SVD to find subspace
            U, S, Vh = torch.linalg.svd(delta_matrix, full_matrices=False)
            V = Vh.t()
            
            # Take top subspace_dim components
            k = min(subspace_dim, delta_matrix.shape[0], delta_matrix.shape[1])
            V_k = V[:, :k]  # [d, k]
            
            # Down projector: residual -> subspace
            down_projector = V_k.T  # [k, d]
            
            # Up projector: subspace -> residual  
            up_projector = V_k  # [d, k]
            
            projectors[parent_id] = (down_projector, up_projector)
            
        return projectors


def ridge_cross_layer_mapping(source_vectors: torch.Tensor,
                            target_vectors: torch.Tensor,
                            alpha: float = 1.0) -> torch.Tensor:
    """
    Learn ridge regression mapping from source layer to target layer.
    
    Args:
        source_vectors: Vectors in source layer [n_concepts, d_source]
        target_vectors: Corresponding vectors in target layer [n_concepts, d_target]
        alpha: Ridge regularization parameter
        
    Returns:
        Mapping matrix [d_target, d_source]
    """
    # Ridge regression: W = (X^T X + αI)^{-1} X^T Y
    # where X = source_vectors, Y = target_vectors
    
    X = source_vectors  # [n, d_source]
    Y = target_vectors  # [n, d_target]
    
    # Add regularization
    XTX = X.T @ X + alpha * torch.eye(X.shape[1])
    XTY = X.T @ Y
    
    # Solve for mapping
    W = torch.linalg.solve(XTX, XTY).T  # [d_target, d_source]
    
    return W


def validate_orthogonality(parent_vectors: Dict[str, torch.Tensor],
                         child_deltas: Dict[str, Dict[str, torch.Tensor]],
                         geometry) -> Dict[str, Any]:
    """
    Validate hierarchical orthogonality: ⟨ℓ_p, δ_{c|p}⟩_c ≈ 0.
    
    Args:
        parent_vectors: Parent concept vectors
        child_deltas: Child delta vectors
        geometry: CausalGeometry instance
        
    Returns:
        Dictionary with orthogonality statistics
    """
    inner_products = []
    angles = []
    
    for parent_id, deltas in child_deltas.items():
        if parent_id not in parent_vectors:
            continue
            
        parent_vector = parent_vectors[parent_id]
        
        for child_id, delta in deltas.items():
            # Compute causal inner product
            inner_prod = geometry.causal_inner_product(parent_vector, delta)
            inner_products.append(inner_prod.item())
            
            # Compute angle
            angle = geometry.causal_angle(parent_vector, delta)
            angles.append(torch.rad2deg(angle).item())
    
    inner_products = np.array(inner_products)
    angles = np.array(angles)
    
    return {
        'inner_products': inner_products,
        'angles_deg': angles,
        'mean_inner_product': np.mean(np.abs(inner_products)),
        'median_angle_deg': np.median(angles),
        'q25_angle_deg': np.percentile(angles, 25),
        'q75_angle_deg': np.percentile(angles, 75),
        'fraction_orthogonal_80deg': np.mean(angles > 80),
        'fraction_orthogonal_85deg': np.mean(angles > 85),
    }