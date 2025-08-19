"""
Geometric computations for causal inner products, whitening, and polytope analysis.

This module implements the core geometric operations needed for analyzing concept
hierarchies in the causal (whitened) metric space.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CausalGeometry:
    """Handles causal inner product computations and geometric operations."""

    def __init__(
        self,
        unembedding_matrix: torch.Tensor,
        shrinkage: float = 0.05,
        cov_device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize causal geometry with unembedding matrix.

        Args:
            unembedding_matrix: U ∈ R^{V×d} unembedding matrix
            shrinkage: Shrinkage parameter for covariance regularization
            cov_device: Device used for covariance computation
            cache_dir: Optional directory to cache Σ and W for reuse
        """
        self.shrinkage = shrinkage
        self.device = unembedding_matrix.device
        self.dtype = torch.float32
        self.cov_device = cov_device
        self.cache_dir = cache_dir
        self.U = unembedding_matrix.detach()

        # Compute whitening matrix
        self.Sigma, self.W = self._compute_whitening_matrix(self.U)

    def _compute_whitening_matrix(
        self, unembedding_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute covariance matrix Σ and whitening matrix W = Σ^{-1/2}.

        Args:
            unembedding_matrix: Unembedding matrix

        Returns:
            Tuple of (Sigma, W) tensors
        """
        sigma_file = w_file = None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            sigma_file = os.path.join(self.cache_dir, "sigma.pt")
            w_file = os.path.join(self.cache_dir, "whitening.pt")
            if os.path.exists(sigma_file) and os.path.exists(w_file):
                logger.info("Loading cached whitening matrices from disk")
                Sigma = torch.load(sigma_file, map_location=self.cov_device)
                W = torch.load(w_file, map_location=self.cov_device)
                return Sigma, W

        logger.info("Computing whitening matrix from unembedding rows")

        U = unembedding_matrix.detach().to(self.cov_device, dtype=torch.float32)
        U_centered = U - U.mean(dim=0, keepdim=True)
        Sigma = torch.cov(U_centered.T) + self.shrinkage * torch.eye(
            U.shape[1], device=self.cov_device
        )

        eigenvals, eigenvecs = torch.linalg.eigh(Sigma)
        eigenvals = torch.clamp(eigenvals, min=1e-8)
        W = eigenvecs @ torch.diag(1.0 / torch.sqrt(eigenvals)) @ eigenvecs.T

        if sigma_file is not None and w_file is not None:
            torch.save(Sigma.cpu(), sigma_file)
            torch.save(W.cpu(), w_file)

        return Sigma, W

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        """Apply whitening transformation: x̃ = Wx."""
        W = self.W.to(
            x.device, dtype=x.dtype if x.dtype.is_floating_point else torch.float32
        )
        return x @ W.T

    def causal_inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute causal inner product: ⟨x,y⟩_c = x^T Σ^{-1} y = (Wx)^T (Wy).

        Args:
            x, y: Vectors in residual space

        Returns:
            Causal inner product
        """
        x_whitened = self.whiten(x)
        y_whitened = self.whiten(y)
        return torch.sum(x_whitened * y_whitened, dim=-1)

    def causal_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal norm: ||x||_c = sqrt(⟨x,x⟩_c)."""
        return torch.sqrt(self.causal_inner_product(x, x))

    def causal_angle(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute causal angle between vectors: ∠_c(x,y) = arccos(⟨x,y⟩_c / (||x||_c ||y||_c)).

        Args:
            x, y: Vectors in residual space

        Returns:
            Angle in radians
        """
        inner_prod = self.causal_inner_product(x, y)
        norm_x = self.causal_norm(x)
        norm_y = self.causal_norm(y)

        # Clamp for numerical stability
        cos_angle = torch.clamp(inner_prod / (norm_x * norm_y), -1.0, 1.0)
        return torch.arccos(cos_angle)

    def normalize_causal(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize vector under causal norm."""
        norm = self.causal_norm(x)
        return x / (norm + 1e-8)

    def to(self, device: str):
        """Move geometry matrices to specified device for training."""
        self.W = self.W.to(device)
        self.Sigma = self.Sigma.to(device)
        return self

    def project_causal(self, x: torch.Tensor, onto: torch.Tensor) -> torch.Tensor:
        """
        Project x onto vector 'onto' using causal inner product.

        Args:
            x: Vector to project
            onto: Vector to project onto

        Returns:
            Projected vector
        """
        onto_normalized = self.normalize_causal(onto)
        projection_coeff = self.causal_inner_product(x, onto_normalized)
        return projection_coeff.unsqueeze(-1) * onto_normalized

    def orthogonalize_causal(
        self, vectors: torch.Tensor, base_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Gram-Schmidt orthogonalization under causal inner product.

        Args:
            vectors: Matrix of vectors to orthogonalize (rows are vectors)
            base_vectors: Optional base vectors to orthogonalize against first

        Returns:
            Orthogonalized vectors
        """
        if base_vectors is not None:
            # First orthogonalize against base vectors
            for i in range(len(vectors)):
                for base_vec in base_vectors:
                    projection = self.project_causal(vectors[i], base_vec)
                    vectors[i] = vectors[i] - projection

        # Gram-Schmidt on remaining vectors
        orthogonal = torch.zeros_like(vectors)
        for i in range(len(vectors)):
            orthogonal[i] = vectors[i].clone()

            # Subtract projections onto previous orthogonal vectors
            for j in range(i):
                projection = self.project_causal(orthogonal[i], orthogonal[j])
                orthogonal[i] = orthogonal[i] - projection

            # Normalize
            orthogonal[i] = self.normalize_causal(orthogonal[i])

        return orthogonal


def compute_polytope_angles(
    parent_vectors: torch.Tensor, child_deltas: torch.Tensor, geometry: CausalGeometry
) -> Dict[str, torch.Tensor]:
    """
    Compute angles between parent vectors and child deltas.

    Args:
        parent_vectors: Parent concept vectors [P, d]
        child_deltas: Child delta vectors [P, C_p, d]
        geometry: CausalGeometry instance

    Returns:
        Dictionary with angle statistics
    """
    P, C_p, d = child_deltas.shape
    angles = torch.zeros(P, C_p)

    for p in range(P):
        for c in range(C_p):
            if torch.norm(child_deltas[p, c]) > 1e-6:  # Skip zero deltas
                angle = geometry.causal_angle(parent_vectors[p], child_deltas[p, c])
                angles[p, c] = angle

    # Convert to degrees for interpretability
    angles_deg = torch.rad2deg(angles)

    return {
        "angles_rad": angles,
        "angles_deg": angles_deg,
        "median_angle_deg": torch.median(angles_deg[angles_deg > 0]),
        "mean_angle_deg": torch.mean(angles_deg[angles_deg > 0]),
        "q25_angle_deg": torch.quantile(angles_deg[angles_deg > 0], 0.25),
        "q75_angle_deg": torch.quantile(angles_deg[angles_deg > 0], 0.75),
    }


def intervention_test(
    model,
    tokenizer,
    geometry: CausalGeometry,
    parent_vector: torch.Tensor,
    alpha_values: list,
    test_prompts: list,
    target_tokens: list,
    sibling_tokens: list,
) -> Dict[str, Any]:
    """
    Test intervention effects by adding α * ℓ_p to residual activations.

    Args:
        model: Language model
        tokenizer: Tokenizer
        geometry: CausalGeometry instance
        parent_vector: Parent concept vector to add
        alpha_values: Intervention magnitudes to test
        test_prompts: List of test prompts
        target_tokens: Target token IDs
        sibling_tokens: Sibling token IDs

    Returns:
        Dictionary with intervention results
    """
    device = next(model.parameters()).device
    parent_vector = parent_vector.to(device)

    results = {
        "alpha_values": alpha_values,
        "target_deltas": [],
        "sibling_deltas": [],
        "effect_sizes": [],
    }

    # Get baseline logits
    baseline_logits = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            baseline_logits.append(outputs.logits[0, -1])  # Last token logits

    baseline_logits = torch.stack(baseline_logits)

    # Test interventions
    for alpha in alpha_values:
        intervention_logits = []

        def intervention_hook(module, input, output):
            # Add α * parent_vector to the last token's residual
            if isinstance(output, tuple):
                h = output[0]
                h = h.clone()
                h[:, -1, :] = h[:, -1, :] + alpha * parent_vector
                return (h,) + output[1:]
            else:
                h = output.clone()
                h[:, -1, :] = h[:, -1, :] + alpha * parent_vector
                return h

        # Register hook (assuming final residual layer)
        hook_handle = None
        for name, module in model.named_modules():
            if "final" in name.lower() or "last" in name.lower():
                hook_handle = module.register_forward_hook(intervention_hook)
                break

        if hook_handle is None:
            logger.warning("Could not find final residual layer for intervention")
            continue

        # Run with intervention
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                intervention_logits.append(outputs.logits[0, -1])

        hook_handle.remove()
        intervention_logits = torch.stack(intervention_logits)

        # Compute deltas
        logit_deltas = intervention_logits - baseline_logits
        target_delta = logit_deltas[:, target_tokens].mean()
        sibling_delta = logit_deltas[:, sibling_tokens].mean()

        results["target_deltas"].append(target_delta.item())
        results["sibling_deltas"].append(sibling_delta.item())

        # Cohen's d effect size
        target_std = logit_deltas[:, target_tokens].std()
        sibling_std = logit_deltas[:, sibling_tokens].std()
        pooled_std = torch.sqrt((target_std**2 + sibling_std**2) / 2)
        effect_size = (target_delta - sibling_delta) / (pooled_std + 1e-8)
        results["effect_sizes"].append(effect_size.item())

    return results
