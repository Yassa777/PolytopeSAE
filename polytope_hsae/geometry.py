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
        whitening: str = "unembedding_rows",
        shrinkage: float = 0.05,
        residual_acts: Optional[torch.Tensor] = None,
        cov_device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize causal geometry.

        Args:
            unembedding_matrix: U ∈ R^{V×d} unembedding matrix
            whitening: Source of covariance (unembedding_rows | residual_acts | diagonal | identity)
            shrinkage: Shrinkage parameter for covariance
            residual_acts: Optional residual activations when whitening="residual_acts"
            cov_device: Device for covariance computation
            cache_dir: Optional cache directory
        """
        self.whitening = whitening
        self.shrinkage = shrinkage
        self.device = unembedding_matrix.device
        self.dtype = torch.float32
        self.cov_device = cov_device
        self.cache_dir = cache_dir
        self.U = unembedding_matrix.detach()
        self.residual_acts = residual_acts

        # Compute whitening matrix
        self.Sigma, self.W = self._compute_whitening_matrix()

    @classmethod
    def from_unembedding(cls, U, shrinkage=0.05):
        """Create geometry from unembedding matrix (standard approach)."""
        return cls(U, whitening="unembedding_rows", shrinkage=shrinkage)
    
    @classmethod
    def from_logit_cov(cls, model, tokenizer, prompts, shrinkage=0.05):
        """Create geometry from logit covariance on sample prompts - MEMORY OPTIMIZED."""
        logger.warning("from_logit_cov: Using diagonal approximation to avoid OOM on large vocabularies")
        
        device = next(model.parameters()).device
        model.eval()
        
        # Use much smaller sample and process one at a time to minimize memory
        all_logits = []
        with torch.no_grad():
            for prompt in prompts[:20]:  # Drastically reduced for memory
                try:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
                    outputs = model(**inputs)
                    # Get logits for last token and immediately move to CPU
                    logits = outputs.logits[0, -1, :].cpu()
                    all_logits.append(logits)
                    # Explicit cleanup to free GPU memory
                    del outputs, inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    continue
        
        if not all_logits:
            raise ValueError("No valid prompts processed for logit covariance")
            
        # Stack logits and use DIAGONAL approximation to avoid OOM
        logits_tensor = torch.stack(all_logits)  # [n_prompts, vocab_size]
        
        # Create dummy unembedding
        vocab_size, d_model = logits_tensor.shape[1], model.config.hidden_size
        dummy_U = torch.randn(vocab_size, d_model)
        
        # Create geometry instance with diagonal whitening
        geometry = cls(dummy_U, whitening="diagonal", shrinkage=shrinkage)
        
        # Use diagonal variance instead of full covariance (memory-efficient)
        logits_centered = logits_tensor - logits_tensor.mean(dim=0, keepdim=True)
        diagonal_var = logits_centered.var(dim=0, unbiased=True) + shrinkage
        
        # Create diagonal matrices instead of full vocab_size x vocab_size matrices
        geometry.Sigma = torch.diag(diagonal_var)
        geometry.W = torch.diag(1.0 / torch.sqrt(diagonal_var))
        
        return geometry
    
    @classmethod  
    def fisher_like(cls, model, tokenizer, prompts, shrinkage=0.05):
        """Create geometry from diagonal Fisher/softmax Hessian approximation - MEMORY OPTIMIZED."""
        device = next(model.parameters()).device
        model.eval()
        
        all_probs = []
        with torch.no_grad():
            for prompt in prompts[:20]:  # Reduced for memory
                try:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
                    outputs = model(**inputs)
                    # Get softmax probabilities for last token and move to CPU immediately
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=0).cpu()
                    all_probs.append(probs)
                    # Explicit cleanup
                    del outputs, inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    continue
        
        if not all_probs:
            raise ValueError("No valid prompts processed for Fisher approximation")
            
        # Stack probabilities and compute diagonal Fisher approximation
        probs_tensor = torch.stack(all_probs)  # [n_prompts, vocab_size]
        
        # Diagonal Fisher: E[p(1-p)] for each vocabulary item
        mean_probs = probs_tensor.mean(dim=0)
        fisher_diag = mean_probs * (1 - mean_probs) + shrinkage
        
        # Create dummy unembedding
        vocab_size, d_model = len(mean_probs), model.config.hidden_size
        dummy_U = torch.randn(vocab_size, d_model)
        
        # Create geometry instance
        geometry = cls(dummy_U, whitening="diagonal", shrinkage=shrinkage)
        
        # Override with Fisher diagonal - keep diagonal representation
        geometry.Sigma = torch.diag(fisher_diag)
        geometry.W = torch.diag(1.0 / torch.sqrt(fisher_diag))
        
        return geometry

    def save(self, path: str):
        """Save geometry tensors to disk."""
        data = {
            "unembedding_matrix": self.U.cpu(),
            "whitening_matrix": self.W.cpu(),
            "Sigma": self.Sigma.cpu(),
            "whitening": self.whitening,
            "shrinkage": self.shrinkage,
            "cov_device": self.cov_device,
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path: str, map_location=None):
        """Load geometry from disk without recomputing whitening."""
        data = torch.load(path, map_location=map_location or "cpu")
        geometry = cls.__new__(cls)
        geometry.U = data["unembedding_matrix"]
        geometry.W = data["whitening_matrix"]
        geometry.Sigma = data.get("Sigma")
        geometry.whitening = data.get("whitening", "unembedding_rows")
        geometry.shrinkage = data.get("shrinkage", 0.05)
        geometry.cov_device = data.get("cov_device", "cpu")
        geometry.device = geometry.U.device
        geometry.dtype = torch.float32
        geometry.cache_dir = None
        geometry.residual_acts = None
        return geometry

    def _compute_whitening_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute covariance matrix Σ and whitening matrix W."""
        dim = self.U.shape[1]

        if self.whitening == "identity":
            Sigma = torch.eye(dim, device=self.cov_device)
            W = torch.eye(dim, device=self.cov_device)
            return Sigma, W

        if self.whitening == "residual_acts" and self.residual_acts is not None:
            data = self.residual_acts.to(self.cov_device, dtype=torch.float32)
        else:
            data = self.U.to(self.cov_device, dtype=torch.float32)

        data = data - data.mean(dim=0, keepdim=True)

        if self.whitening == "diagonal":
            var = data.var(dim=0, unbiased=True) + self.shrinkage
            Sigma = torch.diag(var)
            W = torch.diag(1.0 / torch.sqrt(var))
            return Sigma, W

        Sigma = torch.cov(data.T) + self.shrinkage * torch.eye(dim, device=self.cov_device)
        eigenvals, eigenvecs = torch.linalg.eigh(Sigma)
        eigenvals = torch.clamp(eigenvals, min=1e-8)
        W = eigenvecs @ torch.diag(1.0 / torch.sqrt(eigenvals)) @ eigenvecs.T
        return Sigma, W

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        """Apply whitening transformation: x̃ = Wx."""
        if x.ndim not in (1, 2):
            raise ValueError("whiten expects a 1D or 2D tensor")
        W = self.W.to(
            x.device, dtype=x.dtype if x.dtype.is_floating_point else torch.float32
        )
        return x @ W.T

    def unwhiten(self, x_whitened: torch.Tensor) -> torch.Tensor:
        """Apply inverse whitening transformation: x = x̃W^{-1}."""
        if x_whitened.ndim not in (1, 2):
            raise ValueError("unwhiten expects a 1D or 2D tensor")
        # whiten(x) = x @ W^T  ⇒  unwhiten(x̃) solves W^T · x = x̃
        W = self.W.to(x_whitened.device, dtype=torch.float32)
        xw = x_whitened.to(torch.float32)
        x_orig = torch.linalg.solve(W.T, xw.T).T   # numerically safer than explicit inverse
        return x_orig.to(x_whitened.dtype)

    def causal_inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute causal inner product: ⟨x,y⟩_c = x^T Σ^{-1} y = (Wx)^T (Wy).

        Args:
            x, y: Vectors in residual space

        Returns:
            Causal inner product
        """
        if x.ndim not in (1, 2) or y.ndim not in (1, 2):
            raise ValueError("inputs must be 1D or 2D tensors")
        if x.ndim != y.ndim or x.shape[-1] != y.shape[-1]:
            raise ValueError("inputs must have matching shapes")
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

    def random_rotate(self, x: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """Rotate ``x`` by a given angle in a random causal-orthogonal direction."""
        angle = angle_deg * torch.pi / 180.0
        w = self.whiten(x)
        if torch.allclose(w, torch.zeros_like(w)):
            return x
        rand = torch.randn_like(w)
        rand -= (rand @ w) / (w @ w) * w
        rand = rand / (rand.norm() + 1e-8)
        w_norm = w / (w.norm() + 1e-8)
        rotated = w_norm * torch.cos(angle) + rand * torch.sin(angle)
        rotated = rotated * w.norm()
        return self.unwhiten(rotated.unsqueeze(0)).squeeze(0)

    def to(self, device: str):
        """Move geometry matrices to specified device for training."""
        self.W = self.W.to(device)
        self.Sigma = self.Sigma.to(device)
        return self

    def test_linear_identity(self, x: torch.Tensor, tolerance: float = 1e-4) -> Dict[str, float]:
        """
        Test linear identity: x → whiten → unwhiten → should equal x.
        
        Args:
            x: Test tensor [batch_size, dim]
            tolerance: Numerical tolerance for identity check
            
        Returns:
            Dictionary with identity test metrics
        """
        x_whitened = self.whiten(x)
        x_reconstructed = self.unwhiten(x_whitened)
        
        # Compute reconstruction error
        mse = torch.mean((x - x_reconstructed) ** 2).item()
        max_error = torch.max(torch.abs(x - x_reconstructed)).item()
        
        # Compute explained variance (should be ~1.0)
        from polytope_hsae.metrics import compute_explained_variance
        identity_ev = compute_explained_variance(x, x_reconstructed)
        
        # Check if identity holds within tolerance
        identity_ok = max_error < tolerance
        
        return {
            "identity_mse": mse,
            "identity_max_error": max_error,
            "identity_ev": identity_ev,
            "identity_ok": identity_ok,
            "tolerance": tolerance
        }

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
