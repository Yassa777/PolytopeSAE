"""
Statistical estimators for concept vectors using LDA and related methods.

This module implements LDA-based estimators for finding parent vectors and child
deltas in the whitened causal space.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

logger = logging.getLogger(__name__)


class LDAEstimator:
    """LDA-based estimator for concept directions in causal space."""

    def __init__(
        self,
        shrinkage: float = 0.1,
        min_vocab_count: int = 50,
        class_balance: bool = True,
    ):
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

    def estimate_binary_direction(
        self,
        X_pos: torch.Tensor,
        X_neg: torch.Tensor,
        geometry,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Estimate LDA direction for binary classification in causal space.

        Args:
            X_pos: Positive examples ``[N_pos, d]``
            X_neg: Negative examples ``[N_neg, d]``
            geometry: CausalGeometry instance for whitening
            normalize: Whether to normalize under the causal norm before
                returning.

        Returns:
            LDA direction vector. Normalized if ``normalize`` is ``True``.
        """
        # Whiten the data
        X_pos_whitened = geometry.whiten(X_pos)
        X_neg_whitened = geometry.whiten(X_neg)

        # Check minimum sample requirements
        if (
            len(X_pos_whitened) < self.min_vocab_count
            or len(X_neg_whitened) < self.min_vocab_count
        ):
            logger.warning(
                f"Insufficient samples: pos={len(X_pos_whitened)}, neg={len(X_neg_whitened)}"
            )

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
        S_w = torch.cov(X_centered.T) + self.shrinkage * torch.eye(
            X_centered.shape[1], dtype=X_centered.dtype, device=X_centered.device
        )

        S_w_f32 = S_w.to(torch.float32)
        mean_diff = (mu_pos - mu_neg).to(torch.float32)
        try:
            direction = torch.linalg.solve(S_w_f32, mean_diff)
        except torch.linalg.LinAlgError:
            logger.warning("Singular within-class scatter matrix, using pseudoinverse")
            direction = torch.linalg.pinv(S_w_f32) @ mean_diff
        direction = direction.to(torch.float32)

        direction_original = direction @ geometry.W.T
        if normalize:
            return geometry.normalize_causal(direction_original)
        return direction_original

    def estimate_multiclass_directions(
        self,
        X_list: List[torch.Tensor],
        geometry,
        method: str = "one_vs_rest",
    ) -> torch.Tensor:
        """Estimate directions for multiclass classification.

        This routine mixes PyTorch operations with scikit-learn's LDA
        implementation. Inputs are expected to be ``torch.Tensor`` objects and
        are converted to ``numpy`` arrays where required. The conversion uses
        ``to(torch.float32).cpu().numpy()`` to avoid dtype surprises. For
        deterministic behaviour the numpy random seed is fixed.

        Args:
            X_list: List of tensors, one per class ``[N_i, d]``.
            geometry: CausalGeometry instance.
            method: ``'one_vs_rest'`` or ``'pairwise'``.

        Returns:
            Matrix of class directions ``[n_classes, d]``.
        """
        n_classes = len(X_list)
        d = X_list[0].shape[1]
        directions = torch.zeros(n_classes, d)

        if method == "one_vs_rest":
            for i, X_pos in enumerate(X_list):
                X_neg = torch.cat(
                    [X_list[j] for j in range(n_classes) if j != i], dim=0
                )
                directions[i] = self.estimate_binary_direction(X_pos, X_neg, geometry)

        elif method == "pairwise":
            # Use sklearn's LDA for multiclass
            X_all = torch.cat(X_list, dim=0)
            y_all = torch.cat(
                [torch.full((len(X_list[i]),), i) for i in range(n_classes)]
            )

            # Whiten data
            X_whitened = geometry.whiten(X_all)

            # Ensure deterministic behaviour in scikit-learn
            np.random.seed(0)

            # Fit LDA with proper solver for shrinkage
            lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=self.shrinkage)
            lda.fit(
                X_whitened.to(dtype=torch.float32).cpu().numpy(),
                y_all.to(dtype=torch.int64).cpu().numpy(),
            )

            for i in range(n_classes):
                direction_whitened = torch.from_numpy(lda.coef_[i]).float()
                direction_original = direction_whitened @ geometry.W.T
                directions[i] = geometry.normalize_causal(direction_original)

        return directions


class ConceptVectorEstimator:
    """High-level estimator for parent vectors and child deltas."""

    def __init__(self, lda_estimator: Optional[LDAEstimator] = None, geometry=None):
        """
        Initialize concept vector estimator.

        Args:
            lda_estimator: LDA estimator instance
            geometry: CausalGeometry instance
        """
        self.lda_estimator = lda_estimator or LDAEstimator()
        self.geometry = geometry
        # Store raw, unnormalized parent vectors for delta computation
        self.parent_raw_vectors: Dict[str, torch.Tensor] = {}

    def estimate_parent_vectors(
        self, parent_activations: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
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

            X_pos = data["pos"]  # Activations where parent concept is present
            X_neg = data["neg"]  # Activations where parent concept is absent

            # Get raw parent vector first, then normalize for return
            parent_vector_raw = self.lda_estimator.estimate_binary_direction(
                X_pos, X_neg, self.geometry, normalize=False
            )
            parent_vector = self.geometry.normalize_causal(parent_vector_raw)
            parent_vectors[parent_id] = parent_vector
            self.parent_raw_vectors[parent_id] = parent_vector_raw

        return parent_vectors

    def estimate_child_deltas(
        self,
        parent_vectors: Dict[str, torch.Tensor],
        child_activations: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
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
            parent_vector_raw = self.parent_raw_vectors.get(parent_id)
            if parent_vector_raw is None:
                logger.warning(
                    f"Raw parent vector for {parent_id} not found; using normalized vector"
                )
                parent_vector_raw = parent_vectors[parent_id]
            
            child_deltas[parent_id] = {}

            for child_id, data in children_data.items():
                logger.info(f"Estimating child delta for {parent_id}/{child_id}")

                X_pos = data["pos"]  # Activations where child concept is present
                X_neg = data["neg"]  # Activations where child concept is absent

                # Estimate child vector in raw space
                child_vector_raw = self.lda_estimator.estimate_binary_direction(
                    X_pos, X_neg, self.geometry, normalize=False
                )

                # Compute delta using raw vectors: δ_{c|p} = ℓ_c - ℓ_p
                delta_vector = child_vector_raw - parent_vector_raw
                child_deltas[parent_id][child_id] = delta_vector

        return child_deltas

    def estimate_child_subspace_projectors(
        self,
        parent_vectors: Dict[str, torch.Tensor],
        child_deltas: Dict[str, Dict[str, torch.Tensor]],
        subspace_dim: int = 96,
        energy_threshold: float = 0.85,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Estimate down/up projectors for child subspaces using SVD with energy-based dimension selection.

        Args:
            parent_vectors: Parent concept vectors
            child_deltas: Child delta vectors
            subspace_dim: Global maximum dimension of child subspace
            energy_threshold: Threshold for cumulative energy (default 0.85 for 85%)

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

            # Energy-based dimension selection: pick k = min(global_subspace_dim, k_85%)
            # where k_85% is the smallest rank reaching 85% energy per parent
            cumulative_energy = torch.cumsum(S, dim=0) / S.sum()
            k_energy = (cumulative_energy <= energy_threshold).sum().item() + 1
            k = min(subspace_dim, k_energy, V.shape[1])
            
            logger.info(f"Parent {parent_id}: selected k={k} dims (energy-based: {k_energy}, global cap: {subspace_dim})")
            
            V_k = V[:, :k]  # [d, k]

            # Down projector: residual -> subspace
            down_projector = V_k.T  # [k, d]

            # Up projector: subspace -> residual
            up_projector = V_k  # [d, k]

            projectors[parent_id] = (down_projector, up_projector)

        return projectors


def ridge_cross_layer_mapping(
    source_vectors: torch.Tensor,
    target_vectors: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Learn ridge regression mapping from source layer to target layer.

    The inputs are cast to ``float32`` prior to solving to ensure numerical
    stability. If the regularized system is ill-conditioned, a pseudoinverse is
    used instead of the standard solve.

    Args:
        source_vectors: Vectors in source layer ``[n_concepts, d_source]``
        target_vectors: Corresponding vectors in target layer
            ``[n_concepts, d_target]``
        alpha: Ridge regularization parameter

    Returns:
        Mapping matrix ``[d_target, d_source]``
    """

    X = source_vectors.to(torch.float32)
    Y = target_vectors.to(torch.float32)

    XTX = X.T @ X + alpha * torch.eye(X.shape[1], dtype=torch.float32)
    XTY = X.T @ Y

    try:
        W = torch.linalg.solve(XTX, XTY)
    except torch.linalg.LinAlgError:
        logger.warning("Ill-conditioned matrix in ridge mapping; using pseudoinverse")
        W = torch.linalg.pinv(XTX) @ XTY

    return W.T


def validate_orthogonality(
    parent_vectors: Dict[str, torch.Tensor],
    child_deltas: Dict[str, Dict[str, torch.Tensor]],
    geometry,
) -> "OrthogonalityStats":
    """Validate hierarchical orthogonality: ⟨ℓ_p, δ_{c|p}⟩_c ≈ 0."""

    inner_products: List[float] = []
    angles: List[float] = []

    for parent_id, deltas in child_deltas.items():
        if parent_id not in parent_vectors:
            continue

        parent_vector = parent_vectors[parent_id]

        for child_id, delta in deltas.items():
            inner_prod = geometry.causal_inner_product(parent_vector, delta)
            inner_products.append(inner_prod.item())
            angle = geometry.causal_angle(parent_vector, delta)
            angles.append(torch.rad2deg(angle).item())

    inner_products_arr = np.array(inner_products)
    angles_arr = np.array(angles)

    stats = OrthogonalityStats(
        inner_products=inner_products_arr,
        angles_deg=angles_arr,
        mean_inner_product=np.mean(np.abs(inner_products_arr))
        if inner_products_arr.size
        else 0.0,
        median_angle_deg=np.median(angles_arr) if angles_arr.size else 0.0,
        q25_angle_deg=np.percentile(angles_arr, 25) if angles_arr.size else 0.0,
        q75_angle_deg=np.percentile(angles_arr, 75) if angles_arr.size else 0.0,
        fraction_orthogonal_80deg=np.mean(angles_arr > 80) if angles_arr.size else 0.0,
        fraction_orthogonal_85deg=np.mean(angles_arr > 85) if angles_arr.size else 0.0,
    )
    stats.validate()
    return stats


@dataclass
class OrthogonalityStats:
    """Container for orthogonality statistics with basic validation."""

    inner_products: np.ndarray
    angles_deg: np.ndarray
    mean_inner_product: float
    median_angle_deg: float
    q25_angle_deg: float
    q75_angle_deg: float
    fraction_orthogonal_80deg: float
    fraction_orthogonal_85deg: float

    def validate(self) -> None:
        if self.inner_products.shape != self.angles_deg.shape:
            raise ValueError("Mismatched shapes for inner products and angles")
