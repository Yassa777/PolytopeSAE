import pathlib
import sys

import torch

from polytope_hsae.estimators import (
    LDAEstimator,
    OrthogonalityStats,
    validate_orthogonality,
    ConceptVectorEstimator,
)
from polytope_hsae.geometry import CausalGeometry

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


def test_lda_estimator_outputs_direction_with_correct_shape():
    torch.manual_seed(0)
    U = torch.randn(20, 4)
    geom = CausalGeometry(U)
    X_pos = torch.randn(30, 4)
    X_neg = torch.randn(30, 4)
    lda = LDAEstimator(min_vocab_count=1)
    direction = lda.estimate_binary_direction(X_pos, X_neg, geom)
    assert direction.shape == (4,)
    norm = geom.causal_norm(direction)
    assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)


def test_validate_orthogonality_detects_orthogonal_and_nonorthogonal():
    torch.manual_seed(0)
    d = 3
    U = torch.randn(15, d)
    geom = CausalGeometry(U)
    W_inv_T = torch.linalg.inv(geom.W.T)
    parent_vec = torch.tensor([1.0, 0.0, 0.0]) @ W_inv_T
    delta_ortho = torch.tensor([0.0, 1.0, 0.0]) @ W_inv_T
    delta_non = torch.tensor([1.0, 0.0, 0.0]) @ W_inv_T
    parent_vectors = {"p": parent_vec}
    child_orth = {"p": {"c1": delta_ortho}}
    child_non = {"p": {"c1": delta_non}}
    stats_orth = validate_orthogonality(parent_vectors, child_orth, geom)
    assert isinstance(stats_orth, OrthogonalityStats)
    assert stats_orth.fraction_orthogonal_80deg == 1.0
    assert abs(stats_orth.mean_inner_product) < 1e-5
    stats_non = validate_orthogonality(parent_vectors, child_non, geom)
    assert stats_non.fraction_orthogonal_80deg == 0.0
    assert stats_non.mean_inner_product > 0.1


def test_child_delta_uses_raw_vectors():
    """Child deltas should be computed from unnormalized vectors."""

    class DummyLDA:
        def estimate_binary_direction(self, X_pos, X_neg, geometry, normalize=True):
            # Return a known raw vector depending on inputs
            if torch.allclose(X_pos, torch.tensor([[1.0, 0.0]])):
                vec = torch.tensor([2.0, 0.0])  # parent vector (length 2)
            else:
                vec = torch.tensor([0.0, 3.0])  # child vector (length 3)
            if normalize:
                return geometry.normalize_causal(vec)
            return vec

    geom = CausalGeometry(torch.eye(2))
    estimator = ConceptVectorEstimator(lda_estimator=DummyLDA(), geometry=geom)

    parent_activations = {
        "p": {"pos": torch.tensor([[1.0, 0.0]]), "neg": torch.zeros(1, 2)}
    }
    child_activations = {
        "p": {"c": {"pos": torch.tensor([[0.0, 1.0]]), "neg": torch.zeros(1, 2)}}
    }

    parent_vectors = estimator.estimate_parent_vectors(parent_activations)
    child_vecs, child_deltas = estimator.estimate_child_deltas(parent_vectors, child_activations)
    delta = child_deltas["p"]["c"]

    expected = geom.normalize_causal(
        child_vecs["p"]["c"] - parent_vectors["p"]
    )

    assert torch.allclose(delta, expected, atol=1e-6)
    # All vectors should be unit causal norm
    assert torch.allclose(geom.causal_norm(parent_vectors["p"]), torch.tensor(1.0))
    assert torch.allclose(geom.causal_norm(child_vecs["p"]["c"]), torch.tensor(1.0))
    assert torch.allclose(geom.causal_norm(delta), torch.tensor(1.0))
