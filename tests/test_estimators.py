import torch
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from polytope_hsae.estimators import (
    LDAEstimator,
    validate_orthogonality,
    OrthogonalityStats,
)
from polytope_hsae.geometry import CausalGeometry


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

