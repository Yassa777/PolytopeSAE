import torch
import math
from polytope_hsae.estimators import LDAEstimator
from polytope_hsae.geometry import CausalGeometry


def _to_original_space(geom: CausalGeometry, Z: torch.Tensor) -> torch.Tensor:
    W_inv_T = torch.linalg.inv(geom.W.T)
    return Z @ W_inv_T


def test_lda_direction_points_between_class_means_and_is_unit_causal_norm():
    torch.manual_seed(0)
    d = 6
    geom = CausalGeometry(torch.eye(d))

    mu = torch.zeros(d)
    mu[0] = 1.0
    Z_pos = mu + 0.1 * torch.randn(200, d)
    Z_neg = -mu + 0.1 * torch.randn(200, d)

    X_pos = _to_original_space(geom, Z_pos)
    X_neg = _to_original_space(geom, Z_neg)

    lda = LDAEstimator(min_vocab_count=1)
    w = lda.estimate_binary_direction(X_pos, X_neg, geom)

    angle = geom.causal_angle(w, mu).item()
    assert angle < math.radians(25.0)

    norm = geom.causal_norm(w).item()
    assert abs(norm - 1.0) < 1e-4


def test_lda_handles_singular_within_class_scatter_and_bfloat16():
    torch.manual_seed(0)
    d = 5
    geom = CausalGeometry(torch.eye(d))

    Z_pos = torch.zeros(20, d)
    Z_neg = torch.cat([torch.ones(10, d), torch.zeros(10, d)], dim=0)
    X_pos = _to_original_space(geom, Z_pos)
    X_neg = _to_original_space(geom, Z_neg)

    lda = LDAEstimator(min_vocab_count=1)
    w = lda.estimate_binary_direction(X_pos, X_neg, geom)
    assert torch.isfinite(w).all()

    w_bf16 = lda.estimate_binary_direction(
        X_pos.to(torch.bfloat16), X_neg.to(torch.bfloat16), geom
    )
    assert torch.isfinite(w_bf16).all()
    ang = geom.causal_angle(w, w_bf16).item()
    assert ang < math.radians(5.0)

