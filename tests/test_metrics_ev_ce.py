import math
import torch
import pytest

from polytope_hsae.metrics import (
    compute_explained_variance,
    compute_dual_explained_variance,
    compute_cross_entropy_proxy,
)
from polytope_hsae.geometry import CausalGeometry

DTYPES = [torch.float32, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
def test_explained_variance_identity_and_zero(dtype):
    torch.manual_seed(0)
    x = torch.randn(128, 16).to(dtype)
    ev_same = compute_explained_variance(x, x)
    assert math.isclose(ev_same, 1.0, abs_tol=1e-5)

    x_hat0 = torch.zeros_like(x)
    ev_zero = compute_explained_variance(x, x_hat0)
    assert math.isfinite(ev_zero)
    assert ev_zero < 0.1


def test_dual_explained_variance_matches_ev_in_identity_geometry():
    torch.manual_seed(0)
    n, d = 200, 8
    x = torch.randn(n, d)
    x_hat = x + 0.05 * torch.randn_like(x)

    geom = CausalGeometry(torch.eye(d))

    ev = compute_explained_variance(x, x_hat)
    dual = compute_dual_explained_variance(x, x_hat, geom)
    assert math.isclose(ev, dual["standard_ev"], abs_tol=5e-5)
    assert math.isclose(ev, dual["causal_ev"], abs_tol=5e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_cross_entropy_proxy_shift_invariance_and_stability(dtype):
    torch.manual_seed(0)
    x = torch.randn(64, 11).to(dtype)
    x_hat = x + 0.1 * torch.randn_like(x)
    ce = compute_cross_entropy_proxy(x, x_hat)

    c = 10.0 * torch.ones_like(x_hat)
    ce_shift = compute_cross_entropy_proxy(x, x_hat + c)
    assert math.isclose(ce, ce_shift, abs_tol=1e-6)

    ce32 = compute_cross_entropy_proxy(x.float(), x_hat.float())
    assert math.isfinite(ce) and math.isfinite(ce32)
    assert abs(float(ce) - float(ce32)) < 3e-3

