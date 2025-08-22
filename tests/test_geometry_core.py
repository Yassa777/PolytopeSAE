import math
import torch
import pytest

from polytope_hsae.geometry import CausalGeometry

DTYPES = [torch.float32, torch.bfloat16]


def _spd_from_U(U: torch.Tensor, shrink: float = 1e-6):
    X = U - U.mean(dim=0, keepdim=True)
    Sigma = X.T @ X / max(1, X.shape[0] - 1)
    d = Sigma.shape[0]
    return Sigma + shrink * torch.eye(d)


@pytest.mark.parametrize("dtype", DTYPES)
def test_causal_ip_equals_Sigma_inv_dot(dtype):
    torch.manual_seed(0)
    V, d = 64, 8
    U = torch.randn(V, d)
    geom = CausalGeometry(U, shrinkage=1e-6)
    x = torch.randn(d, dtype=dtype)
    y = torch.randn(d, dtype=dtype)

    ip_geom = geom.causal_inner_product(x, y).item()

    Sigma = _spd_from_U(U).float()
    Sigma_inv = torch.linalg.inv(Sigma)
    ip_true = (x.float() @ Sigma_inv @ y.float()).item()

    assert math.isfinite(ip_geom)
    tol = 5e-2 if dtype == torch.bfloat16 else 1e-4
    assert abs(ip_geom - ip_true) < tol


@pytest.mark.parametrize("dtype", DTYPES)
def test_whiten_shapes_and_dtypes(dtype):
    torch.manual_seed(0)
    V, d = 50, 7
    U = torch.randn(V, d)
    geom = CausalGeometry(U)
    x1 = torch.randn(d, dtype=dtype)
    x2 = torch.randn(10, d, dtype=dtype)

    w1 = geom.whiten(x1)
    w2 = geom.whiten(x2)

    assert w1.shape == x1.shape
    assert w2.shape == x2.shape
    assert torch.isfinite(w1).all() and torch.isfinite(w2).all()


def test_causal_angle_properties_and_zero_vector():
    torch.manual_seed(1)
    V, d = 80, 6
    U = torch.randn(V, d)
    geom = CausalGeometry(U)

    a = torch.randn(d)
    assert torch.allclose(geom.causal_angle(a, 2.0 * a), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(geom.causal_angle(a, -a), torch.tensor(math.pi), atol=1e-6)

    aw = geom.whiten(a)
    rand = torch.randn_like(aw)
    bw = rand - (rand @ aw) / (aw @ aw + 1e-8) * aw
    b = bw @ torch.linalg.pinv(geom.W)
    angle = geom.causal_angle(a, b)
    assert abs(angle.item() - math.pi / 2) < 1e-2

    z = torch.zeros_like(a)
    assert torch.isnan(geom.causal_angle(a, z))


def test_project_and_orthogonalize_causal():
    torch.manual_seed(2)
    V, d = 60, 5
    U = torch.randn(V, d)
    geom = CausalGeometry(U)
    v = torch.randn(d)
    u = torch.randn(d)
    u_perp = geom.orthogonalize_causal(u.unsqueeze(0), v.unsqueeze(0))[0]
    ip = geom.causal_inner_product(v, u_perp).abs().item()
    assert ip < 1e-6


def test_bad_dimensional_inputs_raise():
    torch.manual_seed(0)
    V, d = 32, 4
    U = torch.randn(V, d)
    geom = CausalGeometry(U)
    x = torch.randn(2, 3, d)
    y = torch.randn(d)

    with pytest.raises(Exception):
        geom.whiten(x)

    with pytest.raises(Exception):
        geom.causal_inner_product(x, y)

