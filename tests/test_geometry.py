import math

import torch

from polytope_hsae.geometry import CausalGeometry


def test_causal_geometry_initializes():
    torch.manual_seed(0)
    U = torch.randn(10, 4)
    geom = CausalGeometry(U)
    assert geom.U.shape == (10, 4)
    assert geom.W.shape == (4, 4)
    assert geom.Sigma.shape == (4, 4)


def test_whiten_and_metrics():
    torch.manual_seed(0)
    U = torch.randn(8, 3)
    geom = CausalGeometry(U)
    x = torch.randn(5, 3)
    y = torch.randn(5, 3)
    x_w = geom.whiten(x)
    assert x_w.shape == x.shape
    ip = geom.causal_inner_product(x, y)
    assert ip.shape == (5,)
    expected_ip = (geom.whiten(x) * geom.whiten(y)).sum(dim=-1)
    assert torch.allclose(ip, expected_ip)
    norm_x = geom.causal_norm(x)
    assert norm_x.shape == (5,)
    assert torch.all(norm_x >= 0)
    zero = torch.zeros(3)
    assert geom.causal_norm(zero) == 0
    assert torch.allclose(geom.whiten(zero), torch.zeros_like(zero))


def test_causal_angle_range_and_zero_vector():
    torch.manual_seed(0)
    U = torch.randn(6, 3)
    geom = CausalGeometry(U)
    x = torch.randn(3)
    y = torch.randn(3)
    angle = geom.causal_angle(x, y)
    assert angle.ndim == 0
    assert 0.0 <= angle.item() <= math.pi
    zero = torch.zeros(3)
    angle_zero = geom.causal_angle(x, zero)
    assert torch.isnan(angle_zero)
