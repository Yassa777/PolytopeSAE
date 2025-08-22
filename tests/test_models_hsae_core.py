import torch
import pytest
from polytope_hsae.models import HierarchicalSAE, HSAEConfig
from polytope_hsae.geometry import CausalGeometry


def _tiny_hsae(d=8, P=3, C=2):
    cfg = HSAEConfig(
        input_dim=d,
        n_parents=P,
        topk_parent=1,
        subspace_dim=d,
        n_children_per_parent=C,
        topk_child=1,
        use_tied_decoders_parent=False,
        use_tied_decoders_child=False,
        tie_projectors=False,
        normalize_decoder=True,
    )
    return HierarchicalSAE(cfg)


def test_sample_codes_top1_training_and_eval():
    torch.manual_seed(0)
    model = _tiny_hsae()
    logits = torch.randn(5, model.config.n_parents)

    model.train()
    codes_tr = model._sample_codes(logits, k=1)
    assert torch.all((codes_tr.sum(dim=-1) - 1.0).abs() < 1e-6)

    model.eval()
    codes_ev = model._sample_codes(logits, k=1)
    assert torch.all((codes_ev.sum(dim=-1) - 1.0).abs() < 1e-6)


def test_update_router_temperature_endpoints():
    model = _tiny_hsae()
    start = model.config.router_temp_start
    end = model.config.router_temp_end
    model.update_router_temperature(step=0, total_steps=100)
    assert torch.allclose(model.router_temperature, torch.tensor(start))
    model.update_router_temperature(step=100, total_steps=100)
    assert torch.allclose(model.router_temperature, torch.tensor(end))


def test_normalize_decoder_weights_sets_unit_column_norms():
    torch.manual_seed(0)
    model = _tiny_hsae(d=10, P=4)
    with torch.no_grad():
        model.router.decoder.weight.copy_(torch.randn_like(model.router.decoder.weight))
        for sub in model.subspaces:
            sub.decoder.weight.copy_(torch.randn_like(sub.decoder.weight))
    model.normalize_decoder_weights()

    parent_norms = torch.norm(model.router.decoder.weight, dim=0)
    assert torch.allclose(parent_norms, torch.ones_like(parent_norms), atol=1e-6)
    for sub in model.subspaces:
        cn = torch.norm(sub.decoder.weight, dim=0)
        assert torch.allclose(cn, torch.ones_like(cn), atol=1e-6)


def test_causal_orthogonality_penalty_zero_when_delta_orthogonal():
    torch.manual_seed(0)
    d, P, C = 8, 2, 1
    model = _tiny_hsae(d, P, C)
    geom = CausalGeometry(torch.eye(d))

    with torch.no_grad():
        model.router.decoder.weight.zero_()
        model.router.decoder.weight[0, 0] = 1.0
        model.router.decoder.weight[1, 1] = 1.0
        for p, sub in enumerate(model.subspaces):
            sub.up_projector.weight.copy_(torch.eye(d))
            sub.decoder.weight.zero_()
            delta = torch.zeros(d)
            delta[(p + 2) % d] = 1.0
            child_full = torch.zeros(d)
            child_full[p] = 1.0
            sub.decoder.weight[:, 0] = child_full

    pen0 = model.compute_causal_orthogonality_penalty(geom).item()

    with torch.no_grad():
        for p, sub in enumerate(model.subspaces):
            child_full = torch.zeros(d)
            child_full[p] = 2.0
            sub.decoder.weight[:, 0] = child_full
    pen_parallel = model.compute_causal_orthogonality_penalty(geom).item()

    assert pen0 <= 1e-8
    assert pen_parallel > pen0 + 1e-4

