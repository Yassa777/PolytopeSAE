import torch
from polytope_hsae.models import HierarchicalSAE, HSAEConfig


def _make_orthonormal_rows(P, d):
    W = torch.zeros(P, d)
    for i in range(P):
        W[i, i] = 1.0
    return W


def test_parent_child_biorth_offdiag_zero_when_E_equals_D():
    d, P, C = 6, 4, 3
    cfg = HSAEConfig(
        input_dim=d, n_parents=P, topk_parent=1,
        subspace_dim=d, n_children_per_parent=C, topk_child=1,
        use_tied_decoders_parent=False, use_tied_decoders_child=False,
    )
    model = HierarchicalSAE(cfg)

    with torch.no_grad():
        E = _make_orthonormal_rows(P, d)
        model.router.encoder.weight.copy_(E)
        model.router.decoder.weight.copy_(E.t())

        for sub in model.subspaces:
            Ec = _make_orthonormal_rows(C, d)
            sub.encoder.weight.copy_(Ec)
            sub.decoder.weight.copy_(Ec.t())

    parent_pen = model._biorth_penalty_parent_offdiag().item()
    child_pen = sum(model._biorth_penalty_child_offdiag(i).item() for i in range(P))
    assert parent_pen < 1e-8
    assert child_pen < 1e-8


def test_parent_biorth_offdiag_positive_when_rows_non_orthonormal():
    d, P, C = 6, 4, 3
    cfg = HSAEConfig(
        input_dim=d, n_parents=P, topk_parent=1,
        subspace_dim=d, n_children_per_parent=C, topk_child=1,
    )
    model = HierarchicalSAE(cfg)

    with torch.no_grad():
        model.router.encoder.weight.normal_()
        model.router.decoder.weight.normal_()

    pen = model._biorth_penalty_parent_offdiag().item()
    assert pen > 1e-5

