import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from polytope_hsae.geometry import CausalGeometry
from polytope_hsae.validation import GeometryValidator


def test_hierarchical_orthogonality_returns_dataframe():
    U = torch.eye(4)
    geometry = CausalGeometry(U)
    validator = GeometryValidator(geometry)

    parent_vectors = {"p": torch.tensor([1.0, 0.0, 0.0, 0.0])}
    child_deltas = {"p": {"c": torch.tensor([0.0, 1.0, 0.0, 0.0])}}

    results = validator.test_hierarchical_orthogonality(parent_vectors, child_deltas)
    assert isinstance(results["details"], pd.DataFrame)
    assert set(results["details"].columns) == {
        "parent_id",
        "child_id",
        "angle_deg",
        "cosine",
        "parent_norm",
        "child_norm",
        "delta_norm",
    }
    assert len(results["details"]) == 1


def test_ratio_invariance_dataframe():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    U = torch.eye(model.config.n_embd)
    geometry = CausalGeometry(U)
    validator = GeometryValidator(geometry)

    parent_vectors = {"p": torch.zeros(model.config.n_embd)}
    test_contexts = {"p": ["hello world"]}
    sibling_groups = {"p": ["hello", "world"]}

    results = validator.test_ratio_invariance(
        model,
        tokenizer,
        parent_vectors,
        test_contexts,
        sibling_groups,
        alpha_values=[0.0],
    )
    assert isinstance(results["kl_divergences"], pd.DataFrame)
