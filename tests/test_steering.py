import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from polytope_hsae.steering import ConceptSteering


class DummyHSAE(torch.nn.Module):
    def forward(self, x):
        return x


def test_steer_with_parent_vector_no_change():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    steering = ConceptSteering(model, tokenizer, DummyHSAE(), device="cpu")

    parent_vector = torch.zeros(model.config.n_embd)
    prompts = ["hello world"]

    result = steering.steer_with_parent_vector(
        prompts, parent_vector, magnitude=0.0, output_layer="lm_head"
    )

    assert torch.allclose(result["baseline_logits"], result["steered_logits"])
