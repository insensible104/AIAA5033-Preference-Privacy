import pytest
import torch
from preference_privacy.objective import partial_loss, preference_margin
from preference_privacy.encoding import encode_pair, PairCollator
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model


def tiny_model(vocab_size=16):
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
    )
    model.config._attn_implementation = "eager"
    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    return model


def test_endpoints_and_gradients():
    z = torch.tensor(
        [-20.0, -1.0, 0.0, 1.0, 20.0], dtype=torch.float64, requires_grad=True
    )
    rho = 0.2
    plain = torch.nn.functional.softplus(-z)
    robust = ((1 - rho) * plain - rho * torch.nn.functional.softplus(z)) / (1 - 2 * rho)
    for lam, expected in [(0.0, plain), (1.0, robust), (0.5, (plain + robust) / 2)]:
        actual = partial_loss(z, rho, lam)
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual.sum(), z, retain_graph=True)[0]
        gb = torch.autograd.grad(expected.sum(), z, retain_graph=True)[0]
        torch.testing.assert_close(ga, gb)


def test_unbiased_expectation():
    z = torch.linspace(-10, 10, 21, dtype=torch.float64)
    for rho in [0.01, 0.2, 0.4]:
        expectation = (1 - rho) * partial_loss(z, rho, 1) + rho * partial_loss(
            -z, rho, 1
        )
        torch.testing.assert_close(expectation, torch.nn.functional.softplus(-z))


@pytest.mark.parametrize("rho,lam", [(0.5, 0), (-1, 1), (0.2, 2), (0.2, float("nan"))])
def test_invalid_loss(rho, lam):
    with pytest.raises(ValueError):
        partial_loss(torch.tensor([1.0]), rho, lam)


def test_reference_fixed_and_batch_invariance(row, tokenizer):
    model = tiny_model(len(tokenizer)).eval()
    encoded = encode_pair(row, tokenizer)
    longer = encode_pair(
        row | {"id": "long", "response_b": " Good Good Good"}, tokenizer
    )
    collator = PairCollator(tokenizer.pad_token_id)
    original = {
        k: v.detach().clone()
        for k, v in model.named_parameters()
        if not v.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    model.train()
    margin = preference_margin(model, collator([encoded]), 0.1)
    partial_loss(margin, 0.2, 0.5).mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    optimizer.step()
    for name, p in model.named_parameters():
        if name in original:
            torch.testing.assert_close(p, original[name], rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        single = preference_margin(model, collator([encoded]), 0.1)
        batched = preference_margin(model, collator([encoded, longer]), 0.1)
    torch.testing.assert_close(single, batched[:1], atol=1e-6, rtol=1e-5)
