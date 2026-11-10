import pytest
import torch
from preference_privacy.encoding import parse_hh_prompt, encode_pair, PairCollator


def test_multiturn(row):
    messages = parse_hh_prompt(row["prompt"])
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert [m["content"] for m in messages] == ["Hello", "Hi", "Question"]


@pytest.mark.parametrize(
    "text", ["Hello", "\n\nHuman: Hi", "\n\nAssistant:", "\n\nHuman: \n\nAssistant:"]
)
def test_bad_dialogue(text):
    with pytest.raises(ValueError):
        parse_hh_prompt(text)


def test_mask_and_label_reversal(row, tokenizer):
    a = encode_pair(row, tokenizer)
    b = encode_pair(row | {"label": 0}, tokenizer)
    assert a["chosen_ids"] == b["rejected_ids"]
    assert a["chosen_mask"] == b["rejected_mask"]
    scored = [token for token, mask in zip(a["chosen_ids"], a["chosen_mask"]) if mask]
    assert tokenizer.decode(scored) == "Good [EOS]"
    assert all(x == 0 for x in a["chosen_mask"][: a["prompt_tokens"]])


def test_left_truncation_and_padding(row, tokenizer):
    long = encode_pair(row | {"response_b": " Good " * 20}, tokenizer, max_length=5)
    short = encode_pair(row, tokenizer, max_length=5)
    assert len(long["chosen_ids"]) == 5
    assert long["chosen_truncated_tokens"] > 0
    assert long["chosen_mask"] == [1] * 5
    batch = PairCollator(tokenizer.pad_token_id)([long, short])
    assert batch["input_ids"].shape == (4, 5)
    assert torch.all(batch["completion_mask"] <= batch["attention_mask"])


def test_boundary_failure(row, tokenizer):
    tokenizer.chat_template = (
        "{% if add_generation_prompt %}Hello{% else %}Good Bad{% endif %}"
    )
    with pytest.raises(ValueError, match="prefix"):
        encode_pair(row, tokenizer)


def test_consecutive_roles_and_indented_delimiter():
    prompt = "\n\nHuman: Hello\n\nAssistant: First\n\nAssistant: Second\n\n Human: Question\n\nAssistant:"
    messages = parse_hh_prompt(prompt)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == "First\n\nSecond"


def test_inline_role_text_is_not_guessed():
    messages = parse_hh_prompt("\n\nHuman: Explain Human: as text\n\nAssistant:")
    assert messages[0]["content"] == "Explain Human: as text"
