"""One encoding and collation path for training, evaluation and token audits."""

import re
import torch

FORMAT_VERSION = "shared_completion_v1"
FORMATS = ("chat", "raw_hh")
HH_ROLE = r"\n\n[ \t]*(Human|Assistant):"


def parse_hh_prompt(prompt):
    matches = list(re.finditer(HH_ROLE, prompt))
    if not matches or prompt[: matches[0].start()].strip():
        raise ValueError("Expected HH prompt beginning with a Human turn")
    if matches[-1].group(1) != "Assistant" or prompt[matches[-1].end() :].strip():
        raise ValueError("HH prompt must end with an empty Assistant turn")
    messages = []
    for i, match in enumerate(matches[:-1]):
        role = "user" if match.group(1) == "Human" else "assistant"
        content = prompt[match.end() : matches[i + 1].start()].strip()
        if not content or (not messages and role != "user"):
            raise ValueError("Empty turn or conversation not starting with user")
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n" + content
        else:
            messages.append({"role": role, "content": content})
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Expected a final user message")
    return messages


def encode_pair(row, tokenizer, input_format="chat", max_length=1536):
    if input_format not in FORMATS or max_length < 2:
        raise ValueError("Invalid input format or max_length")
    if row["label"] not in (0, 1):
        raise ValueError("Expected a binary label")
    chosen = row["response_b"] if row["label"] else row["response_a"]
    rejected = row["response_a"] if row["label"] else row["response_b"]
    if input_format == "chat":
        messages = parse_hh_prompt(row["prompt"])
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        sequences = [
            tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": text.strip()}],
                tokenize=True,
                add_generation_prompt=False,
            )
            for text in (chosen, rejected)
        ]
    else:
        if tokenizer.eos_token is None:
            raise ValueError("raw_hh encoding requires an EOS token")
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        sequences = [
            tokenizer.encode(
                row["prompt"]
                + text
                + ("" if text.endswith(tokenizer.eos_token) else tokenizer.eos_token),
                add_special_tokens=False,
            )
            for text in (chosen, rejected)
        ]
    # A boundary merge would silently change which tokens are scored. Fail explicitly.
    if any(ids[: len(prompt_ids)] != prompt_ids for ids in sequences):
        raise ValueError(
            "Prompt tokens are not a prefix of completion tokens; audit tokenizer boundary"
        )
    result = {
        "id": row["id"],
        "prompt_tokens": len(prompt_ids),
        "merged_history_turns": (
            len(list(re.finditer(HH_ROLE, row["prompt"]))) - 1 - len(messages)
        )
        if input_format == "chat"
        else 0,
    }
    for name, ids in zip(("chosen", "rejected"), sequences):
        full_length = len(ids)
        mask = [0] * len(prompt_ids) + [1] * (full_length - len(prompt_ids))
        ids, mask = ids[-max_length:], mask[-max_length:]
        if len(ids) < 2 or not any(mask[1:]):
            raise ValueError(f"No scoreable completion tokens for {row['id']}")
        result[f"{name}_ids"] = ids
        result[f"{name}_mask"] = mask
        result[f"{name}_original_length"] = full_length
        result[f"{name}_truncated_tokens"] = max(0, full_length - max_length)
    return result


class PairCollator:
    def __init__(self, pad_token_id):
        if pad_token_id is None:
            raise ValueError("A padding token is required")
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        # Chosen half followed by rejected half: identical ordering everywhere.
        sequences = [r[f"{side}_ids"] for side in ("chosen", "rejected") for r in rows]
        masks = [r[f"{side}_mask"] for side in ("chosen", "rejected") for r in rows]
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        completion = torch.zeros_like(ids)
        for i, (seq, mask) in enumerate(zip(sequences, masks)):
            ids[i, : len(seq)] = torch.tensor(seq)
            attention[i, : len(seq)] = 1
            completion[i, : len(seq)] = torch.tensor(mask)
        return {
            "input_ids": ids,
            "attention_mask": attention,
            "completion_mask": completion,
        }
