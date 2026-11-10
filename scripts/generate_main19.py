"""Generate fixed, blinded-review candidates after the final experiment is frozen."""

import argparse
from pathlib import Path
import random
import time
from preference_privacy.experiment import read_json, adapter_hash
from preference_privacy.runtime import load_rows, sha256, write_json, register_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    p.add_argument(
        "--name", required=True, help="initial or a preregistered generation run name"
    )
    p.add_argument("--device", choices=["cpu", "cuda", "npu"], default="cpu")
    p.add_argument("--output-json", required=True)
    args = p.parse_args()
    if Path(args.output_json).exists():
        p.error("Output already exists")
    suite = Path(args.suite)
    manifest = read_json(suite / "experiment.json")
    frozen = read_json(suite / "freeze.json")
    if frozen["experiment_sha256"] != sha256(suite / "experiment.json"):
        p.error("Experiment changed after freeze")
    settings = manifest["config"]["generation"]
    if (
        settings["do_sample"] is not False
        or manifest["config"]["training"]["input_format"] != "chat"
    ):
        p.error("This generation protocol requires greedy decoding and chat format")
    if args.name not in settings["names"]:
        p.error("Run not preregistered for generation")
    weights = None
    if args.name != "initial":
        adapter = suite / "models" / args.name / "adapter"
        weights = adapter_hash(adapter)
        if weights != frozen["adapter_sha256"][args.name]:
            p.error("Adapter changed after freeze")
    data = Path(manifest["data_dir"]) / "test.jsonl"
    if sha256(data) != manifest["config"]["split_sha256"]["test"]:
        p.error("Test data changed")
    rows = sorted(load_rows(data), key=lambda x: x["id"])
    rows = random.Random(settings["prompt_seed"]).sample(rows, settings["n_prompts"])
    register_device(args.device)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from preference_privacy.encoding import parse_hh_prompt

    start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(
        manifest["model"], revision=manifest["revision"]
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model"],
        revision=manifest["revision"],
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    if args.name != "initial":
        model = PeftModel.from_pretrained(model, str(adapter))
    model.to(args.device).eval()
    outputs = []
    with torch.no_grad():
        for row in rows:
            messages = parse_hh_prompt(row["prompt"])
            ids = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            original_length = len(ids)
            ids = ids[-settings["max_prompt_tokens"] :]
            tensor = torch.tensor([ids], device=args.device)
            prediction = model.generate(
                input_ids=tensor,
                attention_mask=torch.ones_like(tensor),
                max_new_tokens=settings["max_new_tokens"],
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )
            text = tok.decode(prediction[0, len(ids) :], skip_special_tokens=True)
            outputs.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "response": text,
                    "prompt_tokens": original_length,
                    "truncated_prompt_tokens": original_length - len(ids),
                    "generated_tokens": prediction.shape[1] - len(ids),
                }
            )
    write_json(
        args.output_json,
        {
            "name": args.name,
            "experiment_sha256": sha256(suite / "experiment.json"),
            "test_sha256": sha256(data),
            "settings": settings,
            "adapter_sha256": weights,
            "dtype": "float32",
            "device": args.device,
            "elapsed_seconds": time.monotonic() - start,
            "responses": outputs,
        },
    )


if __name__ == "__main__":
    main()
