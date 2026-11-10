"""Score clean preference pairs through the exact training encoder and completion mask."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--eval-data", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument(
        "--model", help="Override base location, e.g. when moving to another machine"
    )
    p.add_argument("--revision")
    p.add_argument("--input-format", choices=["chat", "raw_hh"])
    p.add_argument("--max-length", type=int)
    p.add_argument("--beta", type=float)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "cuda", "npu"], default="cpu")
    p.add_argument(
        "--allow-protocol-override",
        action="store_true",
        help="For explicitly labelled format audits only",
    )
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    if Path(args.output_json).exists():
        p.error("output-json already exists")
    adapter = Path(args.adapter_dir)
    config_path = next(
        (
            x
            for x in [adapter / "run_config.json", adapter.parent / "run_config.json"]
            if x.exists()
        ),
        None,
    )
    config = json.loads(config_path.read_text()) if config_path else {}
    if not config and (
        not args.model
        or not args.input_format
        or args.max_length is None
        or args.beta is None
    ):
        p.error(
            "Legacy adapters require explicit model, input-format, max-length and beta"
        )
    for key in ["input_format", "max_length", "beta"]:
        supplied = getattr(args, key)
        if (
            supplied is not None
            and key in config
            and supplied != config[key]
            and not args.allow_protocol_override
        ):
            p.error(
                f"{key} differs from training; use --allow-protocol-override for an explicit audit"
            )
    fmt = args.input_format or config["input_format"]
    length = args.max_length if args.max_length is not None else config["max_length"]
    beta = args.beta if args.beta is not None else config["beta"]
    if length < 2 or not __import__("math").isfinite(beta) or beta <= 0:
        p.error("Invalid max-length or beta")
    from preference_privacy.runtime import (
        register_device,
        load_rows,
        sha256,
        write_json,
    )

    register_device(args.device)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from preference_privacy.encoding import encode_pair, PairCollator, FORMAT_VERSION
    from preference_privacy.objective import preference_margin

    if config.get("format_version", FORMAT_VERSION) != FORMAT_VERSION:
        p.error("Unsupported training format version; use the matching code revision")

    base_path = args.model or config["model"]
    revision = args.revision or config.get("model_commit") or config.get("revision")
    tokenizer_source = (
        str(adapter) if (adapter / "tokenizer_config.json").exists() else base_path
    )
    tok = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=revision if tokenizer_source == base_path else None
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        revision=revision,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, str(adapter)).to(args.device).eval()
    rows = load_rows(args.eval_data)
    encoded = [encode_pair(r, tok, fmt, length) for r in rows]
    collator = PairCollator(tok.pad_token_id)
    results = []
    with torch.no_grad():
        for start in range(0, len(encoded), args.batch_size):
            part = encoded[start : start + args.batch_size]
            batch = {k: v.to(args.device) for k, v in collator(part).items()}
            margins = preference_margin(model, batch, beta).cpu().tolist()
            for row, margin in zip(part, margins):
                if not __import__("math").isfinite(margin):
                    raise ValueError("Nonfinite evaluation margin")
                results.append(
                    {k: v for k, v in row.items() if not k.endswith(("_ids", "_mask"))}
                    | {
                        "margin": margin,
                        "correct": float(margin > 0) + 0.5 * float(margin == 0),
                    }
                )
    write_json(
        args.output_json,
        {
            "evaluation_version": FORMAT_VERSION,
            "input_format": fmt,
            "max_length": length,
            "beta": beta,
            "dtype": "float32",
            "adapter_dir": str(adapter),
            "base_model": base_path,
            "revision": revision,
            "protocol_override": args.allow_protocol_override,
            "eval_sha256": sha256(args.eval_data),
            "metrics": {
                "n_pairs": len(results),
                "preference_accuracy_ties_half": sum(r["correct"] for r in results)
                / len(results),
                "mean_margin": sum(r["margin"] for r in results) / len(results),
            },
            "pairs": results,
        },
    )


if __name__ == "__main__":
    main()
