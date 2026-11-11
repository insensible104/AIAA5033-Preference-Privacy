"""Single-device LoRA DPO with a fixed reference and explicit backward/forward correction."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", required=True, help="Local base checkpoint or Hugging Face model ID"
    )
    p.add_argument(
        "--revision", default=None, help="Pin remote model/tokenizer revision"
    )
    p.add_argument("--train-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--input-format", choices=["chat", "raw_hh"], default="chat")
    p.add_argument(
        "--group",
        choices=["dpo", "rr_dpo", "rr_rdpo", "partial", "forward"],
        required=True,
    )
    p.add_argument("--epsilon", type=float)
    p.add_argument("--rr-seed", type=int)
    p.add_argument("--correction", type=float)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--device", choices=["cpu", "cuda", "npu"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--gradient-checkpointing", action="store_true")
    args = p.parse_args()
    from preference_privacy.mechanisms import flip_probability
    from preference_privacy.runtime import (
        register_device,
        load_rows,
        sha256,
        write_json,
        disable_dropout,
    )

    if args.group == "dpo":
        if (
            args.epsilon is not None
            or args.rr_seed is not None
            or args.correction is not None
        ):
            p.error("Clean DPO does not accept epsilon, rr-seed or correction")
        rho, correction = 0.0, 0.0
    else:
        if (
            args.epsilon is None
            or not math.isfinite(args.epsilon)
            or args.epsilon <= 0
            or args.rr_seed is None
        ):
            p.error("RR groups require finite epsilon > 0 and rr-seed")
        if args.group != "partial" and args.correction is not None:
            p.error("Only group partial accepts --correction")
        correction = {"rr_dpo": 0.0, "rr_rdpo": 1.0, "forward": 0.0}.get(
            args.group, args.correction
        )
        if (
            correction is None
            or not math.isfinite(correction)
            or not 0 <= correction <= 1
        ):
            p.error("partial requires correction in [0,1]")
        rho = flip_probability(args.epsilon)
        if rho >= 0.5:
            p.error("epsilon is too small for floating-point correction")
    if not all(
        math.isfinite(x) and x > 0 for x in [args.epochs, args.learning_rate, args.beta]
    ):
        p.error("epochs, learning-rate and beta must be positive and finite")
    if (
        min(
            args.batch_size,
            args.gradient_accumulation,
            args.save_steps,
            args.lora_r,
            args.lora_alpha,
        )
        < 1
        or args.max_length < 2
    ):
        p.error("Invalid batch, checkpoint, LoRA or length setting")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        p.error("output-dir must be empty; existing experiments are never overwritten")
    if args.max_steps == 0 or args.max_steps < -1:
        p.error("max-steps must be -1 or positive")
    data_path = Path(args.train_data)
    data_hash = sha256(data_path)
    if args.group != "dpo":
        manifest_path = data_path.with_suffix(".meta.json")
        if not manifest_path.exists():
            p.error(
                "RR data requires its .meta.json sidecar; generate it with prepare_rr.py"
            )
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["epsilon"] != args.epsilon
            or manifest["rr_seed"] != args.rr_seed
            or manifest["output_sha256"] != data_hash
        ):
            p.error("RR seed, epsilon or file hash does not match the data sidecar")
    from preference_privacy.runtime import reset_peak_memory, peak_memory

    pipeline_started = time.monotonic()
    register_device(args.device)
    reset_peak_memory(args.device)
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model
    from preference_privacy.encoding import encode_pair, PairCollator, FORMAT_VERSION
    from preference_privacy.trainer import PreferenceTrainer

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=getattr(torch, args.dtype),
        attn_implementation="eager",
    )
    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    disable_dropout(model)
    model.config.use_cache = False
    rows = load_rows(data_path)
    encoded = [encode_pair(r, tok, args.input_format, args.max_length) for r in rows]
    config = vars(args) | {
        "objective": "forward" if args.group == "forward" else "backward",
        "rho": rho,
        "correction": correction,
        "data_sha256": data_hash,
        "format_version": FORMAT_VERSION,
        "n_pairs": len(rows),
        "dropout": 0.0,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "versions": {
            k: importlib.metadata.version(k)
            for k in ["torch", "transformers", "peft", "accelerate"]
        },
    }
    trainer = PreferenceTrainer(
        model=model,
        train_dataset=encoded,
        data_collator=PairCollator(tok.pad_token_id),
        processing_class=tok,
        beta=args.beta,
        rho=rho,
        correction=correction,
        objective=config["objective"],
        args=TrainingArguments(
            output_dir=str(out),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=4,
            logging_steps=10,
            report_to=[],
            remove_unused_columns=False,
            label_names=[],
            bf16=args.dtype == "bfloat16",
            use_cpu=args.device == "cpu",
            seed=args.seed,
            dataloader_pin_memory=args.device == "cuda",
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )
    if trainer.args.device.type != args.device:
        raise RuntimeError(
            f"Requested {args.device}, Trainer selected {trainer.args.device}"
        )
    if trainer.args.n_gpu > 1:
        raise RuntimeError("Expose only one accelerator per process")
    project_root = Path(__file__).resolve().parents[1]
    config["source_sha256"] = {
        str(path.relative_to(project_root)): sha256(path)
        for folder in [project_root / "src", project_root / "scripts"]
        for path in sorted(folder.rglob("*.py"))
    }
    write_json(out / "run_config.json", config)
    write_json(
        out / "token_audit.json",
        [
            {k: v for k, v in r.items() if not k.endswith(("_ids", "_mask"))}
            for r in encoded
        ],
    )
    start = time.monotonic()
    result = trainer.train()
    trainer.save_model(str(out / "adapter"))
    tok.save_pretrained(out / "adapter")
    write_json(out / "adapter" / "run_config.json", config)
    write_json(
        out / "train_metrics.json",
        result.metrics
        | {
            "elapsed_seconds": time.monotonic() - start,
            "pipeline_seconds_including_load": time.monotonic() - pipeline_started,
            "device": args.device,
            **peak_memory(args.device),
        },
    )


if __name__ == "__main__":
    main()
