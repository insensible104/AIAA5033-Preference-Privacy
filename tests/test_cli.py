import json
from pathlib import Path
import subprocess
import sys
from transformers import Qwen2Config, Qwen2ForCausalLM

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(script, *args, ok=True):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        capture_output=True,
        text=True,
    )
    if ok:
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
    else:
        assert result.returncode != 0
    return result


def test_train_save_reload_evaluate_and_protocol_guard(tmp_path, tokenizer, row):
    base = tmp_path / "base"
    tokenizer.save_pretrained(base)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
    )
    model.save_pretrained(base)
    data = tmp_path / "clean.jsonl"
    data.write_text("".join(json.dumps(row | {"id": str(i)}) + "\n" for i in range(4)))
    run(
        "prepare_rr.py",
        "--train-data",
        data,
        "--output-dir",
        tmp_path / "rr",
        "--epsilons",
        "2",
        "--rr-seeds",
        "42",
    )
    rr = tmp_path / "rr/train_rr_eps_2_seed_42.jsonl"
    out = tmp_path / "run"
    common = [
        "--model",
        base,
        "--train-data",
        rr,
        "--output-dir",
        out,
        "--group",
        "partial",
        "--epsilon",
        "2",
        "--rr-seed",
        "42",
        "--correction",
        ".5",
        "--max-steps",
        "2",
        "--batch-size",
        "2",
        "--max-length",
        "32",
        "--save-steps",
        "1",
        "--lora-r",
        "2",
        "--lora-alpha",
        "4",
        "--gradient-checkpointing",
    ]
    run("train_dpo.py", *common)
    assert (out / "adapter/adapter_model.safetensors").exists()
    assert (out / "checkpoint-1/adapter_model.safetensors").exists()
    evaluation = tmp_path / "eval.json"
    run(
        "evaluate.py",
        "--adapter-dir",
        out / "adapter",
        "--eval-data",
        data,
        "--output-json",
        evaluation,
    )
    result = json.loads(evaluation.read_text())
    assert result["metrics"]["n_pairs"] == 4
    assert [p["id"] for p in result["pairs"]] == ["0", "1", "2", "3"]
    assert result["input_format"] == "chat"
    run(
        "evaluate.py",
        "--adapter-dir",
        out / "adapter",
        "--eval-data",
        data,
        "--output-json",
        tmp_path / "bad.json",
        "--max-length",
        "16",
        ok=False,
    )
    run("train_dpo.py", *common, ok=False)
    wrong = [str(x) for x in common]
    wrong[wrong.index("--output-dir") + 1] = str(tmp_path / "wrong")
    wrong[wrong.index("--rr-seed") + 1] = "43"
    run("train_dpo.py", *wrong, ok=False)


def test_sweep_pairing(tmp_path, row):
    data = tmp_path / "clean.jsonl"
    data.write_text(json.dumps(row) + "\n")
    out = tmp_path / "sweep"
    run(
        "run_sweep.py",
        "--model",
        "not-loaded-in-plan-mode",
        "--train-data",
        data,
        "--data-dir",
        tmp_path / "rr",
        "--output-dir",
        out,
    )
    jobs = json.loads((out / "sweep.json").read_text())["jobs"]
    assert len(jobs) == 9
    for i in range(0, 9, 3):
        commands = [j["command"] for j in jobs[i : i + 3]]
        assert len({c[c.index("--train-data") + 1] for c in commands}) == 1
        assert len({c[c.index("--seed") + 1] for c in commands}) == 1
