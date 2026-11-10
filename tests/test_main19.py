import json
from pathlib import Path
import subprocess
import sys
import pytest
from preference_privacy.experiment import main19_jobs, adapter_hash
from preference_privacy.runtime import write_json, sha256
from preference_privacy.metrics import paired_difference
from preference_privacy.encoding import FORMAT_VERSION

ROOT = Path(__file__).resolve().parents[1]


def cli(script, *args, ok=True):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result


def test_matrix():
    config = json.loads((ROOT / "configs/main19.json").read_text())
    jobs = main19_jobs(config)
    assert len(jobs) == 19 and sum(j["group"] == "dpo" for j in jobs) == 1
    assert len({j["name"] for j in jobs}) == 19
    for eps in [1, 2]:
        for rr in [42, 43, 44]:
            assert {
                j["correction"]
                for j in jobs
                if j["epsilon"] == eps and j["rr_seed"] == rr
            } == {0, 0.5, 1}
    assert config["checkpoint_steps"] == [1250, 2500, 3750]
    assert config["training"]["save_steps"] == 1250
    assert config["primary_endpoint"] == "adapter"


def test_paired_metrics_align_ids_and_reject_missing():
    a = [{"id": "a", "correct": 1}, {"id": "b", "correct": 0}]
    b = [{"id": "b", "correct": 0}, {"id": "a", "correct": 1}]
    result = paired_difference(a, b, 100, 7)
    assert result["difference_percentage_points"] == 0
    assert result["conditional_example_bootstrap_95_pct"] == [0, 0]
    assert (
        paired_difference(
            a, [{"id": "a", "correct": 0}, {"id": "b", "correct": 0}], 100, 7
        )["difference_percentage_points"]
        == 50
    )
    with pytest.raises(ValueError):
        paired_difference(a, b[:1])


@pytest.fixture
def synthetic_suite(tmp_path):
    suite = tmp_path / "suite"
    config = json.loads((ROOT / "configs/main19.json").read_text())
    jobs = main19_jobs(config)
    for job in jobs:
        job.update(data="fixture", data_sha256="fixture-hash")
    manifest = {
        "config": config,
        "model": "fixture-model",
        "revision": None,
        "jobs": jobs,
        "data_dir": str(tmp_path / "data"),
    }
    write_json(suite / "experiment.json", manifest)
    for job in jobs:
        adapter = suite / "models" / job["name"] / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(
            b"identity-only-fixture-not-real-weights"
        )
        write_json(adapter / "adapter_config.json", {})
        trained = {
            **config["training"],
            **{
                k: job[k]
                for k in ["group", "epsilon", "rr_seed", "correction", "data_sha256"]
            },
            "seed": config["training_seed"],
        }
        pairs = [
            {"id": str(i), "margin": 1.0 if i % 2 else -1.0, "correct": float(i % 2)}
            for i in range(4)
        ]
        result = {
            "eval_sha256": config["split_sha256"]["validation"],
            "evaluation_version": FORMAT_VERSION,
            **{
                k: config["training"][k] for k in ["input_format", "max_length", "beta"]
            },
            "dtype": "float32",
            "base_model": manifest["model"],
            "revision": None,
            "protocol_override": False,
            "training_config": trained,
            "adapter_sha256": adapter_hash(adapter),
            "pairs": pairs,
        }
        write_json(
            suite / "evaluations/validation" / job["name"] / "adapter.json", result
        )
        write_json(
            suite / "completed" / f"{job['name']}.json",
            {
                "experiment_sha256": sha256(suite / "experiment.json"),
                "adapter_sha256": adapter_hash(adapter),
            },
        )
    return suite


def test_summary_and_refuse_missing_or_mismatched(synthetic_suite, tmp_path):
    suite = synthetic_suite
    cli("summarize_main19.py", "--suite", suite, "--output-dir", tmp_path / "summary")
    comparisons = json.loads((tmp_path / "summary/paired_comparisons.json").read_text())
    assert len(comparisons) == 18
    assert all(p["difference_percentage_points"] == 0 for p in comparisons)
    path = suite / "evaluations/validation/clean_dpo/adapter.json"
    result = json.loads(path.read_text())
    result["training_config"]["seed"] = 0
    write_json(path, result)
    cli(
        "summarize_main19.py",
        "--suite",
        suite,
        "--output-dir",
        tmp_path / "bad",
        ok=False,
    )
    path.unlink()
    cli(
        "summarize_main19.py",
        "--suite",
        suite,
        "--output-dir",
        tmp_path / "missing",
        ok=False,
    )


def test_freeze_guards(synthetic_suite):
    suite = synthetic_suite
    cli("run_main19.py", "test", "--suite", suite, ok=False)
    cli("run_main19.py", "freeze", "--suite", suite)
    assert not (suite / "freeze.json").exists()
    cli("run_main19.py", "freeze", "--suite", suite, "--execute")
    assert (suite / "freeze.json").exists()
    cli("run_main19.py", "freeze", "--suite", suite, "--execute", ok=False)
    (suite / "models/clean_dpo/adapter/adapter_model.safetensors").write_bytes(
        b"changed"
    )
    cli("run_main19.py", "test", "--suite", suite, ok=False)


def test_plan_and_one_real_tiny_run(tmp_path, tokenizer, row):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    base = tmp_path / "base"
    tokenizer.save_pretrained(base)
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
    ).save_pretrained(base)
    data = tmp_path / "data"
    data.mkdir()
    for split in ["train", "validation", "test"]:
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(row | {"id": str(i)}) + "\n" for i in range(4))
        )
    config = json.loads((ROOT / "configs/main19.json").read_text())
    config["expected_train_pairs"] = 4
    config["split_sha256"] = {
        s: sha256(data / f"{s}.jsonl") for s in ["train", "validation", "test"]
    }
    config["training"].update(
        epochs=1, save_steps=1, max_length=32, lora_r=2, lora_alpha=4
    )
    config["checkpoint_steps"] = [1]
    config["generation"].update(n_prompts=2, max_new_tokens=3)
    spec = tmp_path / "config.json"
    write_json(spec, config)
    suite = tmp_path / "suite"
    cli(
        "run_main19.py",
        "plan",
        "--suite",
        suite,
        "--config",
        spec,
        "--model",
        base,
        "--data-dir",
        data,
    )
    manifest = json.loads((suite / "experiment.json").read_text())
    assert len(manifest["jobs"]) == 19
    for i in range(1, 19, 3):
        assert len({j["data_sha256"] for j in manifest["jobs"][i : i + 3]}) == 1
    result = cli("run_main19.py", "train", "--suite", suite, "--jobs", "clean_dpo")
    assert "--save-steps 1" in result.stdout
    assert not (suite / "models").exists()
    cli("run_main19.py", "train", "--suite", suite, "--jobs", "clean_dpo", "--execute")
    repeat = cli(
        "run_main19.py", "train", "--suite", suite, "--jobs", "clean_dpo", "--execute"
    )
    assert "Skip completed" in repeat.stdout
    cli(
        "run_main19.py",
        "validation",
        "--suite",
        suite,
        "--jobs",
        "clean_dpo",
        "--execute",
    )
    result = json.loads(
        (suite / "evaluations/validation/clean_dpo/adapter.json").read_text()
    )
    assert result["metrics"]["n_pairs"] == 4
    assert result["cost"]["peak_allocated_bytes"] is None
    assert (suite / "evaluations/validation/clean_dpo/checkpoint-1.json").exists()

    write_json(
        suite / "freeze.json",
        {
            "experiment_sha256": sha256(suite / "experiment.json"),
            "adapter_sha256": {
                "clean_dpo": adapter_hash(suite / "models/clean_dpo/adapter")
            },
        },
    )
    for name in ["initial", "clean_dpo"]:
        cli(
            "generate_main19.py",
            "--suite",
            suite,
            "--name",
            name,
            "--output-json",
            suite / "generation" / f"{name}.json",
        )
    initial = json.loads((suite / "generation/initial.json").read_text())
    trained = json.loads((suite / "generation/clean_dpo.json").read_text())
    assert len(initial["responses"]) == 2
    assert [r["id"] for r in initial["responses"]] == [
        r["id"] for r in trained["responses"]
    ]
    assert all(r["generated_tokens"] <= 3 for r in trained["responses"])
