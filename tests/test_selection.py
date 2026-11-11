import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from preference_privacy.experiment import (
    adapter_hash,
    experiment_jobs,
    validate_paired_suites,
)
from preference_privacy.objective import forward_loss
from preference_privacy.runtime import sha256, write_json, load_rows
from preference_privacy.selection import canonicalize, select_models
from preference_privacy.encoding import FORMAT_VERSION

ROOT = Path(__file__).resolve().parents[1]


def cli(script, *args, ok=True):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result


def test_forward_likelihood_optimum_stability_and_clean_limit():
    z = torch.tensor(
        [-10000.0, -1.0, 0.0, 1.0, 10000.0], dtype=torch.float64, requires_grad=True
    )
    for rho in [0.01, 0.2, 0.499]:
        loss = forward_loss(z, rho)
        assert torch.isfinite(loss).all()
        assert (loss >= -math.log(1 - rho) - 1e-12).all()
        assert (loss <= -math.log(rho) + 1e-12).all()
        assert torch.isfinite(torch.autograd.grad(loss.sum(), z)[0]).all()
        # Population noisy-label likelihood is minimized at the clean log odds.
        true_prob = 0.7
        zstar = torch.tensor(
            math.log(true_prob / (1 - true_prob)),
            dtype=torch.float64,
            requires_grad=True,
        )
        noisy_prob = rho + (1 - 2 * rho) * true_prob
        risk = noisy_prob * forward_loss(zstar, rho) + (1 - noisy_prob) * forward_loss(
            -zstar, rho
        )
        grad = torch.autograd.grad(risk, zstar, create_graph=True)[0]
        assert abs(grad.item()) < 1e-12
        assert torch.autograd.grad(grad, zstar)[0] > 0
    torch.testing.assert_close(forward_loss(z, 0), torch.nn.functional.softplus(-z))
    for rho in [-1, 0.5, float("nan")]:
        with pytest.raises(ValueError):
            forward_loss(z, rho)


def test_trainer_dispatch_without_training(monkeypatch):
    from types import SimpleNamespace
    import preference_privacy.trainer as module

    z = torch.tensor([1.0, -2.0], requires_grad=True)
    monkeypatch.setattr(module, "preference_margin", lambda *a: z)
    trainer = object.__new__(module.PreferenceTrainer)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda m: m)
    trainer.args = SimpleNamespace(world_size=1)
    trainer.beta, trainer.rho, trainer.correction, trainer.objective = (
        0.1,
        0.2,
        0.0,
        "forward",
    )
    loss = trainer.compute_loss(None, {})
    torch.testing.assert_close(loss, forward_loss(z, 0.2).mean())
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_canonicalization_ignores_original_order_and_labels(row):
    pub, gt = canonicalize([row])
    swapped = row | {
        "response_a": row["response_b"],
        "response_b": row["response_a"],
        "label": 1 - row["label"],
    }
    pub2, gt2 = canonicalize([swapped])
    assert pub == pub2 and gt == gt2
    pub3, gt3 = canonicalize([row | {"label": 1 - row["label"]}])
    assert pub3 == pub and next(iter(gt3.values())) != next(iter(gt.values()))
    assert pub[0]["label"] == 0


def bundle(n=1000):
    labels = {str(i): 0 for i in range(n)}
    return {
        "version": 1,
        "alpha": 0.05,
        "contexts": {
            "c": {
                "order": ["base", "good", "tied"],
                "baseline": "base",
                "fixed": {"fixed_rr_dpo": "base"},
                "predictions": {
                    "base": {i: 1 for i in labels},
                    "good": dict(labels),
                    "tied": dict(labels),
                },
            }
        },
        "releases": [dict(context="c", repeat=0, n=n, epsilon=2.0, labels=labels)],
    }


def test_selection_ties_debiasing_and_conservative_fallback():
    b = bundle()
    decision = select_models(b)[0]
    assert decision["selected"]["noisy_argmax"] == "good"
    assert decision["selected"]["conservative"] == "good" and not decision["fallback"]
    assert (
        max(decision["debiased_accuracy"], key=decision["debiased_accuracy"].get)
        == "good"
    )
    assert decision["debiased_accuracy"]["good"] > 1  # Never clip before inference.
    small = select_models(bundle(2))[0]
    assert small["fallback"] and small["selected"]["conservative"] == "base"
    b["contexts"]["c"]["predictions"]["good"] = {
        i: 0.5 for i in b["releases"][0]["labels"]
    }
    assert select_models(b)[0]["noisy_accuracy"]["good"] == 0.5


def test_selector_rejects_extra_clean_fields_and_bad_ids():
    b = bundle()
    b["releases"][0]["clean_labels"] = {}
    with pytest.raises(ValueError):
        select_models(b)
    b = bundle()
    b["contexts"]["c"]["predictions"]["good"].pop("0")
    with pytest.raises(ValueError):
        select_models(b)
    b = bundle()
    b["contexts"]["c"]["predictions"]["good"]["0"] = float("nan")
    with pytest.raises(ValueError):
        select_models(b)
    b = bundle()
    b["releases"].append(copy.deepcopy(b["releases"][0]))
    with pytest.raises(ValueError):
        select_models(b)


@pytest.fixture
def planned(tmp_path, row):
    data = tmp_path / "data"
    data.mkdir()
    for split, n in [("train", 4), ("validation", 24), ("test", 20)]:
        records = [
            row
            | {
                "id": f"{split}-{i}",
                "prompt": row["prompt"].replace("Human:", f"Human: {split} {i}"),
                "label": i % 2,
            }
            for i in range(n)
        ]
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
    base = tmp_path / "local-model"
    base.mkdir()
    config = json.loads((ROOT / "configs/main19.json").read_text())
    config["expected_train_pairs"] = 4
    config["checkpoint_steps"] = [1, 2, 3]
    config["training"]["save_steps"] = 1
    config["split_sha256"] = {
        s: sha256(data / f"{s}.jsonl") for s in ["train", "validation", "test"]
    }
    config["evaluation"]["bootstrap_samples"] = 20
    spec = tmp_path / "main.json"
    write_json(spec, config)
    main = tmp_path / "main"
    forward = tmp_path / "forward"
    selection = tmp_path / "selection"
    cli(
        "run_main19.py",
        "plan",
        "--suite",
        main,
        "--config",
        spec,
        "--data-dir",
        data,
        "--model",
        base,
    )
    config["suite_kind"] = "forward6"
    config["name"] = "fixture_forward"
    fspec = tmp_path / "forward.json"
    write_json(fspec, config)
    cli(
        "run_main19.py",
        "plan",
        "--suite",
        forward,
        "--config",
        fspec,
        "--paired-suite",
        main,
    )
    sc = json.loads((ROOT / "configs/selection.json").read_text())
    sc.update(
        sizes=[5, 12, 24],
        repeats=3,
        endpoints=["checkpoint-1", "checkpoint-2", "adapter"],
    )
    sspec = tmp_path / "selection.json"
    write_json(sspec, sc)
    cli(
        "run_selection.py",
        "plan",
        "--main-suite",
        main,
        "--forward-suite",
        forward,
        "--output-dir",
        selection,
        "--config",
        sspec,
    )
    return main, forward, selection


def populate(planned, split):
    main, forward, root = planned
    plan = json.loads((root / "plan.json").read_text())
    public = load_rows(root / "public_validation.jsonl")
    for source, suite in [("main", main), ("forward", forward)]:
        manifest = json.loads((suite / "experiment.json").read_text())
        for job in manifest["jobs"]:
            trained = {
                **manifest["config"]["training"],
                **{
                    k: job[k]
                    for k in [
                        "group",
                        "epsilon",
                        "rr_seed",
                        "correction",
                        "data_sha256",
                    ]
                },
                "seed": manifest["config"]["training_seed"],
            }
            if job["group"] == "forward":
                trained["objective"] = "forward"
            for endpoint in ["checkpoint-1", "checkpoint-2", "adapter"]:
                adapter = suite / "models" / job["name"] / endpoint
                adapter.mkdir(parents=True, exist_ok=True)
                (adapter / "adapter_model.safetensors").write_bytes(
                    b"identity-fixture-not-real-weights"
                )
                write_json(adapter / "adapter_config.json", {})
                rows = (
                    public
                    if split == "validation"
                    else load_rows(Path(manifest["data_dir"]) / "test.jsonl")
                )
                pairs = [
                    dict(
                        id=r["id"],
                        margin=1.0 if (i + len(job["name"])) % 3 else -1.0,
                        correct=1.0 if (i + len(job["name"])) % 3 else 0.0,
                    )
                    for i, r in enumerate(rows)
                ]
                result = dict(
                    eval_sha256=plan["public_sha256"]
                    if split == "validation"
                    else manifest["config"]["split_sha256"]["test"],
                    evaluation_version=FORMAT_VERSION,
                    dtype="float32",
                    protocol_override=False,
                    base_model=manifest["model"],
                    revision=manifest["revision"],
                    **{
                        k: manifest["config"]["training"][k]
                        for k in ["input_format", "max_length", "beta"]
                    },
                    training_config=trained,
                    adapter_sha256=adapter_hash(adapter),
                    pairs=pairs,
                )
                if job["epsilon"] is not None:
                    write_json(root / split / f"{job['name']}__{endpoint}.json", result)
                # The normal runner evaluates original clean-order validation rows.
                normal = copy.deepcopy(result)
                normal["eval_sha256"] = manifest["config"]["split_sha256"][split]
                if endpoint == "adapter":
                    write_json(
                        suite / "evaluations" / split / job["name"] / "adapter.json",
                        normal,
                    )
            write_json(
                suite / "completed" / f"{job['name']}.json",
                dict(
                    experiment_sha256=sha256(suite / "experiment.json"),
                    adapter_sha256=adapter_hash(
                        suite / "models" / job["name"] / "adapter"
                    ),
                ),
            )


def test_plans_dry_run_and_pairing(planned):
    main, forward, selection = planned
    m = json.loads((main / "experiment.json").read_text())
    f = json.loads((forward / "experiment.json").read_text())
    validate_paired_suites(m, f)
    assert len(f["jobs"]) == 6
    assert all(j["group"] == "forward" for j in experiment_jobs(f["config"]))
    output = cli("run_main19.py", "train", "--suite", forward)
    assert output.stdout.count("--group forward") == 6
    assert not (forward / "models").exists()
    output = cli("run_selection.py", "predict", "--output-dir", selection)
    assert output.stdout.count("evaluate.py") == 72
    assert not (selection / "validation").exists()
    cli("run_selection.py", "test", "--output-dir", selection, ok=False)
    f["jobs"][0]["data_sha256"] = "wrong"
    with pytest.raises(ValueError):
        validate_paired_suites(m, f)
    # RR labels are shared across training seeds and nested across subset sizes.
    releases = json.loads((selection / "releases.json").read_text())
    a = next(
        r
        for r in releases
        if r["context"] == "eps_1_rr_42" and r["repeat"] == 0 and r["n"] == 5
    )
    b = next(
        r
        for r in releases
        if r["context"] == "eps_1_rr_43" and r["repeat"] == 0 and r["n"] == 24
    )
    assert all(b["labels"][i] == y for i, y in a["labels"].items())


def test_selection_freeze_oracle_test_summary_no_training(planned, tmp_path):
    main, forward, root = planned
    populate(planned, "validation")
    cli("run_selection.py", "select", "--output-dir", root)
    choices = json.loads((root / "choices.json").read_text())
    assert len(choices["decisions"]) == 6 * 3 * 3
    assert {d["k"] for d in choices["decisions"]} == {12}
    # Pure selector works even if private oracle labels are unavailable.
    secret = root / "simulation_private/clean_validation.json"
    saved = secret.read_text()
    secret.unlink()
    cli(
        "select_noisy.py",
        "--bundle",
        root / "bundle.json",
        "--output",
        tmp_path / "independent.json",
    )
    assert json.loads((tmp_path / "independent.json").read_text()) == choices
    secret.write_text(saved)
    cli("run_selection.py", "test", "--output-dir", root, ok=False)
    cli("run_selection.py", "oracle", "--output-dir", root)
    cli("run_selection.py", "oracle", "--output-dir", root, ok=False)
    cli("run_selection.py", "test", "--output-dir", root)  # dry-run
    assert not (root / "test_release.json").exists()
    populate(planned, "test")
    # Existing synthetic evaluation receipts are verified; no model is invoked.
    cli("run_selection.py", "test", "--output-dir", root, "--execute")
    cli("run_selection.py", "summarize", "--output-dir", root)
    report = json.loads((root / "summary/results.json").read_text())
    assert len(report["records"]) == 6 * 3 * 3 * 7
    assert len(report["aggregates"]) == 6 * 3 * 7
    assert len(report["by_epsilon"]) == 2 * 3 * 7
    assert all(r["n_training_rr_seeds"] == 3 for r in report["by_epsilon"])
    assert all(r["hindsight_test_regret_points"] >= 0 for r in report["records"])
    for suite in [main, forward]:
        cli("run_main19.py", "freeze", "--suite", suite, "--execute")
    cli(
        "summarize_forward.py",
        "--main-suite",
        main,
        "--forward-suite",
        forward,
        "--split",
        "test",
        "--output-dir",
        tmp_path / "forward_summary",
    )
    comparisons = json.loads(
        (tmp_path / "forward_summary/paired_comparisons.json").read_text()
    )
    assert len(comparisons) == 18
    # Changed selection output invalidates test access, even when evaluations exist.
    choice = json.loads((root / "choices.json").read_text())
    choice["decisions"][0]["fallback"] = not choice["decisions"][0]["fallback"]
    write_json(root / "choices.json", choice)
    cli("run_selection.py", "test", "--output-dir", root, ok=False)


def test_changed_adapter_rejected(planned):
    main, forward, root = planned
    populate(planned, "validation")
    (
        main / "models/eps_1_rr_42_lambda_0/checkpoint-1/adapter_model.safetensors"
    ).write_bytes(b"changed")
    cli("run_selection.py", "select", "--output-dir", root, ok=False)
    assert not (root / "choices.json").exists()
