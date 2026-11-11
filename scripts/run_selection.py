"""Register, predict, select, freeze oracle, test and summarize an offline RR simulation.

No training commands exist here. Prediction/test default to dry-run.
"""

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

from preference_privacy.experiment import (
    read_json,
    adapter_hash,
    validate_evaluation,
    validate_paired_suites,
)
from preference_privacy.runtime import load_rows, sha256, write_json
from preference_privacy.selection import (
    canonicalize,
    seeded_rng,
    accuracy,
    select_models,
)

SCRIPTS = Path(__file__).resolve().parent


def write_new(path, value):
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    write_json(path, value)


def no_test_outputs(suites, root):
    for suite in suites:
        if any((suite / "evaluations/test").rglob("*.json")) or any(
            (suite / "generation").rglob("*.json")
        ):
            raise ValueError(
                "Test/generation already accessed; register an exploratory study separately"
            )
    if (root / "test_release.json").exists() or any((root / "test").rglob("*.json")):
        raise ValueError("Test already accessed")


def source_state(plan):
    manifests = {}
    for name, source in plan["sources"].items():
        path = Path(source["suite"]) / "experiment.json"
        if sha256(path) != source["sha256"]:
            raise ValueError("Source experiment changed after registration")
        manifests[name] = read_json(path)
    validate_paired_suites(manifests["main"], manifests["forward"])
    return manifests


def adapter_path(plan, candidate):
    return (
        Path(plan["sources"][candidate["source"]]["suite"])
        / "models"
        / candidate["job"]
        / candidate["endpoint"]
    )


def checked_result(root, plan, manifests, candidate, split):
    result = read_json(root / split / f"{candidate['id']}.json")
    manifest = copy.deepcopy(manifests[candidate["source"]])
    if split == "validation":
        manifest["config"]["split_sha256"][split] = plan["public_sha256"]
    job = next(j for j in manifest["jobs"] if j["name"] == candidate["job"])
    validate_evaluation(result, manifest, job, split)
    if result["adapter_sha256"] != adapter_hash(adapter_path(plan, candidate)):
        raise ValueError("Changed adapter")
    # Verify a completed training run and its final adapter before using checkpoints.
    suite = Path(plan["sources"][candidate["source"]]["suite"])
    marker = read_json(suite / "completed" / f"{candidate['job']}.json")
    if marker["experiment_sha256"] != plan["sources"][candidate["source"]][
        "sha256"
    ] or marker["adapter_sha256"] != adapter_hash(
        suite / "models" / candidate["job"] / "adapter"
    ):
        raise ValueError("Missing/changed completed training identity")
    rows = result["pairs"]
    expected_ids = (
        {r["id"] for r in load_rows(root / "public_validation.jsonl")}
        if split == "validation"
        else {r["id"] for r in load_rows(Path(manifest["data_dir"]) / "test.jsonl")}
    )
    if len(rows) != len(expected_ids) or {r["id"] for r in rows} != expected_ids:
        raise ValueError("Evaluation sample identity mismatch")
    if any(
        not math.isfinite(r["margin"])
        or r["correct"] != float(r["margin"] > 0) + 0.5 * float(r["margin"] == 0)
        for r in rows
    ):
        raise ValueError("Invalid margin/accuracy")
    return result


def verify_freeze(root):
    frozen = read_json(root / "selection_freeze.json")
    for name, expected in frozen["files"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Selection input/output changed: {name}")
    return frozen


def register(args, root):
    if root.exists() and any(root.iterdir()):
        raise ValueError("Selection directory must be empty")
    if not args.main_suite or not args.forward_suite:
        raise ValueError("plan requires --main-suite and --forward-suite")
    sources = {
        k: dict(
            suite=str(Path(v).resolve()), sha256=sha256(Path(v) / "experiment.json")
        )
        for k, v in [("main", args.main_suite), ("forward", args.forward_suite)]
    }
    manifests = source_state({"sources": sources})
    no_test_outputs([Path(s["suite"]) for s in sources.values()], root)
    config = read_json(args.config)
    if (
        set(config["families"]) != {"rr_dpo", "partial", "rr_rdpo", "forward"}
        or len(config["families"]) != 4
    ):
        raise ValueError("Register four distinct families")
    expected_endpoints = [
        f"checkpoint-{s}" for s in manifests["main"]["config"]["checkpoint_steps"][:-1]
    ] + ["adapter"]
    if config["endpoints"] != expected_endpoints or len(expected_endpoints) != 3:
        raise ValueError("Expected two intermediate checkpoints plus final adapter")
    if (
        config["baseline_family"] != "rr_dpo"
        or config["baseline_endpoint"] != "adapter"
        or config["tie_break"] != "registered_candidate_order"
    ):
        raise ValueError("Unsupported baseline/tie rule")
    if (
        not 0 < config["alpha"] < 1
        or not isinstance(config["repeats"], int)
        or config["repeats"] < 1
    ):
        raise ValueError("Invalid alpha/repeats")
    if (
        not config["sizes"]
        or sorted(set(config["sizes"])) != config["sizes"]
        or any(not isinstance(n, int) or n < 1 for n in config["sizes"])
    ):
        raise ValueError("sizes must be sorted distinct positive integers")
    main = manifests["main"]
    data = Path(main["data_dir"]) / "validation.jsonl"
    if sha256(data) != main["config"]["split_sha256"]["validation"]:
        raise ValueError("Validation split changed")
    public, labels = canonicalize(load_rows(data))
    if max(config["sizes"]) > len(public):
        raise ValueError("Validation subset larger than available data")
    candidates, contexts = [], {}
    for eps in main["config"]["epsilons"]:
        val_eps = config["epsilon_val"][f"{eps:g}"]
        if not math.isfinite(val_eps) or val_eps <= 0 or math.tanh(val_eps / 2) == 0:
            raise ValueError("Invalid epsilon_val")
        for rr in main["config"]["rr_seeds"]:
            context = f"eps_{eps:g}_rr_{rr}"
            contexts[context] = dict(epsilon_train=eps, rr_seed=rr, epsilon_val=val_eps)
            for family in config["families"]:
                source = "forward" if family == "forward" else "main"
                job = next(
                    j
                    for j in manifests[source]["jobs"]
                    if j["epsilon"] == eps
                    and j["rr_seed"] == rr
                    and j["group"] == family
                )
                for endpoint in config["endpoints"]:
                    candidates.append(
                        dict(
                            id=f"{job['name']}__{endpoint}",
                            context=context,
                            family=family,
                            source=source,
                            job=job["name"],
                            endpoint=endpoint,
                        )
                    )
    root.mkdir(parents=True, exist_ok=True)
    public_path = root / "public_validation.jsonl"
    public_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in public)
    )
    write_json(root / "simulation_private/clean_validation.json", labels)
    # One full release per epsilon/repetition; nested sizes and all training seeds share it.
    ids = [r["id"] for r in public]
    seeded_rng(f"selection-subset-v1:{config['subset_seed']}").shuffle(ids)
    releases = []
    for context, info in contexts.items():
        eps = info["epsilon_val"]
        rho = (1 - math.tanh(eps / 2)) / 2
        for repeat in range(config["repeats"]):
            rng = seeded_rng(
                f"selection-validation-rr-v1:{float(eps).hex()}:{config['rr_seed']}:{repeat}"
            )
            noisy = {i: labels[i] ^ int(rng.random() < rho) for i in sorted(labels)}
            for n in config["sizes"]:
                releases.append(
                    dict(
                        context=context,
                        repeat=repeat,
                        n=n,
                        epsilon=eps,
                        labels={i: noisy[i] for i in ids[:n]},
                    )
                )
    write_json(root / "releases.json", releases)
    write_json(
        root / "plan.json",
        dict(
            version=1,
            config=config,
            sources=sources,
            contexts=contexts,
            candidates=candidates,
            public_sha256=sha256(public_path),
            releases_sha256=sha256(root / "releases.json"),
            private_sha256=sha256(root / "simulation_private/clean_validation.json"),
        ),
    )
    print(
        f"Registered {len(candidates)} candidates across {len(contexts)} contexts; no model loaded"
    )


def predict(args, root, plan, manifests):
    split = "validation" if args.stage == "predict" else "test"
    if split == "test":
        verify_freeze(root)
        oracle = root / "oracle.json"
        if not oracle.exists():
            raise ValueError("Freeze clean-validation oracle BEFORE accessing test")
        if read_json(oracle).get("selection_freeze_sha256") != sha256(
            root / "selection_freeze.json"
        ):
            raise ValueError("Oracle belongs to a different selection freeze")
        release = {
            "selection_freeze_sha256": sha256(root / "selection_freeze.json"),
            "oracle_sha256": sha256(oracle),
        }
        receipt = root / "test_release.json"
        if receipt.exists() and read_json(receipt) != release:
            raise ValueError("Choices/oracle changed after test release")
        if args.execute and not receipt.exists():
            write_new(receipt, release)
    elif (root / "selection_freeze.json").exists():
        raise ValueError("Validation predictions already frozen")
    for c in plan["candidates"]:
        manifest = manifests[c["source"]]
        data = (
            root / "public_validation.jsonl"
            if split == "validation"
            else Path(manifest["data_dir"]) / "test.jsonl"
        )
        expected = (
            plan["public_sha256"]
            if split == "validation"
            else manifest["config"]["split_sha256"]["test"]
        )
        if sha256(data) != expected:
            raise ValueError("Evaluation data changed")
        output = root / split / f"{c['id']}.json"
        if output.exists():
            checked_result(root, plan, manifests, c, split)
            continue
        command = [
            sys.executable,
            str(SCRIPTS / "evaluate.py"),
            "--adapter-dir",
            str(adapter_path(plan, c)),
            "--eval-data",
            str(data),
            "--output-json",
            str(output),
            "--device",
            args.device,
            "--batch-size",
            str(manifest["config"]["evaluation"]["batch_size"]),
        ]
        print(shlex.join(command), flush=True)
        if args.execute:
            # Fail before expensive inference if training is incomplete.
            suite = Path(plan["sources"][c["source"]]["suite"])
            marker = read_json(suite / "completed" / f"{c['job']}.json")
            if marker["experiment_sha256"] != plan["sources"][c["source"]][
                "sha256"
            ] or marker["adapter_sha256"] != adapter_hash(
                suite / "models" / c["job"] / "adapter"
            ):
                raise ValueError("Training completion identity mismatch")
            if split == "test":
                checked_result(root, plan, manifests, c, "validation")
            log = root / "logs" / f"{c['id']}_{split}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as handle:
                subprocess.run(
                    command, check=True, stdout=handle, stderr=subprocess.STDOUT
                )
            checked_result(root, plan, manifests, c, split)


def freeze_selection(root, plan, manifests):
    no_test_outputs([Path(s["suite"]) for s in plan["sources"].values()], root)
    if (root / "selection_freeze.json").exists() or (root / "bundle.json").exists():
        raise ValueError("Selection already built; do not overwrite")
    contexts = {}
    files = ["plan.json", "public_validation.jsonl", "releases.json"]
    for context in plan["contexts"]:
        candidates = [c for c in plan["candidates"] if c["context"] == context]
        predictions = {}
        fixed = {}
        for c in candidates:
            result = checked_result(root, plan, manifests, c, "validation")
            predictions[c["id"]] = {
                p["id"]: 0 if p["margin"] > 0 else 1 if p["margin"] < 0 else 0.5
                for p in result["pairs"]
            }
            files.append(f"validation/{c['id']}.json")
            if c["endpoint"] == "adapter":
                fixed[f"fixed_{c['family']}"] = c["id"]
        contexts[context] = dict(
            order=[c["id"] for c in candidates],
            baseline=fixed["fixed_rr_dpo"],
            fixed=fixed,
            predictions=predictions,
        )
    bundle = dict(
        version=1,
        alpha=plan["config"]["alpha"],
        contexts=contexts,
        releases=read_json(root / "releases.json"),
    )
    # Validate before creating immutable outputs; subprocess cannot receive private paths.
    select_models(bundle)
    write_new(root / "bundle.json", bundle)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "select_noisy.py"),
            "--bundle",
            str(root / "bundle.json"),
            "--output",
            str(root / "choices.json"),
        ],
        check=True,
    )
    files += ["bundle.json", "choices.json"]
    write_new(
        root / "selection_freeze.json", {"files": {p: sha256(root / p) for p in files}}
    )
    print("Label-blind choices frozen; clean oracle has not been read")


def freeze_oracle(root, plan):
    verify_freeze(root)
    no_test_outputs([Path(s["suite"]) for s in plan["sources"].values()], root)
    if (root / "test_release.json").exists() or any((root / "test").rglob("*.json")):
        raise ValueError("Cannot choose oracle after test access")
    path = root / "simulation_private/clean_validation.json"
    if sha256(path) != plan["private_sha256"]:
        raise ValueError("Private simulation labels changed")
    labels = read_json(path)
    bundle = read_json(root / "bundle.json")
    outputs = []
    for release in bundle["releases"]:
        context = bundle["contexts"][release["context"]]
        subset = {i: labels[i] for i in release["labels"]}
        best = max(
            context["order"], key=lambda k: accuracy(context["predictions"][k], subset)
        )
        outputs.append(
            dict(
                context=release["context"],
                repeat=release["repeat"],
                n=release["n"],
                selected=best,
            )
        )
    write_new(
        root / "oracle.json",
        dict(
            selection_freeze_sha256=sha256(root / "selection_freeze.json"),
            decisions=outputs,
        ),
    )
    print("Offline clean-validation comparator frozen (not a deployable selector)")


def summarize(root, plan, manifests):
    verify_freeze(root)
    receipt = read_json(root / "test_release.json")
    if receipt != dict(
        selection_freeze_sha256=sha256(root / "selection_freeze.json"),
        oracle_sha256=sha256(root / "oracle.json"),
    ):
        raise ValueError("Test release identity mismatch")
    values = {}
    for c in plan["candidates"]:
        result = checked_result(root, plan, manifests, c, "test")
        # Test adapter must be exactly the candidate observed during selection.
        if (
            result["adapter_sha256"]
            != checked_result(root, plan, manifests, c, "validation")["adapter_sha256"]
        ):
            raise ValueError("Test/selection adapter mismatch")
        values[c["id"]] = statistics.mean(r["correct"] for r in result["pairs"])
    oracle = {
        (x["context"], x["repeat"], x["n"]): x["selected"]
        for x in read_json(root / "oracle.json")["decisions"]
    }
    bundle = read_json(root / "bundle.json")
    records = []
    for decision in read_json(root / "choices.json")["decisions"]:
        context, repeat, n = decision["context"], decision["repeat"], decision["n"]
        clean_best = oracle[context, repeat, n]
        base = bundle["contexts"][context]["baseline"]
        hindsight = max(values[k] for k in bundle["contexts"][context]["order"])
        for rule, selected in {
            **decision["selected"],
            "clean_validation_oracle": clean_best,
        }.items():
            records.append(
                dict(
                    context=context,
                    repeat=repeat,
                    n=n,
                    rule=rule,
                    selected=selected,
                    test_accuracy=values[selected],
                    difference_vs_default_points=100
                    * (values[selected] - values[base]),
                    worse_than_default=values[selected] < values[base],
                    difference_vs_clean_selector_points=100
                    * (values[selected] - values[clean_best]),
                    hindsight_test_regret_points=100 * (hindsight - values[selected]),
                    fallback=decision["fallback"] if rule == "conservative" else None,
                )
            )
    groups = {}
    for row in records:
        groups.setdefault((row["context"], row["n"], row["rule"]), []).append(row)
    aggregates = []
    for (context, n, rule), rows in groups.items():
        aggregates.append(
            dict(
                context=context,
                n=n,
                rule=rule,
                n_validation_rr_repeats=len(rows),
                mean_test_accuracy=statistics.mean(r["test_accuracy"] for r in rows),
                validation_rr_sd_points=statistics.stdev(
                    100 * r["test_accuracy"] for r in rows
                )
                if len(rows) > 1
                else None,
                mean_difference_vs_default_points=statistics.mean(
                    r["difference_vs_default_points"] for r in rows
                ),
                worse_than_default_rate=statistics.mean(
                    r["worse_than_default"] for r in rows
                ),
                mean_difference_vs_clean_selector_points=statistics.mean(
                    r["difference_vs_clean_selector_points"] for r in rows
                ),
                mean_hindsight_test_regret_points=statistics.mean(
                    r["hindsight_test_regret_points"] for r in rows
                ),
                fallback_rate=statistics.mean(r["fallback"] for r in rows)
                if rule == "conservative"
                else None,
            )
        )
    by_epsilon = []
    for eps in sorted({x["epsilon_train"] for x in plan["contexts"].values()}):
        for n in plan["config"]["sizes"]:
            for rule in sorted({r["rule"] for r in aggregates}):
                rows = [
                    r
                    for r in aggregates
                    if r["n"] == n
                    and r["rule"] == rule
                    and plan["contexts"][r["context"]]["epsilon_train"] == eps
                ]
                by_epsilon.append(
                    dict(
                        epsilon_train=eps,
                        n=n,
                        rule=rule,
                        n_training_rr_seeds=len(rows),
                        mean_test_accuracy=statistics.mean(
                            r["mean_test_accuracy"] for r in rows
                        ),
                        training_rr_sd_of_repeat_mean_points=statistics.stdev(
                            100 * r["mean_test_accuracy"] for r in rows
                        )
                        if len(rows) > 1
                        else None,
                        mean_worse_than_default_rate=statistics.mean(
                            r["worse_than_default_rate"] for r in rows
                        ),
                        mean_difference_vs_default_points=statistics.mean(
                            r["mean_difference_vs_default_points"] for r in rows
                        ),
                        mean_difference_vs_clean_selector_points=statistics.mean(
                            r["mean_difference_vs_clean_selector_points"] for r in rows
                        ),
                        mean_fallback_rate=statistics.mean(
                            r["fallback_rate"] for r in rows
                        )
                        if rule == "conservative"
                        else None,
                    )
                )
    out = root / "summary"
    if out.exists():
        raise ValueError("Summary already exists")
    out.mkdir()
    for name, rows in [
        ("decisions", records),
        ("by_context", aggregates),
        ("by_epsilon", by_epsilon),
    ]:
        with (out / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(
        out / "results.json",
        dict(
            records=records,
            aggregates=aggregates,
            by_epsilon=by_epsilon,
            note="Validation RR repetitions are conditional on fixed trained models and a shared test set; not independent training runs or test datasets. Hindsight regret is diagnostic only.",
        ),
    )
    print(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["plan", "predict", "select", "oracle", "test", "summarize"]
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--main-suite")
    p.add_argument("--forward-suite")
    p.add_argument("--config", default="configs/selection.json")
    p.add_argument("--device", choices=["cpu", "cuda", "npu"], default="cpu")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Required only to execute inference in predict/test",
    )
    a = p.parse_args()
    root = Path(a.output_dir).resolve()
    if a.stage == "plan":
        register(a, root)
        return
    plan = read_json(root / "plan.json")
    manifests = source_state(plan)
    for file, key in [
        ("public_validation.jsonl", "public_sha256"),
        ("releases.json", "releases_sha256"),
    ]:
        if sha256(root / file) != plan[key]:
            raise ValueError("Registered validation input changed")
    if a.stage in ("predict", "test"):
        predict(a, root, plan, manifests)
    elif a.stage == "select":
        freeze_selection(root, plan, manifests)
    elif a.stage == "oracle":
        freeze_oracle(root, plan)
    else:
        summarize(root, plan, manifests)


if __name__ == "__main__":
    main()
