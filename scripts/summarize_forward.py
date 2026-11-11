"""Paired final-endpoint forward comparisons against the existing main19 suite."""

import argparse
import csv
from pathlib import Path
import statistics
from preference_privacy.experiment import (
    read_json,
    validate_paired_suites,
    validate_evaluation,
    adapter_hash,
)
from preference_privacy.runtime import sha256, write_json
from preference_privacy.metrics import paired_difference, margin_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-suite", required=True)
    p.add_argument("--forward-suite", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    paths = {"main": Path(a.main_suite), "forward": Path(a.forward_suite)}
    manifests = {k: read_json(v / "experiment.json") for k, v in paths.items()}
    validate_paired_suites(manifests["main"], manifests["forward"])
    out = Path(a.output_dir)
    if out.exists():
        p.error("Output directory must not exist")
    results = {}
    runs = []
    for source, manifest in manifests.items():
        suite = paths[source]
        frozen = read_json(suite / "freeze.json") if a.split == "test" else None
        if frozen and frozen["experiment_sha256"] != sha256(suite / "experiment.json"):
            raise ValueError("Changed test protocol")
        for job in manifest["jobs"]:
            identity = adapter_hash(suite / "models" / job["name"] / "adapter")
            if frozen and frozen["adapter_sha256"][job["name"]] != identity:
                raise ValueError("Changed frozen adapter")
            result = read_json(
                suite / "evaluations" / a.split / job["name"] / "adapter.json"
            )
            validate_evaluation(result, manifest, job, a.split)
            if result["adapter_sha256"] != identity:
                raise ValueError("Evaluation adapter mismatch")
            results[job["name"]] = result
            record = {k: job[k] for k in ["name", "group", "epsilon", "rr_seed"]}
            record.update(margin_summary(result["pairs"]))
            cost = suite / "models" / job["name"] / "train_metrics.json"
            if cost.exists():
                stats = read_json(cost)
                record.update(
                    {
                        k: stats.get(k)
                        for k in [
                            "train_runtime",
                            "pipeline_seconds_including_load",
                            "peak_allocated_bytes",
                        ]
                    }
                )
            runs.append(record)
    comparisons = []
    config = manifests["main"]["config"]
    for job in manifests["forward"]["jobs"]:
        eps, rr = job["epsilon"], job["rr_seed"]
        for lam in [0, 0.5, 1]:
            name = f"eps_{eps:g}_rr_{rr}_lambda_{lam:g}"
            comparisons.append(
                dict(
                    epsilon=eps,
                    rr_seed=rr,
                    reference_lambda=lam,
                    **paired_difference(
                        results[job["name"]]["pairs"],
                        results[name]["pairs"],
                        config["evaluation"]["bootstrap_samples"],
                        config["evaluation"]["bootstrap_seed"],
                    ),
                )
            )
    aggregate = []
    for eps in config["epsilons"]:
        for lam in [0, 0.5, 1]:
            values = [
                r["difference_percentage_points"]
                for r in comparisons
                if r["epsilon"] == eps and r["reference_lambda"] == lam
            ]
            aggregate.append(
                dict(
                    epsilon=eps,
                    reference_lambda=lam,
                    n_rr_seeds=len(values),
                    mean_difference_points=statistics.mean(values),
                    rr_sample_sd_points=statistics.stdev(values),
                )
            )
    out.mkdir(parents=True)
    for name, rows in [("runs", runs), ("paired_rr_summary", aggregate)]:
        with (out / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
            )
            writer.writeheader()
            writer.writerows(rows)
    write_json(out / "paired_comparisons.json", comparisons)
    (out / "README.md").write_text(
        "# Forward correction comparisons\n\nFinal adapters only. Paired differences are forward minus backward-family baselines on identical RR training data. RR SD is conditional on one training seed; example-bootstrap intervals exclude training/RR uncertainty.\n"
    )
    print(out)


if __name__ == "__main__":
    main()
