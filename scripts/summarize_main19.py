"""Summarize all 19 final endpoints; keep RR variation and sampling uncertainty separate."""

import argparse
import csv
from pathlib import Path
import statistics
from preference_privacy.experiment import read_json, main19_jobs, validate_evaluation
from preference_privacy.metrics import paired_difference, margin_summary
from preference_privacy.runtime import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    suite = Path(args.suite)
    manifest = read_json(suite / "experiment.json")
    config = manifest["config"]
    jobs = manifest["jobs"]
    if [j["name"] for j in jobs] != [j["name"] for j in main19_jobs(config)]:
        p.error("Invalid matrix")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        p.error("Output must be empty")
    results = {}
    rows = []
    trajectory = []
    signature = None
    for job in jobs:
        path = suite / "evaluations" / args.split / job["name"] / "adapter.json"
        result = read_json(path)
        validate_evaluation(result, manifest, job, args.split)
        current = tuple(
            result[k]
            for k in [
                "eval_sha256",
                "evaluation_version",
                "input_format",
                "max_length",
                "beta",
                "dtype",
                "base_model",
                "revision",
            ]
        )
        if result["eval_sha256"] != config["split_sha256"][args.split] or result.get(
            "protocol_override"
        ):
            p.error("Wrong split or protocol override")
        if signature is None:
            signature = current
        if current != signature:
            p.error("Incompatible evaluation protocols")
        results[job["name"]] = result
        row = {k: job[k] for k in ["name", "group", "epsilon", "rr_seed", "correction"]}
        row.update(margin_summary(result["pairs"]))
        cost_path = suite / "models" / job["name"] / "train_metrics.json"
        if cost_path.exists():
            cost = read_json(cost_path)
            row.update(
                {
                    k: cost.get(k)
                    for k in [
                        "train_runtime",
                        "pipeline_seconds_including_load",
                        "peak_allocated_bytes",
                    ]
                }
            )
        rows.append(row)
        if args.split == "validation":
            for step in config["checkpoint_steps"]:
                point_path = path.with_name(f"checkpoint-{step}.json")
                if not point_path.exists():
                    continue
                point = read_json(point_path)
                validate_evaluation(point, manifest, job, args.split)
                if (
                    tuple(
                        point[k]
                        for k in [
                            "eval_sha256",
                            "evaluation_version",
                            "input_format",
                            "max_length",
                            "beta",
                            "dtype",
                            "base_model",
                            "revision",
                        ]
                    )
                    != signature
                ):
                    p.error("Trajectory protocol mismatch")
                trajectory.append(
                    {
                        "name": job["name"],
                        "step": step,
                        **margin_summary(point["pairs"]),
                    }
                )
    aggregates = []
    comparisons = []
    for eps in config["epsilons"]:
        for lam in config["corrections"]:
            matching = [
                r for r in rows if r["epsilon"] == eps and r["correction"] == lam
            ]
            values = [r["preference_accuracy_ties_half"] * 100 for r in matching]
            aggregates.append(
                {
                    "epsilon": eps,
                    "correction": lam,
                    "n_rr_seeds": len(values),
                    "accuracy_mean_percent": statistics.mean(values),
                    "difference_vs_clean_points": statistics.mean(values)
                    - rows[0]["preference_accuracy_ties_half"] * 100,
                    "accuracy_rr_sample_sd_points": statistics.stdev(values),
                }
            )
        for rr in config["rr_seeds"]:
            for lam, reference in [(0.5, 0.0), (1.0, 0.0), (0.5, 1.0)]:
                base = results[f"eps_{eps:g}_rr_{rr}_lambda_{reference:g}"]["pairs"]
                candidate = results[f"eps_{eps:g}_rr_{rr}_lambda_{lam:g}"]["pairs"]
                comparisons.append(
                    {
                        "epsilon": eps,
                        "rr_seed": rr,
                        "correction": lam,
                        "reference_correction": reference,
                        **paired_difference(
                            candidate,
                            base,
                            config["evaluation"]["bootstrap_samples"],
                            config["evaluation"]["bootstrap_seed"],
                        ),
                    }
                )
    paired_rr = []
    for eps in config["epsilons"]:
        for lam, reference in [(0.5, 0.0), (1.0, 0.0), (0.5, 1.0)]:
            values = [
                c["difference_percentage_points"]
                for c in comparisons
                if c["epsilon"] == eps
                and c["correction"] == lam
                and c["reference_correction"] == reference
            ]
            paired_rr.append(
                {
                    "epsilon": eps,
                    "correction": lam,
                    "reference_correction": reference,
                    "mean_difference_points": statistics.mean(values),
                    "rr_sample_sd_points": statistics.stdev(values),
                    "n_rr_seeds": len(values),
                }
            )
    output.mkdir(parents=True)
    for name, records in [
        ("runs", rows),
        ("rr_summary", aggregates),
        ("paired_rr_summary", paired_rr),
        ("trajectory", trajectory),
    ]:
        if records:
            keys = list(dict.fromkeys(k for r in records for k in r))
            with (output / f"{name}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)
    write_json(output / "paired_comparisons.json", comparisons)
    lines = [
        "# Main19 results",
        f"Split: {args.split}. Primary endpoint: final epoch 3 adapter.",
        "RR SD measures variation across three RR datasets conditional on one training seed; it is not a confidence interval.",
        "Paired bootstrap intervals resample examples separately for each fixed model pair. They exclude training and RR uncertainty.",
        "",
        "| epsilon | lambda | accuracy mean (%) | RR sample SD (points) |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {r['epsilon']} | {r['correction']} | {r['accuracy_mean_percent']:.2f} | {r['accuracy_rr_sample_sd_points']:.2f} |"
        for r in aggregates
    )
    lines.extend(
        [
            "",
            f"Clean DPO accuracy: {rows[0]['preference_accuracy_ties_half'] * 100:.2f}% (one training seed).",
            "Checkpoint results are diagnostic, not post-hoc replacements for the primary endpoint.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
