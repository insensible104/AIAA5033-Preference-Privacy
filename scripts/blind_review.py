"""Prepare randomized left/right human review; summarize completed votes separately."""

import argparse
import csv
from pathlib import Path
import random
from preference_privacy.experiment import read_json
from preference_privacy.runtime import write_json


def indexed(responses):
    result = {r["id"]: r for r in responses}
    if not result or len(result) != len(responses):
        raise ValueError("Duplicate or empty response IDs")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--reference", required=True)
    prepare.add_argument("--candidates", nargs="+", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--seed", type=int, default=20261110)
    score = sub.add_parser("score")
    score.add_argument("--votes", required=True)
    score.add_argument("--key", required=True)
    score.add_argument("--output-json", required=True)
    args = p.parse_args()
    if args.action == "prepare":
        output = Path(args.output_dir)
        if output.exists() and any(output.iterdir()):
            p.error("Output must be empty")
        reference = read_json(args.reference)
        ref = indexed(reference["responses"])
        rng = random.Random(args.seed)
        tasks = []
        names = set()
        for path in args.candidates:
            candidate = read_json(path)
            rows = indexed(candidate["responses"])
            if candidate["name"] == reference["name"] or candidate["name"] in names:
                p.error("Duplicate candidate or self-comparison")
            names.add(candidate["name"])
            for key in ["experiment_sha256", "test_sha256", "settings", "dtype"]:
                if candidate[key] != reference[key]:
                    p.error("Generation protocol mismatch")
            if ref.keys() != rows.keys():
                p.error("Prompt IDs differ")
            for id in sorted(ref):
                if ref[id]["prompt"] != rows[id]["prompt"]:
                    p.error("Prompt content mismatch")
                candidate_left = bool(rng.getrandbits(1))
                tasks.append(
                    {
                        "id": id,
                        "prompt": ref[id]["prompt"],
                        "left": rows[id]["response"]
                        if candidate_left
                        else ref[id]["response"],
                        "right": ref[id]["response"]
                        if candidate_left
                        else rows[id]["response"],
                        "candidate": candidate["name"],
                        "candidate_side": "left" if candidate_left else "right",
                    }
                )
        rng.shuffle(tasks)
        key = {}
        public = []
        for i, task in enumerate(tasks):
            tid = f"T{i + 1:04d}"
            key[tid] = {k: task[k] for k in ["id", "candidate", "candidate_side"]}
            public.append(
                {
                    "task_id": tid,
                    **{k: task[k] for k in ["prompt", "left", "right"]},
                    "vote": "",
                    "notes": "",
                }
            )
        output.mkdir(parents=True)
        with (output / "review.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["task_id", "prompt", "left", "right", "vote", "notes"],
            )
            writer.writeheader()
            writer.writerows(public)
        write_json(
            output / "private_key.json",
            {"reference": reference["name"], "seed": args.seed, "tasks": key},
        )
        print(
            f"{len(tasks)} tasks. Give reviewers review.csv only, not private_key.json."
        )
    else:
        if Path(args.output_json).exists():
            p.error("Output already exists")
        key = read_json(args.key)
        with Path(args.votes).open() as handle:
            votes = list(csv.DictReader(handle))
        if len({v["task_id"] for v in votes}) != len(votes) or {
            v["task_id"] for v in votes
        } != set(key["tasks"]):
            p.error("Missing/duplicate/unknown tasks")
        totals = {}
        for vote in votes:
            choice = vote["vote"].strip().lower()
            if choice not in ["left", "right", "tie", "skip"]:
                p.error("Every task requires left/right/tie/skip")
            task = key["tasks"][vote["task_id"]]
            count = totals.setdefault(
                task["candidate"], dict(win=0, tie=0, loss=0, skip=0)
            )
            outcome = (
                choice
                if choice in ["tie", "skip"]
                else ("win" if choice == task["candidate_side"] else "loss")
            )
            count[outcome] += 1
        for count in totals.values():
            n = count["win"] + count["tie"] + count["loss"]
            count["n_scored"] = n
            count["win_rate_ties_half"] = (
                (count["win"] + 0.5 * count["tie"]) / n if n else None
            )
        write_json(
            args.output_json,
            {
                "reference": key["reference"],
                "results": totals,
                "note": "Human judgments; skipped tasks excluded from denominator. Report reviewer procedure and disagreements.",
            },
        )


if __name__ == "__main__":
    main()
