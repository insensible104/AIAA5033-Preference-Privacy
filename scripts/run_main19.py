"""Plan, train, validate, freeze and test the fixed 19-run course experiment."""

import argparse
from pathlib import Path
import subprocess
import sys
from preference_privacy.runtime import load_rows, sha256, write_json
from preference_privacy.rr import generate_rr
from preference_privacy.experiment import (
    main19_jobs,
    adapter_hash,
    read_json,
    validate_evaluation,
)

SCRIPTS = Path(__file__).resolve().parent


def run_command(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["plan", "train", "validation", "freeze", "test"])
    p.add_argument("--suite", required=True)
    p.add_argument("--config", default="configs/main19.json")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--device", choices=["cpu", "cuda", "npu"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument(
        "--jobs",
        nargs="+",
        help="Optional subset of named jobs; use disjoint subsets on separate devices",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually train/evaluate/freeze; otherwise print commands",
    )
    args = p.parse_args()
    suite = Path(args.suite).resolve()
    manifest_path = suite / "experiment.json"
    if args.stage == "plan":
        if suite.exists() and any(suite.iterdir()):
            p.error("Suite directory must be empty")
        config = read_json(args.config)
        jobs = main19_jobs(config)
        data_dir = Path(args.data_dir).resolve()
        # Verify all split identities without reading test labels for model selection.
        for split, expected in config["split_sha256"].items():
            if sha256(data_dir / f"{split}.jsonl") != expected:
                p.error(f"{split} hash differs from registered split")
        if len(load_rows(data_dir / "train.jsonl")) != config["expected_train_pairs"]:
            p.error("Unexpected training count")
        model = args.model or config["model"]
        if Path(model).exists():
            model = str(Path(model).resolve())
        # Resolve a remote model to an immutable revision; only metadata is fetched.
        revision = args.revision
        if not Path(model).exists():
            from huggingface_hub import model_info

            revision = model_info(model, revision=revision).sha
        for job in jobs:
            data = data_dir / "train.jsonl"
            if job["epsilon"] is not None:
                data = generate_rr(data, suite / "rr", job["epsilon"], job["rr_seed"])
            job["data"] = str(data)
            job["data_sha256"] = sha256(data)
        write_json(
            manifest_path,
            {
                "config": config,
                "model": model,
                "revision": revision,
                "data_dir": str(data_dir),
                "training_dtype": args.dtype,
                "jobs": jobs,
            },
        )
        print(f"Prepared {len(jobs)} jobs: {manifest_path}")
        return
    manifest = read_json(manifest_path)
    config = manifest["config"]
    if len(manifest["jobs"]) != 19 or [j["name"] for j in manifest["jobs"]] != [
        j["name"] for j in main19_jobs(config)
    ]:
        p.error("Invalid main19 manifest")
    selected = set(args.jobs or [j["name"] for j in manifest["jobs"]])
    if not selected <= {j["name"] for j in manifest["jobs"]}:
        p.error("Unknown job name")
    manifest_sha = sha256(manifest_path)
    if args.stage in ["freeze", "test"]:
        if selected != {j["name"] for j in manifest["jobs"]} and args.stage == "freeze":
            p.error("Freeze requires all 19 jobs")
        current = {
            j["name"]: adapter_hash(suite / "models" / j["name"] / "adapter")
            for j in manifest["jobs"]
        }
        for job in manifest["jobs"]:
            marker = read_json(suite / "completed" / f"{job['name']}.json")
            if (
                marker["experiment_sha256"] != manifest_sha
                or marker["adapter_sha256"] != current[job["name"]]
            ):
                p.error("Missing or changed completed training identity")
        if args.stage == "freeze":
            if (suite / "freeze.json").exists():
                p.error("Freeze already exists")
            validations = {}
            for job in manifest["jobs"]:
                result = read_json(
                    suite / "evaluations" / "validation" / job["name"] / "adapter.json"
                )
                validate_evaluation(result, manifest, job, "validation")
                if (
                    result["eval_sha256"] != config["split_sha256"]["validation"]
                    or result.get("adapter_sha256") != current[job["name"]]
                ):
                    p.error("Validation identity mismatch")
                validations[job["name"]] = sha256(
                    suite / "evaluations" / "validation" / job["name"] / "adapter.json"
                )
            if args.execute:
                write_json(
                    suite / "freeze.json",
                    {
                        "experiment_sha256": manifest_sha,
                        "endpoint": "adapter",
                        "adapter_sha256": current,
                        "validation_sha256": validations,
                    },
                )
            print(
                "Final endpoint fixed to epoch 3; freeze written"
                if args.execute
                else "Freeze verified; add --execute to write"
            )
            return
        frozen = read_json(suite / "freeze.json")
        if (
            frozen["experiment_sha256"] != manifest_sha
            or frozen["adapter_sha256"] != current
        ):
            p.error("Configuration or adapters changed after freeze")
    for job in manifest["jobs"]:
        if job["name"] not in selected:
            continue
        run = suite / "models" / job["name"]
        if args.stage == "train":
            if sha256(job["data"]) != job["data_sha256"]:
                p.error("Training data changed")
            marker = suite / "completed" / f"{job['name']}.json"
            if marker.exists():
                state = read_json(marker)
                if state["experiment_sha256"] != manifest_sha or state[
                    "adapter_sha256"
                ] != adapter_hash(run / "adapter"):
                    p.error("Completed run changed")
                print("Skip completed:", job["name"])
                continue
            command = [
                sys.executable,
                str(SCRIPTS / "train_dpo.py"),
                "--model",
                manifest["model"],
                "--train-data",
                job["data"],
                "--output-dir",
                str(run),
                "--group",
                job["group"],
                "--seed",
                str(config["training_seed"]),
                "--device",
                args.device,
                "--dtype",
                manifest["training_dtype"],
            ]
            if manifest["revision"]:
                command.extend(["--revision", manifest["revision"]])
            for key, value in config["training"].items():
                if isinstance(value, bool):
                    if value:
                        command.append("--" + key.replace("_", "-"))
                else:
                    command.extend(["--" + key.replace("_", "-"), str(value)])
            if job["epsilon"] is not None:
                command.extend(
                    ["--epsilon", str(job["epsilon"]), "--rr-seed", str(job["rr_seed"])]
                )
            if job["group"] == "partial":
                command.extend(["--correction", str(job["correction"])])
            print("Train:", job["name"], flush=True)
            if args.execute:
                run_command(command, suite / "logs" / f"{job['name']}_train.log")
                write_json(
                    marker,
                    {
                        "experiment_sha256": manifest_sha,
                        "adapter_sha256": adapter_hash(run / "adapter"),
                    },
                )
            else:
                print(__import__("shlex").join(command))
        else:
            split = args.stage
            data = Path(manifest["data_dir"]) / f"{split}.jsonl"
            if sha256(data) != config["split_sha256"][split]:
                p.error("Evaluation data changed")
            endpoints = (
                ["adapter"]
                if split == "test"
                else [f"checkpoint-{s}" for s in config["checkpoint_steps"]]
                + ["adapter"]
            )
            for endpoint in endpoints:
                adapter = run / endpoint
                output = (
                    suite / "evaluations" / split / job["name"] / f"{endpoint}.json"
                )
                if output.exists():
                    result = read_json(output)
                    validate_evaluation(result, manifest, job, split)
                    if result["eval_sha256"] != sha256(data) or result.get(
                        "adapter_sha256"
                    ) != adapter_hash(adapter):
                        p.error("Existing evaluation changed or mismatched")
                    print("Skip evaluated:", job["name"], endpoint)
                    continue
                command = [
                    sys.executable,
                    str(SCRIPTS / "evaluate.py"),
                    "--adapter-dir",
                    str(adapter),
                    "--eval-data",
                    str(data),
                    "--output-json",
                    str(output),
                    "--device",
                    args.device,
                    "--batch-size",
                    str(config["evaluation"]["batch_size"]),
                ]
                print("Evaluate:", job["name"], split, endpoint, flush=True)
                if args.execute:
                    run_command(
                        command,
                        suite / "logs" / f"{job['name']}_{split}_{endpoint}.log",
                    )
                else:
                    print(__import__("shlex").join(command))


if __name__ == "__main__":
    main()
