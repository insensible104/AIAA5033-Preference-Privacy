"""Write a paired sweep manifest; execution is explicit with --execute."""

import argparse
from pathlib import Path
import subprocess
import sys
from preference_privacy.rr import generate_rr
from preference_privacy.runtime import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--train-data", required=True)
    p.add_argument("--data-dir", default="data/rr")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epsilons", nargs="+", type=float, default=[2.0])
    p.add_argument("--rr-seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--training-seeds", nargs="+", type=int, default=[20261030])
    p.add_argument("--corrections", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--device", choices=["cpu", "cuda", "npu"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    import math

    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in args.corrections):
        p.error("corrections must be in [0,1]")
    for values in [args.epsilons, args.rr_seeds, args.training_seeds, args.corrections]:
        if len(values) != len(set(values)):
            p.error("Sweep dimensions must not contain duplicates")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        p.error("output-dir must be empty")
    train_script = Path(__file__).with_name("train_dpo.py").resolve()
    jobs = []
    for eps in args.epsilons:
        for rr_seed in args.rr_seeds:
            data = generate_rr(args.train_data, args.data_dir, eps, rr_seed)
            for seed in args.training_seeds:
                for correction in args.corrections:
                    name = f"eps_{eps:g}_rr_{rr_seed}_seed_{seed}_lambda_{correction:g}"
                    command = [
                        sys.executable,
                        str(train_script),
                        "--model",
                        args.model,
                        "--train-data",
                        str(data.resolve()),
                        "--output-dir",
                        str((output / name).resolve()),
                        "--group",
                        "partial",
                        "--epsilon",
                        str(eps),
                        "--rr-seed",
                        str(rr_seed),
                        "--seed",
                        str(seed),
                        "--correction",
                        str(correction),
                        "--epochs",
                        str(args.epochs),
                        "--device",
                        args.device,
                        "--dtype",
                        args.dtype,
                        "--gradient-checkpointing",
                    ]
                    if args.revision:
                        command.extend(["--revision", args.revision])
                    jobs.append({"name": name, "command": command})
    write_json(output / "sweep.json", {"jobs": jobs, "executed": args.execute})
    print(f"{len(jobs)} paired jobs written to {output}/sweep.json")
    if args.execute:
        for job in jobs:
            subprocess.run(job["command"], check=True)


if __name__ == "__main__":
    main()
