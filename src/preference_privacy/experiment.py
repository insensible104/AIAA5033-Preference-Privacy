"""Frozen main-experiment matrix and output identities (no accelerator imports)."""

import json
from pathlib import Path
from .runtime import sha256


def main19_jobs(config):
    if (
        config["epsilons"] != [1.0, 2.0]
        or config["rr_seeds"] != [42, 43, 44]
        or config["corrections"] != [0.0, 0.5, 1.0]
    ):
        raise ValueError("main19 requires epsilon=1/2, RR=42/43/44 and lambda=0/.5/1")
    jobs = [
        {
            "name": "clean_dpo",
            "group": "dpo",
            "epsilon": None,
            "rr_seed": None,
            "correction": 0.0,
        }
    ]
    for eps in config["epsilons"]:
        for rr in config["rr_seeds"]:
            for lam in config["corrections"]:
                jobs.append(
                    {
                        "name": f"eps_{eps:g}_rr_{rr}_lambda_{lam:g}",
                        "group": {0.0: "rr_dpo", 0.5: "partial", 1.0: "rr_rdpo"}[lam],
                        "epsilon": eps,
                        "rr_seed": rr,
                        "correction": lam,
                    }
                )
    return jobs


def adapter_hash(path):
    path = Path(path)
    files = sorted(path.glob("adapter_model*.safetensors")) + sorted(
        path.glob("adapter_model*.bin")
    )
    if not files or not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Incomplete adapter: {path}")
    return {f.name: sha256(f) for f in files + [path / "adapter_config.json"]}


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_evaluation(result, manifest, job, split):
    """Reject stale, mismatched or exploratory evaluations in the primary table."""
    from .encoding import FORMAT_VERSION

    config = manifest["config"]
    expected = {
        "eval_sha256": config["split_sha256"][split],
        "evaluation_version": FORMAT_VERSION,
        "dtype": "float32",
        "protocol_override": False,
        "base_model": manifest["model"],
        "revision": manifest["revision"],
    }
    expected.update(
        {k: config["training"][k] for k in ["input_format", "max_length", "beta"]}
    )
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Evaluation protocol mismatch: {key}")
    trained = result["training_config"]
    expected_training = {
        "group": job["group"],
        "epsilon": job["epsilon"],
        "rr_seed": job["rr_seed"],
        "correction": job["correction"],
        "seed": config["training_seed"],
        "data_sha256": job["data_sha256"],
    }
    expected_training.update(config["training"])
    for key, value in expected_training.items():
        if trained.get(key) != value:
            raise ValueError(f"Training configuration mismatch: {key}")
