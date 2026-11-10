"""Rebuild the original split from included local raw files or pinned upstream data."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/data.json")
    p.add_argument("--data-root", default="data", help="Contains raw/ and processed/")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(args.data_root)
    for split, expected in config["raw_sha256"].items():
        target = root / "raw" / f"{split}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            url = f"https://huggingface.co/datasets/{config['dataset']}/resolve/{config['revision']}/{config['subset']}/{split}.jsonl.gz"
            packed = target.with_suffix(".jsonl.gz.partial")
            temporary = target.with_suffix(".jsonl.partial")
            try:
                urllib.request.urlretrieve(url, packed)
                with gzip.open(packed, "rb") as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Raw decompressed checksum mismatch: {split}")
                temporary.replace(target)
            finally:
                packed.unlink(missing_ok=True)
                temporary.unlink(missing_ok=True)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Existing raw checksum mismatch: {split}")
    # read_split in the original project expected root/data/raw; provide a direct adapter.
    from preference_privacy.data import read_split_from_data_root

    train, _ = read_split_from_data_root(root, "train")
    test, _ = read_split_from_data_root(root, "test")
    test_ids = {r["prompt_id"] for r in test}
    train = [r for r in train if r["prompt_id"] not in test_ids]
    rng = random.Random(config["split_seed"])
    rng.shuffle(train)
    rng.shuffle(test)
    n = config["sizes"]
    splits = {
        "validation": train[: n["validation"]],
        "train": train[n["validation"] : n["validation"] + n["train"]],
        "test": test[: n["test"]],
    }
    if any(len(rows) != n[name] for name, rows in splits.items()):
        raise ValueError("Not enough valid pairs")
    (root / "processed").mkdir(parents=True, exist_ok=True)
    # Preflight all outputs before writing any of them.
    payloads = {
        name: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        for name, rows in splits.items()
    }
    for name, payload in payloads.items():
        dest = root / "processed" / f"{name}.jsonl"
        if dest.exists() and dest.read_text() != payload:
            raise FileExistsError(f"Refusing to overwrite {dest}")
    for name, payload in payloads.items():
        (root / "processed" / f"{name}.jsonl").write_text(payload)
    print({name: len(rows) for name, rows in splits.items()})


if __name__ == "__main__":
    main()
