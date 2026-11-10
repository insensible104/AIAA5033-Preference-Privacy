"""HH-RLHF parsing, deterministic public-text ordering, and output utilities."""

import gzip
import hashlib
import json


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_rows(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stats(values):
    s = sorted(values)
    return {
        "min": s[0],
        "median": s[len(s) // 2],
        "p95": s[int(0.95 * (len(s) - 1))],
        "max": s[-1],
        "mean": sum(s) / len(s),
    }


def read_split(root, split):
    return read_split_from_data_root(root / "data", split)


def read_split_from_data_root(root, split):
    gz = root / "raw" / f"{split}.jsonl.gz"
    plain = root / "raw" / f"{split}.jsonl"
    path = gz if gz.exists() else plain
    info = {
        "raw_records": 0,
        "invalid_records": 0,
        "duplicate_prompt_records": 0,
        "conflicting_pairs": 0,
        "file_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    by_prompt = {}
    seen_labels = {}
    conflicts = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            info["raw_records"] += 1
            item = json.loads(line)
            chosen = item["chosen"]
            rejected = item["rejected"]
            sep = "\n\nAssistant:"
            pc, sc, ac = chosen.rpartition(sep)
            pr, sr, ar = rejected.rpartition(sep)
            if (
                not sc
                or not sr
                or pc != pr
                or not ac.strip()
                or not ar.strip()
                or ac == ar
            ):
                info["invalid_records"] += 1
                continue
            a, b = sorted((ac, ar))
            prompt = pc + sep
            pid = digest(prompt)
            pair_id = digest(json.dumps([prompt, a, b]))
            row = {
                "id": pair_id,
                "prompt_id": pid,
                "prompt": prompt,
                "response_a": a,
                "response_b": b,
                "label": int(ac == b),
            }
            if pair_id in seen_labels and seen_labels[pair_id] != row["label"]:
                conflicts.add(pair_id)
            seen_labels[pair_id] = row["label"]
            if pid in by_prompt:
                info["duplicate_prompt_records"] += 1
            if pid not in by_prompt or pair_id < by_prompt[pid]["id"]:
                by_prompt[pid] = row
    info["conflicting_pairs"] = len(conflicts)
    if conflicts:
        raise ValueError("Conflicting labels require review before proceeding")
    rows = sorted(by_prompt.values(), key=lambda r: r["prompt_id"])
    info["unique_valid_prompts"] = len(rows)
    return rows, info
