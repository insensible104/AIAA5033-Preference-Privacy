"""Audit all pairs without loading model weights or evaluating model quality."""

import argparse
from pathlib import Path
from transformers import AutoTokenizer
from preference_privacy.encoding import encode_pair, FORMAT_VERSION
from preference_privacy.runtime import load_rows, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument(
        "--formats", nargs="+", choices=["chat", "raw_hh"], default=["chat", "raw_hh"]
    )
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--output-json", required=True)
    args = p.parse_args()
    if Path(args.output_json).exists():
        p.error("output-json already exists")
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    summary = []
    for path in args.data:
        rows = load_rows(path)
        for fmt in args.formats:
            failures = []
            truncated = 0
            merged = 0
            for row in rows:
                try:
                    result = encode_pair(row, tok, fmt, args.max_length)
                    truncated += int(
                        result["chosen_truncated_tokens"] > 0
                        or result["rejected_truncated_tokens"] > 0
                    )
                    merged += int(result["merged_history_turns"] > 0)
                except ValueError as exc:
                    failures.append({"id": row["id"], "error": str(exc)})
            summary.append(
                {
                    "file": str(path),
                    "sha256": sha256(path),
                    "format": fmt,
                    "n_pairs": len(rows),
                    "truncated_pairs": truncated,
                    "merged_history_pairs": merged,
                    "failures": failures,
                }
            )
    write_json(
        args.output_json,
        {
            "format_version": FORMAT_VERSION,
            "model": args.model,
            "revision": args.revision,
            "max_length": args.max_length,
            "datasets": summary,
        },
    )
    for result in summary:
        print(
            result["file"],
            result["format"],
            result["n_pairs"],
            "failures:",
            len(result["failures"]),
        )
    if any(result["failures"] for result in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
