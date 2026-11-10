import csv
from preference_privacy.runtime import write_json
from test_main19 import cli
import json


def test_blinding_and_score(tmp_path):
    shared = {
        "experiment_sha256": "x",
        "test_sha256": "y",
        "settings": {},
        "dtype": "float32",
    }
    responses = [
        {"id": str(i), "prompt": f"question{i}", "response": "reference"}
        for i in range(4)
    ]
    reference = tmp_path / "ref.json"
    candidate = tmp_path / "candidate.json"
    write_json(reference, shared | {"name": "clean", "responses": responses})
    write_json(
        candidate,
        shared
        | {
            "name": "partial",
            "responses": [r | {"response": "candidate"} for r in responses],
        },
    )
    out = tmp_path / "review"
    cli(
        "blind_review.py",
        "prepare",
        "--reference",
        reference,
        "--candidates",
        candidate,
        "--output-dir",
        out,
    )
    key = json.loads((out / "private_key.json").read_text())
    with (out / "review.csv").open() as f:
        votes = list(csv.DictReader(f))
    assert len(votes) == 4 and "candidate" not in votes[0]
    for v in votes:
        v["vote"] = key["tasks"][v["task_id"]]["candidate_side"]
    with (out / "review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=votes[0].keys())
        writer.writeheader()
        writer.writerows(votes)
    result = tmp_path / "scores.json"
    cli(
        "blind_review.py",
        "score",
        "--votes",
        out / "review.csv",
        "--key",
        out / "private_key.json",
        "--output-json",
        result,
    )
    score = json.loads(result.read_text())["results"]["partial"]
    assert score["win"] == 4 and score["win_rate_ties_half"] == 1
    votes[0]["vote"] = ""
    with (out / "review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=votes[0].keys())
        writer.writeheader()
        writer.writerows(votes)
    cli(
        "blind_review.py",
        "score",
        "--votes",
        out / "review.csv",
        "--key",
        out / "private_key.json",
        "--output-json",
        tmp_path / "bad.json",
        ok=False,
    )
