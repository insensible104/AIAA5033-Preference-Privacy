import json
import math
import pytest
from preference_privacy.rr import generate_rr
from preference_privacy.mechanisms import flip_probability


def test_rr_reproducibility_and_separation(tmp_path, row):
    source = tmp_path / "clean.jsonl"
    rows = [row | {"id": str(i)} for i in range(1000)]
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    output = generate_rr(source, tmp_path / "rr", 2.0, 42)
    assert output == generate_rr(source, tmp_path / "rr", 2.0, 42)
    noisy = [json.loads(s) for s in output.read_text().splitlines()]
    assert all(
        {k: v for k, v in a.items() if k != "label"}
        == {k: v for k, v in b.items() if k != "label"}
        for a, b in zip(rows, noisy)
    )
    other = [
        json.loads(s)
        for s in generate_rr(source, tmp_path / "rr", 2.0, 43).read_text().splitlines()
    ]
    assert [r["label"] for r in noisy] != [r["label"] for r in other]
    weak = [
        json.loads(s)
        for s in generate_rr(source, tmp_path / "rr", 1.0, 42).read_text().splitlines()
    ]
    # Unlike coupled uniforms, flips at epsilon=2 are not a subset of epsilon=1 flips.
    assert any(a["label"] == 0 and b["label"] == 1 for a, b in zip(noisy, weak))
    output.write_text("changed")
    with pytest.raises(FileExistsError):
        generate_rr(source, tmp_path / "rr", 2.0, 42)


def test_rr_privacy_ratio():
    for eps in [0.5, 1, 2, 4]:
        rho = flip_probability(eps)
        assert (1 - rho) / rho == pytest.approx(math.exp(eps))
