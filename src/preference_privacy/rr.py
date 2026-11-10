"""Independent RR streams across (epsilon, seed); shared files within paired methods."""

import hashlib
import json
import math
from pathlib import Path
import random
from .mechanisms import flip_probability
from .runtime import load_rows, sha256, write_json


def generate_rr(source, output_dir, epsilon, rr_seed):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    rows = load_rows(source)
    rho = flip_probability(epsilon)
    # Domain separation prevents accidental reuse of uniforms across budgets.
    stream = f"rr-v2:{epsilon.hex()}:{rr_seed}"
    rng = random.Random(int.from_bytes(hashlib.sha256(stream.encode()).digest(), "big"))
    noisy = []
    flips = 0
    for row in rows:
        if row["label"] not in (0, 1):
            raise ValueError("Labels must be binary")
        flip = int(rng.random() < rho)
        flips += flip
        noisy.append(row | {"label": row["label"] ^ flip})
    output = Path(output_dir) / f"train_rr_eps_{epsilon:g}_seed_{rr_seed}.jsonl"
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in noisy)
    metadata = {
        "mechanism": "binary_randomized_response",
        "version": "rr-v2",
        "epsilon": epsilon,
        "rho": rho,
        "rr_seed": rr_seed,
        "n_pairs": len(rows),
        "source_sha256": sha256(source),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "flips_simulation_only": flips,
        "privacy_note": "Public-data simulation. Seeds and flip counts are not private-release artifacts.",
    }
    sidecar = output.with_suffix(".meta.json")
    if output.exists() or sidecar.exists():
        if (
            output.exists()
            and sidecar.exists()
            and output.read_text() == payload
            and json.loads(sidecar.read_text()) == metadata
        ):
            return output
        raise FileExistsError(
            f"Refusing to replace a different or incomplete RR dataset: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload)
    write_json(sidecar, metadata)
    return output
