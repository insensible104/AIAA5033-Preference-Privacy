"""Preference summaries and example-level paired uncertainty, conditional on a run."""

import numpy as np


def margin_summary(pairs):
    margins = np.array([p["margin"] for p in pairs], dtype=float)
    if not len(margins) or not np.isfinite(margins).all():
        raise ValueError("Empty/nonfinite margins")
    if len({p["id"] for p in pairs}) != len(pairs):
        raise ValueError("Duplicate sample IDs")
    for pair in pairs:
        expected = float(pair["margin"] > 0) + 0.5 * float(pair["margin"] == 0)
        if "correct" in pair and pair["correct"] != expected:
            raise ValueError("Correctness inconsistent with saved margin")
    return {
        "n_pairs": len(pairs),
        "preference_accuracy_ties_half": float(
            ((margins > 0) + 0.5 * (margins == 0)).mean()
        ),
        "mean_margin": float(margins.mean()),
        "median_margin": float(np.median(margins)),
        "margin_p05": float(np.quantile(margins, 0.05)),
        "margin_p95": float(np.quantile(margins, 0.95)),
        "mean_absolute_margin": float(np.abs(margins).mean()),
        "tie_fraction": float((margins == 0).mean()),
    }


def paired_difference(first, second, repetitions=2000, seed=20261110):
    def index(pairs):
        result = {p["id"]: float(p["correct"]) for p in pairs}
        if (
            len(result) != len(pairs)
            or not result
            or any(v not in (0.0, 0.5, 1.0) for v in result.values())
        ):
            raise ValueError("Invalid pair IDs or correctness")
        return result

    a, b = index(first), index(second)
    if a.keys() != b.keys():
        raise ValueError("Paired comparison requires identical example IDs")
    delta = np.array([a[k] - b[k] for k in sorted(a)])
    if repetitions < 1:
        raise ValueError("Positive bootstrap count required")
    rng = np.random.default_rng(seed)
    means = np.array(
        [
            rng.choice(delta, size=len(delta), replace=True).mean()
            for _ in range(repetitions)
        ]
    )
    return {
        "difference_percentage_points": float(delta.mean() * 100),
        "conditional_example_bootstrap_95_pct": [
            float(x * 100) for x in np.quantile(means, [0.025, 0.975])
        ],
        "n_pairs": len(delta),
        "repetitions": repetitions,
        "seed": seed,
        "scope": "Example resampling for this fixed trained model pair; excludes RR and training randomness.",
    }
