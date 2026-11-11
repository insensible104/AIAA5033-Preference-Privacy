"""Label-blind selectors. This module has no clean-data or model-file access."""

import hashlib
import math
import random


def seeded_rng(domain):
    return random.Random(
        int.from_bytes(hashlib.sha256(domain.encode()).digest(), "big")
    )


def canonicalize(rows):
    """Trusted simulation preparation; public order/IDs never depend on label."""
    import json

    public, labels = [], {}
    for row in rows:
        a, b = sorted([row["response_a"], row["response_b"]])
        if a == b or row["label"] not in (0, 1):
            raise ValueError("Distinct responses and binary labels required")
        identifier = hashlib.sha256(
            json.dumps([row["prompt"], a, b], ensure_ascii=False).encode()
        ).hexdigest()
        if identifier in labels:
            raise ValueError("Duplicate canonical pair")
        chosen = row["response_b"] if row["label"] else row["response_a"]
        labels[identifier] = int(chosen == b)
        # label=0 is a PUBLIC scoring convention, not ground truth.
        public.append(
            dict(
                id=identifier, prompt=row["prompt"], response_a=a, response_b=b, label=0
            )
        )
    public.sort(key=lambda r: r["id"])
    return public, labels


def validate_bundle(bundle):
    if (
        set(bundle) != {"version", "alpha", "contexts", "releases"}
        or bundle["version"] != 1
    ):
        raise ValueError("Unexpected selector input fields")
    if not 0 < bundle["alpha"] < 1:
        raise ValueError("alpha must be in (0,1)")
    contexts = bundle["contexts"]
    if not contexts or not bundle["releases"]:
        raise ValueError("Empty selection problem")
    for context in contexts.values():
        if set(context) != {"order", "baseline", "fixed", "predictions"}:
            raise ValueError("Unexpected context fields")
        order, pred = context["order"], context["predictions"]
        if len(order) < 2 or len(set(order)) != len(order) or set(order) != set(pred):
            raise ValueError("Invalid candidate order")
        if context["baseline"] not in order or not set(
            context["fixed"].values()
        ) <= set(order):
            raise ValueError("Unknown baseline")
        ids = set(pred[order[0]])
        if not ids:
            raise ValueError("Empty predictions")
        for values in pred.values():
            if set(values) != ids or any(v not in (0, 0.5, 1) for v in values.values()):
                raise ValueError("Mismatched IDs or nonfinite/non-discrete predictions")
    seen = set()
    for release in bundle["releases"]:
        if set(release) != {"context", "repeat", "n", "epsilon", "labels"}:
            raise ValueError(
                "Unexpected release fields (clean labels/masks/seeds forbidden)"
            )
        key = release["context"], release["repeat"], release["n"]
        if key in seen or release["context"] not in contexts:
            raise ValueError("Duplicate or unknown release")
        seen.add(key)
        eps = release["epsilon"]
        if not math.isfinite(eps) or eps <= 0 or math.tanh(eps / 2) == 0:
            raise ValueError("Invalid validation epsilon")
        ids = set(
            contexts[release["context"]]["predictions"][
                contexts[release["context"]]["baseline"]
            ]
        )
        if (
            not release["labels"]
            or len(release["labels"]) != release["n"]
            or not set(release["labels"]) <= ids
        ):
            raise ValueError("Invalid release IDs/count")
        if any(v not in (0, 1) for v in release["labels"].values()):
            raise ValueError("Expected binary RR labels")


def accuracy(predictions, labels):
    if not labels or not set(labels) <= set(predictions):
        raise ValueError("Labels/predictions mismatch")
    return sum(1 - abs(predictions[k] - y) for k, y in labels.items()) / len(labels)


def select_models(bundle):
    """Only predictions and released RR labels enter; deterministic ordered ties."""
    validate_bundle(bundle)
    decisions = []
    for release in bundle["releases"]:
        context = bundle["contexts"][release["context"]]
        order = context["order"]
        scores = {
            k: accuracy(context["predictions"][k], release["labels"]) for k in order
        }
        best = max(order, key=lambda k: scores[k])
        baseline = context["baseline"]
        attenuation = math.tanh(release["epsilon"] / 2)
        rho = (1 - attenuation) / 2
        radius = (
            math.sqrt(
                2 * math.log(2 * (len(order) - 1) / bundle["alpha"]) / release["n"]
            )
            / attenuation
        )
        improvement = (scores[best] - scores[baseline]) / attenuation
        chosen = best if improvement > radius else baseline
        decisions.append(
            {
                "context": release["context"],
                "repeat": release["repeat"],
                "n": release["n"],
                "epsilon_val": release["epsilon"],
                "k": len(order),
                "selected": {
                    "noisy_argmax": best,
                    "conservative": chosen,
                    **context["fixed"],
                },
                "fallback": improvement <= radius,
                "radius": radius,
                "best_paired_improvement": improvement,
                "noisy_accuracy": scores,
                "debiased_accuracy": {
                    k: (v - rho) / attenuation for k, v in scores.items()
                },
            }
        )
    return decisions
