"""Binary randomized response and unbiased DPO-loss correction (stdlib only)."""

import math


def flip_probability(epsilon):
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    z = math.exp(-epsilon)
    return z / (1 + z)


def randomized_response(label, epsilon, rng):
    if label not in (0, 1):
        raise ValueError("binary labels only")
    return label ^ int(rng.random() < flip_probability(epsilon))


def logistic_loss(margin, label):
    if label not in (0, 1):
        raise ValueError("binary labels only")
    z = -(2 * label - 1) * margin
    return max(z, 0) + math.log1p(math.exp(-abs(z)))


def corrected_loss(margin, observed_label, rho):
    if not 0 <= rho < 0.5:
        raise ValueError("correction requires 0 <= rho < 0.5")
    return (
        (1 - rho) * logistic_loss(margin, observed_label)
        - rho * logistic_loss(margin, 1 - observed_label)
    ) / (1 - 2 * rho)
