"""Explicit partial backward correction; epsilon and correction strength are distinct."""

import math
import torch
import torch.nn.functional as F


def partial_loss(margin, rho, correction):
    if not math.isfinite(rho) or not 0 <= rho < 0.5:
        raise ValueError("rho must be in [0, 0.5)")
    if not math.isfinite(correction) or not 0 <= correction <= 1:
        raise ValueError("correction must be in [0, 1]")
    ordinary = F.softplus(-margin)
    corrected = ((1 - rho) * ordinary - rho * F.softplus(margin)) / (1 - 2 * rho)
    return (1 - correction) * ordinary + correction * corrected


def completion_logps(model, batch):
    logits = (
        model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        .logits[:, :-1, :]
        .float()
    )
    targets = batch["input_ids"][:, 1:]
    token_logps = logits.gather(-1, targets.unsqueeze(-1)).squeeze(
        -1
    ) - logits.logsumexp(-1)
    mask = (
        batch["completion_mask"][:, 1:].bool() & batch["attention_mask"][:, 1:].bool()
    )
    return token_logps.masked_fill(~mask, 0).sum(-1)


def preference_margin(model, batch, beta, reference_model=None):
    policy = completion_logps(model, batch)
    # Eval mode fixes reference dropout without interfering with policy autograd.
    reference = reference_model if reference_model is not None else model
    was_training = reference.training
    reference.eval()
    try:
        with torch.no_grad():
            if reference_model is None:
                with model.disable_adapter():
                    ref = completion_logps(model, batch)
            else:
                ref = completion_logps(reference, batch)
    finally:
        reference.train(was_training)
    chosen, rejected = (policy - ref).chunk(2)
    return beta * (chosen - rejected)
