import torch
from transformers import Trainer
from .objective import partial_loss, preference_margin


class PreferenceTrainer(Trainer):
    def __init__(self, *args, beta=0.1, rho=0.0, correction=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta, self.rho, self.correction = beta, rho, correction
        # Our batch mean is over pairs, not Trainer's language-model token count.
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        unwrapped = self.accelerator.unwrap_model(model)
        # Single-process training only; no silently incorrect DDP forward bypass.
        if self.args.world_size != 1:
            raise ValueError(
                "Distributed training is not supported; launch independent single-device runs"
            )
        margin = preference_margin(unwrapped, inputs, self.beta)
        loss = partial_loss(margin, self.rho, self.correction).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Nonfinite preference loss; inspect margins and precision"
            )
        return (loss, {"margins": margin}) if return_outputs else loss
