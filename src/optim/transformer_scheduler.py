# src/optim/transformer_scheduler.py
import math

from torch.optim.lr_scheduler import _LRScheduler


class TransformerLRScheduler(_LRScheduler):
    def __init__(
        self, optimizer, d_model=144, warmup_steps=10000, factor=0.05, last_epoch=-1
    ):
        self.d_model = d_model
        self.warmup = warmup_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = (
            self.factor
            * (self.d_model**-0.5)
            * min(step**-0.5, step * (self.warmup**-1.5))
        )
        return [base_lr * scale for base_lr in self.base_lrs]
