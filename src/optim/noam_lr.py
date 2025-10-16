# src/schedulers/noam_lr.py

import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        d_model (int): Dimensionality of the model (e.g., 512, 768).
        warmup_steps (int): Number of warmup steps.
        factor (float): Scaling factor (default: 1.0).
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, d_model, warmup_steps, factor=1.0, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)  # step starts from 1
        scale = (
            self.factor
            * (self.d_model**-0.5)
            * min(step**-0.5, step * (self.warmup_steps**-1.5))
        )
        return [base_lr * scale for base_lr in self.base_lrs]
