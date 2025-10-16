import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class OneCycleLR(_LRScheduler):

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        epochs=None,
        steps_per_epoch=None,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
    ):
        total_steps = int(epochs) * int(steps_per_epoch)

        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase

        self._inner = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
        )

        super(OneCycleLR, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            return self._inner.get_last_lr()
