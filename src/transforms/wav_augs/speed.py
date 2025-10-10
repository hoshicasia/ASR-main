import torch
import torchaudio.transforms as tat
from torch import Tensor, nn


class Speed(nn.Module):
    """Random speech rate change"""

    def __init__(self, min_speed=0.85, max_speed=1.15, p=0.7, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.factors = [min_speed, 1.0, max_speed]
        self._aug = tat.SpeedPerturbation(
            orig_freq=self.sample_rate, factors=self.factors
        )

    def forward(self, data: Tensor) -> Tensor:
        if data.dim() == 1:
            data = data.unsqueeze(0).unsqueeze(1)
            result, _ = self._aug(data)
            return result.squeeze(1).squeeze(0)
        elif data.dim() == 2:
            data = data.unsqueeze(1)
            result, _ = self._aug(data)
            return result.squeeze(1)
        else:
            raise ValueError("Expected shape [time] or [batch, time].")
