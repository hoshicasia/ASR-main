import torch
import torch_audiomentations as ta
from torch import Tensor, nn

# Inspired by https://arxiv.org/pdf/2505.20606


class PitchShift(nn.Module):
    "Random pitch shifting in semitones."

    def __init__(
        self,
        sample_rate: int = 16000,
        min_transpose_semitones=-4.0,
        max_transpose_semitones=6.0,
        p=0.7,
    ):
        super().__init__()
        self._aug = ta.PitchShift(
            sample_rate=sample_rate,
            min_transpose_semitones=min_transpose_semitones,
            max_transpose_semitones=max_transpose_semitones,
            p=p,
        )

    def forward(self, data: Tensor) -> Tensor:
        if data.dim() == 1:
            data = data.unsqueeze(0).unsqueeze(1)
            return self._aug(data).squeeze(1).squeeze(0)
        elif data.dim() == 2:
            data = data.unsqueeze(1)
            return self._aug(data).squeeze(1)
        else:
            raise ValueError("Expected shape [time] or [batch, time].")
