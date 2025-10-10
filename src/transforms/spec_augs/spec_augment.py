import torch
import torchaudio
from torch import nn


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=15, time_mask_param=35, p=1.0):
        super().__init__()
        self.transforms = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param),
        )
        self.p = p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            return self.transforms(spec)
        return spec
