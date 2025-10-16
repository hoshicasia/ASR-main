import torch_audiomentations
from torch import Tensor, nn


class Noise(nn.Module):
    def __init__(self, min_amplitude=0.001, max_amplitude=0.02, p=0.5):
        """Noise addition augmentation."""
        super().__init__()
        self._aug = torch_audiomentations.AddGaussianNoise(
            min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p
        )

    def forward(self, data: Tensor):
        """
        Args:
            data: Tensor for augmentation. Shape [time] or [batch, time].
        Returns:
            Augmented tensor
        """
        x = data.unsqueeze(1) if data.dim() == 2 else data.unsqueeze(0).unsqueeze(1)
        x_aug = self._aug(x)
        return x_aug.squeeze(1)
