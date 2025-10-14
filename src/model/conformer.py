"""
Conformer model for CTC-based ASR.
Source: https://arxiv.org/abs/2005.08100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer as TorchAudioConformer


class Conformer(nn.Module):
    """
    Conformer model for CTC-based ASR
    """

    def __init__(
        self,
        n_feats=128,
        n_tokens=28,
        input_dim=144,
        num_heads=4,
        ffn_dim=576,
        num_layers=16,
        depthwise_conv_kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()

        self.n_feats = n_feats
        self.input_dim = input_dim

        # Following the article's suggestion for subsampling
        self.subsampling = nn.Sequential(
            nn.Conv1d(
                in_channels=n_feats,
                out_channels=input_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=input_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.conformer = TorchAudioConformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )

        self.output_projection = nn.Linear(input_dim, n_tokens)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Args:
            spectrogram
            spectrogram_length: lengths of the spectrograms before padding
        Returns:
            dict with:
                - log_probs: log probabilities for CTC
                - log_probs_length: output lengths
        """
        spectrogram_length = spectrogram_length.to(spectrogram.device)
        x = self.subsampling(spectrogram)
        x = x.permute(0, 2, 1).contiguous()
        output_lengths = self.transform_input_lengths(spectrogram_length)

        output, output_lengths = self.conformer(x, output_lengths)

        logits = self.output_projection(output)
        log_probs = F.log_softmax(logits, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": output_lengths}

    def transform_input_lengths(self, input_lengths):
        """
        Input lengths stay the same.
        """
        return (input_lengths + 3) // 4
