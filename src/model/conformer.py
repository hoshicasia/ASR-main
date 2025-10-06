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
        input_dim=128,
        num_heads=8,
        ffn_dim=512,
        num_layers=12,
        depthwise_conv_kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()

        self.n_feats = n_feats
        self.input_dim = input_dim

        if n_feats != input_dim:
            self.input_projection = nn.Linear(n_feats, input_dim)
        else:
            self.input_projection = None

        self.conformer = TorchAudioConformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )

        self.output_projection = nn.Linear(input_dim, n_tokens)

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
        spectrogram = spectrogram.transpose(1, 2)
        if self.input_projection is not None:
            x = self.input_projection(spectrogram)
        else:
            x = spectrogram

        spectrogram_length = spectrogram_length.to(x.device)
        output, output_lengths = self.conformer(x, spectrogram_length)
        logits = self.output_projection(output)
        log_probs = F.log_softmax(logits, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": output_lengths}

    def transform_input_lengths(self, input_lengths):
        """
        Input lengths stay the same.
        """
        return input_lengths
