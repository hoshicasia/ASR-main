import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def __init__(self, blank=0, reduction="mean", zero_infinity=True):
        super().__init__(blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ):
        """
        Accepts:
          log_probs: (B, T, V) - log softmax
          log_probs_length: (B,)
          text_encoded: either (B, L) padded with 0 or concatenated 1D targets
          text_encoded_length: (B,)
        Returns {'loss': loss_tensor}
        """
        device = log_probs.device
        logp_t = log_probs.transpose(0, 1).contiguous()

        if text_encoded.dim() == 2:
            targets = []
            for i in range(text_encoded.size(0)):
                L = int(text_encoded_length[i].item())
                targets.append(text_encoded[i, :L])
            targets = torch.cat(targets).to(device)
        else:
            targets = text_encoded.to(device)

        input_lengths = log_probs_length.to(device)
        target_lengths = text_encoded_length.to(device)

        loss = super().forward(logp_t, targets, input_lengths, target_lengths)
        return {"loss": loss}
