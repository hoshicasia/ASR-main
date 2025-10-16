from typing import List

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer, calc_wer


class UniversalWERMetric(BaseMetric):
    """
    WER metric that uses pre-decoded predictions from the batch.
    The decoding strategy is determined by the trainer configuration.
    """

    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, text: List[str], predictions: List[str] = None, **kwargs):
        """
        Calculate WER using pre-decoded predictions.

        Args:
            text: Ground truth text
            predictions: Pre-decoded predictions from trainer
            **kwargs: Other batch elements (ignored)

        Returns:
            Average WER across the batch
        """

        wers = []
        for pred_text, target_text in zip(predictions, text):
            target_text = self.text_encoder.normalize_text(target_text)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers) if len(wers) > 0 else 1.0


class UniversalCERMetric(BaseMetric):
    """
    CER metric that uses pre-decoded predictions from the batch.
    The decoding strategy is determined by the trainer configuration.
    """

    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, text: List[str], predictions: List[str] = None, **kwargs):
        """
        Calculate CER using pre-decoded predictions.

        Args:
            text: Ground truth text
            predictions: Pre-decoded predictions from trainer
            **kwargs: Other batch elements (ignored)

        Returns:
            Average CER across the batch
        """

        cers = []
        for pred_text, target_text in zip(predictions, text):
            target_text = self.text_encoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers) if len(cers) > 0 else 1.0
