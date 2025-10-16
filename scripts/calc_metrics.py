import sys

from src.metrics.universal_metrics import UniversalCERMetric, UniversalWERMetric
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    gt_path = sys.argv[1]
    pred_path = sys.argv[2]

    ground_truths = read_lines(gt_path)
    predictions = read_lines(pred_path)

    encoder = CTCTextEncoder()
    wer_metric = UniversalWERMetric(encoder)
    cer_metric = UniversalCERMetric(encoder)

    wer = wer_metric(ground_truths, predictions)
    cer = cer_metric(ground_truths, predictions)
    print(f"WER: {wer*100:.2f}%")
    print(f"CER: {cer*100:.2f}%")
