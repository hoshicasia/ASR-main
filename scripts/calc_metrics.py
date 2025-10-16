import argparse
import os
import sys
from pathlib import Path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E402

from src.metrics.universal_metrics import (  # noqa: E402
    UniversalCERMetric,
    UniversalWERMetric,
)
from src.text_encoder.ctc_text_encoder import CTCTextEncoder  # noqa: E402


def read_text_file(path):
    with open(path) as f:
        return f.read().replace("\n", " ").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    args = parser.parse_args()

    pred_path = Path(args.pred_dir).resolve()
    target_path = Path(args.target_dir).resolve()

    predictions = []
    ground_truths = []

    for pred_file in pred_path.glob("*.txt"):
        target_file = target_path / pred_file.name
        pred_text = read_text_file(pred_file)
        target_text = read_text_file(target_file)
        predictions.append(pred_text)
        ground_truths.append(target_text)

    encoder = CTCTextEncoder()
    wer_metric = UniversalWERMetric(encoder)
    cer_metric = UniversalCERMetric(encoder)

    wer = wer_metric(ground_truths, predictions)
    cer = cer_metric(ground_truths, predictions)

    print(f"WER: {wer}")
    print(f"CER: {cer}")


if __name__ == "__main__":
    main()
