import sys
from pathlib import Path

from src.metrics.universal_metrics import UniversalCERMetric, UniversalWERMetric
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def read_text_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read().replace("\n", " ").strip()


def main(pred_dir: str, target_dir: str):
    pred_path = Path(pred_dir)
    target_path = Path(target_dir)

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

    print(f"WER: {wer * 100:.2f}%")
    print(f"CER: {cer * 100:.2f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
