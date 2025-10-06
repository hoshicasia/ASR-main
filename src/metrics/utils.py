# Based on seminar materials

# Don't forget to support cases when target_text == ''

from torchmetrics.text import CharErrorRate, WordErrorRate


def calc_cer(target_text, predicted_text) -> float:
    if target_text == "":
        return 0.0 if predicted_text == "" else 1.0
    cer_metric = CharErrorRate()
    return cer_metric([predicted_text], [target_text]).item()


def calc_wer(target_text, predicted_text) -> float:
    if target_text == "":
        return 0.0 if predicted_text == "" else 1.0
    wer_metric = WordErrorRate()
    return wer_metric([predicted_text], [target_text]).item()
