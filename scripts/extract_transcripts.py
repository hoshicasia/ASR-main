import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=Path, default="data/datasets/librispeech")
parser.add_argument("--output", type=Path, default="text_corpus.txt")
args = parser.parse_args()

texts = []
for f in sorted(args.dataset_path.rglob("*.trans.txt")):
    for line in f.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) < 2:
            continue
        t = parts[1]
        t = re.sub(r"[^a-z ]", "", t.lower())
        t = " ".join(t.split())
        if t:
            texts.append(t)

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("\n".join(texts), encoding="utf-8")

print(f"Extracted {len(texts)} transcripts")
total_words = sum(len(t.split()) for t in texts)
avg_words = total_words / len(texts) if texts else 0
