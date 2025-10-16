import argparse
import re
from pathlib import Path

import sentencepiece as spm

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--input", type=Path)
group.add_argument("--dataset", choices=["librispeech"])
parser.add_argument("--data-dir", type=Path, default=Path("data/datasets/librispeech"))
parser.add_argument("--vocab-size", type=int, default=1000)
parser.add_argument("--model-type", choices=["bpe", "unigram"], default="bpe")
parser.add_argument("--output", type=str, default="bpe_model")
parser.add_argument("--char-coverage", type=float, default=1.0)
args = parser.parse_args()

texts = []
if args.input:
    texts = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            texts.append(line)
else:
    for f in args.data_dir.glob("**/*.trans.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                texts.append(parts[1])


corpus_file = Path(f"{args.output}_corpus.txt")
corpus = []
for t in texts:
    t = re.sub(r"[^A-Za-z ]", "", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip().lower()
    if t:
        corpus.append(t)

corpus_file.parent.mkdir(parents=True, exist_ok=True)
corpus_file.write_text("\n".join(corpus), encoding="utf-8")


spm.SentencePieceTrainer.train(
    input=str(corpus_file),
    model_prefix=args.output,
    vocab_size=args.vocab_size,
    model_type=args.model_type,
    character_coverage=args.char_coverage,
    unk_id=0,
    bos_id=-1,
    eos_id=-1,
    pad_id=-1,
    split_by_whitespace=True,
    remove_extra_whitespaces=True,
    add_dummy_prefix=True,
)
