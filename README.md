# Automatic Speech Recognition with Confomer-CTC

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Inference](#inference)
- [Training](#training)
- [Demo](#demo)
- [Notes](#notes)
- [Useful Links](#useful)
- [Credits](#credits)
- [License](#license)


## About

This repository contains an implementation of **Automatic Speech Recognition (ASR)** using slightly modified [**Conformer**](https://arxiv.org/abs/2005.08100) architecture trained on the [**LibriSpeech**](https://www.openslr.org/12) dataset.
The model was evaluated on the standard LibriSpeech test sets. Below are the final metrics:

| Dataset       | Test Method         | WER   | CER   |
|---------------|---------------------|-------|-------|
| `test-clean`  | beam-search-LM      | 12.4  | 5.4   |
| `test-other`  | beam-search-LM      | 27.25 | 14.2  |



## Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/hoshicasia/ASR-main.git
cd ASR-main
pip install -r requirements.txt
```

Download pretrained checkpoints:

```bash
chmod +x download_checkpoints.sh
./download_checkpoints.sh
```

## Inference
To run inference on the LibriSpeech test dataset (default: test-clean):

```bash
python inference.py -cn=inference_bpe writer=none text_encoder.model_path=data/bpe_model.model inferencer.from_pretrained=data/best_model/
```
Among the options for evaluation: argmax, beam_search, beam_search_lm. The latter one is recommended.

You can also use your own dataset in the following format:

```arduino
NameOfTheDirectoryWithUtterances/
├── audio/
│   ├── UtteranceID1.wav      # may also be .flac or .mp3
│   ├── UtteranceID2.wav
│   └── ...
└── transcriptions/           # ground truth, optional
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    └── ...
```

Run inference on your custom dataset:

```bash
python inference.py -cn=inference_bpe datasets=custom_dir \
    inferencer.save_dir=inference_results \
    datasets.val.data_dir=test_data/
```

If you already have ground truth and prediction files, you can evaluate them directly with the following metrics script:

```bash
python scripts/calc_metrics.py --pred_dir=your/predictions/dir --target_dir=your/ground_truth/dir
```
## Training

If you wish to replicate training process, follow this steps:

0. (Optional) By default this project uses BPE model with vocabulary size 300 trained on librispeech text corpus.
   If you want to train your own BPE model, run the following:


```bash
python train_bpe.py --input your_text_corpus.txt --vocab-size VOCAB_SIZE --output bpe_model
```

1. Baseline training

```bash
python train.py -cn=baseline_bpe text_encoder.model_path=data/bpe_model.model writer=null
```

2. Fine-tuning

```bash
python train.py -cn=finetune_bpe_p1 text_encoder.model_path=data/bpe_model.model trainer.resume_from=path/to/last/model/checkpoint
```

## Demo

 The Installation and Inference steps described above are demonstrated in the ASR-Project-Demo.ipynb notebook included in this repository.

## Notes

This project uses CometML for logging, to use it set up your API key or choose writer=none option.

## Useful links

1. [Conformer](https://arxiv.org/abs/2005.08100) - article with the architecture that was used in the project
2. [LibriSpeech dataset](https://www.openslr.org/12) - dataset that was used for training and evaluation
3. [Hydra Documentation](https://hydra.cc/docs/intro/) — configuration framework used in this project
4. [SentencePiece (BPE)](https://github.com/google/sentencepiece) — subword tokenizer used for BPE model
5. [PyTorch](https://pytorch.org/) — DL framework used in the project
6. [CometML](https://www.comet.com/site/) - platform for logging used in the project.
7. [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) + [KenLM](https://github.com/kpu/kenlm) - combo used for LM-fusion evaluation.


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
