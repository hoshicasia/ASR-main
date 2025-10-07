import json
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from datasets import DownloadMode, load_dataset
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        valid_parts = [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
            "train_all",
        ]
        assert part in valid_parts

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
        else:
            data_dir = Path(data_dir)

        data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir

        if part == "train_all":
            index = []
            train_parts = ["train-clean-100", "train-clean-360", "train-other-500"]
            for train_part in train_parts:
                index.extend(self._get_or_load_index(train_part))
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            print(f"Loading index from cache: {index_path}")
            with index_path.open() as f:
                index = json.load(f)
        else:
            print(f"Index not found, creating new index for: {part}")
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        print(f"Loading dataset part: {part}")

        config_mapping = {
            "train-clean-100": ("clean", "train.100"),
            "train-clean-360": ("clean", "train.360"),
            "train-other-500": ("other", "train.500"),
            "dev-clean": ("clean", "validation"),
            "dev-other": ("other", "validation"),
            "test-clean": ("clean", "test"),
            "test-other": ("other", "test"),
        }

        if part not in config_mapping:
            raise ValueError(
                f"Unknown part: {part}. Valid parts: {list(config_mapping.keys())}"
            )

        config, split = config_mapping[part]

        print(
            f"Loading HuggingFace dataset: librispeech_asr, config='{config}', split='{split}'"
        )
        dataset = load_dataset(
            "librispeech_asr",
            config,
            split=split,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
            streaming=True,
        )
        index = []

        for i, item in tqdm(enumerate(dataset), desc=f"Processing {part}"):
            audio = item["audio"]
            text = item["text"]

            waveform = audio["array"]
            sample_rate = audio["sampling_rate"]
            length = len(waveform) / sample_rate

            audio_path = self._data_dir / f"{part}_{i}.flac"

            if not audio_path.exists():
                if waveform.ndim == 1:
                    waveform_tensor = torch.tensor(waveform).unsqueeze(0)
                else:
                    waveform_tensor = torch.tensor(waveform).T

                torchaudio.save(str(audio_path), waveform_tensor, sample_rate)

            index.append(
                {
                    "path": str(audio_path.absolute().resolve()),
                    "text": text.lower(),
                    "audio_len": length,
                }
            )

        return index
