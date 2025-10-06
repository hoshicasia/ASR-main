from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    """
    Dataset for loading audio files from a custom directory structure of the following format:
        data_dir/
        ├── audio/
        │   ├── UtteranceID1.wav  # may be .flac, .mp3, .m4a
        │   ├── UtteranceID2.wav
        │   └── ...
        └── transcriptions/  # optional, ground truth
            ├── UtteranceID1.txt
            ├── UtteranceID2.txt
            └── ...

    Args:
        data_dir (str | Path): path to the root directory containing
            'audio' and optionally 'transcriptions' subdirectories.
    """

    def __init__(self, data_dir, *args, **kwargs):
        """
        Initialize CustomDirAudioDataset.

        Args:
            data_dir (str | Path): path to the root directory.
            *args: additional arguments for BaseDataset.
            **kwargs: additional keyword arguments for BaseDataset.
        """
        data_dir = Path(data_dir)
        audio_dir = data_dir / "audio"
        transcription_dir = data_dir / "transcriptions"

        index = []
        audio_extensions = {".mp3", ".wav", ".flac", ".m4a"}

        for audio_path in sorted(audio_dir.iterdir()):
            if audio_path.suffix.lower() in audio_extensions:
                entry = {"path": str(audio_path.absolute().resolve())}

                if transcription_dir.exists():
                    transcription_path = transcription_dir / (audio_path.stem + ".txt")
                    if transcription_path.exists():
                        with transcription_path.open("r", encoding="utf-8") as f:
                            entry["text"] = f.read().strip().lower()
                    else:
                        entry["text"] = ""
                else:
                    entry["text"] = ""

                audio_info = torchaudio.info(str(audio_path))
                entry["audio_len"] = audio_info.num_frames / audio_info.sample_rate
                index.append(entry)

        super().__init__(index, *args, **kwargs)
