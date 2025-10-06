import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    batch_size = len(dataset_items)

    audios = [item["audio"] for item in dataset_items]
    audio_lengths = torch.tensor(
        [audio.shape[-1] for audio in audios], dtype=torch.long
    )
    max_audio_length = int(audio_lengths.max().item())
    audio_channels = audios[0].shape[0]
    audio_batch = audios[0].new_zeros((batch_size, audio_channels, max_audio_length))
    for idx, audio in enumerate(audios):
        audio_batch[idx, :, : audio.shape[-1]] = audio

    spectrograms = []
    for item in dataset_items:
        spectrogram = item["spectrogram"]
        if spectrogram.dim() == 3 and spectrogram.shape[0] == 1:
            spectrogram = spectrogram.squeeze(0)
        spectrograms.append(spectrogram)

    spectrogram_lengths = torch.tensor(
        [spec.shape[-1] for spec in spectrograms], dtype=torch.long
    )
    max_spec_length = int(spectrogram_lengths.max().item())
    n_mels = spectrograms[0].shape[0]
    spectrogram_batch = spectrograms[0].new_zeros((batch_size, n_mels, max_spec_length))
    for idx, spectrogram in enumerate(spectrograms):
        spectrogram_batch[idx, :, : spectrogram.shape[-1]] = spectrogram

    encoded_sequences = []
    encoded_lengths = []
    for item in dataset_items:
        encoded = item["text_encoded"].squeeze(0).to(dtype=torch.long)
        encoded_sequences.append(encoded)
        encoded_lengths.append(encoded.numel())

    text_encoded = torch.cat(encoded_sequences)
    text_encoded_length = torch.tensor(encoded_lengths, dtype=torch.long)

    text = [item["text"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]

    result_batch = {
        "audio": audio_batch,
        "audio_length": audio_lengths,
        "spectrogram": spectrogram_batch,
        "spectrogram_length": spectrogram_lengths,
        "text": text,
        "text_encoded": text_encoded,
        "text_encoded_length": text_encoded_length,
        "audio_path": audio_paths,
    }

    return result_batch
