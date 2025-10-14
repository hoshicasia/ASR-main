import torch
from torch.nn.utils.rnn import pad_sequence


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
    audio_lengths = torch.tensor([a.shape[-1] for a in audios], dtype=torch.long)
    max_audio_len = int(audio_lengths.max().item())
    audio_channels = audios[0].shape[0]

    audio_batch = audios[0].new_zeros((batch_size, audio_channels, max_audio_len))
    for i, a in enumerate(audios):
        audio_batch[i, :, : a.shape[-1]] = a

    specs = []
    for item in dataset_items:
        s = item["spectrogram"]
        if s.dim() == 3 and s.shape[0] == 1:
            s = s.squeeze(0)
        specs.append(s)

    spec_lengths = torch.tensor([s.shape[-1] for s in specs], dtype=torch.long)
    max_spec_len = int(spec_lengths.max().item())
    n_mels = specs[0].shape[0]

    spec_batch = specs[0].new_zeros((batch_size, n_mels, max_spec_len))
    for i, s in enumerate(specs):
        spec_batch[i, :, : s.shape[-1]] = s

    encoded_seqs = [item["text_encoded"].view(-1).long() for item in dataset_items]
    encoded_lengths = torch.tensor(
        [seq.size(0) for seq in encoded_seqs], dtype=torch.long
    )

    padded_texts = pad_sequence(encoded_seqs, batch_first=True, padding_value=0)
    texts = [item["text"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]

    result_batch = {
        "audio": audio_batch,
        "audio_length": audio_lengths,
        "spectrogram": spec_batch,
        "spectrogram_length": spec_lengths,
        "text": texts,
        "text_encoded": padded_texts,
        "text_encoded_length": encoded_lengths,
        "audio_path": audio_paths,
    }
    return result_batch
