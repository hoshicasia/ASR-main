from abc import ABC, abstractmethod
from string import ascii_lowercase

import torch

from src.text_encoder.base_ctc_encoder import BaseCTCTextEncoder


class CTCTextEncoder(BaseCTCTextEncoder):
    EMPTY = ""

    def __init__(self, alphabet=None):
        super().__init__()
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")
        self.alphabet = alphabet
        self.vocab = [self.EMPTY] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {c: i for i, c in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def encode(self, text):
        text = self.normalize_text(text)
        return torch.tensor(
            [self.char2ind[c] for c in text], dtype=torch.long
        ).unsqueeze(0)

    def raw_decode(self, inds):
        return "".join(self.ind2char[int(i)] for i in inds).strip()

    def ind_to_token(self, ind):
        return self.ind2char[ind]
