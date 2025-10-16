import sentencepiece as spm
import torch

from src.text_encoder.base_ctc_encoder import BaseCTCTextEncoder


class BPETextEncoder(BaseCTCTextEncoder):
    def __init__(self, model_path):
        super().__init__()
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))
        self.vocab_size = self.sp.vocab_size()
        self.ind2token = {0: ""}
        self.token2ind = {"": 0}
        for i in range(self.sp.vocab_size()):
            tok = self.sp.id_to_piece(i)
            ctc_idx = i + 1
            self.ind2token[ctc_idx] = tok
            self.token2ind[tok] = ctc_idx

    def __len__(self):
        return self.vocab_size + 1

    def encode(self, text):
        text = self.normalize_text(text)
        sp_inds = self.sp.encode(text)
        ctc_inds = [i + 1 for i in sp_inds]
        return torch.tensor(ctc_inds, dtype=torch.long).unsqueeze(0)

    def raw_decode(self, inds):
        sp_inds = [int(i) - 1 for i in inds if int(i) > 0]
        if not sp_inds:
            return ""
        return self.sp.decode(sp_inds).strip()

    def ind_to_token(self, ind):
        return self.ind2token[int(ind)]
