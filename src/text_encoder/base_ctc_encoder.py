from abc import abstractmethod
from collections import defaultdict

import numpy as np
import torch
from pyctcdecode import build_ctcdecoder


class BaseCTCTextEncoder:
    """Base class for CTC text encoders."""

    def __init__(self):
        self._lm_decoder = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        pass

    @abstractmethod
    def raw_decode(self, inds) -> str:
        """Decode indices directly (no CTC collapsing)."""
        pass

    @abstractmethod
    def ind_to_token(self, ind: int) -> str:
        pass

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        import re

        text = re.sub(r"[^a-z ]", "", text)
        return text

    def ctc_decode(self, inds) -> str:
        """Basic decoding: collapsing repeats and removing blanks."""
        result = []
        prev = None
        for ind in inds:
            ind = int(ind)
            if ind != prev and ind != 0:
                result.append(ind)
            prev = ind
        if not result:
            return ""
        return self.raw_decode([i for i in result])

    def ctc_argmax_decode(self, log_probs, log_probs_length):
        """Argmax decoding"""
        decoded = []
        raw = []
        for i in range(len(log_probs)):
            lp = log_probs[i][: log_probs_length[i]]
            argm = torch.argmax(lp, dim=-1)
            decoded.append(self.ctc_decode(argm))
            raw.append(self.raw_decode(argm))
        return decoded, raw

    def ctc_beam_search_decode(self, probs, probs_length, beam_width=10):
        """BeamSearch without LM"""
        import torch

        probs = torch.softmax(probs, dim=-1).cpu().numpy()

        out = []
        for row, L in zip(probs, probs_length):
            L = int(L)
            beam = {("", 0): 1.0}
            for step in row[:L]:
                new_beam = defaultdict(float)
                for (pref, last), p_pref in beam.items():
                    for idx, p_t in enumerate(step):
                        ch = self.ind2token[idx]
                        if ch == last:
                            new_pref = pref
                        else:
                            if ch == 0:
                                new_pref = pref
                            else:
                                new_pref = pref + ch

                        new_beam[(new_pref, ch)] += p_pref * float(p_t)
                items = sorted(new_beam.items(), key=lambda x: -x[1])[:beam_width]
                beam = dict(items)
            final = {}
            for (pref, _), p in beam.items():
                final[pref] = final.get(pref, 0.0) + p
            best = max(final.items(), key=lambda x: x[1])[0] if final else ""
            out.append(best)
        return out

    def build_lm_decoder(self, labels, lm_path, alpha=0.5, beta=1.0):
        if (
            self._lm_decoder is None
            or getattr(self._lm_decoder, "model_path", None) != lm_path
        ):
            self._lm_decoder = build_ctcdecoder(
                labels=labels, kenlm_model_path=lm_path, alpha=alpha, beta=beta
            )
            self._lm_decoder.model_path = lm_path

        return self._lm_decoder

    def ctc_beam_search_lm_decode(
        self, probs, probs_length, lm_path, beam_width=50, alpha=0.5, beta=1.0
    ):
        """BeamSearch-LM with pyctcdecode"""

        probs = probs.detach().cpu().numpy()
        V = probs.shape[-1]
        labels = [self.ind_to_token(i) for i in range(V)]
        decoder = self.build_lm_decoder(labels, lm_path, alpha=alpha, beta=beta)
        results = []
        for i in range(probs.shape[0]):
            logp = probs[i][: probs_length[i]].astype(np.float32)
            decoded = decoder.decode(logp, beam_width=beam_width)
            results.append(decoded.strip())
        return results
