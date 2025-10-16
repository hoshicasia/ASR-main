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
        decoded = []
        raw = []
        for i in range(len(log_probs)):
            lp = log_probs[i][: log_probs_length[i]]
            argm = torch.argmax(lp, dim=-1)
            decoded.append(self.ctc_decode(argm))
            raw.append(self.raw_decode(argm))
        return decoded, raw

    def ctc_beam_search_decode(self, log_probs, probs_length, beam_width=10):
        """Prefix beam search"""
        import numpy as np
        import torch

        lp = torch.tensor(log_probs.detach().cpu().numpy())
        B = lp.shape[0]
        out = []
        for b in range(B):
            p = lp[b][: probs_length[b]]
            T, V = p.shape
            beam = {(): (0.0, -np.inf)}
            for t in range(T):
                pt = p[t]
                new = {}
                for prefix, (pb, pnb) in beam.items():
                    cb, cn = new.get(prefix, (-np.inf, -np.inf))
                    new[prefix] = (np.logaddexp(cb, np.logaddexp(pb, pnb) + pt[0]), cn)
                    last = prefix[-1] if prefix else None
                    for c in range(1, V):
                        npref = prefix + (c,)
                        cb2, cn2 = new.get(npref, (-np.inf, -np.inf))
                        if last == c:
                            cn2 = np.logaddexp(cn2, pb + pt[c])
                        else:
                            cn2 = np.logaddexp(cn2, np.logaddexp(pb, pnb) + pt[c])
                        new[npref] = (cb2, cn2)
                items = sorted(
                    new.items(),
                    key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]),
                    reverse=True,
                )[:beam_width]
                beam = dict(items)
            if not beam:
                out.append("")
                continue
            best = max(beam.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))[0]
            out.append(self.raw_decode(list(best)) if best else "")
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
