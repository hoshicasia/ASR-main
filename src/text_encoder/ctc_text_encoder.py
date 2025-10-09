import re
from collections import defaultdict
from string import ascii_lowercase

import numpy as np
import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        result = []
        prev_ind = None
        for ind in inds:
            ind = int(ind)
            if ind != prev_ind and ind != 0:
                result.append(self.ind2char[ind])
            prev_ind = ind
        return "".join(result).strip()

    def ctc_argmax_decode(self, log_probs, log_probs_length):
        decoded_texts = []
        raw_texts = []
        for i in range(len(log_probs)):
            length = log_probs_length[i]
            lp = log_probs[i][:length]
            argmax_pred = torch.argmax(lp, dim=-1)

            decoded_texts.append(self.ctc_decode(argmax_pred))
            raw_texts.append(self.decode(argmax_pred))
        return decoded_texts, raw_texts

    def ctc_beam_search_decode(self, probs, probs_length, beam_width=10):
        results = []
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        for i in range(len(probs)):
            log_probs = probs[i][: probs_length[i]]
            T, vocab_size = log_probs.shape

            beam = {(): (0.0, float("-inf"))}

            for t in range(T):
                new_beam = defaultdict(lambda: (float("-inf"), float("-inf")))

                for prefix, (p_b, p_nb) in beam.items():
                    p_blank = log_probs[t, 0]
                    new_p_b, new_p_nb = new_beam[prefix]
                    new_p_b = np.logaddexp(new_p_b, np.logaddexp(p_b, p_nb) + p_blank)
                    new_beam[prefix] = (new_p_b, new_p_nb)

                    for c in range(1, vocab_size):
                        p_c = log_probs[t, c]
                        new_prefix = prefix + (c,)
                        new_p_b, new_p_nb = new_beam[new_prefix]
                        if len(prefix) > 0 and prefix[-1] == c:
                            new_p_nb = np.logaddexp(new_p_nb, p_b + p_c)
                        else:
                            new_p_nb = np.logaddexp(
                                new_p_nb, np.logaddexp(p_b, p_nb) + p_c
                            )

                        new_beam[new_prefix] = (new_p_b, new_p_nb)

                    if len(prefix) > 0:
                        last_c = prefix[-1]
                        p_c = log_probs[t, last_c]
                        new_p_b, new_p_nb = new_beam[prefix]
                        new_p_nb = np.logaddexp(new_p_nb, p_nb + p_c)
                        new_beam[prefix] = (new_p_b, new_p_nb)

                beam = dict(
                    sorted(
                        new_beam.items(),
                        key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                        reverse=True,
                    )[:beam_width]
                )

            best_prefix = max(
                beam.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1])
            )[0]
            decoded_text = "".join([self.ind2char[c] for c in best_prefix])
            results.append(decoded_text.strip())
        return results

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
