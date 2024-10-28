import re
from collections import defaultdict
from string import ascii_lowercase

import numpy as np
import sentencepiece as spm
import torch
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files

class CTCTextEncoder:
    empty_token = ""

    def __init__(
            self,
            alphabet = None,
            lm_decoding = False,
            use_bpe_tokenizer = False,
            bpe_model_path = None,
            **kwargs,
    ):
        self.use_bpe_tokenizer = use_bpe_tokenizer
        self.do_lm_decoding = lm_decoding

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.empty_token] + list(self.alphabet)

        # Use pretrained set of tokens for bpe tokenizer
        if self.use_bpe_tokenizer:
            self._add_bpe_tokens(bpe_model_path)

        self.mx_token_len = max([len(item) for item in self.vocab])

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        if lm_decoding:
            self._init_lm_decoder()

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        return self.ind2char[item]
    
    def encode(self, text):
        text = self.normalize_text(text)
        try:
            # For BPE tokenizer try to split on as large tokens as possible
            if self.use_bpe_tokenizer:
                res_list = []
                ind = 0
                while ind < len(text):
                    for substr_len in range(
                        min(len(text) - ind, self.mx_token_len), 0, -1
                    ):
                        if text[ind:ind + substr_len] in self.char2ind:
                            res_list.append(self.char2ind[text[ind:ind + substr_len]])
                            ind += substr_len
                            break

                return torch.Tensor(res_list).unsqueeze(0)

            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )
    def decode(self, inds):
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()
    def ctc_decode(self, inds):
        decoded = []
        last_ind = -1
        for ind in inds:
            if ind == last_ind:
                continue

            last_ind = ind
            if ind == 0:
                continue
            decoded.append(self.ind2char[ind])

        return "".join(decoded).strip()
    
    def _expand_and_merge_path(self, dp, log_probs):
        next_dp = defaultdict(float)
        for char_ind, next_token_prob in enumerate(log_probs):
            char = self.vocab[char_ind]
            for (prefix, last_char), cur_prob in dp:
                if last_char == char:
                    new_prefix = prefix
                else:
                    if char != self.empty_token:
                        new_prefix = prefix + char
                    else:
                        new_prefix = prefix

                next_dp[(new_prefix, char)] += cur_prob * np.exp(next_token_prob)
        return next_dp
    
    def _truncate_paths(self, dp, beam_size):
        """
        Leave only beam size most probable paths.

        Args:
            dp (defaultdict): dict with (prefix, last_ind) -> probability
        Returns:
            dp (defaultdict): dict with (prefix, last_ind) -> probability with exactly beam_size keys
        """
        return sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size]
    def get_best_pred_with_beam_search(self, log_probs, beam_size):
        """
        Get single best prediction with CTC beam search.

        Args:
            log_probs (torch.tensor): 2D torch tensor of shape (seq_len, n_tokens)
            beam_size (int): width of beam search
        Returns:
            best_prefix (str): decoded best prediction
        """
        dp = [[("", self.empty_token), 1.0]]

        for log_prob_row in log_probs:
            dp = self._expand_and_merge_path(dp, log_prob_row)
            dp = self._truncate_paths(dp, beam_size)

        (best_prefix, _), _ = sorted(dp, key=lambda x: -x[1])[0]
        return best_prefix

    def _add_bpe_tokens(self, bpe_model_path: str):
        """
        Add bpe tokens to model vocabulary. Use SentecePieceProcessor for getting tokens.

        Args:
            bpe_model_path (str): path to SentencePieceProcessor saved state.
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_path)

        additional_vocab = [self.sp.id_to_piece(i) for i in range(self.sp.vocab_size())]
        additional_vocab = additional_vocab
        for item in additional_vocab:
            if "<" in item or ">" in item:  # Exclude sentencepiece special symbols
                continue
            correct_item = item.replace(
                "▁", ""
            ).lower()  # Exclude sentencepiece specific separators
            if correct_item not in self.vocab:
                self.vocab.append(correct_item)

        self.vocab = [self.empty_token] + self.vocab
    def _init_lm_decoder(self):
        """
        Download LM params and make LM and our vocabulary and tokens compatible
        """
        files = download_pretrained_files("librispeech-4-gram")

        txt = None
        with open(files.lexicon, "r") as f:
            txt = f.read()
        with open(files.lexicon, "w") as f:
            f.write(txt.replace("'", ""))  # We do not predict ' in our model

        self.vocab = [
            item if item != " " else "|" for item in self.vocab
        ]  # LM cannot work with silence token being space

        self.lm_decoder = ctc_decoder(
            lexicon=files.lexicon,
            tokens=self.vocab,
            lm=files.lm,
            nbest=1,
            blank_token="",
            sil_token="|",
        )

@staticmethod
def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text
