from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

class ArgmaxCERMetric(BaseMetric):
    def __init__(
        self, text_encoder, use_beam_search=False, beam_size=3, *args, **kwargs
    ):
        """
        Calculate Character Error Metric with the choosen decoding method.
        Args:
            use_beam_search (bool): use beam_search
                (if text_encoder is initialized with LM, premade version with LM is used)
            beam_size (int): beam size for custom beam search (if text_encoder does not use LM)
        """
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.use_beam_search = use_beam_search

        if self.use_beam_search:
            self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        if self.use_beam_search and self.text_encoder.do_lm_decoding:
            predictions = self.text_encoder.lm_decoder(log_probs)

        lengths = log_probs_length.detach().numpy()
        for prediction, log_probs_matrix, length, target_text in zip(
            predictions, log_probs, lengths, text
        ):
            target_text = self.text_encoder.normalize_text(target_text)

            if self.use_beam_search and self.text_encoder.do_lm_decoding:
                pred_text = " ".join(prediction[0].words)
            elif self.use_beam_search:
                pred_text = self.text_encoder.get_best_pred_with_beam_search(
                    log_probs_matrix[:length, :], self.beam_size
                )
            else:
                pred_text = self.text_encoder.ctc_decode(prediction[:length])

            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)