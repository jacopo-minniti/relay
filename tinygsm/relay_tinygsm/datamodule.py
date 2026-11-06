from typing import List

import torch

from mlm.datamodule_mlm import MLMSeq2SeqTrainCollator
from xlm.datamodule import Seq2SeqCollatorInput


class MLMSeq2SeqTrainMaskAllCollator(MLMSeq2SeqTrainCollator):
    """Seq2seq train collator that fully masks the visible suffix.

    The parent draws ``t ~ U[0, 1]`` masks. This subclass then sets every
    attention-visible, non-fixed position (the suffix block, including in-block
    pads) to the mask token. That is the rollout-buffer start state: the model
    always begins a trajectory from a fully masked answer.
    """

    def __call__(self, examples: List[Seq2SeqCollatorInput]):
        batch = super().__call__(examples)
        fixed = batch["fixed_positions_mask"].to(dtype=torch.bool)
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        visible_suffix = (~fixed) & attention_mask
        input_ids = batch["input_ids"].clone()
        input_ids[visible_suffix] = self.tokenizer.mask_token_id
        batch["input_ids"] = input_ids
        return batch
