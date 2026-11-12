"""Vanilla MLM predictor with the relay raw-cumsum unmask rule.

Seq2seq prediction matches ``RelayConfidencePredictor`` without the relay
carry. The prefix is left-padded, ``max_new_tokens`` mask slots are appended,
and ``to_dict`` emits ``generated_text`` from ``output_start_idx``. Unmask
selection is the raw cumsum of ``1 - top_prob``, the same budget as
``RelayConfidencePredictor.compute_confidence_unmask``.
"""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT

from mlm.datamodule_mlm import extend_fixed_positions_mask
from mlm.unbatch import unbatch
from xlm.datamodule import Tokenizer
from xlm.harness import Predictor
from xlm.utils.nn import sample_from_logits, sample_from_top_k, sample_from_top_p

StepResults = Dict[str, Any]


class CumTopProbMLMPredictor(torch.nn.Module, Predictor):
    def __init__(
        self,
        max_steps: int,
        max_new_tokens: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[torch.nn.Module] = None,
        noise_schedule: Optional[torch.nn.Module] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        skip_special_tokens: bool = True,
        confidence: str = "top_prob",
        threshold: Optional[float] = None,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        if confidence != "top_prob":
            raise ValueError(
                "CumTopProbMLMPredictor requires confidence='top_prob', "
                f"got {confidence!r}."
            )
        if threshold is None:
            raise ValueError("CumTopProbMLMPredictor requires threshold.")
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        if top_k is not None and top_p is not None:
            raise ValueError("Both top_k and top_p cannot be non-None")
        if top_k is not None:
            self.sampling_function = partial(sample_from_top_k, top_k)
        elif top_p is not None:
            self.sampling_function = partial(sample_from_top_p, top_p)
        else:
            self.sampling_function = sample_from_logits
        self.confidence = confidence
        self.threshold = threshold
        self.skip_special_tokens = skip_special_tokens

    def reset(self) -> None:
        return None

    def compute_confidence(
        self,
        logits: Float[TT, " batch seq_len vocab_size"],
    ) -> Float[TT, " batch seq_len"]:
        return logits.softmax(dim=-1).max(dim=-1)[0]

    def compute_confidence_unmask(
        self,
        logits: Float[TT, " batch seq_len vocab_size"],
        masked_mutable: Bool[TT, " batch seq_len"],
        threshold: Optional[Union[float, torch.Tensor]] = None,
        done: Optional[Bool[TT, " batch"]] = None,
    ) -> Bool[TT, " batch seq_len"]:
        """Cumsum-threshold unmask. Higher top-prob unmasks first."""
        with torch.no_grad():
            confidence = self.compute_confidence(logits).clone()
            confidence[~masked_mutable] = float("-inf")
            top_confidence_values, top_confidence_indices = torch.sort(
                1 - confidence, dim=-1
            )
            threshold_value = self.threshold if threshold is None else threshold
            if threshold_value is None:
                raise ValueError(
                    "compute_confidence_unmask requires a threshold override or "
                    "predictor.threshold to be set."
                )
            unmask = torch.cumsum(top_confidence_values, dim=-1) < threshold_value
            unmask = unmask.scatter(-1, top_confidence_indices, unmask)

            no_unmask = ~unmask.any(dim=-1)
            has_mutable = masked_mutable.any(dim=-1)
            force_unmask = no_unmask & has_mutable
            best_idx = top_confidence_indices[:, 0]
            force_update = torch.zeros_like(unmask).scatter_(
                -1, best_idx.unsqueeze(-1), True
            )
            unmask = unmask | (force_update & force_unmask.unsqueeze(-1))
            if done is not None:
                unmask = unmask & ~done.unsqueeze(-1)
            return unmask

    def stop(self, step_results: StepResults) -> bool:
        fixed = step_results["fixed"]
        x = step_results["x"]
        masked_mutable = (x == self.tokenizer.mask_token_id) & ~fixed
        steps_done = bool(step_results["done"].all())
        all_filled = bool(masked_mutable.sum() == 0)
        max_steps_reached = bool(
            (step_results["steps_taken"] >= self.max_steps).all()
        )
        return steps_done or all_filled or max_steps_reached

    def _forward_logits(
        self,
        x: TT,
        attention_mask: TT,
        positions: TT,
    ) -> TT:
        assert self.model is not None, "Model is not initialized"
        flash_attn = getattr(self.model, "force_flash_attn", False)
        if flash_attn and bool((~attention_mask).any()):
            raise ValueError(
                "Attention mask is not all True and flash attention is enabled"
            )
        return self.model(x, attention_mask, positions)

    def predict_single_step(
        self,
        step_results: StepResults,
        final_step: bool = False,
    ) -> StepResults:
        x = step_results["x"]
        fixed = step_results["fixed"]
        attention_mask = step_results["attention_mask"]
        positions = step_results["positions"]
        masked_mutable = (x == self.tokenizer.mask_token_id) & ~fixed
        logits = self._forward_logits(x, attention_mask, positions)

        if not final_step:
            unmask = self.compute_confidence_unmask(
                logits, masked_mutable, done=step_results["done"]
            )
        else:
            unmask = masked_mutable
        unmask = unmask & masked_mutable
        x = x.clone()
        x[unmask] = self.sampling_function(logits[unmask])

        step_results["steps_taken"] = step_results["steps_taken"] + 1
        masked_after = (x == self.tokenizer.mask_token_id) & ~fixed
        per_sample_filled = ~masked_after.any(dim=-1)
        done = (step_results["steps_taken"] >= self.max_steps) | per_sample_filled
        return {
            "x": x,
            "fixed": fixed,
            "attention_mask": attention_mask,
            "positions": positions,
            "logits": logits,
            "steps_taken": step_results["steps_taken"],
            "done": done,
        }

    def _extend_prefix(self, batch: Dict[str, Any]) -> Tuple[TT, TT, TT, TT, int]:
        x = batch["input_ids"].clone()
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                x.shape[0], x.shape[1], dtype=torch.bool, device=x.device
            )
        else:
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
        fixed = batch.get("fixed_positions_mask")
        if fixed is None:
            fixed = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        else:
            fixed = fixed.to(device=x.device, dtype=torch.bool)
        output_start_idx = x.shape[-1]
        if self.max_new_tokens is not None:
            x = torch.cat(
                [
                    x,
                    torch.full(
                        (x.shape[0], self.max_new_tokens),
                        self.tokenizer.mask_token_id,
                        device=x.device,
                    ),
                ],
                dim=-1,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], self.max_new_tokens),
                        device=attention_mask.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=-1,
            )
            fixed = extend_fixed_positions_mask(
                fixed, self.max_new_tokens, fill=False
            )
        positions = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        positions = positions * attention_mask.long()
        return x, attention_mask, fixed, positions, output_start_idx

    def decode(self, x: Integer[TT, " batch seq_len"]) -> List[str]:
        return self.tokenizer.batch_decode(
            x, skip_special_tokens=self.skip_special_tokens
        )

    def to_dict(
        self,
        batch: Dict[str, Any],
        preds: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if "generated_text" not in preds and "output_start_idx" in preds:
            start = int(preds["output_start_idx"])
            preds = {
                **preds,
                "generated_text": self.tokenizer.batch_decode(
                    preds["ids"][:, start:],
                    skip_special_tokens=self.skip_special_tokens,
                ),
            }
        return unbatch(preds, length=len(preds["text"]))

    @torch._dynamo.disable()
    def predict(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        import time

        _start_time = time.time()
        x, attention_mask, fixed, positions, output_start_idx = self._extend_prefix(
            batch
        )
        step_results: StepResults = {
            "x": x,
            "fixed": fixed,
            "attention_mask": attention_mask,
            "positions": positions,
            "steps_taken": torch.zeros(x.shape[0], dtype=torch.int, device=x.device),
            "done": torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
        }
        while not self.stop(step_results):
            step_results = self.predict_single_step(step_results)
        step_results = self.predict_single_step(step_results, final_step=True)

        final_x = step_results["x"]
        text = self.decode(final_x)
        _time_taken = time.time() - _start_time
        self.reset()
        steps_taken = step_results["steps_taken"].detach().cpu().tolist()
        return {
            "text": text,
            "ids": final_x,
            "loss": None,
            "time_taken": [_time_taken] * len(text),
            "output_start_idx": output_start_idx,
            "steps_taken": steps_taken,
        }
