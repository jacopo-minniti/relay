"""Confidence-threshold predictor with an optional relay carry.

Seq2seq prediction matches ``mlm.predictor_mlm.MLMPredictor``: the prefix is
left-padded, ``max_new_tokens`` mask slots are appended, and ``to_dict`` emits
``generated_text`` from ``output_start_idx`` so ``Gsm8kCodeEval`` can execute
the suffix. Unmask selection is the Sudoku cumsum-threshold rule.
"""

from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT

from mlm.datamodule_mlm import extend_fixed_positions_mask
from mlm.unbatch import unbatch
from xlm.datamodule import Tokenizer
from xlm.harness import Predictor
from xlm.utils.nn import (
    sample_from_logits,
    sample_from_top_k,
    sample_from_top_p,
    select_random_indices,
)
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

StepResults = Dict[str, Any]


class RelayConfidencePredictor(torch.nn.Module, Predictor):
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
        confidence: Optional[Literal["prob_diff", "entropy", "top_prob"]] = None,
        threshold: Optional[float] = 0.15,
        with_relay: bool = False,
        log_rollout_diagnostics: bool = False,
        log_rollout_diagnostics_max_steps: int = 8,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.with_relay = with_relay
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
        self.log_rollout_diagnostics = log_rollout_diagnostics
        self.log_rollout_diagnostics_max_steps = log_rollout_diagnostics_max_steps

    def reset(self) -> None:
        return None

    def compute_confidence(
        self,
        logits: Float[TT, " batch seq_len vocab_size"],
    ) -> Float[TT, " batch seq_len"]:
        if self.confidence == "prob_diff":
            temp = logits.softmax(dim=-1)
            top2_probs, _ = torch.topk(temp, k=2, dim=-1)
            return top2_probs[:, :, 0] - top2_probs[:, :, 1]
        if self.confidence == "entropy":
            temp = logits.softmax(dim=-1)
            return torch.sum(temp * torch.log(temp + 1e-10), dim=-1)
        if self.confidence == "top_prob":
            return logits.softmax(dim=-1).max(dim=-1)[0]
        raise ValueError(f"Unknown confidence: {self.confidence}")

    def compute_confidence_unmask(
        self,
        logits: Float[TT, " batch seq_len vocab_size"],
        masked_mutable: Bool[TT, " batch seq_len"],
        threshold: Optional[Union[float, torch.Tensor]] = None,
        done: Optional[Bool[TT, " batch"]] = None,
    ) -> Bool[TT, " batch seq_len"]:
        """Cumsum-threshold unmask (Sudoku rule). Higher confidence unmasks first."""
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

    def _record_rollout_step_diagnostics(
        self,
        step_results: StepResults,
        logits: TT,
        masked_mutable: TT,
        h_new: Optional[TT],
    ) -> None:
        if not self.with_relay or h_new is None:
            return
        h_prev = step_results.get("h")
        if h_prev is None:
            return
        with torch.no_grad():
            step_idx = int(step_results["steps_taken"][0].item())
            masked_count = masked_mutable.sum(dim=-1).float().mean().item()
            if masked_mutable.any():
                post_conf = logits.softmax(dim=-1).max(dim=-1)[0]
                mean_conf_masked = post_conf[masked_mutable].mean().item()
            else:
                mean_conf_masked = 0.0
            step_results.setdefault("rollout_diagnostics_steps", []).append(
                {
                    "step": step_idx,
                    "masked_tokens_per_sample": float(masked_count),
                    "mean_confidence_masked": float(mean_conf_masked),
                    "h_t_norm": float(h_prev.norm(dim=-1).mean().item()),
                    "h_s_norm": float(h_new.norm(dim=-1).mean().item()),
                }
            )

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
        assert self.model is not None, "Model is not initialized"
        if self.with_relay:
            logits, h_new = self.model(
                x,
                step_results["h"],
                attention_mask=attention_mask,
                positions=positions,
            )
            self._record_rollout_step_diagnostics(
                step_results, logits, masked_mutable, h_new
            )
        else:
            logits = self.model(x, attention_mask, positions)
            h_new = None

        if not final_step:
            if not self.confidence:
                steps_left = self.max_steps - step_results["steps_taken"]
                num_unmask = (
                    masked_mutable.sum(dim=-1) / steps_left.clamp(min=1)
                ).long()
                unmask = select_random_indices(
                    inp_shape=x.shape,
                    num_unmask=num_unmask,
                    select_from_mask=masked_mutable,
                    selection_score=None,
                    selection_mode="sample",
                )
            else:
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
        result: StepResults = {
            "x": x,
            "fixed": fixed,
            "attention_mask": attention_mask,
            "positions": positions,
            "logits": logits,
            "steps_taken": step_results["steps_taken"],
            "done": done,
        }
        if self.with_relay:
            result["h"] = h_new
        if "rollout_diagnostics_steps" in step_results:
            result["rollout_diagnostics_steps"] = step_results[
                "rollout_diagnostics_steps"
            ]
        return result

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
        if self.with_relay and self.model is not None:
            step_results["h"] = torch.zeros(
                x.shape[0],
                x.shape[1],
                self.model.d_model,
                device=x.device,
            )
            step_results["rollout_diagnostics_steps"] = []

        while not self.stop(step_results):
            step_results = self.predict_single_step(step_results)
        step_results = self.predict_single_step(step_results, final_step=True)

        final_x = step_results["x"]
        text = self.decode(final_x)
        rollout_steps = step_results["steps_taken"].float().mean().item()
        _time_taken = time.time() - _start_time

        rollout_diagnostics: Dict[str, Any] = {}
        if self.with_relay:
            rollout_diagnostics["steps"] = step_results.get(
                "rollout_diagnostics_steps", []
            )
            if self.log_rollout_diagnostics:
                for step_diag in rollout_diagnostics["steps"][
                    : self.log_rollout_diagnostics_max_steps
                ]:
                    logger.info(
                        "Rollout step=%d masked=%.3f conf=%.4f h_t_norm=%.4f h_s_norm=%.4f",
                        step_diag["step"],
                        step_diag["masked_tokens_per_sample"],
                        step_diag["mean_confidence_masked"],
                        step_diag["h_t_norm"],
                        step_diag["h_s_norm"],
                    )
        self.reset()
        result: Dict[str, Any] = {
            "text": text,
            "ids": final_x,
            "loss": None,
            "time_taken": [_time_taken] * len(text),
            "output_start_idx": output_start_idx,
            "rollout_steps": rollout_steps,
            "rollout_diagnostics": rollout_diagnostics,
        }
        if self.log_rollout_diagnostics:
            result["rollout_steps_per_sample"] = (
                step_results["steps_taken"].detach().cpu().tolist()
            )
            result["inference_steps"] = result["rollout_steps_per_sample"]
        else:
            result["inference_steps"] = (
                step_results["steps_taken"].detach().cpu().tolist()
            )
        return result
