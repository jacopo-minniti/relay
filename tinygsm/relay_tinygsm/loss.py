from typing import Any, Dict, List, Optional

import torch

from xlm.datamodule import Tokenizer
from xlm.harness import Harness, LossFunction
from xlm.utils.nn import masked_mean, select_random_indices
from xlm.utils.rank_zero import RankedLogger

from .streaming_batch import StreamingBatch

logger = RankedLogger(__name__, rank_zero_only=True)


class RelayBPTTSeq2SeqLoss(LossFunction):
    """Multi-step relay rollout loss for seq2seq MLM batches.

    Same schedule as the Sudoku ``RelayBPTTLoss``: a rollout buffer of partially
    unmasked examples, ``num_steps`` teacher-forced confidence unmask steps, and
    an optional stop-gradient on the carried hidden state.

    Batch contract (``mlm`` seq2seq collators):

    * ``fixed_positions_mask`` is the clue / prefix mask (never remasked).
    * ``attention_mask`` hides left pads and the right canvas tail.
    * ``target_ids == -100`` marks positions with no loss (the canvas tail when
      ``loss_on_padding=True``). In-block pads keep the pad id as the target.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Tokenizer] = None,
        predictor: Optional[Any] = None,
        loss_on_visible_tokens: bool = False,
        use_model_confidence: bool = True,
        batch_shape_key: str = "input_ids",
        num_steps: int = 2,
        stop_grad_h_s: bool = False,
        with_relay: bool = True,
        threshold_sampling_sigma: float = 0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.predictor = predictor
        self.loss_on_visible_tokens = loss_on_visible_tokens
        self.use_model_confidence = use_model_confidence
        self.batch_shape_key = batch_shape_key
        self.num_steps = num_steps
        self.stop_grad_h_s = stop_grad_h_s
        self.with_relay = with_relay
        self.threshold_sampling_sigma = threshold_sampling_sigma
        self.pl_module = None
        self.mask_token_id_tensor = None
        if self.threshold_sampling_sigma < 0:
            raise ValueError(
                "threshold_sampling_sigma must be >= 0, "
                f"got {self.threshold_sampling_sigma}."
            )

    def configure(self, pl_module: Harness) -> None:
        self.pl_module = pl_module
        if self.predictor is None:
            self.predictor = pl_module.predictor
        self.mask_token_id_tensor = torch.tensor(
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )
        self.streaming_batch = StreamingBatch()
        self.val_streaming_batch = StreamingBatch()

    def __call__(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.loss_fn(batch, batch_idx, dataloader_idx, dataloader_name)

    def _positions(self, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = (attention_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        return positions * attention_mask.long()

    def _compute_ce(
        self,
        logits: torch.Tensor,
        x_clean: torch.Tensor,
        x_input: torch.Tensor,
        fixed: torch.Tensor,
    ) -> torch.Tensor:
        targets = x_clean.clone()
        ignore = torch.zeros_like(x_input, dtype=torch.bool)
        if not self.loss_on_visible_tokens:
            ignore = ignore.logical_or(x_input != self.mask_token_id_tensor)
        else:
            ignore = ignore.logical_or(fixed)
        # Canvas tail (and any other non-target slot) is -100 in target_ids.
        # Tie the masked_mean denominator to that, so a 0 from ignore_index
        # is not counted.
        ignore = ignore.logical_or(x_clean == -100)
        targets[ignore] = -100
        ce = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2),
            targets,
            reduction="none",
            ignore_index=-100,
        )
        return masked_mean(ce.flatten(), ~ignore.flatten(), dim=-1)

    def _select_unmask(
        self,
        logits: torch.Tensor,
        masked_mutable: torch.Tensor,
        confidence_threshold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.use_model_confidence:
                return self.predictor.compute_confidence_unmask(
                    logits,
                    masked_mutable,
                    threshold=confidence_threshold,
                )
            num_unmask = (masked_mutable.sum(dim=-1) // 2).clamp(min=1)
            return select_random_indices(
                inp_shape=logits.shape[:2],
                num_unmask=num_unmask,
                select_from_mask=masked_mutable,
                selection_score=None,
                selection_mode="sample",
            )

    def _sample_confidence_threshold(
        self,
        *,
        device: torch.device,
        training: bool,
    ) -> Optional[float]:
        if not self.use_model_confidence:
            return None
        mu = getattr(self.predictor, "threshold", None)
        if mu is None:
            return None
        mu = float(mu)
        if (not training) or self.threshold_sampling_sigma == 0:
            return mu
        sampled = mu + self.threshold_sampling_sigma * torch.randn(
            (), device=device
        ).item()
        return max(0.0, sampled)

    def loss_fn(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        is_training = (
            bool(self.pl_module.training) if self.pl_module is not None else True
        )
        if "fixed_positions_mask" not in batch:
            raise KeyError(
                "RelayBPTTSeq2SeqLoss requires batch['fixed_positions_mask']"
            )

        if not is_training:
            self.val_streaming_batch.reset()
        sb = self.val_streaming_batch if not is_training else self.streaming_batch
        evicted_candidates: Optional[torch.Tensor] = None
        if sb.storage is not None:
            evicted_candidates = sb.ready_to_evict.nonzero(as_tuple=True)[0]

        evicted_count = sb.evict_and_fill(
            batch,
            mask_token_id=self.mask_token_id_tensor,
            batch_shape_key=self.batch_shape_key,
        )

        device = sb.storage["input_ids"].device
        capacity, seq_len = sb.capacity, sb.seq_len

        if self.with_relay:
            d_model = self.model.d_model
            if "h_s" not in sb.storage:
                sb.storage["h_s"] = torch.zeros(
                    capacity,
                    seq_len,
                    d_model,
                    device=device,
                    dtype=torch.float32,
                )
            if evicted_count > 0 and evicted_candidates is not None:
                evicted_slots = evicted_candidates[:evicted_count]
                sb.storage["h_s"][evicted_slots] = 0.0
        else:
            sb.storage.pop("h_s", None)

        x = sb.storage["input_ids"].clone()
        x_clean = sb.storage["target_ids"]
        fixed = sb.storage["fixed_positions_mask"]
        attention_mask = sb.storage["attention_mask"].to(dtype=torch.bool)
        positions = self._positions(attention_mask)
        h: Optional[torch.Tensor] = (
            sb.storage["h_s"].clone() if self.with_relay else None
        )

        loss_terms: List[torch.Tensor] = []
        last_unmask: Optional[torch.Tensor] = None

        for _t in range(self.num_steps):
            masked_mutable = (x == self.tokenizer.mask_token_id) & ~fixed
            has_any_mask = masked_mutable.any()

            if self.with_relay:
                assert h is not None
                logits, h_s = self.model(
                    x, h, attention_mask=attention_mask, positions=positions
                )
            else:
                logits = self.model(x, attention_mask, positions)
                h_s = None

            L_t = self._compute_ce(logits, x_clean, x, fixed)
            anchor = 0.0 * logits.sum()
            L_t = L_t + anchor
            loss_terms.append(L_t)

            sampled_threshold = self._sample_confidence_threshold(
                device=logits.device,
                training=is_training,
            )
            threshold_t = (
                torch.tensor(
                    sampled_threshold, device=logits.device, dtype=logits.dtype
                )
                if sampled_threshold is not None
                else None
            )
            unmask = self._select_unmask(
                logits.detach(),
                masked_mutable,
                confidence_threshold=threshold_t,
            )
            unmask = unmask & masked_mutable
            last_unmask = unmask

            x = x.clone()
            with torch.no_grad():
                # -100 targets are never masked (canvas tail), so this write
                # only copies real token ids into mutable mask slots.
                x[unmask] = x_clean[unmask]

            if self.with_relay:
                assert h is not None and h_s is not None
                h_old = h
                h_next = h_s.detach() if self.stop_grad_h_s else h_s
                h = torch.where(has_any_mask, h_next, h_old)

        total_loss = torch.stack(loss_terms).sum()
        with torch.no_grad():
            sb.storage["input_ids"] = x.detach().clone()
            if self.with_relay:
                assert h is not None
                sb.storage["h_s"] = h.detach().clone()
        sb.update_after_unmask(last_unmask, x_clean, self.mask_token_id_tensor)

        return {"loss": total_loss}
