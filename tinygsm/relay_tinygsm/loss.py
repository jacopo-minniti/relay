from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from xlm.datamodule import Tokenizer
from xlm.harness import Harness, LossFunction
from relay_tinygsm.predictor import contiguous_block_allow
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
        log_step_diagnostics: bool = False,
        log_grad_diagnostics: bool = False,
        grad_diagnostics_every_n_steps: int = 50,
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
        self.log_step_diagnostics = log_step_diagnostics
        self.log_grad_diagnostics = log_grad_diagnostics
        self.grad_diagnostics_every_n_steps = grad_diagnostics_every_n_steps
        self.pl_module = None
        self.mask_token_id_tensor = None
        self._grad_diag: Optional[Dict[str, Any]] = None
        self._grad_diag_skip_warned = False
        if self.grad_diagnostics_every_n_steps < 1:
            raise ValueError(
                "grad_diagnostics_every_n_steps must be >= 1, "
                f"got {self.grad_diagnostics_every_n_steps}."
            )
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
        return_per_token: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
        loss = masked_mean(ce.flatten(), ~ignore.flatten(), dim=-1)
        if return_per_token:
            return loss, ce.detach(), ignore
        return loss

    def _select_unmask(
        self,
        logits: torch.Tensor,
        masked_mutable: torch.Tensor,
        confidence_threshold: Optional[torch.Tensor] = None,
        fixed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.use_model_confidence:
                return self.predictor.compute_confidence_unmask(
                    logits,
                    masked_mutable,
                    threshold=confidence_threshold,
                    fixed=fixed,
                )
            eligible = masked_mutable
            block = getattr(self.predictor, "allow_block_size", None)
            if block:
                if fixed is None:
                    raise ValueError(
                        "allow_block_size requires fixed so the suffix start "
                        "is known."
                    )
                eligible = masked_mutable & contiguous_block_allow(
                    masked_mutable, fixed, int(block)
                )
            num_unmask = (eligible.sum(dim=-1) // 2).clamp(min=1)
            return select_random_indices(
                inp_shape=logits.shape[:2],
                num_unmask=num_unmask,
                select_from_mask=eligible,
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

    def _wants_grad_diag(self, batch_idx: Optional[int], is_training: bool) -> bool:
        if not (
            self.log_grad_diagnostics
            and is_training
            and self.pl_module is not None
            and self.num_steps == 2
            and batch_idx is not None
        ):
            return False
        trainer = getattr(self.pl_module, "trainer", None)
        if trainer is None:
            return False
        accum = int(trainer.accumulate_grad_batches)
        world = int(getattr(trainer, "world_size", 1) or 1)
        if accum == 1 and world > 1:
            if not self._grad_diag_skip_warned:
                logger.warning(
                    "Skipping grad diagnostics: accumulate_grad_batches=1 "
                    "under DDP would run the split backward on the sync microbatch."
                )
                self._grad_diag_skip_warned = True
            return False
        if int(trainer.global_step) % self.grad_diagnostics_every_n_steps != 0:
            return False
        return int(batch_idx) % accum == 0

    def _collect_step_diag(
        self,
        *,
        t: int,
        logits: torch.Tensor,
        ce_tok: torch.Tensor,
        ignore: torch.Tensor,
        x_clean: torch.Tensor,
        x_input: torch.Tensor,
        unmask: torch.Tensor,
        h_read: Optional[torch.Tensor],
        h_written: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            valid = ~ignore
            m = valid.sum().to(dtype=torch.float32)
            u = (unmask & valid).sum().to(dtype=torch.float32) / m.clamp_min(1.0)
            tgt = x_clean.clamp(min=0).unsqueeze(-1)
            lt = logits.gather(-1, tgt).squeeze(-1)
            mx, am = logits.max(dim=-1)
            conf = torch.exp(mx.float() - ce_tok.float() - lt.float())
            acc = (am == x_clean).to(dtype=torch.float32)
            flat_valid = valid.flatten()
            c_bar = masked_mean(conf.flatten(), flat_valid, dim=-1)
            a = masked_mean(acc.flatten(), flat_valid, dim=-1)
            l_scalar = masked_mean(ce_tok.flatten().float(), flat_valid, dim=-1)
            record: Dict[str, torch.Tensor] = {
                "t": torch.tensor(t),
                "ce": ce_tok.detach(),
                "valid": valid,
                "L_scalar": l_scalar,
                "m": m,
                "u": u,
                "c_bar": c_bar,
                "a": a,
            }
            if h_read is not None and h_written is not None:
                record.update(self._carry_stats(t, h_read, h_written, x_input))
            return record

    def _carry_stats(
        self,
        t: int,
        h_read: torch.Tensor,
        h_written: torch.Tensor,
        x_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        del t
        ht = h_read.detach().float()
        hs = h_written.detach().float()
        row = ht.pow(2).sum(dim=(-1, -2)) > 0
        rms_in = self._rms_on_rows(ht, row)
        rms_out = self._rms_on_rows(hs, row)
        num = (ht * hs).sum(dim=-1)
        den = ht.norm(dim=-1).clamp_min(1e-12) * hs.norm(dim=-1).clamp_min(1e-12)
        cos = num / den
        pos = row.unsqueeze(-1) & (ht.norm(dim=-1) > 0)
        cos_mean = masked_mean(cos.flatten(), pos.flatten(), dim=-1)
        out: Dict[str, torch.Tensor] = {
            "rms_h": rms_in,
            "rms_h_next": rms_out,
            "cos_h": cos_mean,
        }
        if self.with_relay and hasattr(self.model, "relay_layer_norm"):
            ln_h = self.model.relay_layer_norm(h_read.detach()).float()
            emb = self.model.embed_tokens(x_input).detach().float()
            out["rho"] = self._rms(ln_h) / self._rms(emb).clamp_min(1e-12)
        return out

    @staticmethod
    def _rms(v: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(v.float().pow(2).mean())

    @staticmethod
    def _rms_on_rows(v: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
        if int(row.sum()) == 0:
            return v.new_zeros(())
        keep = row.view(-1, *([1] * (v.ndim - 1))).to(dtype=v.dtype)
        total = (v.pow(2) * keep).sum()
        count = row.sum().to(dtype=v.dtype) * v.shape[-1] * v.shape[-2]
        return torch.sqrt(total / count.clamp_min(1.0))

    @torch.compiler.disable
    def _log_step_diag(
        self,
        records: List[Dict[str, torch.Tensor]],
        *,
        h0: Optional[torch.Tensor],
        evicted_count: int,
        capacity: int,
    ) -> None:
        assert self.pl_module is not None
        log_kw = dict(
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
            rank_zero_only=True,
            logger=True,
            add_dataloader_idx=False,
        )
        for rec in records:
            t = int(rec["t"])
            self.pl_module.log(f"train/diag/L{t}", rec["L_scalar"], **log_kw)
            self.pl_module.log(f"train/diag/m{t}", rec["m"], **log_kw)
            self.pl_module.log(f"train/diag/u{t}", rec["u"], **log_kw)
            self.pl_module.log(f"train/diag/c_bar{t}", rec["c_bar"], **log_kw)
            self.pl_module.log(f"train/diag/a{t}", rec["a"], **log_kw)
            self.pl_module.log(
                f"train/diag/c_bar_minus_a{t}",
                rec["c_bar"] - rec["a"],
                **log_kw,
            )
            if "rms_h" in rec:
                self.pl_module.log(f"train/diag/rms_h{t}", rec["rms_h"], **log_kw)
                self.pl_module.log(
                    f"train/diag/rms_h{t + 1}", rec["rms_h_next"], **log_kw
                )
                self.pl_module.log(f"train/diag/cos_h{t}", rec["cos_h"], **log_kw)
            if "rho" in rec:
                self.pl_module.log(f"train/diag/rho{t}", rec["rho"], **log_kw)
        if h0 is not None and records:
            valid0 = records[0]["valid"]
            ce0 = records[0]["ce"]
            carried = h0.detach().float().pow(2).sum(dim=(-1, -2)) > 0
            row_valid = valid0 & carried.unsqueeze(-1)
            l0c = masked_mean(ce0.flatten().float(), row_valid.flatten(), dim=-1)
            zeta = (h0.detach().abs().sum(dim=-1) == 0).float().mean()
            self.pl_module.log("train/diag/L0_carried", l0c, **log_kw)
            self.pl_module.log("train/diag/zeta", zeta, **log_kw)
        eps = torch.tensor(
            float(evicted_count) / float(max(capacity, 1)),
            device=records[0]["m"].device if records else None,
        )
        self.pl_module.log("train/diag/epsilon", eps, **log_kw)

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

        step_diag = bool(
            self.log_step_diagnostics and is_training and self.pl_module is not None
        )
        grad_diag = self._wants_grad_diag(batch_idx, is_training)
        if grad_diag and not getattr(self.pl_module, "consumes_grad_diag", False):
            raise RuntimeError(
                "loss.log_grad_diagnostics=true requires "
                "lightning_module._target_=relay_tinygsm.harness.RelayDiagnosticsHarness"
            )
        h0_diag = h.detach() if (step_diag and h is not None) else None
        diag_records: List[Dict[str, torch.Tensor]] = []

        loss_terms: List[torch.Tensor] = []
        last_unmask: Optional[torch.Tensor] = None
        attached_h1: Optional[torch.Tensor] = None
        leaf: Optional[torch.Tensor] = None

        for t in range(self.num_steps):
            masked_mutable = (x == self.tokenizer.mask_token_id) & ~fixed
            has_any_mask = masked_mutable.any()

            h_for_model = h
            if grad_diag and t == 1 and self.with_relay:
                assert h is not None
                attached_h1 = h
                leaf = h.detach().requires_grad_(True)
                h_for_model = leaf

            if self.with_relay:
                assert h_for_model is not None
                logits, h_s = self.model(
                    x,
                    h_for_model,
                    attention_mask=attention_mask,
                    positions=positions,
                )
            else:
                logits = self.model(x, attention_mask, positions)
                h_s = None

            if step_diag:
                L_t, ce_tok, ignore = self._compute_ce(
                    logits, x_clean, x, fixed, return_per_token=True
                )
            else:
                L_t = self._compute_ce(logits, x_clean, x, fixed)
                ce_tok = None
                ignore = None
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
                fixed=fixed,
            )
            unmask = unmask & masked_mutable
            last_unmask = unmask
            if step_diag:
                assert ce_tok is not None and ignore is not None
                diag_records.append(
                    self._collect_step_diag(
                        t=t,
                        logits=logits,
                        ce_tok=ce_tok,
                        ignore=ignore,
                        x_clean=x_clean,
                        x_input=x,
                        unmask=unmask,
                        h_read=h,
                        h_written=h_s,
                    )
                )

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

        if step_diag:
            self._log_step_diag(
                diag_records,
                h0=h0_diag,
                evicted_count=evicted_count,
                capacity=capacity,
            )
        if grad_diag:
            through_h = bool(
                self.with_relay
                and not self.stop_grad_h_s
                and attached_h1 is not None
                and attached_h1.requires_grad
            )
            self._grad_diag = {
                "L0": loss_terms[0],
                "L1": loss_terms[1],
                "h1": attached_h1,
                "leaf": leaf,
                "through_h": through_h,
            }
        else:
            self._grad_diag = None

        return {"loss": total_loss}
