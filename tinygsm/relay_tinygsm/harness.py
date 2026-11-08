"""Harness that splits the relay training gradient into direct and carry terms.

Selected only by ``lightning_module._target_=relay_tinygsm.harness.RelayDiagnosticsHarness``.
With ``loss.log_grad_diagnostics`` off, ``backward`` is the base implementation and
no parameter hooks are registered.
"""

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from xlm.harness import Harness

_NO_SCALER_PRECISIONS = {
    "32",
    "32-true",
    "64",
    "64-true",
    "bf16",
    "bf16-mixed",
    "bf16-true",
}


def param_group_name(name: str) -> str:
    """Map a ``model.named_parameters`` name onto a diagnostic group."""
    if name.startswith("embed_tokens"):
        return "E"
    if name.startswith("relay_layer_norm"):
        return "N"
    if name.startswith("encoder."):
        return name[len("encoder.") :].split(".", 1)[0]
    if name.startswith("output_layer"):
        return "out"
    return "other"


class RelayDiagnosticsHarness(Harness):
    """``Harness`` plus the Tier B split backward.

    ``consumes_grad_diag`` tells the loss it is safe to feed step 1 a detached
    leaf: this class puts the carry-path gradient back in ``backward``.
    """

    consumes_grad_diag = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._hook_phase: Optional[str] = None
        self._grad_sq: Dict[str, Dict[str, Tensor]] = {}
        self._hooks_installed = False
        self._pending_grad_logs: Optional[Dict[str, Tensor]] = None
        self._capture_split_grads = False
        self._captured_gd: Optional[Dict[str, Tensor]] = None

    def on_fit_start(self) -> None:
        super().on_fit_start()
        loss = self.loss_function
        if not getattr(loss, "log_grad_diagnostics", False):
            return
        if self.fsdp_root() is not None:
            raise RuntimeError(
                "Grad diagnostics are DDP-only. FSDP shards .grad, so the "
                "parameter-name groups do not match the tensors the hooks see."
            )
        self._install_grad_hooks()

    def _install_grad_hooks(self) -> None:
        if self._hooks_installed:
            return
        seen = set()
        for name, param in self.model.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            param.register_hook(self._make_hook(param_group_name(name)))
        self._hooks_installed = True

    def _make_hook(self, group: str):
        def hook(grad: Tensor) -> None:
            phase = self._hook_phase
            if phase is None or grad is None:
                return
            sq = grad.detach().double().pow(2).sum()
            acc = self._grad_sq.setdefault(phase, {})
            acc[group] = acc[group] + sq if group in acc else sq
            acc["total"] = acc["total"] + sq if "total" in acc else sq

        return hook

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        loss_fn = self.loss_function
        stash = getattr(loss_fn, "_grad_diag", None) if loss_fn is not None else None
        if not stash:
            super().backward(loss, *args, **kwargs)
            return
        loss_fn._grad_diag = None
        precision = str(getattr(self.trainer, "precision", ""))
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        scaler_on = bool(scaler is not None and getattr(scaler, "is_enabled", lambda: True)())
        if scaler_on or precision not in _NO_SCALER_PRECISIONS:
            raise RuntimeError(
                "Grad diagnostics do not support GradScaler. "
                f"Got precision {precision!r}."
            )
        scale = 1.0 / float(self.trainer.accumulate_grad_batches)
        self._split_backward(stash, scale=scale)
        self._pending_grad_logs = self._metrics_from_split()

    def on_after_backward(self) -> None:
        pending = self._pending_grad_logs
        if not pending:
            return
        self._pending_grad_logs = None
        log_kw = dict(
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
            rank_zero_only=True,
            logger=True,
            add_dataloader_idx=False,
        )
        for key, value in pending.items():
            self.log(key, value, **log_kw)

    def _split_backward(self, stash: Dict[str, Any], *, scale: float) -> None:
        """Three backward calls. Leaves ``.grad`` equal to the ordinary ``g``."""
        if not self._hooks_installed:
            self._install_grad_hooks()
        self._captured_gd = None
        self._grad_sq = {}
        through_h = bool(stash["through_h"])
        self._hook_phase = "g1d"
        (stash["L1"] * scale).backward()
        retain = through_h
        self._hook_phase = "g0"
        (stash["L0"] * scale).backward(retain_graph=retain)
        self._gd_sq = self._sq_from_grad()
        if self._capture_split_grads:
            self._captured_gd = self._clone_grads()
        if through_h:
            leaf = stash["leaf"]
            assert leaf is not None and leaf.grad is not None
            self._hook_phase = "gh"
            stash["h1"].backward(leaf.grad.detach())
        self._hook_phase = None
        self._g_sq = self._sq_from_grad()

    def _clone_grads(self) -> Dict[str, Tensor]:
        out = {}
        seen = set()
        for name, param in self.model.named_parameters():
            if param.grad is None or id(param) in seen:
                continue
            seen.add(id(param))
            out[name] = param.grad.detach().clone()
        return out

    def _sq_from_grad(self) -> Dict[str, Tensor]:
        acc: Dict[str, Tensor] = {}
        seen = set()
        total: Optional[Tensor] = None
        for name, param in self.model.named_parameters():
            if param.grad is None or id(param) in seen:
                continue
            seen.add(id(param))
            sq = param.grad.detach().double().pow(2).sum()
            group = param_group_name(name)
            acc[group] = acc[group] + sq if group in acc else sq
            total = sq if total is None else total + sq
        if total is None:
            total = torch.zeros((), dtype=torch.float64)
        acc["total"] = total
        return acc

    def _metrics_from_split(self) -> Dict[str, Tensor]:
        g0 = self._grad_sq.get("g0", {})
        g1d = self._grad_sq.get("g1d", {})
        gh = self._grad_sq.get("gh", {})
        gd = self._gd_sq
        g = self._g_sq
        groups = set(g0) | set(g1d) | set(gh) | set(gd) | set(g)
        logs: Dict[str, Tensor] = {}
        for group in groups:
            z = _zero_like_any(g0, g1d, gh, gd, g)
            n0 = g0.get(group, z)
            n1 = g1d.get(group, z)
            nh = gh.get(group, z)
            nd = gd.get(group, z)
            ng = g.get(group, z)
            logs[f"train/gradB/gd_norm/{group}"] = torch.sqrt(nd)
            logs[f"train/gradB/gh_norm/{group}"] = torch.sqrt(nh)
            logs[f"train/gradB/gh_over_gd/{group}"] = torch.sqrt(nh) / torch.sqrt(nd).clamp_min(1e-12)
            logs[f"train/gradB/cos_gh_gd/{group}"] = _cos_from_sq(nh, nd, ng)
            logs[f"train/gradB/kappa/{group}"] = torch.sqrt(ng) / (
                torch.sqrt(nd) + torch.sqrt(nh)
            ).clamp_min(1e-12)
            logs[f"train/gradB/g1d_over_g0/{group}"] = torch.sqrt(n1) / torch.sqrt(n0).clamp_min(1e-12)
            logs[f"train/gradB/cos_g0_g1d/{group}"] = _cos_from_sq(n0, n1, nd)
        return {k: v.float() for k, v in logs.items()}


def _zero_like_any(*dicts: Dict[str, Tensor]) -> Tensor:
    for d in dicts:
        for v in d.values():
            return v.new_zeros(())
    return torch.zeros((), dtype=torch.float64)


def _cos_from_sq(na: Tensor, nb: Tensor, nab: Tensor) -> Tensor:
    """Cosine of two pieces whose sum has squared norm ``nab``."""
    dot = 0.5 * (nab - na - nb)
    small = (na < 1e-24) | (nb < 1e-24)
    denom = torch.sqrt(na * nb).clamp_min(1e-12)
    return torch.where(small, torch.zeros_like(dot), dot / denom)
