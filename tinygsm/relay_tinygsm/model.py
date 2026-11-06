from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT

from xlm.modules.rotary_transformer import (
    RotaryEmbedding,
    RotaryTransformerFinalLayer,
    RotaryTransformerLayer,
    RotaryTransformerLayerList,
)
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class RotaryTransformerRelayMLMModel(torch.nn.Module):
    """Rotary transformer with a learned relay, seq2seq attention aware.

    On each forward the model receives the previous-step hidden state ``h_t``
    and returns ``(logits, h_s)`` where ``h_s`` is the hidden state at
    ``relay_layer`` (default ``-1`` = last layer).

    Injection: ``x = token_emb + LayerNorm(h_t)`` before encoder layer 0.
    ``attention_mask`` and ``positions`` are forwarded to every block so left
    pads (prediction) and the right canvas tail (training) are not attended.
    """

    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
        final_layer_without_normalization: bool = False,
        weight_decay_all_params: bool = False,
        relay_layer: int = -1,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.weight_decay_all_params = weight_decay_all_params
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.d_model = d_model
        self.num_layers = num_layers
        self.force_flash_attn = force_flash_attn
        self._relay_layer_idx = relay_layer % num_layers
        self.embed_tokens = nn.Embedding(
            num_embeddings, d_model, padding_idx=padding_idx
        )
        self.relay_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dim_feedforward = dim_feedforward or 4 * d_model
        encoder_layer = RotaryTransformerLayer(
            d_model,
            nhead,
            self.dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            force_flash_attn=force_flash_attn,
        )
        self.max_length = max_length
        self.encoder = RotaryTransformerLayerList.from_layer(
            encoder_layer,
            num_layers,
            RotaryEmbedding(
                rotary_emb_dim, head_first=True, cache_size=max_length
            ),
        )
        self.output_layer = RotaryTransformerFinalLayer(
            d_model,
            num_embeddings,
            layer_norm_eps,
            use_final_layer_norm=not final_layer_without_normalization,
            zero_init=False,
        )
        if tie_embeddings:
            nn.init.normal_(
                self.embed_tokens.weight, mean=0.0, std=d_model ** -0.5
            )
            with torch.no_grad():
                self.embed_tokens.weight[padding_idx].zero_()
            self.output_layer.linear.weight = self.embed_tokens.weight

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        h_t: Float[TT, " *batch seq_len d_model"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        block_mask: Any = None,
    ) -> Tuple[
        Float[TT, " *batch seq_len vocab_size"],
        Float[TT, " *batch seq_len d_model"],
    ]:
        del block_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(x_t, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)
        if positions is None:
            positions = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            positions = positions * attention_mask.long()

        x = self.embed_tokens(x_t) + self.relay_layer_norm(h_t)

        h_s: Optional[torch.Tensor] = None
        for i, block in enumerate(self.encoder):
            x = block(x, attention_mask, positions=positions)
            if i == self._relay_layer_idx:
                h_s = x

        vocab_logits = self.output_layer(x)
        return vocab_logits, h_s

    def get_named_params_for_weight_decay(self):
        for name, param in self.named_parameters():
            if not self.weight_decay_all_params and (
                "bias" in name or "norm" in name
            ):
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        if self.weight_decay_all_params:
            return
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                yield (name, param)
