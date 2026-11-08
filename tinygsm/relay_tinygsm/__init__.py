"""relay_tinygsm: RELAY ablation ladder on TinyGSM seq2seq MLM.

Ports the Sudoku rollout buffer, relay carry, and truncated BPTT loss onto the
xlm-core TinyGSM batch contract (attention masks, fixed positions, prefix-only
prediction). Datasets, seq2seq collators, and GSM8K code-exec eval stay in
xlm-core / the mlm package.
"""

from .datamodule import MLMSeq2SeqTrainMaskAllCollator
from .harness import RelayDiagnosticsHarness
from .loss import RelayBPTTSeq2SeqLoss
from .model import RotaryTransformerRelayMLMModel
from .predictor import RelayConfidencePredictor

__all__ = [
    "MLMSeq2SeqTrainMaskAllCollator",
    "RelayDiagnosticsHarness",
    "RelayBPTTSeq2SeqLoss",
    "RotaryTransformerRelayMLMModel",
    "RelayConfidencePredictor",
]
