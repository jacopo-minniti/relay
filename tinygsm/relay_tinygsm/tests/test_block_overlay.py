"""CPU checks for the contiguous block overlay on the relay predictor."""

import torch

from relay_tinygsm.predictor import RelayConfidencePredictor, contiguous_block_allow


class StubTokenizer:
    mask_token_id = 1

    def batch_decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(str(int(token)) for token in row) for row in ids]


class StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.force_flash_attn = False

    def forward(self, x, attention_mask, positions):
        del attention_mask, positions
        logits = torch.zeros(*x.shape, 2)
        logits[..., 0] = 10.0
        return logits


def _predictor(**kwargs) -> RelayConfidencePredictor:
    return RelayConfidencePredictor(
        max_steps=8,
        tokenizer=StubTokenizer(),
        confidence="top_prob",
        threshold=0.15,
        top_k=1,
        with_relay=False,
        model=StubModel(),
        **kwargs,
    )


def test_allow_keeps_only_the_first_incomplete_block():
    fixed = torch.tensor([[True, True, False, False, False, False, False, False]])
    masked = torch.tensor([[False, False, True, True, True, True, True, True]])
    allow = contiguous_block_allow(masked, fixed, block_size=4)
    assert allow[0].tolist() == [False, False, True, True, True, True, False, False]


def test_allow_opens_the_next_block_once_the_first_is_full():
    fixed = torch.tensor([[True, True, False, False, False, False, False, False]])
    masked = torch.tensor([[False, False, False, False, False, False, True, True]])
    allow = contiguous_block_allow(masked, fixed, block_size=4)
    assert allow[0].tolist() == [False, False, False, False, False, False, True, True]


def test_allow_keeps_holes_inside_the_active_block():
    fixed = torch.tensor([[True, True, False, False, False, False, False, False]])
    masked = torch.tensor([[False, False, False, True, False, True, True, True]])
    allow = contiguous_block_allow(masked, fixed, block_size=4)
    assert allow[0].tolist() == [False, False, True, True, True, True, False, False]


def test_allow_uses_each_rows_suffix_start():
    fixed = torch.tensor(
        [
            [True, False, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False],
        ]
    )
    masked = ~fixed
    allow = contiguous_block_allow(masked, fixed, block_size=4)
    assert allow[0].tolist() == [False, True, True, True, True, False, False, False]
    assert allow[1].tolist() == [False, False, False, True, True, True, True, False]


def test_overlay_ignores_a_confident_token_past_the_block():
    predictor = _predictor(allow_block_size=4)
    logits = torch.zeros(1, 8, 2)
    logits[0, 3, 0] = 0.4
    logits[0, 2, 0] = 0.1
    logits[0, 4, 0] = 0.1
    logits[0, 5, 0] = 0.1
    logits[0, 6, 0] = 10.0
    logits[0, 7, 0] = 10.0
    masked = torch.tensor([[False, False, True, True, True, True, True, True]])
    fixed = torch.tensor([[True, True, False, False, False, False, False, False]])
    unmask = predictor.compute_confidence_unmask(logits, masked, fixed=fixed)
    assert unmask[0].tolist() == [False, False, False, True, False, False, False, False]


def test_no_overlay_selects_the_later_token():
    predictor = _predictor(allow_block_size=None)
    logits = torch.zeros(1, 8, 2)
    logits[0, 3, 0] = 0.4
    logits[0, 6, 0] = 10.0
    logits[0, 7, 0] = 10.0
    masked = torch.tensor([[False, False, True, True, True, True, True, True]])
    fixed = torch.tensor([[True, True, False, False, False, False, False, False]])
    unmask = predictor.compute_confidence_unmask(logits, masked, fixed=fixed)
    assert unmask[0, 6] and unmask[0, 7]
    assert not unmask[0, 3]


def test_overlay_requires_fixed():
    predictor = _predictor(allow_block_size=4)
    logits = torch.zeros(1, 4, 2)
    masked = torch.ones(1, 4, dtype=torch.bool)
    try:
        predictor.compute_confidence_unmask(logits, masked)
    except ValueError as exc:
        assert "fixed" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_final_step_fills_masks_outside_the_active_block():
    predictor = _predictor(allow_block_size=4)
    x = torch.tensor([[2, 2, 1, 1, 1, 1, 1, 1]])
    fixed = torch.tensor([[True, True, False, False, False, False, False, False]])
    attention = torch.ones_like(x, dtype=torch.bool)
    step = {
        "x": x,
        "fixed": fixed,
        "attention_mask": attention,
        "positions": torch.arange(x.shape[-1]).unsqueeze(0),
        "steps_taken": torch.zeros(1, dtype=torch.int),
        "done": torch.zeros(1, dtype=torch.bool),
    }
    out = predictor.predict_single_step(step, final_step=True)
    assert (out["x"][0, 2:] != 1).all()


def test_one_step_stays_inside_the_active_block():
    predictor = _predictor(allow_block_size=4)
    x = torch.tensor([[2, 2, 1, 1, 1, 1, 1, 1]])
    fixed = torch.tensor([[True, True, False, False, False, False, False, False]])

    class Biased(StubModel):
        def forward(self, x, attention_mask, positions):
            del attention_mask, positions
            logits = torch.zeros(*x.shape, 2)
            logits[..., 0] = 0.1
            logits[:, 3, 0] = 0.4
            logits[:, 6:, 0] = 10.0
            return logits

    predictor.model = Biased()
    attention = torch.ones_like(x, dtype=torch.bool)
    step = {
        "x": x,
        "fixed": fixed,
        "attention_mask": attention,
        "positions": torch.arange(x.shape[-1]).unsqueeze(0),
        "steps_taken": torch.zeros(1, dtype=torch.int),
        "done": torch.zeros(1, dtype=torch.bool),
    }
    out = predictor.predict_single_step(step, final_step=False)
    assert out["x"][0, 3].item() != 1
    assert out["x"][0, 6].item() == 1
    assert out["x"][0, 7].item() == 1
