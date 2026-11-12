"""CPU checks for the raw top-prob cumsum MLM predictor."""

import math

import torch

from relay_tinygsm.cum_top_prob_mlm_predictor import CumTopProbMLMPredictor


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


def _predictor(threshold: float, **kwargs) -> CumTopProbMLMPredictor:
    return CumTopProbMLMPredictor(
        max_steps=8,
        tokenizer=StubTokenizer(),
        confidence="top_prob",
        threshold=threshold,
        top_k=1,
        **kwargs,
    )


def _logits_for_probs(probs, vocab_size: int = 10) -> torch.Tensor:
    """Logits whose max-softmax equals each probability.

    The winning class is 0. The other classes share the remaining mass, so a
    target below 0.5 is reachable.
    """
    logits = torch.zeros(1, len(probs), vocab_size, dtype=torch.float64)
    others = vocab_size - 1
    for index, prob in enumerate(probs):
        logits[0, index, 0] = math.log(prob * others / (1.0 - prob))
    return logits


def test_cumsum_budget_selects_highest_confidence_first():
    predictor = _predictor(0.15)
    logits = _logits_for_probs([0.9, 0.5, 0.2])
    masked = torch.ones(1, 3, dtype=torch.bool)

    low = predictor.compute_confidence_unmask(logits, masked, threshold=0.05)
    mid = predictor.compute_confidence_unmask(logits, masked, threshold=0.15)
    high = predictor.compute_confidence_unmask(logits, masked, threshold=0.7)

    assert low.tolist() == [[True, False, False]]
    assert mid.tolist() == [[True, False, False]]
    assert high.tolist() == [[True, True, False]]


def test_non_mutable_position_is_never_selected():
    predictor = _predictor(0.7)
    logits = _logits_for_probs([0.99, 0.5, 0.2])
    masked = torch.tensor([[False, True, True]])

    unmask = predictor.compute_confidence_unmask(logits, masked, threshold=0.7)

    assert unmask.tolist() == [[False, True, False]]


def test_done_sample_unmasks_nothing():
    predictor = _predictor(0.7)
    logits = _logits_for_probs([0.9, 0.5, 0.2])
    masked = torch.ones(1, 3, dtype=torch.bool)
    done = torch.tensor([True])

    unmask = predictor.compute_confidence_unmask(
        logits, masked, threshold=0.7, done=done
    )

    assert unmask.tolist() == [[False, False, False]]


def test_predict_appends_masks_and_counts_the_final_fill():
    predictor = _predictor(0.15, max_new_tokens=3, model=StubModel())
    prompt = torch.tensor([[7, 8], [7, 8]])
    batch = {
        "input_ids": prompt.clone(),
        "attention_mask": torch.ones_like(prompt, dtype=torch.bool),
    }

    preds = predictor.predict(batch)

    assert preds["output_start_idx"] == 2
    assert preds["ids"].shape == (2, 5)
    assert torch.equal(preds["ids"][:, :2], prompt)
    assert torch.all(preds["ids"][:, 2:] == 0)
    assert len(preds["steps_taken"]) == 2
    assert preds["steps_taken"] == [2, 2]


if __name__ == "__main__":
    test_cumsum_budget_selects_highest_confidence_first()
    test_non_mutable_position_is_never_selected()
    test_done_sample_unmasks_nothing()
    test_predict_appends_masks_and_counts_the_final_fill()
    print("ok")
