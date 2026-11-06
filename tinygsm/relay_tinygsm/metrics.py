from typing import Any, Dict


def rollout_steps_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    """Extract mean rollout steps (NFE) from the predictor output."""
    return {"value": loss_dict["rollout_steps"]}


def mean_metric_update_fn(
    batch: Dict[str, Any], loss_dict: Dict[str, Any], tokenizer: Any = None
) -> Dict[str, Any]:
    return {"value": loss_dict["loss"]}
