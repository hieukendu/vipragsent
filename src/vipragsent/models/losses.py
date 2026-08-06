from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..constants import PRAGMATIC_LABELS, RATIONALE_BETA


def _masked_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        return values.mean()
    active = mask.to(device=values.device, dtype=values.dtype)
    denominator = active.sum()
    if denominator.item() == 0:
        return values.sum() * 0.0
    return (values * active).sum() / denominator


def classification_losses(
    logits: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    pragmatic_pos_weight: Mapping[str, float] | None = None,
    polarity_weight: Tensor | None = None,
    emotion_weight: Tensor | None = None,
    active_tasks: set[str] | None = None,
    target_masks: Mapping[str, Tensor] | None = None,
    sarcasm_target_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    if active_tasks is None:
        active_tasks = {"pragmatic", "polarity", "emotion"}
    target_masks = target_masks or {}
    losses: dict[str, Tensor] = {}
    if "pragmatic" in active_tasks:
        for key in PRAGMATIC_LABELS:
            weight = None
            if pragmatic_pos_weight and key in pragmatic_pos_weight:
                weight = torch.as_tensor(float(pragmatic_pos_weight[key]), device=logits[key].device, dtype=logits[key].dtype)
            values = F.binary_cross_entropy_with_logits(logits[key], targets[key].float(), pos_weight=weight, reduction="none")
            mask = sarcasm_target_mask if key == "sarcasm" and sarcasm_target_mask is not None else target_masks.get(key)
            losses[key] = _masked_mean(values, mask)
    if "polarity" in active_tasks:
        values = F.cross_entropy(logits["polarity"], targets["polarity"].long(), weight=polarity_weight, reduction="none")
        losses["polarity"] = _masked_mean(values, target_masks.get("polarity"))
    if "emotion" in active_tasks:
        values = F.cross_entropy(logits["emotion"], targets["emotion"].long(), weight=emotion_weight, reduction="none")
        losses["emotion"] = _masked_mean(values, target_masks.get("emotion"))
    return losses


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """Eight independent homoscedastic parameters and the locked 0.5 coefficients."""

    def __init__(self, rationale_beta: float = RATIONALE_BETA, tasks: Iterable[str] | None = None) -> None:
        super().__init__()
        task_keys = tuple(tasks) if tasks is not None else (*PRAGMATIC_LABELS, "polarity", "emotion")
        allowed = set(PRAGMATIC_LABELS) | {"polarity", "emotion"}
        if not task_keys or not set(task_keys).issubset(allowed) or len(set(task_keys)) != len(task_keys):
            raise ValueError("Uncertainty tasks must be a non-empty subset of canonical classification tasks")
        self.log_variances = nn.ParameterDict({key: nn.Parameter(torch.zeros(())) for key in task_keys})
        self.rationale_beta = rationale_beta

    def forward(self, losses: Mapping[str, Tensor], rationale_loss: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        if losses:
            total = next(iter(losses.values())).new_zeros(())
        elif rationale_loss is not None:
            total = rationale_loss.new_zeros(())
        else:
            total = self.log_variances["polarity"].new_zeros(())
        components: dict[str, Tensor] = {}
        for key, loss in losses.items():
            if key not in self.log_variances:
                raise ValueError(f"Unknown uncertainty task: {key}")
            variance = self.log_variances[key].clamp(-5.0, 5.0)
            weighted = 0.5 * torch.exp(-variance) * loss + 0.5 * variance
            components[key] = weighted
            total = total + weighted
        if rationale_loss is not None:
            components["rationale"] = self.rationale_beta * rationale_loss
            total = total + components["rationale"]
        return total, components


def equal_weight_loss(
    losses: Mapping[str, Tensor],
    rationale_loss: Tensor | None = None,
    *,
    rationale_beta: float = RATIONALE_BETA,
    reference: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if losses:
        total = next(iter(losses.values())).new_zeros(())
    elif rationale_loss is not None:
        total = rationale_loss.new_zeros(())
    elif reference is not None:
        total = reference.new_zeros(())
    else:
        raise ValueError("A device reference is required when no classification or rationale loss exists")
    components = dict(losses)
    total = total + sum(losses.values(), total.new_zeros(()))
    if rationale_loss is not None:
        components["rationale"] = rationale_beta * rationale_loss
        total = total + components["rationale"]
    return total, components


def token_cross_entropy(
    logits: Tensor,
    target_ids: Tensor,
    ignore_index: int = -100,
    sample_mask: Tensor | None = None,
) -> Tensor:
    if logits.ndim != 3 or target_ids.ndim != 2:
        raise ValueError("Expected logits [batch, target, vocab] and target IDs [batch, target]")
    values = F.cross_entropy(logits.transpose(1, 2), target_ids, ignore_index=ignore_index, reduction="none", label_smoothing=0.0)
    active = target_ids.ne(ignore_index)
    if sample_mask is not None:
        if sample_mask.ndim != 1 or sample_mask.size(0) != target_ids.size(0):
            raise ValueError("Rationale sample mask must have shape [batch]")
        active = active & sample_mask.to(device=target_ids.device, dtype=torch.bool)[:, None]
    denominator = active.sum()
    if denominator.item() == 0:
        return values.sum() * 0.0
    return values.masked_select(active).sum() / denominator
