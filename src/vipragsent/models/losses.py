from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..constants import PRAGMATIC_LABELS, RATIONALE_BETA


def classification_losses(
    logits: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    pragmatic_pos_weight: Mapping[str, float] | None = None,
    polarity_weight: Tensor | None = None,
    emotion_weight: Tensor | None = None,
    active_tasks: set[str] | None = None,
) -> dict[str, Tensor]:
    active_tasks = active_tasks or {"pragmatic", "polarity", "emotion"}
    losses: dict[str, Tensor] = {}
    if "pragmatic" in active_tasks:
        for key in PRAGMATIC_LABELS:
            weight = None
            if pragmatic_pos_weight and key in pragmatic_pos_weight:
                weight = torch.tensor(float(pragmatic_pos_weight[key]), device=logits[key].device)
            losses[key] = F.binary_cross_entropy_with_logits(logits[key], targets[key].float(), pos_weight=weight)
    if "polarity" in active_tasks:
        losses["polarity"] = F.cross_entropy(logits["polarity"], targets["polarity"].long(), weight=polarity_weight)
    if "emotion" in active_tasks:
        losses["emotion"] = F.cross_entropy(logits["emotion"], targets["emotion"].long(), weight=emotion_weight)
    return losses


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """Eight independent homoscedastic uncertainty parameters plus a fixed rationale coefficient."""

    def __init__(self, rationale_beta: float = RATIONALE_BETA) -> None:
        super().__init__()
        self.log_variances = nn.ParameterDict({key: nn.Parameter(torch.zeros(())) for key in (*PRAGMATIC_LABELS, "polarity", "emotion")})
        self.rationale_beta = rationale_beta

    def forward(self, losses: Mapping[str, Tensor], rationale_loss: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        total = next(iter(losses.values())).new_zeros(()) if losses else torch.zeros(())
        components: dict[str, Tensor] = {}
        for key, loss in losses.items():
            if key not in self.log_variances:
                raise ValueError(f"Unknown uncertainty task: {key}")
            precision = torch.exp(-self.log_variances[key])
            weighted = precision * loss + self.log_variances[key]
            components[key] = weighted
            total = total + weighted
        if rationale_loss is not None:
            components["rationale"] = self.rationale_beta * rationale_loss
            total = total + components["rationale"]
        return total, components


def token_cross_entropy(logits: Tensor, target_ids: Tensor, ignore_index: int = -100) -> Tensor:
    if logits.ndim != 3 or target_ids.ndim != 2:
        raise ValueError("Expected logits [batch, target, vocab] and target IDs [batch, target]")
    return F.cross_entropy(logits.transpose(1, 2), target_ids, ignore_index=ignore_index, label_smoothing=0.0)
