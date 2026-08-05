from __future__ import annotations

import torch
from torch import Tensor, nn

from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS


class ClassificationHeads(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1, active_tasks: set[str] | None = None) -> None:
        super().__init__()
        active_tasks = {"pragmatic", "polarity", "emotion"} if active_tasks is None else set(active_tasks)
        self.dropout = nn.Dropout(dropout)
        self.pragmatic = nn.ModuleDict({key: nn.Linear(hidden_size, 1) for key in PRAGMATIC_LABELS}) if "pragmatic" in active_tasks else nn.ModuleDict()
        self.polarity = nn.Linear(hidden_size, len(POLARITY_LABELS)) if "polarity" in active_tasks else None
        self.emotion = nn.Linear(hidden_size, len(EMOTION_LABELS)) if "emotion" in active_tasks else None

    def forward(self, pooled: Tensor, *, active_tasks: set[str] | None = None) -> dict[str, Tensor]:
        active_tasks = {"pragmatic", "polarity", "emotion"} if active_tasks is None else active_tasks
        outputs: dict[str, Tensor] = {}
        if "pragmatic" in active_tasks:
            outputs.update({key: layer(self.dropout(pooled)).squeeze(-1) for key, layer in self.pragmatic.items()})
        if "polarity" in active_tasks and self.polarity is not None:
            outputs["polarity"] = self.polarity(self.dropout(pooled))
        if "emotion" in active_tasks and self.emotion is not None:
            outputs["emotion"] = self.emotion(self.dropout(pooled))
        return outputs
