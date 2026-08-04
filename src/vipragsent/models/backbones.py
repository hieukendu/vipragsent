from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    family: str
    hidden_size: int
    revision: str | None = None


class DummyBackbone(nn.Module):
    """Small deterministic-compatible backbone used by tests and fixture runs."""

    def __init__(self, vocab_size: int = 512, hidden_size: int = 32) -> None:
        super().__init__()
        self.config = type("DummyConfig", (), {"hidden_size": hidden_size, "vocab_size": vocab_size})()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **_: Any) -> Any:
        hidden = self.activation(self.projection(self.embedding(input_ids)))
        return type("BackboneOutput", (), {"last_hidden_state": hidden})()


def pool_hidden_states(hidden: Tensor, attention_mask: Tensor, family: str) -> Tensor:
    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden must be [batch, sequence, hidden] and attention_mask must be [batch, sequence]")
    mask = attention_mask.to(dtype=torch.bool)
    if family in {"encoder", "phobert", "xlmr"}:
        indices = mask.long().argmax(dim=1)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), indices]
    if family in {"causal", "sailor", "vistral"}:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (hidden * weights).sum(dim=1) / denominator
    raise ValueError(f"Unknown backbone family: {family}")


def load_pretrained_backbone(
    repo_id: str,
    *,
    revision: str,
    family: str,
    trust_remote_code: bool = False,
) -> nn.Module:
    """Load only the base transformer; no unused pretrained LM head is allocated."""
    if not revision:
        raise ValueError("An immutable model revision is required")
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("transformers is required for real model loading") from exc
    model = AutoModel.from_pretrained(repo_id, revision=revision, trust_remote_code=trust_remote_code)
    model._vipragsent_backbone_family = family
    return model
