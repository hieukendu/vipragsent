from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    local_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    transformers_module: Any | None = None,
) -> nn.Module:
    """Load only the base transformer; no unused pretrained LM head is allocated."""
    if not revision:
        raise ValueError("An immutable model revision is required")
    try:
        transformers = transformers_module or __import__("transformers")
        AutoModel = transformers.AutoModel
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("transformers is required for real model loading") from exc
    if trust_remote_code:
        raise ValueError("trust_remote_code is prohibited without a reviewed ADR")
    source = str(local_path) if local_path else repo_id
    if local_path is not None and not Path(local_path).exists():
        raise RuntimeError(f"Pinned local model snapshot is missing: {local_path}")
    try:
        model = AutoModel.from_pretrained(
            source,
            revision=revision,
            trust_remote_code=False,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only or local_path is not None,
            output_hidden_states=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to load pinned backbone {repo_id}@{revision}: {exc}") from exc
    model._vipragsent_backbone_family = family
    return model


def load_pretrained_causal_lm(
    repo_id: str,
    *,
    revision: str,
    trust_remote_code: bool = False,
    local_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    transformers_module: Any | None = None,
) -> nn.Module:
    """Load the native causal-LM interface used by the approved CoT system."""
    if not revision:
        raise ValueError("An immutable model revision is required")
    try:
        transformers = transformers_module or __import__("transformers")
        AutoModelForCausalLM = transformers.AutoModelForCausalLM
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("transformers with AutoModelForCausalLM is required for causal-LM loading") from exc
    if trust_remote_code:
        raise ValueError("trust_remote_code is prohibited without a reviewed ADR")
    source = str(local_path) if local_path else repo_id
    if local_path is not None and not Path(local_path).exists():
        raise RuntimeError(f"Pinned local model snapshot is missing: {local_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            revision=revision,
            trust_remote_code=False,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only or local_path is not None,
            output_hidden_states=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to load pinned causal LM {repo_id}@{revision}: {exc}") from exc
    if not callable(getattr(model, "generate", None)):
        raise RuntimeError("the pinned causal-LM does not expose generate()")
    if not callable(getattr(model, "forward", None)):
        raise RuntimeError("the pinned causal-LM does not expose forward()")
    model._vipragsent_backbone_family = "causal"
    model._vipragsent_causal_lm = True
    return model
