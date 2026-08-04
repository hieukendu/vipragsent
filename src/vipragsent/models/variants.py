from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..constants import RATIONALE_BETA
from .backbones import DummyBackbone, pool_hidden_states
from .heads import ClassificationHeads
from .rationale_decoder import RationaleDecoder


@dataclass(frozen=True)
class VariantConfig:
    name: str
    backbone_family: str = "encoder"
    hidden_size: int = 32
    vocab_size: int = 512
    rationale_vocab_size: int = 512
    rationale_enabled_for_training: bool = False
    rationale_beta: float = RATIONALE_BETA

    @property
    def active_tasks(self) -> set[str]:
        if self.name.startswith("cot_only"):
            return set()
        if self.name.startswith("explanation_only"):
            return {"pragmatic"}
        if self.name.startswith("pragmatic") or self.name == "no_auxiliary":
            return {"pragmatic"}
        return {"pragmatic", "polarity", "emotion"}


class ViPragSentModel(nn.Module):
    def __init__(self, backbone: nn.Module, config: VariantConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.heads = ClassificationHeads(config.hidden_size)
        self.rationale_decoder = RationaleDecoder(config.hidden_size, config.rationale_vocab_size) if config.rationale_enabled_for_training else None

    @property
    def inference_output_source(self) -> str:
        return "parsed_generated_labels" if self.config.name.startswith("cot_only") else "classification_heads"

    @property
    def rationale_decoder_enabled_at_inference(self) -> bool:
        return False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        rationale_input_ids: Tensor | None = None,
    ) -> dict[str, Any]:
        encoded = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoded.last_hidden_state
        pooled = pool_hidden_states(hidden, attention_mask, self.config.backbone_family)
        outputs: dict[str, Any] = {"logits": self.heads(pooled, active_tasks=self.config.active_tasks)}
        if self.rationale_decoder is not None and rationale_input_ids is not None:
            outputs["rationale_logits"] = self.rationale_decoder(rationale_input_ids, hidden)
        return outputs


def build_dummy_model(config: VariantConfig | None = None) -> ViPragSentModel:
    config = config or VariantConfig(name="vipragsent_full", rationale_enabled_for_training=True)
    return ViPragSentModel(DummyBackbone(config.vocab_size, config.hidden_size), config)
