from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..constants import PRAGMATIC_LABELS, RATIONALE_BETA
from .backbones import DummyBackbone, pool_hidden_states
from .heads import ClassificationHeads
from .rationale_decoder import RationaleDecoder

VARIANT_IDS = {
    "full",
    "no_emotion_auxiliary",
    "no_polarity_auxiliary",
    "no_rationale",
    "no_multitask",
    "no_uncertainty_weighting",
    "vipragsent_full",
    "vipragsent_full_phobert",
    "vipragsent_full_vistral",
    "phobert_pragmatic_finetune",
    "phobert_pragmatic_single_task",
    "xlmr_pragmatic_finetune",
    "vistral_pragmatic_sft",
    "vipragsent_no_auxiliary_vistral",
    "sailor_pragmatic_sft",
    "phobert_multitask_8head",
    "xlmr_multitask_8head",
    "sailor_multitask_8head",
    "vistral_multitask_8head",
    "phobert_pol_single",
    "phobert_emo_single",
    "cot_only_vistral",
    "explanation_only_vistral",
}


@dataclass(frozen=True)
class VariantConfig:
    name: str
    backbone_family: str = "encoder"
    hidden_size: int = 32
    vocab_size: int = 512
    rationale_vocab_size: int = 512
    rationale_enabled_for_training: bool | None = None
    rationale_beta: float = RATIONALE_BETA
    use_uncertainty_weighting: bool | None = None

    def __post_init__(self) -> None:
        if self.name not in VARIANT_IDS:
            raise ValueError(f"Unknown locked variant ID: {self.name}")

    @property
    def active_tasks(self) -> set[str]:
        if self.name in {"cot_only_vistral", "explanation_only_vistral"}:
            return set()
        if self.name in {"phobert_pol_single"}:
            return {"polarity"}
        if self.name in {"phobert_emo_single"}:
            return {"emotion"}
        if self.name in {"phobert_pragmatic_finetune", "phobert_pragmatic_single_task", "xlmr_pragmatic_finetune", "sailor_pragmatic_sft", "vistral_pragmatic_sft", "vipragsent_no_auxiliary_vistral"}:
            return {"pragmatic"}
        if self.name == "no_emotion_auxiliary":
            return {"pragmatic", "polarity"}
        if self.name == "no_polarity_auxiliary":
            return {"pragmatic", "emotion"}
        if self.name == "no_multitask":
            return {"pragmatic", "polarity", "emotion"}
        return {"pragmatic", "polarity", "emotion"}

    @property
    def has_rationale_decoder(self) -> bool:
        if self.rationale_enabled_for_training is not None:
            return self.rationale_enabled_for_training
        return self.name in {"full", "vipragsent_full", "vipragsent_full_phobert", "vipragsent_full_vistral", "no_emotion_auxiliary", "no_polarity_auxiliary", "no_uncertainty_weighting"}

    @property
    def has_uncertainty_weighting(self) -> bool:
        if self.use_uncertainty_weighting is not None:
            return self.use_uncertainty_weighting
        return self.name in {
            "full",
            "vipragsent_full",
            "vipragsent_full_phobert",
            "vipragsent_full_vistral",
            "no_emotion_auxiliary",
            "no_polarity_auxiliary",
            "no_rationale",
            "vipragsent_no_auxiliary_vistral",
        }

    @property
    def loss_aggregation(self) -> str:
        return "homoscedastic_uncertainty" if self.has_uncertainty_weighting else "equal_weight"

    @property
    def uncertainty_task_keys(self) -> tuple[str, ...]:
        if not self.has_uncertainty_weighting:
            return ()
        if self.name == "vipragsent_no_auxiliary_vistral":
            return tuple(PRAGMATIC_LABELS)
        keys: list[str] = []
        if "pragmatic" in self.active_tasks:
            keys.extend(PRAGMATIC_LABELS)
        if "polarity" in self.active_tasks:
            keys.append("polarity")
        if "emotion" in self.active_tasks:
            keys.append("emotion")
        return tuple(keys)

    @property
    def is_checkpoint_bundle(self) -> bool:
        return self.name == "no_multitask"


class ViPragSentModel(nn.Module):
    def __init__(self, backbone: nn.Module, config: VariantConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.heads = ClassificationHeads(config.hidden_size, active_tasks=config.active_tasks)
        self.rationale_decoder = RationaleDecoder(config.hidden_size, config.rationale_vocab_size) if config.has_rationale_decoder else None

    @property
    def inference_output_source(self) -> str:
        return "classification_heads"

    @property
    def rationale_decoder_enabled_at_inference(self) -> bool:
        return False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        rationale_input_ids: Tensor | None = None,
        rationale_attention_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        encoded = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoded.last_hidden_state
        pooled = pool_hidden_states(hidden, attention_mask, self.config.backbone_family)
        outputs: dict[str, Any] = {"logits": self.heads(pooled, active_tasks=self.config.active_tasks)}
        if self.training and self.rationale_decoder is not None and rationale_input_ids is not None:
            target_attention = rationale_attention_mask if rationale_attention_mask is not None else torch.ones_like(rationale_input_ids)
            outputs["rationale_logits"], outputs["rationale_labels"], outputs["rationale_padding_mask"] = self.rationale_decoder.teacher_forcing(
                rationale_input_ids, target_attention, hidden, attention_mask
            )
        return outputs


class SingleTaskClassifier(nn.Module):
    """One independent checkpoint component used by the no-multitask bundle."""

    def __init__(self, backbone: nn.Module, config: VariantConfig, *, output_key: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.output_key = output_key
        output_size = 1 if output_key in {"implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking"} else 3 if output_key == "polarity" else 7
        self.classifier = nn.Linear(config.hidden_size, output_size)

    @property
    def active_head_keys(self) -> tuple[str, ...]:
        return (self.output_key,)

    @property
    def inference_output_source(self) -> str:
        return "classification_heads"

    @property
    def rationale_decoder_enabled_at_inference(self) -> bool:
        return False

    def forward(self, input_ids: Tensor, attention_mask: Tensor, **_: Tensor | None) -> dict[str, Any]:
        encoded = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool_hidden_states(encoded.last_hidden_state, attention_mask, self.config.backbone_family)
        logits = self.classifier(pooled)
        if logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return {"logits": {self.output_key: logits}}


class IndependentCheckpointBundle(nn.Module):
    """The no-multitask variant is a bundle of independent checkpoint components."""

    def __init__(self, backbone_factory: Callable[[], nn.Module], config: VariantConfig) -> None:
        super().__init__()
        self.config = config
        pragmatic_keys = ("implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking")
        components: dict[str, nn.Module] = {}
        for key in pragmatic_keys:
            components[f"pragmatic_{key}"] = SingleTaskClassifier(
                backbone_factory(),
                VariantConfig("phobert_pragmatic_finetune", config.backbone_family, config.hidden_size, config.vocab_size),
                output_key=key,
            )
        components["polarity"] = SingleTaskClassifier(
            backbone_factory(),
            VariantConfig("phobert_pol_single", config.backbone_family, config.hidden_size, config.vocab_size),
            output_key="polarity",
        )
        components["emotion"] = SingleTaskClassifier(
            backbone_factory(),
            VariantConfig("phobert_emo_single", config.backbone_family, config.hidden_size, config.vocab_size),
            output_key="emotion",
        )
        self.components = nn.ModuleDict(components)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **_: Tensor | None,
    ) -> dict[str, Any]:
        logits: dict[str, Tensor] = {}
        for component in self.components.values():
            output = component(input_ids, attention_mask)
            logits.update(output["logits"])
        return {"logits": logits}

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        return tuple([f"phobert_{key}_single" for key in ("implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking")] + ["phobert_pol_single", "phobert_emo_single"])

    @property
    def active_head_keys(self) -> tuple[str, ...]:
        return tuple(component.active_head_keys[0] for component in self.components.values())  # type: ignore[attr-defined]


class SingleTaskPragmaticBundle(nn.Module):
    """Six independent pragmatic heads for the approved ordinary single-task baseline."""

    def __init__(self, backbone_factory: Callable[[], nn.Module], config: VariantConfig) -> None:
        super().__init__()
        self.config = config
        self.components = nn.ModuleDict(
            {
                key: SingleTaskClassifier(
                    backbone_factory(),
                    VariantConfig("phobert_pragmatic_finetune", config.backbone_family, config.hidden_size, config.vocab_size),
                    output_key=key,
                )
                for key in PRAGMATIC_LABELS
            }
        )

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(f"phobert_{key}_single" for key in PRAGMATIC_LABELS)

    @property
    def active_head_keys(self) -> tuple[str, ...]:
        return tuple(PRAGMATIC_LABELS)

    @property
    def inference_output_source(self) -> str:
        return "classification_heads"

    @property
    def rationale_decoder_enabled_at_inference(self) -> bool:
        return False

    def forward(self, input_ids: Tensor, attention_mask: Tensor, **_: Tensor | None) -> dict[str, Any]:
        logits: dict[str, Tensor] = {}
        for component in self.components.values():
            logits.update(component(input_ids, attention_mask)["logits"])
        return {"logits": logits}


class GenerationBaselineModel(nn.Module):
    """Production construction marker for generation baselines.

    Label parsing and generation are owned by the generation executor; this
    model deliberately exposes no classification heads.
    """

    def __init__(self, backbone: nn.Module, config: VariantConfig, *, fixture_compatibility: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.fixture_compatibility = fixture_compatibility
        if not fixture_compatibility and not callable(getattr(backbone, "generate", None)):
            raise RuntimeError("production generation baseline requires a native causal-LM generate() method")

    @property
    def inference_output_source(self) -> str:
        if self.fixture_compatibility:
            return "parsed_generated_labels"
        return "judge_of_generated_reasoning"

    @property
    def rationale_decoder_enabled_at_inference(self) -> bool:
        return False

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        generator = getattr(self.backbone, "generate", None)
        if not callable(generator):
            raise RuntimeError("production generation baseline does not expose generate()")
        return generator(*args, **kwargs)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, *, labels: Tensor | None = None, **kwargs: Tensor | None) -> Any:
        if not self.fixture_compatibility:
            inputs: dict[str, Any] = {"input_ids": input_ids, "labels": labels, **kwargs}
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask
            return self.backbone(**inputs)
        encoded = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return {"generation_hidden_states": encoded.last_hidden_state, "logits": {}}


def build_dummy_model(config: VariantConfig | None = None) -> nn.Module:
    config = config or VariantConfig(name="vipragsent_full", rationale_enabled_for_training=True)
    if config.name == "no_multitask":
        return IndependentCheckpointBundle(lambda: DummyBackbone(config.vocab_size, config.hidden_size), config)
    if config.name == "phobert_pragmatic_single_task":
        return SingleTaskPragmaticBundle(lambda: DummyBackbone(config.vocab_size, config.hidden_size), config)
    if config.name in {"cot_only_vistral", "explanation_only_vistral"}:
        return GenerationBaselineModel(DummyBackbone(config.vocab_size, config.hidden_size), config, fixture_compatibility=True)
    return ViPragSentModel(DummyBackbone(config.vocab_size, config.hidden_size), config)
