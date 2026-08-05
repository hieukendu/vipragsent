from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from torch import nn

from ..orchestration.status import RuntimeBlocked
from ..runtime.device import (
    place_non_quantized_model,
    place_task_modules,
    resolve_selected_cuda_device,
)
from .backbones import load_pretrained_backbone
from .qlora import build_qlora_backbone
from .variants import (
    GenerationBaselineModel,
    IndependentCheckpointBundle,
    SingleTaskClassifier,
    SingleTaskPragmaticBundle,
    VariantConfig,
    ViPragSentModel,
)


@dataclass(frozen=True)
class ModelRuntimeSpec:
    backbone: str
    repo_id: str
    revision: str
    tokenizer_revision: str
    architecture: str
    quantization: str
    trust_remote_code: bool
    pooling: str


def load_model_registry(path: str | Path = "configs/models/model_registry.yaml") -> dict[str, ModelRuntimeSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    specs: dict[str, ModelRuntimeSpec] = {}
    for name, item in raw.get("models", {}).items():
        if not item.get("repo_id") or not item.get("revision") or not item.get("tokenizer_revision"):
            raise ValueError(f"Model {name} is not pinned")
        specs[name] = ModelRuntimeSpec(
            name,
            item["repo_id"],
            item["revision"],
            item["tokenizer_revision"],
            item["architecture"],
            item.get("quantization", "none"),
            bool(item.get("trust_remote_code", False)),
            "first_non_padding_bos" if item["architecture"] == "encoder" else "attention_mask_mean",
        )
    return specs


def build_production_model(
    backbone: str,
    variant: str,
    *,
    registry_path: str | Path = "configs/models/model_registry.yaml",
    local_snapshot: str | Path | None = None,
    execution_mode: str = "production",
    hidden_size: int | None = None,
    vocab_size: int | None = None,
    selected_device: str | int | None = None,
) -> tuple[nn.Module, ModelRuntimeSpec]:
    specs = load_model_registry(registry_path)
    if backbone not in specs:
        raise ValueError(f"Unknown locked backbone {backbone}")
    spec = specs[backbone]
    if execution_mode == "fixture":
        raise ValueError("Use build_dummy_model explicitly for fixture mode")
    if not local_snapshot:
        raise RuntimeBlocked(f"Pinned local snapshot is required before loading {backbone}")
    family = "encoder" if spec.architecture == "encoder" else "causal"
    selected = resolve_selected_cuda_device(selected_device)

    def load_base() -> nn.Module:
        if spec.quantization == "nf4":
            return build_qlora_backbone(
                spec.repo_id,
                revision=spec.revision,
                local_path=str(local_snapshot),
                selected_device=selected,
            )
        return load_pretrained_backbone(
            spec.repo_id,
            revision=spec.revision,
            family=family,
            trust_remote_code=spec.trust_remote_code,
            local_path=local_snapshot,
            local_files_only=True,
        )

    base = load_base()
    config = VariantConfig(
        name=variant,
        backbone_family=family,
        hidden_size=hidden_size or int(getattr(base.config, "hidden_size", 0)),
        vocab_size=vocab_size or int(getattr(base.config, "vocab_size", 0)),
        rationale_vocab_size=vocab_size or int(getattr(base.config, "vocab_size", 0)),
    )
    if config.hidden_size <= 0 or config.vocab_size <= 0:
        raise RuntimeBlocked("Loaded backbone did not expose hidden_size and vocab_size")
    if variant == "phobert_pragmatic_single_task":
        model = SingleTaskPragmaticBundle(load_base, config)
    elif variant == "no_multitask":
        model = IndependentCheckpointBundle(load_base, config)
    elif variant == "cot_only_vistral":
        model = GenerationBaselineModel(base, config)
    elif variant == "explanation_only_vistral":
        full_config = VariantConfig(
            name="vipragsent_full_vistral",
            backbone_family=family,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            rationale_vocab_size=config.vocab_size,
        )
        model = ViPragSentModel(base, full_config)
        model.baseline_variant = "explanation_only_vistral"  # type: ignore[attr-defined]
    else:
        model = ViPragSentModel(base, config)
    if spec.quantization == "nf4":
        place_task_modules(model, selected)
    else:
        place_non_quantized_model(model, selected, model_family=backbone)
    return model, spec


def build_production_component_model(
    backbone: str,
    component: str,
    *,
    registry_path: str | Path = "configs/models/model_registry.yaml",
    local_snapshot: str | Path | None = None,
    execution_mode: str = "production",
    selected_device: str | int | None = None,
) -> tuple[nn.Module, ModelRuntimeSpec]:
    """Build exactly one independent component model for bundle execution."""
    allowed = {"implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking", "polarity", "emotion"}
    if component not in allowed:
        raise ValueError(f"Unknown component: {component}")
    specs = load_model_registry(registry_path)
    if backbone not in specs:
        raise ValueError(f"Unknown locked backbone {backbone}")
    spec = specs[backbone]
    if execution_mode == "fixture":
        raise ValueError("Use build_dummy_model explicitly for fixture mode")
    if not local_snapshot:
        raise RuntimeBlocked(f"Pinned local snapshot is required before loading {backbone}")
    selected = resolve_selected_cuda_device(selected_device)
    family = "encoder" if spec.architecture == "encoder" else "causal"
    if spec.quantization == "nf4":
        base = build_qlora_backbone(spec.repo_id, revision=spec.revision, local_path=str(local_snapshot), selected_device=selected)
    else:
        base = load_pretrained_backbone(spec.repo_id, revision=spec.revision, family=family, trust_remote_code=spec.trust_remote_code, local_path=local_snapshot, local_files_only=True)
    config = VariantConfig(
        name=f"component_{component}",
        backbone_family=family,
        hidden_size=int(getattr(base.config, "hidden_size", 0)),
        vocab_size=int(getattr(base.config, "vocab_size", 0)),
    )
    if config.hidden_size <= 0 or config.vocab_size <= 0:
        raise RuntimeBlocked("Loaded backbone did not expose hidden_size and vocab_size")
    model = SingleTaskClassifier(base, config, output_key=component)
    if spec.quantization == "nf4":
        place_task_modules(model, selected)
    else:
        place_non_quantized_model(model, selected, model_family=backbone)
    return model, spec
