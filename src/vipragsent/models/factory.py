from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from torch import nn

from ..orchestration.status import RuntimeBlocked
from .backbones import load_pretrained_backbone
from .qlora import build_qlora_backbone
from .variants import (
    GenerationBaselineModel,
    IndependentCheckpointBundle,
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
    def load_base() -> nn.Module:
        if spec.quantization == "nf4":
            return build_qlora_backbone(spec.repo_id, revision=spec.revision, local_path=str(local_snapshot))
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
        first = True

        def independent_pragmatic_backbone() -> nn.Module:
            nonlocal first
            if first:
                first = False
                return base
            return load_base()

        return SingleTaskPragmaticBundle(independent_pragmatic_backbone, config), spec
    if config.is_checkpoint_bundle:
        first = True

        def independent_backbone() -> nn.Module:
            nonlocal first
            if first:
                first = False
                return base
            return load_base()

        return IndependentCheckpointBundle(independent_backbone, config), spec
    if variant in {"cot_only_vistral", "explanation_only_vistral"}:
        return GenerationBaselineModel(base, config), spec
    return ViPragSentModel(base, config), spec
