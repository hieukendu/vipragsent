from __future__ import annotations

from pathlib import Path
from typing import Any

from ..orchestration.status import RuntimeBlocked
from .preprocessing import DummyTokenizer

BACKBONE_REPOSITORIES = {
    "phobert_base": "vinai/phobert-base",
    "xlmr_large": "FacebookAI/xlm-roberta-large",
    "sailor_7b": "sail/Sailor-7B",
    "vistral_7b": "Viet-Mistral/Vistral-7B-Chat",
}


def create_tokenizer(
    backbone: str,
    *,
    revision: str,
    local_path: str | Path | None = None,
    execution_mode: str = "production",
    use_fast: bool | None = None,
) -> Any:
    if execution_mode == "fixture":
        return DummyTokenizer()
    if backbone not in BACKBONE_REPOSITORIES:
        raise ValueError(f"Unknown locked backbone: {backbone}")
    if not revision or revision == "fixture":
        raise RuntimeBlocked("An immutable tokenizer revision is required")
    source = str(local_path) if local_path else BACKBONE_REPOSITORIES[backbone]
    if local_path is not None and not Path(local_path).exists():
        raise RuntimeBlocked(f"Local tokenizer snapshot is missing: {local_path}")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeBlocked("transformers is required for production tokenizers") from exc
    fast = False if backbone == "phobert_base" else True if use_fast is None else use_fast
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, use_fast=fast, local_files_only=local_path is not None)
    except Exception as exc:
        raise RuntimeBlocked(f"Unable to load pinned tokenizer {backbone}@{revision}: {exc}") from exc
    if backbone in {"sailor_7b", "vistral_7b"} and tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeBlocked(f"{backbone} tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer._vipragsent_revision = revision
    return tokenizer
