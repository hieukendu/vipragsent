from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from ..constants import EMOTION_LABELS, POLARITY_LABELS
from ..hashing import sha256_file
from ..orchestration.status import RuntimeBlocked

MATRIX_KEY_BY_SYSTEM = {
    "phobert_pol_single": "phobert_ordinary_single_task",
    "phobert_emo_single": "phobert_ordinary_single_task",
    "phobert_multitask_8head": "phobert_multitask",
    "xlmr_multitask_8head": "xlmr_multitask",
    "sailor_multitask_8head": "sailor_multitask",
    "vistral_multitask_8head": "vistral_multitask",
    "vipragsent_full_phobert": "vipragsent",
}
DATASET_TASK = {"vsfc": "polarity", "aivivn": "polarity", "vsmec": "emotion"}


@dataclass(frozen=True)
class Q1BSource:
    system_id: str
    matrix_key: str
    checkpoint_key: str
    run_id: str
    run_root: Path
    seed: int | str | None
    checkpoint_path: Path
    checkpoint_sha256: str
    review_summary_sha256: str
    approval_sha256: str
    checksum_file_sha256: str
    variant_fingerprint: str

    def as_dict(self, root: Path) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "matrix_key": self.matrix_key,
            "checkpoint_key": self.checkpoint_key,
            "run_id": self.run_id,
            "run_root": str(self.run_root.relative_to(root)).replace("\\", "/"),
            "seed": self.seed,
            "checkpoint_path": str(self.checkpoint_path.relative_to(root)).replace("\\", "/"),
            "checkpoint_sha256": self.checkpoint_sha256,
            "review_summary_sha256": self.review_summary_sha256,
            "approval_sha256": self.approval_sha256,
            "checksum_file_sha256": self.checksum_file_sha256,
            "variant_fingerprint": self.variant_fingerprint,
        }


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeBlocked(f"Q1b source artifact is invalid: {path}") from exc
    return dict(payload) if isinstance(payload, Mapping) else {}


def resolve_checkpoint_matrix_entry(root: str | Path, entry: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    root = Path(root)
    system_id = str(entry.get("system_id", ""))
    matrix_key = MATRIX_KEY_BY_SYSTEM.get(system_id)
    if not matrix_key:
        raise RuntimeBlocked(f"Q1b system has no exact checkpoint-matrix mapping: {system_id}")
    payload = yaml.safe_load((root / "configs/experiments/q1b/checkpoint_matrix.yaml").read_text(encoding="utf-8")) or {}
    raw = payload.get("systems", {}).get(matrix_key)
    if not isinstance(raw, Mapping):
        raise RuntimeBlocked(f"Q1b checkpoint matrix row is missing: {matrix_key}")
    row = dict(raw)
    if system_id == "phobert_pol_single":
        checkpoint_key = str(row.get("polarity_checkpoint", ""))
    elif system_id == "phobert_emo_single":
        checkpoint_key = str(row.get("emotion_checkpoint", ""))
    else:
        checkpoint_key = str(row.get("checkpoint", ""))
    if not checkpoint_key:
        raise RuntimeBlocked(f"Q1b checkpoint matrix has no exact checkpoint key for {system_id}")
    return matrix_key, {"system_id": system_id, "checkpoint_key": checkpoint_key, **row}


def resolve_exact_q1b_source(root: str | Path, entry: Mapping[str, Any]) -> Q1BSource:
    root = Path(root)
    matrix_key, matrix = resolve_checkpoint_matrix_entry(root, entry)
    system_id = str(entry.get("system_id"))
    seed = entry.get("seed")
    checkpoint_key = str(entry.get("source_checkpoint_id") or entry.get("reusable_checkpoint_key") or f"{matrix['checkpoint_key']}:{seed}")
    expected_key = f"{matrix['checkpoint_key']}:{seed}"
    if checkpoint_key != expected_key:
        raise RuntimeBlocked(f"Q1b source key mismatch: expected {expected_key}, got {checkpoint_key}")
    index_path = root / "results/approved_run_index.json"
    if not index_path.exists():
        raise RuntimeBlocked("Q1b approved-run index is missing")
    index = _load(index_path)
    candidates: list[Q1BSource] = []
    for row in index.get("runs", []):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("system")) != system_id or str(row.get("seed")) != str(seed):
            continue
        run_id = str(row.get("run_id", ""))
        run_root = root / "results/runs" / run_id
        summary_path = run_root / "review_summary.json"
        approval_path = run_root / "approval_status.json"
        state_path = run_root / "state.json"
        manifest_path = run_root / "checkpoints/checkpoint_manifest.json"
        checksums_path = run_root / "checksums.sha256"
        if not all(path.exists() for path in (summary_path, approval_path, state_path, manifest_path, checksums_path)):
            continue
        summary, approval, state, manifest = (_load(summary_path), _load(approval_path), _load(state_path), _load(manifest_path))
        if str(summary.get("system_id")) != system_id or str(summary.get("seed")) != str(seed):
            continue
        if str(summary.get("reusable_checkpoint_key")) != checkpoint_key or approval.get("status") != "APPROVED":
            continue
        if state.get("run_status") not in {"COMPLETED_PENDING_APPROVAL", "APPROVED"}:
            continue
        summary_hash = sha256_file(summary_path)
        checksum_hash = sha256_file(checksums_path)
        if approval.get("review_summary_sha256") != summary_hash or approval.get("artifact_checksum_file_sha256") != checksum_hash:
            continue
        checkpoint_value = manifest.get("best") or manifest.get("checkpoint_path")
        checkpoint_path = run_root / str(checkpoint_value or "")
        if not checkpoint_path.exists() or manifest.get("checkpoint_sha256") != sha256_file(checkpoint_path):
            continue
        variant_fingerprint = str(manifest.get("variant_fingerprint") or summary.get("variant_fingerprint") or "")
        if not variant_fingerprint:
            continue
        expected_variant_fingerprint = entry.get("variant_fingerprint")
        if expected_variant_fingerprint not in (None, "", variant_fingerprint):
            continue
        candidates.append(Q1BSource(system_id, matrix_key, checkpoint_key, run_id, run_root, seed, checkpoint_path, sha256_file(checkpoint_path), summary_hash, sha256_file(approval_path), checksum_hash, variant_fingerprint))
    if len(candidates) != 1:
        raise RuntimeBlocked(f"Q1b requires exactly one approved source for exact system {system_id}, seed {seed}; found {len(candidates)}")
    return candidates[0]


class DiskBackedQ1BPredictor:
    """Self-contained, source-backed predictor used by the public Q1b CLI."""

    def __init__(self, root: str | Path, entry: Mapping[str, Any], *, source: Q1BSource | None = None, model_loader: Any | None = None, tokenizer: Any | None = None, model: Any | None = None) -> None:
        self.root = Path(root)
        self.entry = dict(entry)
        self.matrix_key, self.matrix = resolve_checkpoint_matrix_entry(self.root, entry)
        self.source = source or resolve_exact_q1b_source(self.root, entry)
        self.model_loader = model_loader
        self.model = model
        self.tokenizer = tokenizer
        self._loaded = model is not None
        self.applicable_datasets = tuple(dataset for dataset, task in DATASET_TASK.items() if self._task_applicable(task))

    def _task_applicable(self, task: str) -> bool:
        system_id = str(self.entry.get("system_id"))
        if system_id == "phobert_pol_single":
            return task == "polarity"
        if system_id == "phobert_emo_single":
            return task == "emotion"
        return True

    def _load(self) -> None:
        if self._loaded:
            return
        if self.model_loader is not None:
            loaded = self.model_loader(self.source)
            self.model = loaded[0] if isinstance(loaded, tuple) else loaded
            if isinstance(loaded, tuple) and len(loaded) > 1:
                self.tokenizer = loaded[1]
            self._loaded = True
            return
        system_id = str(self.entry.get("system_id"))
        family = str(self.entry.get("backbone") or "phobert_base")
        variant = system_id if system_id in {"phobert_pol_single", "phobert_emo_single", "phobert_multitask_8head", "xlmr_multitask_8head", "sailor_multitask_8head", "vistral_multitask_8head"} else "vipragsent_full_phobert"
        from ..data.tokenizers import create_tokenizer
        from ..models.factory import build_production_model
        from ..runtime.model_assets import read_family_status

        snapshot = read_family_status(self.root, family, "cache").get("local_path")
        if not snapshot:
            raise RuntimeBlocked(f"Phase 15 local snapshot is unavailable for {family}")
        self.model, runtime_spec = build_production_model(family, variant, local_snapshot=snapshot, execution_mode="production")
        self.tokenizer = create_tokenizer(family, revision=runtime_spec.tokenizer_revision, local_path=snapshot, execution_mode="production")
        state = torch.load(self.source.checkpoint_path, map_location="cpu")
        state_dict = state.get("model_state_dict") if isinstance(state, Mapping) else None
        if isinstance(state_dict, Mapping):
            self.model.load_state_dict(state_dict, strict=False)
        self._loaded = True

    def predict(self, dataset: str, example: Any) -> str:
        if dataset not in DATASET_TASK or dataset not in self.applicable_datasets:
            raise RuntimeBlocked(f"dataset {dataset} is not applicable to Q1b source {self.entry.get('system_id')}")
        self._load()
        if self.model is None or self.tokenizer is None:
            raise RuntimeBlocked("Q1b predictor model/tokenizer is unavailable")
        from ..orchestration.executors.generation import _encode_text

        input_ids, attention = _encode_text(self.tokenizer, str(example.text))
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention)
        logits = output.get("logits", {}) if isinstance(output, Mapping) else getattr(output, "logits", {})
        task = DATASET_TASK[dataset]
        task_logits = logits.get(task) if isinstance(logits, Mapping) else None
        if task_logits is None:
            raise RuntimeBlocked(f"source model did not expose the routed {task} head")
        values = task_logits[0] if getattr(task_logits, "ndim", 0) > 1 else task_logits
        index = int(torch.argmax(values).item())
        return (POLARITY_LABELS if task == "polarity" else EMOTION_LABELS)[index]

    def provenance(self) -> dict[str, Any]:
        return {"source": self.source.as_dict(self.root), "matrix": {"key": self.matrix_key, **self.matrix}, "applicable_datasets": list(self.applicable_datasets), "external_finetuning": False, "optimizer_steps": 0, "backward_calls": 0, "predictor_factory": "disk_backed_q1b_v1"}
