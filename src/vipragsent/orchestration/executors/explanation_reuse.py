from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...atomic import atomic_write_json, atomic_write_text
from ...evaluation.reasoning_judge import (
    ReasoningJudge,
    build_reasoning_prediction_row,
    compute_reasoning_metrics,
)
from ...hashing import sha256_file


@dataclass(frozen=True)
class ApprovedFullVistralSource:
    run_id: str
    run_root: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    review_summary_sha256: str
    approval_sha256: str
    checksum_file_sha256: str
    config_sha256: str
    variant_fingerprint: str
    seed: int | str
    model_revision: str
    tokenizer_revision: str

    def as_dict(self, root: Path) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "checkpoint_path": str(self.checkpoint_path.relative_to(root)).replace("\\", "/"),
            "checkpoint_sha256": self.checkpoint_sha256,
            "review_summary_sha256": self.review_summary_sha256,
            "approval_sha256": self.approval_sha256,
            "checksum_file_sha256": self.checksum_file_sha256,
            "config_sha256": self.config_sha256,
            "variant_fingerprint": self.variant_fingerprint,
            "seed": self.seed,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
        }


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid source artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"source artifact must be an object: {path}")
    return payload


def _approved_index(root: Path) -> list[dict[str, Any]]:
    path = root / "results/approved_run_index.json"
    if not path.exists():
        return []
    payload = _load(path)
    return [dict(item) for item in payload.get("runs", []) if isinstance(item, Mapping)]


def resolve_approved_full_vistral_source(root: str | Path, entry: Mapping[str, Any]) -> ApprovedFullVistralSource:
    """Resolve one exact approved source; no substring or first-match fallback."""
    root = Path(root)
    seed = entry.get("seed")
    source_key = str(entry.get("source_checkpoint_id") or entry.get("reusable_checkpoint_key") or f"vipragsent_full_vistral:{seed}")
    expected_key = f"vipragsent_full_vistral:{seed}"
    if source_key != expected_key:
        raise RuntimeError(f"explanation-only source key must be {expected_key}, got {source_key}")
    candidates: list[Path] = []
    index = _approved_index(root)
    if index:
        for row in index:
            if str(row.get("system")) == "vipragsent_full_vistral" and str(row.get("seed")) == str(seed) and str(row.get("run_id")):
                candidates.append(root / "results/runs" / str(row["run_id"]))
    else:
        candidates = [path.parent for path in sorted((root / "results/runs").glob("*/review_summary.json"))]
    matched: list[ApprovedFullVistralSource] = []
    for run_root in candidates:
        summary_path = run_root / "review_summary.json"
        approval_path = run_root / "approval_status.json"
        state_path = run_root / "state.json"
        manifest_path = run_root / "checkpoints/checkpoint_manifest.json"
        checksums_path = run_root / "checksums.sha256"
        if not all(path.exists() for path in (summary_path, approval_path, state_path, manifest_path, checksums_path)):
            continue
        summary = _load(summary_path)
        approval = _load(approval_path)
        state = _load(state_path)
        manifest = _load(manifest_path)
        if str(summary.get("system_id")) != "vipragsent_full_vistral" or str(summary.get("seed")) != str(seed):
            continue
        if str(summary.get("reusable_checkpoint_key") or summary.get("source_checkpoint_id")) != expected_key:
            continue
        if approval.get("status") != "APPROVED" or state.get("run_status") not in {"COMPLETED_PENDING_APPROVAL", "APPROVED"}:
            continue
        summary_hash = sha256_file(summary_path)
        if approval.get("review_summary_sha256") != summary_hash:
            continue
        checksum_hash = sha256_file(checksums_path)
        if approval.get("artifact_checksum_file_sha256") != checksum_hash:
            continue
        checkpoint_value = manifest.get("best") or manifest.get("checkpoint_path")
        if not checkpoint_value:
            continue
        checkpoint_path = run_root / str(checkpoint_value)
        if not checkpoint_path.exists():
            continue
        checkpoint_hash = sha256_file(checkpoint_path)
        if str(manifest.get("checkpoint_sha256")) != checkpoint_hash:
            continue
        config_path = run_root / "config_snapshot.yaml"
        config_hash = sha256_file(config_path) if config_path.exists() else ""
        if summary.get("config_hash") and str(summary["config_hash"]) != config_hash:
            continue
        variant_fingerprint = str(manifest.get("variant_fingerprint") or summary.get("variant_fingerprint") or "")
        if not variant_fingerprint:
            continue
        model_revision = str(summary.get("model_revision") or manifest.get("model_revision") or "")
        tokenizer_revision = str(summary.get("tokenizer_revision") or manifest.get("tokenizer_revision") or "")
        if entry.get("model_revision") not in (None, "", model_revision) or entry.get("tokenizer_revision") not in (None, "", tokenizer_revision):
            continue
        matched.append(ApprovedFullVistralSource(str(run_root.name), run_root, checkpoint_path, checkpoint_hash, summary_hash, sha256_file(approval_path), checksum_hash, config_hash, variant_fingerprint, seed, model_revision, tokenizer_revision))
    if len(matched) != 1:
        raise RuntimeError(f"exactly one approved full Vistral source is required for {expected_key}; found {len(matched)}")
    return matched[0]


def validate_source_checkpoint(root: str | Path, source: ApprovedFullVistralSource) -> dict[str, Any]:
    root = Path(root)
    manifest = _load(source.run_root / "checkpoints/checkpoint_manifest.json")
    errors: list[str] = []
    if not source.checkpoint_path.exists() or sha256_file(source.checkpoint_path) != source.checkpoint_sha256:
        errors.append("source checkpoint hash mismatch")
    if manifest.get("checkpoint_sha256") != source.checkpoint_sha256:
        errors.append("checkpoint manifest hash binding mismatch")
    if not source.variant_fingerprint:
        errors.append("source variant fingerprint is missing")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors, "source": source.as_dict(root)}


def _decode(tokenizer: Any, ids: Any) -> str:
    if isinstance(ids, str):
        return ids
    if not hasattr(tokenizer, "decode"):
        raise ValueError("rationale decoder inference requires tokenizer.decode")
    return str(tokenizer.decode(ids, skip_special_tokens=True)).strip()


class ExplanationReuseExecutor:
    """Rationale-decoder-only inference over an approved full Vistral model."""

    def __init__(self, root: str | Path, *, model: torch.nn.Module, tokenizer: Any, judge: ReasoningJudge, run_root: str | Path, source: ApprovedFullVistralSource) -> None:
        self.root = Path(root)
        self.model = model
        self.tokenizer = tokenizer
        self.judge = judge
        self.run_root = Path(run_root)
        self.source = source
        if getattr(model, "rationale_decoder", None) is None:
            raise RuntimeError("approved full model does not expose a rationale decoder")

    def generate_reasoning_split(self, split: str, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        self.model.eval()
        bos = int(getattr(self.tokenizer, "bos_token_id", 1))
        eos = int(getattr(self.tokenizer, "eos_token_id", 2))
        with torch.no_grad():
            for record in records:
                input_ids = record["input_ids"] if isinstance(record["input_ids"], torch.Tensor) else torch.tensor(record["input_ids"], dtype=torch.long)
                attention = record.get("attention_mask")
                if attention is None:
                    attention = torch.ones_like(input_ids)
                elif not isinstance(attention, torch.Tensor):
                    attention = torch.tensor(attention, dtype=torch.long)
                if input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(0)
                if attention.ndim == 1:
                    attention = attention.unsqueeze(0)
                encoded = self.model.backbone(input_ids=input_ids, attention_mask=attention)
                decoded = self.model.rationale_decoder.greedy_decode(encoded.last_hidden_state, attention, bos, eos, 160)
                reasoning = _decode(self.tokenizer, decoded[0])
                rows.append({"sample_id": str(record["sample_id"]), "split": split, "generated_reasoning": reasoning, "raw_generation": reasoning, "generation_status": "PASS" if reasoning else "INVALID", "failure_reason": None if reasoning else "empty_rationale", "truncated": bool(len(decoded[0]) >= 160 and eos not in decoded[0].tolist()), "inference_output_source": "judge_of_rationale_decoder_output"})
        atomic_write_text(self.run_root / f"reasoning/{split}_reasoning.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        return rows

    def judge_and_write(self, split: str, generated: Iterable[Mapping[str, Any]], gold: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        predictions: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for row in generated:
            decision = self.judge.judge(str(row.get("generated_reasoning", "")))
            decisions.append({"sample_id": row["sample_id"], **decision})
            predictions.append(build_reasoning_prediction_row(str(row["sample_id"]), gold[str(row["sample_id"])], str(row.get("generated_reasoning", "")), decision, truncated=bool(row.get("truncated"))))
        self.judge.write_artifacts(self.run_root, split, predictions, decisions)
        atomic_write_text(self.run_root / f"predictions/{split}_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions))
        metrics = compute_reasoning_metrics(predictions, diagnostics=self.judge.diagnostics) | {"status": "PASS", "split": split, "inference_output_source": "judge_of_rationale_decoder_output"}
        atomic_write_json(self.run_root / f"metrics/{split}_reasoning_metrics.json", metrics)
        return metrics

    def write_source_provenance(self) -> dict[str, Any]:
        provenance = {"status": "PASS", "source": self.source.as_dict(self.root), "additional_training": False, "optimizer_created": False, "scheduler_created": False, "backward_calls": 0, "direct_classification_outputs_used": False, "inference_output_source": "judge_of_rationale_decoder_output"}
        atomic_write_json(self.run_root / "source/source_provenance.json", provenance)
        return provenance
