from __future__ import annotations

import json
from pathlib import Path

import pytest

from vipragsent.artifacts.exporter import _read_production_runs
from vipragsent.constants import TRAINING_SEEDS


def _manifest(*, seed: int, budget: str | None = None) -> dict[str, object]:
    return {
        "mode": "full",
        "system": "phobert_finetune",
        "backbone": "phobert_base",
        "seed": seed,
        "research_question": "Q3" if budget else "Q1a",
        "budget": budget,
        "model_revision": "pinned-model-revision",
        "tokenizer_revision": "pinned-tokenizer-revision",
        "preprocessing_name": "vncorenlp_rdrsegmenter",
        "preprocessing_version": "vncorenlp-1",
        "physical_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "effective_batch_size": 32,
        "inference_output_source": "classification_heads",
        "rationale_decoder_enabled_at_inference": False,
        "data_fingerprint": "data-hash",
        "config_hash": "config-hash",
        "code_commit": "code-commit",
    }


def _write_run(root: Path, payload: dict[str, object]) -> None:
    run = root / "results/runs" / str(payload["system"]) / str(payload["seed"])
    run.mkdir(parents=True, exist_ok=True)
    (run / "test_predictions.jsonl").write_text('{"sample_id":"synthetic"}\n', encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_production_contract_rejects_missing_seed(tmp_path: Path) -> None:
    _write_run(tmp_path, _manifest(seed=TRAINING_SEEDS[0]))
    with pytest.raises(ValueError, match="required training seeds"):
        _read_production_runs(tmp_path)


def test_production_contract_rejects_missing_q3_budget(tmp_path: Path) -> None:
    for seed in TRAINING_SEEDS:
        _write_run(tmp_path, _manifest(seed=seed, budget="32"))
    with pytest.raises(ValueError, match="required Q3 budgets"):
        _read_production_runs(tmp_path)
