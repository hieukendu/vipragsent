from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vipragsent.models.variants import VariantConfig, build_dummy_model
from vipragsent.training.engine import TrainingConfig, TrainingEngine


def _batch() -> dict[str, object]:
    input_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2], [1, 8, 9, 2], [1, 10, 11, 2]])
    targets = {key: torch.tensor([0.0, 1.0, 0.0, 1.0]) for key in ("implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking")}
    targets.update({"polarity": torch.tensor([0, 1, 2, 0]), "emotion": torch.tensor([0, 1, 2, 3])})
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "targets": targets, "sample_ids": [f"fixture_{i}" for i in range(4)]}


def _engine(root: Path, run_id: str, *, hooks: dict[str, object] | None = None) -> TrainingEngine:
    torch.manual_seed(11)
    model = build_dummy_model(VariantConfig(name="no_rationale", hidden_size=16, vocab_size=32))
    return TrainingEngine(
        model,
        TrainingConfig(max_epochs=2, patience=10, learning_rate=0.05, precision="fp32", gradient_accumulation_steps=2),
        run_id=run_id,
        checkpoint_root=root / "checkpoints",
        runtime_hooks=hooks,
    )


def test_training_engine_selects_dev_checkpoint_and_freezes_thresholds(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "selection")
    with pytest.raises(RuntimeError):
        engine.assert_test_access()
    state = engine.train([_batch()], seed=20260521, dev_batches=[_batch()], test_batches=[_batch()], output_root=tmp_path / "outputs")
    assert state.status == "PASS"
    assert state.best_epoch is not None
    assert engine.gate.checkpoint_frozen is True
    assert len({round(row["dev_metric"], 8) for row in state.history}) >= 1
    for name in ("thresholds.json", "dev_predictions.jsonl", "test_predictions.jsonl", "training_history.json", "run_manifest.json"):
        assert (tmp_path / "outputs" / name).exists()
    assert json.loads((tmp_path / "outputs" / "run_manifest.json").read_text(encoding="utf-8"))["thresholds_frozen"] is True


def test_resume_after_interruption_matches_uninterrupted_model(tmp_path: Path) -> None:
    clean = _engine(tmp_path / "clean", "run")
    clean_state = clean.train([_batch()], seed=20260521, dev_batches=[_batch()])
    interrupted = {"raised": False}

    def stop_after_first_epoch(epoch: int, *_: object) -> None:
        if epoch == 1 and not interrupted["raised"]:
            interrupted["raised"] = True
            raise RuntimeError("test interruption")

    first = _engine(tmp_path / "resume", "run", hooks={"on_epoch_end": stop_after_first_epoch})
    with pytest.raises(RuntimeError):
        first.train([_batch()], seed=20260521, dev_batches=[_batch()])
    resumed = _engine(tmp_path / "resume", "run")
    resumed_state = resumed.train([_batch()], seed=20260521, dev_batches=[_batch()], resume=True)
    assert resumed_state.best_metric == clean_state.best_metric
    assert resumed_state.best_epoch == clean_state.best_epoch
