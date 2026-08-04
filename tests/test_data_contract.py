from __future__ import annotations

import json
from pathlib import Path

from vipragsent.constants import EXPECTED_SPLIT_COUNTS, PRAGMATIC_LABELS
from vipragsent.data.loaders import calculate_loss_weights, load_vipragsent
from vipragsent.data.masks import validate_q3_masks
from vipragsent.data.rationales import iter_rationale_inputs


def test_frozen_vipragsent_counts_and_split_immutability() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = load_vipragsent(root / "data/processed/vipragsent")
    assert {key: len(value) for key, value in bundle.splits.items()} == EXPECTED_SPLIT_COUNTS
    ids = [example.sample_id for examples in bundle.splits.values() for example in examples]
    assert len(ids) == len(set(ids)) == 11997
    assert sum(example.labels["sarcasm"] for example in bundle.train) == 545
    assert not set(item.sample_id for item in bundle.train) & set(item.sample_id for item in bundle.test)


def test_q3_masks_are_nested_and_keep_other_tasks_active() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = load_vipragsent(root / "data/processed/vipragsent")
    report = validate_q3_masks(root / "data/processed/q3_low_resource_sarcasm", {item.sample_id: item for item in bundle.train})
    assert report["nested"] is True
    assert report["selected_positive_counts"]["full"] == 545
    assert report["fixed_negative_count"] == 7453


def test_loss_weights_are_train_only() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = load_vipragsent(root / "data/processed/vipragsent")
    weights = calculate_loss_weights(bundle.train)
    assert weights["source_split"] == "train"
    for key in PRAGMATIC_LABELS:
        assert weights["pragmatic_pos_weight"][key] > 0


def test_active_rationale_input_has_no_legacy_generator_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data/processed/rationales/azure_rationale_input_train.jsonl"
    rows = list(iter_rationale_inputs(path))
    assert len(rows) == 7998
    assert set(rows[0]) == {"sample_id", "comment", "gold_labels"}
    assert "generator_metadata_required" not in rows[0]
