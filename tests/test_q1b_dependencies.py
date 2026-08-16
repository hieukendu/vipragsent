from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vipragsent.hashing import sha256_file
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.q1b_composition import compose_ordinary_single_task
from vipragsent.orchestration.q1b_dependencies import (
    build_q1b_dependency_graph,
    resolve_q1b_producer,
)
from vipragsent.orchestration.q1b_predictor import DiskBackedQ1BPredictor
from vipragsent.orchestration.status import RuntimeBlocked

ROOT = Path(__file__).resolve().parents[1]


def _write_approved_status(run_root: Path, summary: Path, checksums: Path) -> None:
    timestamp = "2026-08-16T00:00:00Z"
    (run_root / "approval_status.json").write_text(
        json.dumps(
            {
                "run_id": run_root.name,
                "status": "APPROVED",
                "approved_by": "fixture-reviewer",
                "approved_at": timestamp,
                "record": {
                    "run_id": run_root.name,
                    "decision": "approve",
                    "review_note": "fixture approval",
                    "approved_or_rejected_by": "fixture-reviewer",
                    "timestamp": timestamp,
                    "review_summary_sha256": sha256_file(summary),
                    "artifact_checksum_file_sha256": sha256_file(checksums),
                },
            }
        ),
        encoding="utf-8",
    )


def test_q1b_every_consumer_has_exact_producer() -> None:
    graph = build_q1b_dependency_graph(ROOT)
    assert graph["status"] == "PASS", graph
    assert graph["paper_inventory_count_before"] == 162
    assert graph["paper_inventory_count_after"] == 162
    assert graph["paper_inventory_changed"] is False
    q1b_edges = [edge for edge in graph["edges"] if str(edge["consumer_id"]).startswith("q1b_")]
    assert len(q1b_edges) == 21
    assert len({edge["consumer_id"] for edge in q1b_edges}) == 21
    assert all(edge["expected_checkpoint_key"] == edge["produced_checkpoint_key"] for edge in q1b_edges)
    assert all(edge["producer_kind"] == "trainable_checkpoint" for edge in q1b_edges)


def test_q1b_dependency_graph_is_acyclic() -> None:
    graph = build_q1b_dependency_graph(ROOT)
    assert graph["status"] == "PASS", graph
    assert len(graph["topological_order"]) == len(graph["nodes"])
    positions = {node: index for index, node in enumerate(graph["topological_order"])}
    assert all(positions[edge["from"]] < positions[edge["to"]] for edge in graph["edges"])


def test_q1b_producer_seed_matches_consumer() -> None:
    graph = build_q1b_dependency_graph(ROOT)
    for edge in graph["edges"]:
        if str(edge["consumer_id"]).startswith("q1b_") and edge["seed"] is not None:
            assert str(edge["producer_run_id"]).endswith(f":{edge['seed']}")
            assert f":{edge['seed']}" in edge["produced_checkpoint_key"]


def test_q1b_reusable_checkpoint_key_matches() -> None:
    inventory = build_expected_runs(ROOT)
    graph = build_q1b_dependency_graph(ROOT, inventory_rows=inventory["rows"])
    by_id = {row["experiment_id"]: row for row in inventory["rows"]}
    for edge in graph["edges"]:
        consumer = by_id.get(edge["consumer_id"])
        if consumer is not None and consumer["research_question"] == "Q1b":
            assert edge["expected_checkpoint_key"] == consumer["reusable_checkpoint_key"]


def test_q1b_ordinary_single_task_same_seed_composition() -> None:
    polarity = {
        "seed": 20260521,
        "source_checkpoint": "phobert_pol_single:20260521",
        "predictions": {
            "vsfc": [{"gold": "positive", "prediction": "positive"}],
            "aivivn": [{"gold": "neutral", "prediction": "neutral"}],
        },
    }
    emotion = {
        "seed": 20260521,
        "source_checkpoint": "phobert_emo_single:20260521",
        "predictions": {"vsmec": [{"gold": "anger", "prediction": "anger"}]},
    }
    composed = compose_ordinary_single_task(polarity_results=polarity, emotion_results=emotion)
    assert composed["source_seed"] == 20260521
    assert set(composed["predictions"]) == {"vsfc", "aivivn", "vsmec"}
    assert composed["ord_f1"] == pytest.approx((composed["vsfc_macro_f1"] + composed["aivivn_macro_f1"] + composed["vsmec_macro_f1"]) / 3)
    with pytest.raises(ValueError, match="same training seed"):
        compose_ordinary_single_task(polarity_results=polarity, emotion_results={**emotion, "seed": 20260522})


def test_q1b_no_fake_non_applicable_predictions(tmp_path: Path) -> None:
    # This uses the real resolver only for the source metadata; prediction routing is injected.
    config_root = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", config_root)
    run_root = tmp_path / "results/runs/source"
    checkpoint = run_root / "checkpoints/best/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"approved")
    summary = run_root / "review_summary.json"
    summary.write_text(json.dumps({"system_id": "phobert_pol_single", "seed": 20260521, "reusable_checkpoint_key": "phobert_pol_single:20260521", "variant_fingerprint": "variant"}), encoding="utf-8")
    checksums = run_root / "checksums.sha256"
    checksums.write_text("checkpoint-entry\n", encoding="utf-8")
    (run_root / "state.json").write_text(json.dumps({"run_id": run_root.name, "run_status": "APPROVED", "approval_status": "APPROVED"}), encoding="utf-8")
    (run_root / "checkpoints/checkpoint_manifest.json").write_text(json.dumps({"best": "checkpoints/best/model.pt", "checkpoint_sha256": sha256_file(checkpoint), "variant_fingerprint": "variant"}), encoding="utf-8")
    _write_approved_status(run_root, summary, checksums)
    index = tmp_path / "results/approved_run_index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"runs": [{"system": "phobert_pol_single", "seed": 20260521, "run_id": "source"}]}), encoding="utf-8")
    model = SimpleNamespace(__call__=lambda **_: {"logits": {"polarity": torch.tensor([[0.0, 1.0, 2.0]])}})
    predictor = DiskBackedQ1BPredictor(tmp_path, {"experiment_id": "q1b_phobert_pol_single_20260521", "system_id": "phobert_pol_single", "seed": 20260521, "backbone": "phobert_base"}, model=model, tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: [1]))
    assert predictor.applicable_datasets == ("vsfc", "aivivn")
    assert predictor.source.producer_id == "q1b_train_phobert_pol_single"
    with pytest.raises(RuntimeBlocked, match="not applicable"):
        predictor.predict("vsmec", SimpleNamespace(text="fixture"))


def test_q1b_public_cli_resolves_source_without_injection(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    run_root = tmp_path / "results/runs/source"
    checkpoint = run_root / "checkpoints/best/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"approved")
    summary = run_root / "review_summary.json"
    summary.write_text(json.dumps({"system_id": "phobert_pol_single", "seed": 20260521, "reusable_checkpoint_key": "phobert_pol_single:20260521", "variant_fingerprint": "variant"}), encoding="utf-8")
    checksums = run_root / "checksums.sha256"
    checksums.write_text("checkpoint-entry\n", encoding="utf-8")
    (run_root / "state.json").write_text(json.dumps({"run_id": run_root.name, "run_status": "APPROVED", "approval_status": "APPROVED"}), encoding="utf-8")
    (run_root / "checkpoints/checkpoint_manifest.json").write_text(json.dumps({"best": "checkpoints/best/model.pt", "checkpoint_sha256": sha256_file(checkpoint), "variant_fingerprint": "variant"}), encoding="utf-8")
    _write_approved_status(run_root, summary, checksums)
    index = tmp_path / "results/approved_run_index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"runs": [{"system": "phobert_pol_single", "seed": 20260521, "run_id": "source"}]}), encoding="utf-8")
    resolved = resolve_q1b_producer(tmp_path, {"experiment_id": "q1b_phobert_pol_single_20260521", "system_id": "phobert_pol_single", "seed": 20260521})
    assert resolved["edge"]["producer_id"] == "q1b_train_phobert_pol_single"
    predictor = DiskBackedQ1BPredictor(tmp_path, {"experiment_id": "q1b_phobert_pol_single_20260521", "system_id": "phobert_pol_single", "seed": 20260521, "backbone": "phobert_base"}, model=SimpleNamespace(), tokenizer=SimpleNamespace())
    assert predictor.source.producer_run_id == "q1b_train_phobert_pol_single:20260521"
