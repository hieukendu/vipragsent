from __future__ import annotations

import json
import shutil
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vipragsent.hashing import sha256_file
from vipragsent.orchestration import aggregation
from vipragsent.orchestration.contracts import RunContext, RunEntry
from vipragsent.orchestration.executors.external_retention import (
    evaluate_external_retention_from_disk,
)
from vipragsent.orchestration.q1b_predictor import DiskBackedQ1BPredictor
from vipragsent.orchestration.review import build_review_summary
from vipragsent.orchestration.run_store import RunStore
from vipragsent.orchestration.sequential import load_inventory
from vipragsent.runtime.naacl_profile import (
    ProfileValidationError,
    build_naacl_profile_snapshot,
    validate_q3_profile_rows,
)

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SYSTEMS = (
    "phobert_pragmatic_finetune",
    "vistral_pragmatic_sft",
    "vipragsent_full_vistral",
)
RETAINED_BUDGETS = ("32", "128", "512", "full")
LOCKED_SEEDS = (20260521, 20260522, 20260523)
AZURE = "azure_gpt41_mini_8shot"


def _profile_rows() -> list[dict[str, object]]:
    return [
        {"system_id": system, "budget": budget, "seed": seed}
        for system, budget, seed in product(LOCAL_SYSTEMS, RETAINED_BUDGETS, LOCKED_SEEDS)
    ] + [{"system_id": AZURE, "budget": budget, "seed": None} for budget in RETAINED_BUDGETS]


def test_q3_profile_accepts_exact_36_local_cells_and_four_seedless_azure_rows() -> None:
    result = validate_q3_profile_rows(_profile_rows())
    assert result == {"local_cell_count": 36, "azure_row_count": 4, "total_row_count": 40, "azure_seed": None}
    profile = build_naacl_profile_snapshot(ROOT)
    assert aggregation._profile_q3_record_blockers(
        [{"summary": row} for row in _profile_rows()], profile
    ) == []


def test_q3_profile_rejects_the_legacy_72_row_cartesian_shape() -> None:
    old_rows = [
        {"system_id": system, "budget": budget, "seed": seed}
        for system, budget, seed in product(
            (*LOCAL_SYSTEMS, "xlmr_pragmatic_finetune"),
            ("32", "64", "128", "256", "512", "full"),
            LOCKED_SEEDS,
        )
    ]
    with pytest.raises(ProfileValidationError, match="missing retained Azure|out-of-profile"):
        validate_q3_profile_rows(old_rows)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing retained Azure"),
        ("extra", "out-of-profile"),
        ("azure_seed", "must not define a seed axis"),
    ],
)
def test_q3_profile_fails_closed_on_missing_extra_and_invented_azure_seed(
    mutation: str, message: str
) -> None:
    rows = _profile_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append({"system_id": "phobert_pragmatic_finetune", "budget": "64", "seed": 20260521})
    else:
        rows.append({"system_id": AZURE, "budget": "32", "seed": 20260521})
    with pytest.raises(ProfileValidationError, match=message):
        validate_q3_profile_rows(rows)


def test_profile_aggregation_rejects_q3_inventory_rows_outside_authorized_source_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = build_naacl_profile_snapshot(ROOT)
    inventory = load_inventory(ROOT)
    q3_rows = [row for row in inventory if str(row.get("research_question")) == "Q3"]
    q3_rows.append({"system_id": "rogue_q3_system", "budget": "32", "seed": LOCKED_SEEDS[0]})
    monkeypatch.setattr(aggregation, "_scope_rows", lambda _root, scope: q3_rows if scope == "Q3" else [])
    with pytest.raises(ProfileValidationError, match="complete authorized source matrix"):
        aggregation._profile_q3_inventory_rows(ROOT, profile)


def test_aggregation_entry_validates_the_current_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def reject(_root: Path) -> dict[str, object]:
        raise ProfileValidationError("current graph/source digest drift")

    monkeypatch.setattr(aggregation, "validate_naacl_profile", reject)
    result = aggregation.aggregate_approved_scope(tmp_path, "Q3")
    assert result["status"] == "BLOCKED"
    assert "current graph/source digest drift" in result["blockers"][0]


def test_approved_run_validation_fails_closed_on_incomplete_approval_record(tmp_path: Path) -> None:
    run_root = tmp_path / "results/runs/run"
    run_root.mkdir(parents=True)
    (run_root / "state.json").write_text(
        json.dumps({"run_id": "run", "run_status": "APPROVED", "approval_status": "APPROVED"}),
        encoding="utf-8",
    )
    (run_root / "approval_status.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "status": "APPROVED",
                "approved_by": "reviewer",
                "approved_at": "2026-08-16T00:00:00Z",
                "record": {"approved_or_rejected_by": "reviewer", "timestamp": "2026-08-16T00:00:00Z"},
            }
        ),
        encoding="utf-8",
    )

    record, errors = aggregation._validate_approved_run(tmp_path, "run")

    assert record is None
    assert any("approval decision record is missing fields" in error for error in errors)


def test_q1b_profile_rejects_unresolved_producer_and_training_metrics(tmp_path: Path) -> None:
    profile = build_naacl_profile_snapshot(ROOT)
    edge = next(edge for edge in profile["q1b"]["consumer_edges"] if edge["seed"] is not None)
    record = {
        "run_id": edge["consumer_id"],
        "run_root": str(tmp_path),
        "summary": {
            "producer_id": edge["producer_id"],
            "producer_run_id": edge["producer_run_id"],
            "producer_kind": edge["producer_kind"],
            "source_checkpoint_id": edge["checkpoint_key"],
            "source_seed": edge["seed"],
        },
    }
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics/external_retention_metrics.json").write_text('{"optimizer_steps": 1}', encoding="utf-8")
    blockers = aggregation._profile_q1b_record_blockers([record], profile)
    assert any("profile-bound consumers" in blocker for blocker in blockers)
    assert any("optimizer steps must be zero" in blocker for blocker in blockers)

    unresolved = {**record, "summary": {"source_seed": edge["seed"]}}
    blockers = aggregation._profile_q1b_record_blockers([unresolved], profile)
    assert any("unresolved Q1b producer/checkpoint/seed binding" in blocker for blocker in blockers)


def test_q1b_profile_requires_canonical_digests_and_rejects_conflicts_or_type_coercion(tmp_path: Path) -> None:
    profile = build_naacl_profile_snapshot(ROOT)
    edge = next(edge for edge in profile["q1b"]["consumer_edges"] if edge["seed"] is not None)
    canonical = {
        "producer_id": edge["producer_id"],
        "producer_run_id": edge["producer_run_id"],
        "producer_kind": edge["producer_kind"],
        "checkpoint_key": edge["checkpoint_key"],
        "source_seed": edge["seed"],
        "dependency_graph_sha256": profile["graph"]["sha256"],
        "dependency_source_sha256": profile["source"]["sha256"],
        "external_finetuning": False,
        "train_loader_created": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "training_applicability": "NOT_APPLICABLE",
    }
    record = {"run_id": edge["consumer_id"], "run_root": str(tmp_path), "summary": dict(canonical)}
    (tmp_path / "external").mkdir()
    (tmp_path / "external/external_evaluation_manifest.json").write_text(json.dumps(canonical), encoding="utf-8")
    blockers = aggregation._profile_q1b_record_blockers([record], profile)
    assert not any("binding disagrees" in blocker or "unresolved Q1b" in blocker for blocker in blockers)

    conflicting = dict(canonical, producer_kind="wrong_kind")
    (tmp_path / "external/external_evaluation_manifest.json").write_text(json.dumps(conflicting), encoding="utf-8")
    blockers = aggregation._profile_q1b_record_blockers([record], profile)
    assert any("conflicting Q1b producer_kind" in blocker for blocker in blockers)

    typed = dict(canonical, external_finetuning="true")
    (tmp_path / "external/external_evaluation_manifest.json").write_text(json.dumps(typed), encoding="utf-8")
    blockers = aggregation._profile_q1b_record_blockers([record], profile)
    assert any("external_finetuning must be a JSON boolean" in blocker for blocker in blockers)

    string_seed = dict(canonical, external_finetuning=False, source_seed=str(edge["seed"]))
    record["summary"] = dict(string_seed)
    (tmp_path / "external/external_evaluation_manifest.json").write_text(json.dumps(string_seed), encoding="utf-8")
    blockers = aggregation._profile_q1b_record_blockers([record], profile)
    assert any("Q1b seed binding disagrees" in blocker for blocker in blockers)

    alias_conflict = dict(canonical)
    alias_conflict["source_checkpoint_id"] = "rogue-checkpoint"
    (tmp_path / "external/external_evaluation_manifest.json").write_text(json.dumps(alias_conflict), encoding="utf-8")
    blockers = aggregation._profile_q1b_record_blockers([record], profile)
    assert any("conflicting Q1b checkpoint_key" in blocker for blocker in blockers)


def test_q1b_profile_requires_explicit_training_prohibition_fields(tmp_path: Path) -> None:
    profile = build_naacl_profile_snapshot(ROOT)
    edge = next(edge for edge in profile["q1b"]["consumer_edges"] if edge["seed"] is not None)
    payload = {
        "producer_id": edge["producer_id"],
        "producer_run_id": edge["producer_run_id"],
        "producer_kind": edge["producer_kind"],
        "checkpoint_key": edge["checkpoint_key"],
        "source_seed": edge["seed"],
        "dependency_graph_sha256": profile["graph"]["sha256"],
        "dependency_source_sha256": profile["source"]["sha256"],
    }
    record = {"run_id": edge["consumer_id"], "run_root": str(tmp_path), "summary": payload}
    blockers = aggregation._profile_q1b_record_blockers([record], profile)
    assert any("external_finetuning must be explicitly false" in blocker for blocker in blockers)
    assert any("optimizer steps must be explicitly zero" in blocker for blocker in blockers)
    assert any("training applicability must be explicitly NOT_APPLICABLE" in blocker for blocker in blockers)


def test_q1b_profile_rejects_null_nested_provenance(tmp_path: Path) -> None:
    profile = build_naacl_profile_snapshot(ROOT)
    edge = next(edge for edge in profile["q1b"]["consumer_edges"] if edge["seed"] is not None)
    payload = {
        "producer_id": edge["producer_id"],
        "producer_run_id": edge["producer_run_id"],
        "producer_kind": edge["producer_kind"],
        "checkpoint_key": edge["checkpoint_key"],
        "source_seed": edge["seed"],
        "dependency_graph_sha256": profile["graph"]["sha256"],
        "dependency_source_sha256": profile["source"]["sha256"],
        "external_finetuning": False,
        "train_loader_created": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "training_applicability": "NOT_APPLICABLE",
    }
    record = {"run_id": edge["consumer_id"], "run_root": str(tmp_path), "summary": payload}
    (tmp_path / "external").mkdir()
    (tmp_path / "external/external_evaluation_manifest.json").write_text(
        json.dumps({"producer": {"producer_id": None}}), encoding="utf-8"
    )
    blockers = aggregation._profile_q1b_record_blockers([record], profile)
    assert any("conflicting Q1b producer_id" in blocker for blocker in blockers)


def test_q1b_fixture_chain_emits_predictor_provenance_into_retention_and_review(tmp_path: Path) -> None:
    """Exercise the real predictor -> retention -> review propagation on CPU fixtures."""

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "src", tmp_path / "src")

    source_root = tmp_path / "results/runs/source"
    checkpoint = source_root / "checkpoints/best/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture-approved-checkpoint")
    summary = {
        "system_id": "phobert_multitask_8head",
        "seed": 20260521,
        "reusable_checkpoint_key": "phobert_multitask_8head:20260521",
        "variant_fingerprint": "fixture-multitask",
        "USER_REVIEW_STATUS": "PENDING",
        "NEXT_RUN_ALLOWED": "NO",
    }
    summary_path = source_root / "review_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    checksums_path = source_root / "checksums.sha256"
    checksums_path.write_text("fixture checksum list\n", encoding="utf-8")
    (source_root / "state.json").write_text(json.dumps({"run_id": source_root.name, "run_status": "APPROVED", "approval_status": "APPROVED"}), encoding="utf-8")
    (source_root / "checkpoints/checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "best": "checkpoints/best/model.pt",
                "checkpoint_sha256": sha256_file(checkpoint),
                "variant_fingerprint": "fixture-multitask",
            }
        ),
        encoding="utf-8",
    )
    timestamp = "2026-08-16T00:00:00Z"
    (source_root / "approval_status.json").write_text(
        json.dumps(
            {
                "run_id": source_root.name,
                "status": "APPROVED",
                "approved_by": "fixture-reviewer",
                "approved_at": timestamp,
                "record": {
                    "run_id": source_root.name,
                    "decision": "approve",
                    "review_note": "fixture approval",
                    "approved_or_rejected_by": "fixture-reviewer",
                    "timestamp": timestamp,
                    "review_summary_sha256": sha256_file(summary_path),
                    "artifact_checksum_file_sha256": sha256_file(checksums_path),
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "results/approved_run_index.json").write_text(
        json.dumps({"runs": [{"system": "phobert_multitask_8head", "seed": 20260521, "run_id": source_root.name}]}),
        encoding="utf-8",
    )

    external_root = tmp_path / "data/processed/external"
    dataset_specs = {
        "uit_vsfc": ("vsfc", "polarity", "positive"),
        "uit_vsmec": ("vsmec", "emotion", "enjoyment"),
        "aivivn_human_derived_3way": ("aivivn", "polarity", "neutral"),
    }
    manifest_items: dict[str, dict[str, str]] = {}
    for manifest_key, (dataset, label_column, label) in dataset_specs.items():
        path = external_root / dataset / "test.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"sample_id,text,{label_column}\n{dataset}-0,fixture,{label}\n", encoding="utf-8")
        manifest_items[manifest_key] = {"status": "PASS", "normalized_path": path.relative_to(tmp_path).as_posix(), "checksum": sha256_file(path)}
    (tmp_path / "data/manifests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/manifests/external_datasets.json").write_text(json.dumps({"datasets": manifest_items}), encoding="utf-8")

    class FixtureMultiTaskModel(nn.Module):
        def forward(self, **_: object) -> dict[str, object]:
            return {
                "logits": {
                    "polarity": torch.tensor([[0.0, 0.0, 1.0]]),
                    "emotion": torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                }
            }

    raw_entry = {
        "experiment_id": "q1b_phobert_multitask_8head_20260521",
        "research_question": "Q1b",
        "system_id": "phobert_multitask_8head",
        "display_name": "fixture Q1b multitask",
        "variant": "table3_checkpoint",
        "backbone": "phobert_base",
        "seed": 20260521,
        "execution_kind": "evaluation_only",
        "source_checkpoint_id": "phobert_multitask_8head:20260521",
        "external_finetuning": False,
        "_repository_root": str(tmp_path),
    }
    entry = RunEntry.from_mapping(raw_entry)
    run_root = tmp_path / "runs/fixture/results/runs" / entry.run_id
    context = RunContext(tmp_path, entry, fixture=True, run_root=run_root)
    store = RunStore(context)
    state = store.initialize()
    predictor = DiskBackedQ1BPredictor(
        tmp_path,
        raw_entry,
        model=FixtureMultiTaskModel(),
        tokenizer=SimpleNamespace(encode=lambda *_args, **_kwargs: [1, 2]),
    )
    result = evaluate_external_retention_from_disk(tmp_path, raw_entry, output_root=run_root, predictor=predictor)
    review = build_review_summary(context, entry, state)

    assert result["producer_id"] == predictor.source.producer_id
    assert result["producer_run_id"] == predictor.source.producer_run_id
    assert result["producer_kind"] == "trainable_checkpoint"
    assert result["dependency_graph_sha256"] == predictor.source.dependency_graph_sha256
    assert result["dependency_source_sha256"] == predictor.source.dependency_source_sha256
    assert review["producer_id"] == result["producer_id"]
    assert review["producer_run_id"] == result["producer_run_id"]
    assert review["producer_kind"] == result["producer_kind"]
    assert review["source_seed"] == 20260521
    assert review["external_finetuning"] is False
    assert review["train_loader_created"] is False
    assert review["optimizer_steps"] == 0
    assert review["backward_calls"] == 0
    assert review["training_applicability"] == "NOT_APPLICABLE"


def test_q3_aggregation_does_not_normalize_azure_seed_sentinels() -> None:
    profile = build_naacl_profile_snapshot(ROOT)
    rows = _profile_rows()
    rows[-1]["seed"] = "NOT_APPLICABLE"
    blockers = aggregation._profile_q3_record_blockers([{"summary": row} for row in rows], profile)
    assert blockers and "missing retained Azure" in blockers[0]


def test_q2_profile_requires_six_variants_and_three_locked_seeds() -> None:
    profile = build_naacl_profile_snapshot(ROOT)
    variants = profile["protocol_binding"]["q2"]["retained_variants"]
    records = [
        {"summary": {"variant": variant, "seed": seed}}
        for variant, seed in product(variants, LOCKED_SEEDS)
    ]
    assert aggregation._profile_q2_record_blockers(records, profile) == []
    records[-1]["summary"]["seed"] = 20260524
    blockers = aggregation._profile_q2_record_blockers(records, profile)
    assert blockers and "six retained variants and three locked seeds" in blockers[0]
