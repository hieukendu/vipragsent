from __future__ import annotations

from itertools import product
from pathlib import Path

import pytest

from vipragsent.orchestration import aggregation
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


def test_aggregation_entry_validates_the_current_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def reject(_root: Path) -> dict[str, object]:
        raise ProfileValidationError("current graph/source digest drift")

    monkeypatch.setattr(aggregation, "validate_naacl_profile", reject)
    result = aggregation.aggregate_approved_scope(tmp_path, "Q3")
    assert result["status"] == "BLOCKED"
    assert "current graph/source digest drift" in result["blockers"][0]


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
