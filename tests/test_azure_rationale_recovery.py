from __future__ import annotations

import json
from pathlib import Path

from vipragsent.azure.schemas import strict_rationale_schema
from vipragsent.hashing import sha256_file, sha256_json
from vipragsent.orchestration.azure_rationale_recovery import (
    CANDIDATE_MANIFEST_PATH,
    CANDIDATE_PATH,
    SUPPLEMENT_COST_PATH,
    SUPPLEMENT_MANIFEST_PATH,
    SUPPLEMENT_RECOVERY_PATH,
    SUPPLEMENT_SUBMITTED_PATH,
    materialize_recovery_artifacts,
    validate_supplemental_recovery,
)
from vipragsent.orchestration.provenance import expected_inference_provenance
from vipragsent.orchestration.rationale_promotion import (
    load_approved_rationales,
    promote_approved_rationales,
)
from vipragsent.orchestration.review import COMMON_FIELDS
from vipragsent.orchestration.run_store import RunStore

RUN_ID = "azure_rationale_generation"
SCHEMA_HASH = sha256_json({"strict": True, "schema": strict_rationale_schema()})


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _fixture_root(tmp_path: Path) -> tuple[Path, list[str], list[str]]:
    root = tmp_path
    run_root = root / "results/runs" / RUN_ID
    azure = run_root / "azure"
    frozen_ids = [f"sample-{index:04d}" for index in range(7998)]
    failure_ids = frozen_ids[-65:]
    frozen = [{"sample_id": sample_id, "comment": f"bình luận {sample_id}", "gold_labels": {}} for sample_id in frozen_ids]
    frozen_hashes = {row["sample_id"]: sha256_json(row) for row in frozen}
    _write_jsonl(root / "data/processed/rationales/azure_rationale_input_train.jsonl", frozen)
    _write_json(azure / "rationale_failures.json", [{"sample_id": sample_id, "status": "FAILED", "error": "content policy"} for sample_id in failure_ids])
    original = []
    for sample_id in frozen_ids[:-65]:
        comment = next(row["comment"] for row in frozen if row["sample_id"] == sample_id)
        original.append(
            {
                "sample_id": sample_id,
                "rationale_target": "<RATIONALE>verified cue</RATIONALE>",
                "deployment": "gpt-4.1-mini",
                "observed_model": "gpt-4.1-mini",
                "observed_model_version": "2025-04-14",
                "prompt_hash": sha256_json({"prompt": f"Generate a rationale for this Vietnamese comment:\n{comment}"}),
                "schema_hash": SCHEMA_HASH,
                "response_id": f"response-{sample_id}",
                "usage": {},
            }
        )
    _write_jsonl(azure / "rationale.jsonl", original)
    _write_json(azure / "request_manifest.json", {"deployment": "gpt-4.1-mini", "requested": 7998})
    _write_json(azure / "response_manifest.json", {"requested": 7998, "successful": 7933, "invalid": 65, "failed": 0, "missing": 0})
    _write_json(azure / "usage.json", {"request_count": 7998, "total_azure_cost_usd": 1.36426, "input_tokens": 1, "output_tokens": 1})
    _write_json(azure / "cost_ledger.json", {"status": "PASS", "cost_accounting_method": "USER_SUPPLIED_RATES_ACTUAL_SUCCESSFUL_USAGE", "cost_verification_status": "LOCAL_USAGE_ACCOUNTING"})
    _write_json(azure / "cache_manifest.json", {"cache_entries": 0})
    _write_jsonl(azure / "usage_records.jsonl", [])
    _write_jsonl(azure / "invalid_outputs.jsonl", [])
    _write_json(root / "reports/azure_pricing_snapshot.json", {"model_version": "2025-04-14"})
    submitted = [
        {
            "sample_id": sample_id,
            "comment": next(row["comment"] for row in frozen if row["sample_id"] == sample_id),
            "rationale_target": "<RATIONALE>recovered cue</RATIONALE>",
            "source_input_hash": frozen_hashes[sample_id],
        }
        for sample_id in failure_ids
    ]
    _write_jsonl(root / SUPPLEMENT_SUBMITTED_PATH, submitted)
    return root, frozen_ids, failure_ids


def test_supplemental_recovery_validates_exact_ids_hashes_and_schema(tmp_path: Path) -> None:
    root, _, failure_ids = _fixture_root(tmp_path)

    validation = validate_supplemental_recovery(root)

    assert validation["frozen_ids"][-65:] == failure_ids
    assert set(validation["failure_ids"]) == set(validation["submitted_ids"])
    assert len(validation["original"]) == 7933


def test_recovery_artifacts_are_idempotent_and_do_not_mix_model_families(tmp_path: Path) -> None:
    root, _, _ = _fixture_root(tmp_path)
    first = materialize_recovery_artifacts(root)
    hashes_first = {path: sha256_file(root / path) for path in (SUPPLEMENT_RECOVERY_PATH, SUPPLEMENT_MANIFEST_PATH, SUPPLEMENT_COST_PATH, CANDIDATE_PATH, CANDIDATE_MANIFEST_PATH)}

    second = materialize_recovery_artifacts(root)
    hashes_second = {path: sha256_file(root / path) for path in hashes_first}

    assert first["cost"]["combined_conservative_upper_bound_usd"] == second["cost"]["combined_conservative_upper_bound_usd"]
    assert hashes_first == hashes_second
    rows = [json.loads(line) for line in (root / SUPPLEMENT_RECOVERY_PATH).read_text(encoding="utf-8").splitlines()]
    assert {row["generation_source"] for row in rows} == {"azure_gpt_4_1_mini"}
    assert {row["generation_phase"] for row in rows} == {"supplemental_azure_recovery"}
    assert all(row["provider_metadata_status"] == "response_id_and_usage_unavailable_in_submitted_artifact" for row in rows)
    assert all(row["response_id"] is None and row["usage"] is None for row in rows)


def _approved_source(root: Path, frozen_ids: list[str]) -> None:
    run_root = root / "results/runs" / RUN_ID
    _write_json(run_root / "state.json", {"run_id": RUN_ID, "run_status": "APPROVED", "approval_status": "APPROVED"})
    summary = {field: "NOT_APPLICABLE" for field in COMMON_FIELDS}
    summary.update(
        {
            "run_id": RUN_ID,
            "research_question": "setup",
            "system_id": RUN_ID,
            "execution_kind": "azure",
            "run_status": "PASS",
            "user_review_status": "PENDING",
            "next_run_allowed": "NO",
            "RUN_STATUS": "PASS",
            "USER_REVIEW_STATUS": "PENDING",
            "NEXT_RUN_ALLOWED": "NO",
            "validation_status": "PASS",
            "artifact_paths": ["azure/rationale.jsonl"],
            "artifact_sha256": {"azure/rationale.jsonl": sha256_file(run_root / "azure/rationale.jsonl")},
            "not_applicable_reason": "Azure rationale generation has no local training checkpoint.",
            **expected_inference_provenance(RUN_ID, execution_kind="azure"),
        }
    )
    _write_json(run_root / "review_summary.json", summary)
    _write_json(run_root / "approval_status.json", {"run_id": RUN_ID, "status": "PENDING_USER_APPROVAL"})
    context = type("Context", (), {"root": root, "entry": type("Entry", (), {"run_id": RUN_ID, "is_azure": True})(), "fixture": False, "run_root": run_root})()
    RunStore(context).write_checksums()
    summary_hash = sha256_file(run_root / "review_summary.json")
    checksum_hash = sha256_file(run_root / "checksums.sha256")
    timestamp = "2026-08-16T00:00:00Z"
    _write_json(
        run_root / "approval_status.json",
        {
            "run_id": RUN_ID,
            "status": "APPROVED",
            "approved_by": "fixture-reviewer",
            "approved_at": timestamp,
            "record": {
                "run_id": RUN_ID,
                "decision": "approve",
                "review_note": "fixture approval",
                "approved_or_rejected_by": "fixture-reviewer",
                "timestamp": timestamp,
                "review_summary_sha256": summary_hash,
                "artifact_checksum_file_sha256": checksum_hash,
            },
        },
    )


def test_official_promotion_merges_recovery_and_preserves_original_failure_history(tmp_path: Path) -> None:
    root, frozen_ids, _ = _fixture_root(tmp_path)
    materialize_recovery_artifacts(root)
    _approved_source(root, frozen_ids)

    result = promote_approved_rationales(root, train_ids=frozen_ids)
    records = load_approved_rationales(root)

    assert result["status"] == "PASS"
    assert len(records) == 7998
    assert sum(row.get("generation_phase") == "supplemental_azure_recovery" for row in records.values()) == 65
    assert result["manifest"]["historical_original_failed_count"] == 65
    assert result["manifest"]["failed_count"] == 0
    assert result["manifest"]["unresolved_count"] == 0
    assert sum(row.get("source_response_id") is None for row in records.values()) == 65
