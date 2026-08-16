from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json, atomic_write_text
from ..azure.schemas import strict_rationale_schema
from ..hashing import sha256_file, sha256_json
from .approval import validate_approval_record
from .provenance import expected_inference_provenance
from .review import COMMON_FIELDS, validate_review_summary
from .run_store import RunStore, artifact_hashes, utc_now

RUN_ID = "azure_rationale_generation"
SUPPLEMENT_SUBMITTED_PATH = Path("results/runs/azure_rationale_generation/azure/manual_rationale_supplement.jsonl")
SUPPLEMENT_RECOVERY_PATH = Path("results/runs/azure_rationale_generation/azure/supplemental_azure_recovery.jsonl")
SUPPLEMENT_MANIFEST_PATH = Path("results/runs/azure_rationale_generation/azure/supplemental_recovery_manifest.json")
SUPPLEMENT_COST_PATH = Path("results/runs/azure_rationale_generation/azure/supplemental_cost_ledger.json")
CANDIDATE_PATH = Path("results/runs/azure_rationale_generation/azure/complete_rationale_candidate.jsonl")
CANDIDATE_MANIFEST_PATH = Path("results/runs/azure_rationale_generation/azure/complete_rationale_candidate_manifest.json")
HISTORY_PATH = Path("results/runs/azure_rationale_generation/azure/rationale_generation_history.json")
INTEGRATION_REPORT_PATH = Path("reports/azure_supplemental_azure_recovery.json")
INTEGRATION_MARKDOWN_PATH = Path("reports/azure_supplemental_azure_recovery.md")
LEGACY_REPORT_PATH = Path("reports/azure_manual_rationale_update.json")
LEGACY_MARKDOWN_PATH = Path("reports/azure_manual_rationale_update.md")

GENERATION_SOURCE = "azure_gpt_4_1_mini"
ORIGINAL_GENERATION_PHASE = "original_azure_rationale_generation"
SUPPLEMENTAL_GENERATION_PHASE = "supplemental_azure_recovery"
RECOVERY_REASON = "original_request_content_policy_blocked"
MODEL = "gpt-4.1-mini"
MODEL_VERSION = "2025-04-14"
DEPLOYMENT = "gpt-4.1-mini"
PROVIDER = "Azure OpenAI"
PROMPT_TEMPLATE = "Generate a rationale for this Vietnamese comment:\n{comment}"
MAX_OUTPUT_TOKENS = 256
RESPONSE_ID_UNAVAILABLE = "response_id_unavailable_in_submitted_artifact"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jsonl_text(rows: list[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _rationale_is_valid(value: Any, *, require_wrapper: bool = False) -> bool:
    if not isinstance(value, str) or not value.strip() or "<LABELS>" in value or "</LABELS>" in value:
        return False
    return not require_wrapper or bool(re.fullmatch(r"<RATIONALE>[\s\S]*</RATIONALE>", value.strip()))


def _prompt_hash(comment: str) -> str:
    return sha256_json({"prompt": PROMPT_TEMPLATE.format(comment=comment)})


def _schema_hash() -> str:
    return sha256_json({"strict": True, "schema": strict_rationale_schema()})


def _ids_hash(ids: set[str] | list[str]) -> str:
    return sha256_json(sorted(str(value) for value in ids))


def _recovery_context(root: Path) -> Any:
    return type(
        "RecoveryContext",
        (),
        {
            "root": root,
            "entry": type("Entry", (), {"run_id": RUN_ID, "is_azure": True})(),
            "fixture": False,
            "run_root": root / "results/runs" / RUN_ID,
        },
    )()


def _append_event_once(root: Path, event: str, payload: Mapping[str, Any]) -> None:
    store = RunStore(_recovery_context(root))
    if store.events_path.exists():
        for line in store.events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    if json.loads(line).get("event") == event:
                        return
                except json.JSONDecodeError:
                    continue
    store.append_event(event, payload)


def validate_supplemental_recovery(root: str | Path = ".") -> dict[str, Any]:
    """Validate the submitted 65-row recovery batch without making any API call."""

    root = Path(root)
    run_root = root / "results/runs" / RUN_ID
    input_path = root / "data/processed/rationales/azure_rationale_input_train.jsonl"
    failure_path = run_root / "azure/rationale_failures.json"
    original_path = run_root / "azure/rationale.jsonl"
    request_path = run_root / "azure/request_manifest.json"
    response_path = run_root / "azure/response_manifest.json"
    usage_path = run_root / "azure/usage.json"
    pricing_path = root / "reports/azure_pricing_snapshot.json"
    required = [input_path, failure_path, original_path, request_path, response_path, usage_path, pricing_path, root / SUPPLEMENT_SUBMITTED_PATH]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise ValueError("Azure recovery validation is missing: " + ", ".join(missing))

    frozen_rows = _jsonl(input_path)
    frozen_by_id = {str(row.get("sample_id")): row for row in frozen_rows}
    frozen_ids = list(frozen_by_id)
    if len(frozen_rows) != 7998 or len(frozen_by_id) != len(frozen_rows):
        raise ValueError("frozen Azure rationale input must contain 7,998 unique rows")

    failures = _load(failure_path)
    if not isinstance(failures, list):
        raise ValueError("Azure rationale failure artifact must be a list")
    failure_ids = [str(row.get("sample_id", "")) for row in failures]
    if len(failure_ids) != 65 or len(set(failure_ids)) != 65 or any(sample_id not in frozen_by_id for sample_id in failure_ids):
        raise ValueError("the authoritative Azure failure set must contain exactly 65 frozen sample IDs")

    submitted = _jsonl(root / SUPPLEMENT_SUBMITTED_PATH)
    submitted_ids = [str(row.get("sample_id", "")) for row in submitted]
    if len(submitted) != 65 or len(set(submitted_ids)) != 65 or set(submitted_ids) != set(failure_ids):
        raise ValueError("supplemental IDs do not exactly match the authoritative 65-row failure set")

    frozen_hashes = {sample_id: sha256_json(row) for sample_id, row in frozen_by_id.items()}
    for row in submitted:
        sample_id = str(row.get("sample_id", ""))
        frozen = frozen_by_id[sample_id]
        if row.get("comment") != frozen.get("comment"):
            raise ValueError(f"supplemental comment differs from frozen input: {sample_id}")
        if row.get("source_input_hash") != frozen_hashes[sample_id]:
            raise ValueError(f"supplemental source_input_hash differs from frozen input: {sample_id}")
        if not _rationale_is_valid(row.get("rationale_target"), require_wrapper=True):
            raise ValueError(f"supplemental rationale schema is invalid: {sample_id}")

    original = _jsonl(original_path)
    original_by_id: dict[str, dict[str, Any]] = {}
    for row in original:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in original_by_id or sample_id not in frozen_by_id:
            raise ValueError("original Azure rationale records contain duplicate or out-of-split IDs")
        if not _rationale_is_valid(row.get("rationale_target")):
            raise ValueError(f"original Azure rationale schema is invalid: {sample_id}")
        original_by_id[sample_id] = row
    if len(original_by_id) != 7933 or set(original_by_id) & set(submitted_ids):
        raise ValueError("original and supplemental Azure rationale IDs do not form disjoint 7,998-row coverage")

    request = _load(request_path)
    response = _load(response_path)
    usage = _load(usage_path)
    pricing = _load(pricing_path)
    if response.get("requested") != 7998 or response.get("successful") != 7933 or response.get("invalid") != 65 or response.get("failed") != 0 or response.get("missing") != 0:
        raise ValueError("original Azure response accounting is not the locked 7,933/65 result")
    if usage.get("total_azure_cost_usd") != 1.36426:
        raise ValueError("original Azure cost ledger no longer contains the verified $1.36426 cost")
    if str(request.get("deployment") or DEPLOYMENT) != DEPLOYMENT:
        raise ValueError("supplemental deployment does not match the locked Azure deployment")
    if pricing.get("model_version") not in (None, MODEL_VERSION):
        raise ValueError("supplemental model version does not match the locked deployment version")

    observed_models = {str(row.get("observed_model")) for row in original if row.get("observed_model")}
    observed_versions = {str(row.get("observed_model_version")) for row in original if row.get("observed_model_version")}
    schema_hashes = {str(row.get("schema_hash")) for row in original if row.get("schema_hash")}
    if observed_models and observed_models != {MODEL}:
        raise ValueError("original Azure response model provenance is inconsistent")
    if observed_versions and observed_versions != {MODEL_VERSION}:
        raise ValueError("original Azure response version provenance is inconsistent")
    if schema_hashes and len(schema_hashes) != 1:
        raise ValueError("original Azure rationale schema hashes are inconsistent")

    source_hashes = {
        "frozen_input_sha256": sha256_file(input_path),
        "failure_report_sha256": sha256_file(failure_path),
        "original_rationale_sha256": sha256_file(original_path),
        "submitted_supplement_sha256": sha256_file(root / SUPPLEMENT_SUBMITTED_PATH),
        "request_manifest_sha256": sha256_file(request_path),
        "response_manifest_sha256": sha256_file(response_path),
        "usage_sha256": sha256_file(usage_path),
        "pricing_snapshot_sha256": sha256_file(pricing_path),
    }
    return {
        "root": root,
        "run_root": run_root,
        "frozen_rows": frozen_rows,
        "frozen_by_id": frozen_by_id,
        "frozen_ids": frozen_ids,
        "frozen_hashes": frozen_hashes,
        "failures": failures,
        "failure_ids": failure_ids,
        "submitted": submitted,
        "submitted_ids": submitted_ids,
        "original": original,
        "original_by_id": original_by_id,
        "request": request,
        "response": response,
        "usage": usage,
        "pricing": pricing,
        "source_hashes": source_hashes,
        "schema_hash": next(iter(schema_hashes), _schema_hash()),
        "observed_model": next(iter(observed_models), MODEL),
        "observed_model_version": next(iter(observed_versions), MODEL_VERSION),
    }


def _supplemental_rows(validation: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in validation["submitted"]:
        sample_id = str(source["sample_id"])
        row = {
            "sample_id": sample_id,
            "comment": source["comment"],
            "rationale_target": source["rationale_target"],
            "source_input_hash": source["source_input_hash"],
            "generation_source": GENERATION_SOURCE,
            "generation_phase": SUPPLEMENTAL_GENERATION_PHASE,
            "recovery_reason": RECOVERY_REASON,
            "provider": PROVIDER,
            "model": MODEL,
            "deployment": str(validation["request"].get("deployment") or DEPLOYMENT),
            "configured_model_version": str(validation["pricing"].get("model_version") or MODEL_VERSION),
            "observed_model": None,
            "observed_model_version": None,
            "prompt_hash": _prompt_hash(str(source["comment"])),
            "schema_hash": validation["schema_hash"],
            "response_id": None,
            "usage": None,
            "provider_metadata_status": "response_id_and_usage_unavailable_in_submitted_artifact",
            "provenance_basis": "user_supplied_successful_azure_outputs",
        }
        row["source_record_hash"] = sha256_json(row)
        rows.append(row)
    return [row for row in sorted(rows, key=lambda item: item["sample_id"])]


def _candidate_rows(validation: Mapping[str, Any], supplemental: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    supplemental_by_id = {str(row["sample_id"]): dict(row) for row in supplemental}
    candidates: list[dict[str, Any]] = []
    for sample_id in validation["frozen_ids"]:
        if sample_id in validation["original_by_id"]:
            raw = dict(validation["original_by_id"][sample_id])
            raw.update(
                {
                    "comment": validation["frozen_by_id"][sample_id]["comment"],
                    "source_input_hash": validation["frozen_hashes"][sample_id],
                    "generation_source": GENERATION_SOURCE,
                    "generation_phase": ORIGINAL_GENERATION_PHASE,
                    "recovery_reason": None,
                    "provider": PROVIDER,
                    "source_kind": "original_successful_azure_response",
                    "source_record_hash": sha256_json(raw),
                }
            )
            candidates.append(raw)
        elif sample_id in supplemental_by_id:
            row = dict(supplemental_by_id[sample_id])
            row["source_kind"] = "supplemental_successful_azure_recovery"
            candidates.append(row)
        else:
            raise ValueError(f"complete rationale candidate is missing frozen sample ID: {sample_id}")
    if len(candidates) != len(validation["frozen_ids"]) or len({str(row["sample_id"]) for row in candidates}) != len(candidates):
        raise ValueError("complete rationale candidate does not have exactly one row per frozen sample")
    return candidates


def _cost_ledger(validation: Mapping[str, Any]) -> dict[str, Any]:
    prompt_bytes = sum(len(PROMPT_TEMPLATE.format(comment=str(row["comment"])).encode("utf-8")) for row in validation["submitted"])
    output_upper_bound = len(validation["submitted"]) * MAX_OUTPUT_TOKENS
    input_cost = prompt_bytes / 1_000_000 * 0.40
    cached_cost = 0.0
    output_cost = output_upper_bound / 1_000_000 * 1.60
    upper_bound = input_cost + cached_cost + output_cost
    known_azure_cost = float(validation["usage"]["total_azure_cost_usd"])
    return {
        "schema_version": 1,
        "status": "PASS",
        "cost_classification": "supplemental Azure GPT-4.1-mini generation estimated cost bound",
        "provider": PROVIDER,
        "model": MODEL,
        "model_version": str(validation["pricing"].get("model_version") or MODEL_VERSION),
        "deployment": str(validation["request"].get("deployment") or DEPLOYMENT),
        "logical_request_count": len(validation["submitted"]),
        "successful_response_records": len(validation["submitted"]),
        "retry_attempts_are_not_costed_separately": True,
        "provider_usage_available": False,
        "provider_usage_status": "unavailable_in_submitted_artifact",
        "non_cached_input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "exact_supplemental_cost_usd": None,
        "conservative_upper_bound": {
            "non_cached_input_tokens": prompt_bytes,
            "cached_input_tokens": 0,
            "output_tokens": output_upper_bound,
            "input_cost_usd": input_cost,
            "cached_input_cost_usd": cached_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": upper_bound,
            "basis": "one token per UTF-8 byte for frozen prompt text plus the locked 256-token output cap per recovered request",
        },
        "rates": {
            "currency": "USD",
            "unit": "per_1_million_tokens",
            "input_usd_per_1m": 0.40,
            "cached_input_usd_per_1m": 0.10,
            "output_usd_per_1m": 1.60,
        },
        "original_azure_cost_usd": known_azure_cost,
        "combined_conservative_upper_bound_usd": known_azure_cost + upper_bound,
        "exact_combined_billed_cost_usd": None,
        "exact_cost_unavailable_reason": "supplemental response usage was not included in the submitted successful-output artifact",
    }


def materialize_recovery_artifacts(root: str | Path = ".", validation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write deterministic, non-destructive recovery and candidate artifacts."""

    root = Path(root)
    validation = dict(validation or validate_supplemental_recovery(root))
    supplemental = _supplemental_rows(validation)
    candidates = _candidate_rows(validation, supplemental)
    cost = _cost_ledger(validation)

    atomic_write_text(root / SUPPLEMENT_RECOVERY_PATH, _jsonl_text(supplemental))
    atomic_write_text(root / CANDIDATE_PATH, _jsonl_text(candidates))

    candidate_manifest = {
        "schema_version": 1,
        "status": "PASS",
        "candidate_scope": "frozen_vipragsent_train",
        "candidate_record_count": len(candidates),
        "original_successful_count": len(validation["original"]),
        "supplemental_recovered_count": len(supplemental),
        "unresolved_count": 0,
        "sample_id_sha256": _ids_hash({str(row["sample_id"]) for row in candidates}),
        "frozen_input_sha256": validation["source_hashes"]["frozen_input_sha256"],
        "failure_report_sha256": validation["source_hashes"]["failure_report_sha256"],
        "candidate_file_sha256": sha256_file(root / CANDIDATE_PATH),
        "candidate_is_not_canonical": True,
        "canonical_promotion_required": True,
    }
    atomic_write_json(root / CANDIDATE_MANIFEST_PATH, candidate_manifest)

    supplemental_manifest = {
        "schema_version": 1,
        "status": "PASS",
        "job_id": RUN_ID,
        "generation_source": GENERATION_SOURCE,
        "generation_phase": SUPPLEMENTAL_GENERATION_PHASE,
        "recovery_reason": RECOVERY_REASON,
        "provider": PROVIDER,
        "model": MODEL,
        "model_version": cost["model_version"],
        "deployment": cost["deployment"],
        "provider_response_metadata": "response_id_and_usage_unavailable_in_submitted_artifact",
        "submitted_source_path": SUPPLEMENT_SUBMITTED_PATH.as_posix(),
        "submitted_source_sha256": validation["source_hashes"]["submitted_supplement_sha256"],
        "recovery_records_path": SUPPLEMENT_RECOVERY_PATH.as_posix(),
        "recovery_records_sha256": sha256_file(root / SUPPLEMENT_RECOVERY_PATH),
        "recovered_count": len(supplemental),
        "recovered_id_sha256": _ids_hash(set(validation["submitted_ids"])),
        "original_failure_report_path": "results/runs/azure_rationale_generation/azure/rationale_failures.json",
        "original_failure_report_sha256": validation["source_hashes"]["failure_report_sha256"],
        "frozen_input_path": "data/processed/rationales/azure_rationale_input_train.jsonl",
        "frozen_input_sha256": validation["source_hashes"]["frozen_input_sha256"],
        "original_failure_records_retained": True,
        "source_response_ids_fabricated": False,
        "usage_fabricated": False,
        "cost_ledger_path": SUPPLEMENT_COST_PATH.as_posix(),
    }
    atomic_write_json(root / SUPPLEMENT_MANIFEST_PATH, supplemental_manifest)
    atomic_write_json(root / SUPPLEMENT_COST_PATH, cost)

    history = [
        {
            "event_id": ORIGINAL_GENERATION_PHASE,
            "generation_source": GENERATION_SOURCE,
            "generation_phase": ORIGINAL_GENERATION_PHASE,
            "status": "PARTIAL_CONTENT_POLICY_BLOCKED",
            "successful_count": 7933,
            "content_policy_failure_count": 65,
            "unresolved_count_at_event": 65,
            "cost_usd": 1.36426,
            "response_manifest_sha256": validation["source_hashes"]["response_manifest_sha256"],
            "failure_report_sha256": validation["source_hashes"]["failure_report_sha256"],
            "failure_records_retained": True,
        },
        {
            "event_id": SUPPLEMENTAL_GENERATION_PHASE,
            "generation_source": GENERATION_SOURCE,
            "generation_phase": SUPPLEMENTAL_GENERATION_PHASE,
            "recovery_reason": RECOVERY_REASON,
            "status": "PASS",
            "successful_count": 65,
            "unresolved_count_at_event": 0,
            "provider": PROVIDER,
            "model": MODEL,
            "model_version": cost["model_version"],
            "deployment": cost["deployment"],
            "provider_response_metadata": "response_id_and_usage_unavailable_in_submitted_artifact",
            "estimated_cost_upper_bound_usd": cost["conservative_upper_bound"]["total_cost_usd"],
            "cost_classification": cost["cost_classification"],
            "recovery_manifest_sha256": sha256_file(root / SUPPLEMENT_MANIFEST_PATH),
        },
    ]
    atomic_write_json(root / HISTORY_PATH, history)

    report = {
        "schema_version": 2,
        "status": "VALIDATED_CANDIDATE",
        "report_type": "azure_supplemental_recovery_integration",
        "job_id": RUN_ID,
        "coverage": {
            "frozen_train_input_rows": 7998,
            "original_successful_azure_rows": 7933,
            "original_content_policy_failure_rows": 65,
            "supplemental_successful_azure_rows": 65,
            "candidate_logical_rationale_rows": len(candidates),
            "unresolved_rows": 0,
        },
        "validation": {
            "status": "PASS",
            "ids_exact_match": True,
            "source_hashes_exact_match": True,
            "frozen_comments_exact_match": True,
            "rationale_schema_validation": "PASS",
            "empty_rationales": 0,
            "label_blocks_found": 0,
            "model_family_mixing": False,
            "original_failure_records_retained": True,
        },
        "provenance": {
            **validation["source_hashes"],
            "generation_source": GENERATION_SOURCE,
            "original_generation_phase": ORIGINAL_GENERATION_PHASE,
            "supplemental_generation_phase": SUPPLEMENTAL_GENERATION_PHASE,
            "recovery_reason": RECOVERY_REASON,
            "provider": PROVIDER,
            "model": MODEL,
            "model_version": cost["model_version"],
            "deployment": cost["deployment"],
            "response_ids_for_supplement": "unavailable_and_not_fabricated",
            "usage_for_supplement": "unavailable_and_not_fabricated",
            "history_path": HISTORY_PATH.as_posix(),
        },
        "artifacts": {
            "supplemental_recovery_path": SUPPLEMENT_RECOVERY_PATH.as_posix(),
            "supplemental_manifest_path": SUPPLEMENT_MANIFEST_PATH.as_posix(),
            "supplemental_cost_path": SUPPLEMENT_COST_PATH.as_posix(),
            "candidate_path": CANDIDATE_PATH.as_posix(),
            "candidate_manifest_path": CANDIDATE_MANIFEST_PATH.as_posix(),
        },
        "cost": cost,
        "canonical_promotion": "PENDING",
        "downstream_started": False,
        "scientific_protocol_changed": False,
    }
    atomic_write_json(root / INTEGRATION_REPORT_PATH, report)
    atomic_write_text(root / INTEGRATION_MARKDOWN_PATH, _render_report_markdown(report))
    return {"validation": validation, "supplemental": supplemental, "candidates": candidates, "cost": cost, "report": report}


def _render_report_markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    cost = report["cost"]
    bound = cost["conservative_upper_bound"]
    return "\n".join(
        [
            "# Supplemental Azure rationale recovery",
            "",
            f"Status: **{report['status']}**",
            "",
            "The original Azure history is retained: 7,933 successful responses and 65 content-policy failures. The 65 later successful recovery outputs are recorded as a supplemental Azure GPT-4.1-mini recovery batch.",
            "",
            "## Coverage",
            "",
            f"- Frozen input rows: **{coverage['frozen_train_input_rows']}**",
            f"- Original successful Azure rows: **{coverage['original_successful_azure_rows']}**",
            f"- Original content-policy failures retained: **{coverage['original_content_policy_failure_rows']}**",
            f"- Supplemental successful Azure rows: **{coverage['supplemental_successful_azure_rows']}**",
            f"- Complete candidate rows: **{coverage['candidate_logical_rationale_rows']}**",
            f"- Unresolved rows: **{coverage['unresolved_rows']}**",
            "",
            "## Provenance",
            "",
            f"- Generation source: `{report['provenance']['generation_source']}`",
            f"- Supplemental phase: `{report['provenance']['supplemental_generation_phase']}`",
            f"- Recovery reason: `{report['provenance']['recovery_reason']}`",
            f"- Provider/model/deployment: `{report['provenance']['provider']}` / `{report['provenance']['model']}` / `{report['provenance']['deployment']}`",
            f"- Configured model version: `{report['provenance']['model_version']}`",
            "- Supplemental response IDs and provider usage are unavailable in the submitted artifact and were not fabricated.",
            "- The original failure report remains unchanged.",
            "",
            "## Cost",
            "",
            "The original verified Azure cost remains **$1.36426 USD**. Supplemental provider usage is unavailable, so this is an estimated upper bound, not an exact billed amount.",
            "",
            f"- Supplemental input upper bound: `{bound['non_cached_input_tokens']}` tokens",
            f"- Supplemental cached input upper bound: `{bound['cached_input_tokens']}` tokens",
            f"- Supplemental output upper bound: `{bound['output_tokens']}` tokens",
            f"- Supplemental estimated cost upper bound: **${bound['total_cost_usd']:.7f} USD**",
            f"- Combined conservative upper bound: **${cost['combined_conservative_upper_bound_usd']:.7f} USD**",
            "- Exact combined billed cost: unavailable because supplemental provider usage was not supplied.",
            "",
            "## Artifacts",
            "",
            f"- Recovery records: `{report['artifacts']['supplemental_recovery_path']}`",
            f"- Complete candidate: `{report['artifacts']['candidate_path']}`",
            f"- Generation history: `{report['provenance']['history_path']}`",
            "- The candidate is not canonical until the official approval and promotion mechanism succeeds.",
            "",
        ]
    )


def _build_azure_review_summary(root: Path, state: Mapping[str, Any], recovery: Mapping[str, Any]) -> dict[str, Any]:
    run_root = root / "results/runs" / RUN_ID
    usage = recovery["validation"]["usage"]
    validated_at = state.get("rationale_recovery", {}).get("validated_at", "NOT_APPLICABLE")
    data_manifest = root / "data/manifests/dataset_manifest.json"
    artifacts = artifact_hashes(run_root)
    summary: dict[str, Any] = {field: "NOT_APPLICABLE" for field in COMMON_FIELDS}
    summary.update(
        {
            "run_id": RUN_ID,
            "research_question": "setup",
            "system_id": RUN_ID,
            "display_name": "Azure rationale generation",
            "variant": "rationale_generation",
            "backbone": "azure",
            "seed": "NOT_APPLICABLE",
            "budget": "NOT_APPLICABLE",
            "execution_kind": "azure",
            "execution_mode": "production_sequential_review_gated",
            "run_status": "PASS",
            "user_review_status": "PENDING",
            "next_run_allowed": "NO",
            "dataset_fingerprint": sha256_file(data_manifest) if data_manifest.exists() else recovery["validation"]["source_hashes"]["frozen_input_sha256"],
            "split_hashes": {"train": recovery["validation"]["source_hashes"]["frozen_input_sha256"], "dev": "NOT_APPLICABLE", "test": "NOT_APPLICABLE"},
            "model_repository": "azure_openai",
            "model_revision": recovery["cost"]["model_version"],
            "tokenizer_revision": "NOT_APPLICABLE",
            "preprocessing_name": "NOT_APPLICABLE",
            "preprocessing_version": "NOT_APPLICABLE",
            "configuration_hash": sha256_file(run_root / "config_snapshot.yaml") if (run_root / "config_snapshot.yaml").exists() else "NOT_APPLICABLE",
            "code_commit": state.get("code_commit") or "unknown",
            "start_time": state.get("created_at") or "NOT_APPLICABLE",
            "end_time": validated_at,
            "wall_clock_seconds": 0.0,
            "warnings": ["Original 65 content-policy failures are retained as immutable history; supplemental recovery evidence completes logical coverage."],
            "blockers": [],
            "validation_status": "PASS",
            "artifact_paths": sorted(artifacts),
            "artifact_sha256": artifacts,
            "RUN_STATUS": "PASS",
            "USER_REVIEW_STATUS": "PENDING",
            "NEXT_RUN_ALLOWED": "NO",
            "not_applicable_reason": "Azure rationale generation has no local model training checkpoint or prediction split.",
            "azure_request_count": int(usage.get("request_count", 7998)),
            "azure_input_tokens": int(usage.get("input_tokens", 0)),
            "azure_cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
            "azure_non_cached_input_tokens": int(usage.get("non_cached_input_tokens", 0)),
            "azure_output_tokens": int(usage.get("output_tokens", 0)),
            "azure_cost_usd": float(usage.get("total_azure_cost_usd", 1.36426)),
            "azure_non_cached_input_cost_usd": float(usage.get("non_cached_input_cost_usd", 0.0)),
            "azure_cached_input_cost_usd": float(usage.get("cached_input_cost_usd", 0.0)),
            "azure_output_cost_usd": float(usage.get("output_cost_usd", 0.0)),
            "azure_cost_accounting_method": usage.get("cost_accounting_method", "NOT_APPLICABLE"),
            "azure_cost_verification_status": usage.get("cost_verification_status", "NOT_APPLICABLE"),
            "azure_usage_records_path": usage.get("usage_records_path", "azure/usage_records.jsonl"),
            "azure_cost_ledger_path": usage.get("cost_ledger_path", "azure/cost_ledger.json"),
            "azure_invalid_output_rate": 0.0,
            "azure_cache_hits": int(usage.get("cache_hits", 0)),
            "azure_cache_misses": int(usage.get("cache_misses", 0)),
            "azure_failed_requests": 0,
            "azure_retried_requests": int(usage.get("retried_requests", 0)),
            "rationale_recovery": {
                "original_successful_count": 7933,
                "original_content_policy_failure_count": 65,
                "supplemental_successful_count": 65,
                "cumulative_successful_count": 7998,
                "unresolved_count": 0,
                "supplemental_cost_classification": recovery["cost"]["cost_classification"],
            },
            "azure_original_response_invalid_count": 65,
            "azure_original_response_invalid_rate": float(recovery["validation"]["response"].get("invalid", 65)) / 7998,
            "azure_supplemental_usage_status": "UNAVAILABLE_NOT_FABRICATED",
        }
    )
    summary.update(expected_inference_provenance(RUN_ID, execution_kind="azure"))
    summary["summary_hash_input"] = sha256_json({key: value for key, value in summary.items() if key not in {"artifact_paths", "artifact_sha256", "summary_hash_input"}})
    errors = validate_review_summary(summary, completed=True)
    if errors:
        raise ValueError("recovered Azure review summary is invalid: " + "; ".join(errors))
    return summary


def _write_source_state(root: Path, recovery: Mapping[str, Any]) -> dict[str, Any]:
    run_root = root / "results/runs" / RUN_ID
    state_path = run_root / "state.json"
    state = _load(state_path)
    approval = _load(run_root / "approval_status.json") if (run_root / "approval_status.json").exists() else {"status": "PENDING_USER_APPROVAL"}
    if state.get("run_status") == "REJECTED" or approval.get("status") == "REJECTED":
        raise ValueError("a rejected Azure rationale run cannot be silently reopened")
    if "original_validate_responses_stage" not in state:
        state["original_validate_responses_stage"] = dict(state.get("stages", {}).get("validate_responses", {}))
    recovery_meta = dict(state.get("rationale_recovery", {}))
    recovery_meta.setdefault("validated_at", utc_now())
    recovery_meta.update(
        {
            "status": "PASS",
            "generation_source": GENERATION_SOURCE,
            "generation_phase": SUPPLEMENTAL_GENERATION_PHASE,
            "recovery_reason": RECOVERY_REASON,
            "original_successful_count": 7933,
            "original_content_policy_failure_count": 65,
            "supplemental_successful_count": 65,
            "cumulative_successful_count": 7998,
            "unresolved_count": 0,
            "original_failure_report_retained": True,
            "canonical_promotion_status": recovery_meta.get("canonical_promotion_status", "PENDING"),
            "approval_basis": "standing_user_authorization_after_successful_audit",
        }
    )
    state["rationale_recovery"] = recovery_meta
    stages = dict(state.get("stages", {}))
    stages["validate_responses"] = {
        "status": "PASS",
        "recovery_finalization": True,
        "summary": {
            "original_stage_status": "FAIL",
            "original_content_policy_failure_count": 65,
            "supplemental_recovered_count": 65,
            "cumulative_successful_count": 7998,
            "unresolved_count": 0,
        },
        "blockers": [],
        "warnings": ["Original content-policy failures remain preserved in azure/rationale_failures.json."],
    }
    for stage in ("export_artifacts", "validate_artifacts", "generate_review_summary"):
        stages[stage] = {
            "status": "PASS",
            "recovery_finalization": True,
            "summary": {"rationale_coverage": "7998/7998", "canonical_promotion_required": stage != "generate_review_summary"},
            "blockers": [],
            "warnings": [],
        }
    state["stages"] = stages
    state["run_status"] = "APPROVED" if approval.get("status") == "APPROVED" else "COMPLETED_PENDING_APPROVAL"
    state["approval_status"] = "APPROVED" if approval.get("status") == "APPROVED" else "PENDING_USER_APPROVAL"
    state["next_run_allowed"] = "NO"
    state["blockers"] = []
    state["rationale_dependency_status"] = "PASS_CANDIDATE_VALIDATED_PENDING_CANONICAL_PROMOTION"
    if state.get("run_status") != "APPROVED":
        state["updated_at"] = utc_now()
    atomic_write_json(state_path, state)
    _append_event_once(
        root,
        "azure_supplemental_recovery_validated",
        {
            "successful_count": 65,
            "cumulative_successful_count": 7998,
            "unresolved_count": 0,
            "original_failure_report_retained": True,
            "generation_phase": SUPPLEMENTAL_GENERATION_PHASE,
        },
    )

    metrics_path = run_root / "metrics.json"
    metrics = _load(metrics_path) if metrics_path.exists() else {"run_id": RUN_ID}
    metrics.update(
        {
            "status": "PASS",
            "rationale_coverage": {"original_successful": 7933, "supplemental_recovered": 65, "cumulative_successful": 7998, "unresolved": 0},
            "supplemental_recovery_manifest": SUPPLEMENT_MANIFEST_PATH.as_posix(),
            "supplemental_cost_ledger": SUPPLEMENT_COST_PATH.as_posix(),
            "original_cost_usd_preserved": 1.36426,
            "supplemental_cost_classification": recovery["cost"]["cost_classification"],
            "supplemental_cost_upper_bound_usd": recovery["cost"]["conservative_upper_bound"]["total_cost_usd"],
            "combined_cost_upper_bound_usd": recovery["cost"]["combined_conservative_upper_bound_usd"],
        }
    )
    atomic_write_json(metrics_path, metrics)
    provenance_path = run_root / "provenance.json"
    provenance = _load(provenance_path) if provenance_path.exists() else {"run_id": RUN_ID}
    provenance.update(
        {
            "rationale_generation_history_path": HISTORY_PATH.as_posix(),
            "rationale_generation_history_sha256": sha256_file(root / HISTORY_PATH),
            "supplemental_recovery_manifest_path": SUPPLEMENT_MANIFEST_PATH.as_posix(),
            "supplemental_recovery_manifest_sha256": sha256_file(root / SUPPLEMENT_MANIFEST_PATH),
            "complete_rationale_candidate_path": CANDIDATE_PATH.as_posix(),
            "complete_rationale_candidate_sha256": sha256_file(root / CANDIDATE_PATH),
            "original_failure_report_sha256": recovery["validation"]["source_hashes"]["failure_report_sha256"],
            "generation_source": GENERATION_SOURCE,
            "supplemental_generation_phase": SUPPLEMENTAL_GENERATION_PHASE,
            "recovery_reason": RECOVERY_REASON,
            "provider_metadata_status": "response_id_and_usage_unavailable_in_submitted_artifact",
        }
    )
    atomic_write_json(provenance_path, provenance)
    return state


def integrate_azure_rationale_recovery(root: str | Path = ".", *, reviewer: str = "standing_user_authorization_after_successful_audit") -> dict[str, Any]:
    """Validate, finalize, officially approve, and promote the supplied recovery batch."""

    root = Path(root)
    validation = validate_supplemental_recovery(root)
    recovery = materialize_recovery_artifacts(root, validation)
    state = _write_source_state(root, recovery)
    run_root = root / "results/runs" / RUN_ID
    approval_path = run_root / "approval_status.json"
    approval = _load(approval_path) if approval_path.exists() else {"status": "PENDING_USER_APPROVAL"}

    store = RunStore(_recovery_context(root))
    store.write_checksums()
    if approval.get("status") != "APPROVED":
        review = _build_azure_review_summary(root, state, recovery)
        atomic_write_json(run_root / "review_summary.json", review)
        atomic_write_text(root / "results/runs/azure_rationale_generation/review_summary.md", "# Sequential Run Review Summary\n\nRUN_STATUS: PASS\nUSER_REVIEW_STATUS: PENDING\nNEXT_RUN_ALLOWED: NO\n\n" + "Recovery validation passed; awaiting the explicitly authorized official approval record.\n")
        from .approval import record_run_approval

        record_run_approval(
            root,
            RUN_ID,
            decision="approve",
            review_note="Supplemental Azure GPT-4.1-mini recovery validated against the frozen 65-sample failure set; original failure history and $1.36426 Azure cost preserved; approval_basis=standing_user_authorization_after_successful_audit.",
            reviewer=reviewer,
        )
        approval = _load(approval_path)
    else:
        approval_errors = validate_approval_record(run_root, expected_run_id=RUN_ID)
        if approval_errors:
            raise ValueError("existing Azure approval is incomplete: " + "; ".join(approval_errors))
        existing_review = _load(run_root / "review_summary.json")
        errors = validate_review_summary(existing_review, completed=True)
        if errors:
            raise ValueError("existing approved Azure review summary is invalid: " + "; ".join(errors))

    from .rationale_promotion import promote_approved_rationales

    promotion = promote_approved_rationales(root, source_run_id=RUN_ID, train_ids=validation["frozen_ids"])
    state = _load(run_root / "state.json")
    state["rationale_recovery"]["canonical_promotion_status"] = "PASS"
    state["rationale_dependency_status"] = "PASS"
    state["canonical_rationale_promotion"] = {
        "status": "PASS",
        "canonical_path": promotion["canonical_path"],
        "manifest_path": promotion["manifest_path"],
        "canonical_file_sha256": promotion["canonical_file_sha256"],
        "coverage_count": promotion["manifest"]["successful_count"],
        "unresolved_count": promotion["manifest"].get("unresolved_count", 0),
    }
    state["next_action"] = "Run official preflight for q1a_cot_only_vistral_20260521; continue only after PASS."
    atomic_write_json(run_root / "state.json", state)
    _append_event_once(root, "canonical_rationale_promoted", {"canonical_file_sha256": promotion["canonical_file_sha256"], "coverage_count": 7998})

    report = recovery["report"]
    report["status"] = "PASS"
    report["canonical_promotion"] = promotion
    report["approval"] = {
        "status": approval.get("status"),
        "approval_basis": "standing_user_authorization_after_successful_audit",
        "approval_record_sha256": sha256_json(approval.get("record", {})),
    }
    report["protocol_state"] = {
        "original_partial_content_policy_result_retained": True,
        "supplemental_recovery_pass": True,
        "cumulative_coverage": 7998,
        "unresolved": 0,
        "rationale_dependency": "PASS",
        "canonical_promotion": "PASS",
        "scientific_protocol_changed": False,
        "downstream_started": False,
        "next_action": "q1a_cot_only_vistral_20260521 official preflight",
    }
    atomic_write_json(root / INTEGRATION_REPORT_PATH, report)
    atomic_write_text(root / INTEGRATION_MARKDOWN_PATH, _render_report_markdown(report) + "\nCanonical promotion: **PASS**\nApproval basis: `standing_user_authorization_after_successful_audit`\n")
    legacy = {
        "status": "SUPERSEDED",
        "superseded_by": INTEGRATION_REPORT_PATH.as_posix(),
        "reason": "The submitted 65-row recovery batch was explicitly clarified as successful supplemental Azure GPT-4.1-mini output.",
        "source_artifact_retained": SUPPLEMENT_SUBMITTED_PATH.as_posix(),
    }
    atomic_write_json(root / LEGACY_REPORT_PATH, legacy)
    atomic_write_text(root / LEGACY_MARKDOWN_PATH, "# Superseded Azure recovery report\n\nThis report is superseded by [the authoritative supplemental Azure recovery report](azure_supplemental_azure_recovery.md). The original submitted artifact remains retained and its provenance is recorded there.\n")
    return {"status": "PASS", "validation": recovery["report"]["validation"], "cost": recovery["cost"], "promotion": promotion, "canonical_path": promotion["canonical_path"], "approval": approval}
