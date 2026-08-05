from __future__ import annotations

import importlib.util
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..hashing import sha256_file, sha256_json
from ..protocol import validate_protocol_resolution
from ..runtime.disk import derive_minimum_free_disk_bytes
from ..runtime.hardware import validate_hardware
from ..runtime.model_assets import read_family_status
from .contracts import VALID_EXECUTION_KINDS, RunEntry
from .run_store import git_commit, git_worktree_clean
from .system_registry import resolve_execution_spec


def _registry(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "configs/models/model_registry.yaml"
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(name): dict(value) for name, value in payload.get("models", {}).items()}


def _check(checks: dict[str, dict[str, Any]], name: str, passed: bool, *, detail: str, required: bool = True) -> None:
    checks[name] = {"passed": bool(passed), "required": required, "detail": detail}


def _mask_hash(root: Path, budget: str | int | None) -> tuple[Path | None, str | None]:
    if budget in (None, "", "None"):
        return None, None
    path = root / "data/processed/q3_low_resource_sarcasm" / f"budget_{budget}_masks.csv"
    return path, sha256_file(path) if path.exists() else None


def run_single_preflight(
    root: str | Path,
    entry_mapping: Mapping[str, Any] | RunEntry,
    *,
    kind: str = "experiment",
    run_id: str | None = None,
    fixture: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run exact, per-entry checks without touching model weights, Azure, or training."""
    root = Path(root)
    entry = entry_mapping if isinstance(entry_mapping, RunEntry) else RunEntry.from_mapping(entry_mapping, run_id=run_id)
    checks: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    warnings: list[str] = []

    protocol = validate_protocol_resolution(root)
    protocol_conflicts = list(protocol.get("scientific_protocol_conflicts", []))
    _check(checks, "protocol_conflicts_empty", not protocol_conflicts, detail="; ".join(protocol_conflicts) if protocol_conflicts else "no scientific protocol conflicts")
    blockers.extend(protocol_conflicts)

    required = ["run_id", "research_question", "system_id", "display_name", "variant", "backbone", "execution_kind"]
    missing = [field for field in required if getattr(entry, field, None) in (None, "")]
    _check(checks, "inventory_entry_schema", not missing, detail="missing fields: " + ", ".join(missing) if missing else "all required entry fields are present")
    if missing:
        blockers.append("inventory entry is missing: " + ", ".join(missing))
    _check(checks, "execution_kind_exact", entry.execution_kind in VALID_EXECUTION_KINDS, detail=f"execution_kind={entry.execution_kind}")
    if entry.execution_kind not in VALID_EXECUTION_KINDS:
        blockers.append(f"unsupported execution_kind: {entry.execution_kind}")

    execution_spec = None
    if not entry.is_azure:
        try:
            execution_spec = resolve_execution_spec(root, entry.system_id)
            registry_ok = execution_spec.variant_id != ""
            registry_detail = f"executor={execution_spec.executor_kind}; variant={execution_spec.variant_id}"
        except ValueError as exc:
            registry_ok = False
            registry_detail = str(exc)
        _check(checks, "exact_system_execution_registry", registry_ok, detail=registry_detail)
        if not registry_ok:
            blockers.append(registry_detail)
    else:
        _check(checks, "exact_system_execution_registry", True, detail="Azure jobs use the dedicated Azure job inventory", required=False)

    inventory_path = root / "reports/expected_experiment_runs.json"
    rows: list[Mapping[str, Any]] = []
    if inventory_path.exists():
        try:
            rows = list(json.loads(inventory_path.read_text(encoding="utf-8")).get("rows", []))
        except (OSError, json.JSONDecodeError):
            rows = []
    matches = [row for row in rows if str(row.get("experiment_id") or row.get("run_id")) == entry.run_id or str(row.get("job_id")) == entry.run_id]
    unique = kind == "azure" or len(matches) == 1
    _check(checks, "exact_entry_exists_once", unique, detail=f"matched_entries={len(matches)}")
    if not unique:
        blockers.append(f"exact run ID {entry.run_id!r} does not resolve to exactly one inventory entry")

    data_manifest = root / "data/manifests/dataset_manifest.json"
    data_present = data_manifest.exists() and (root / "data/processed/vipragsent").exists()
    _check(checks, "dataset_manifest_and_processed_data", data_present, detail=str(data_manifest) if data_present else "dataset manifest or processed data is missing", required=not fixture)
    if not data_present and not fixture:
        blockers.append("exact dataset manifest and processed ViPragSent data are required")
    if entry.raw.get("dataset_fingerprint") and data_manifest.exists():
        expected = str(entry.raw["dataset_fingerprint"])
        actual = sha256_file(data_manifest)
        matches_hash = expected == actual
        _check(checks, "dataset_fingerprint_exact", matches_hash, detail=f"expected={expected}; actual={actual}")
        if not matches_hash:
            blockers.append("dataset fingerprint does not match the inventory entry")
    else:
        _check(checks, "dataset_fingerprint_exact", fixture or data_manifest.exists(), detail="inventory has no per-entry override; frozen dataset manifest is authoritative", required=not fixture)

    registry = _registry(root)
    if entry.is_azure:
        _check(checks, "model_family_exact", entry.backbone == "azure", detail="Azure job has no local model family")
        from ..azure.client import AzureSettings

        try:
            settings = AzureSettings.from_env()
            azure_settings_ok = True
            azure_detail = settings.redacted()
        except ValueError as exc:
            azure_settings_ok = False
            azure_detail = str(exc)
        _check(checks, "azure_settings_exact", fixture or azure_settings_ok, detail="fixture transport" if fixture else str(azure_detail), required=not fixture)
        if not azure_settings_ok and not fixture:
            blockers.append(f"Azure deployment configuration is invalid: {azure_detail}")
        if entry.variant == "rationale_generation":
            prompt_path = root / "data/manifests/prompts/pragmatic_v1.json"
            input_path = root / "data/processed/rationales/azure_rationale_input_train.jsonl"
        elif entry.research_question == "Q3":
            prompt_path = root / "data/manifests/prompts" / f"q3_budget_{entry.budget}_v1.json"
            input_path = root / "data/processed/vipragsent/test.csv"
        else:
            prompt_path = root / "data/manifests/prompts" / f"{entry.task}_v1.json"
            input_path = root / "data/processed/vipragsent/test.csv"
        _check(checks, "azure_prompt_manifest_exact", fixture or prompt_path.exists(), detail=str(prompt_path), required=not fixture)
        _check(checks, "azure_input_split_exact", fixture or input_path.exists(), detail=str(input_path), required=not fixture)
        if not fixture and not prompt_path.exists():
            blockers.append(f"Azure prompt manifest is missing: {prompt_path.name}")
        if not fixture and not input_path.exists():
            blockers.append(f"Azure input split is missing: {input_path}")
    else:
        model_family = str(entry.raw.get("model_family") or entry.backbone)
        spec = registry.get(model_family)
        _check(checks, "model_family_exact", spec is not None, detail=f"model_family={model_family}")
        if spec is None and not fixture:
            blockers.append(f"exact model family is not in the locked registry: {model_family}")
        if spec is not None:
            pinned = bool(spec.get("revision")) and bool(spec.get("tokenizer_revision"))
            _check(checks, "model_revisions_pinned", pinned, detail=f"revision={spec.get('revision')}; tokenizer_revision={spec.get('tokenizer_revision')}")
            if not pinned:
                blockers.append(f"model family {model_family} has an unpinned revision")
            expected_revision = entry.model_revision or entry.raw.get("model_revision")
            expected_tokenizer = entry.tokenizer_revision or entry.raw.get("tokenizer_revision")
            revision_match = fixture or expected_revision in (None, "", spec.get("revision"))
            tokenizer_match = fixture or expected_tokenizer in (None, "", spec.get("tokenizer_revision"))
            _check(checks, "model_revision_exact", revision_match, detail=f"expected={expected_revision}; locked={spec.get('revision')}")
            _check(checks, "tokenizer_revision_exact", tokenizer_match, detail=f"expected={expected_tokenizer}; locked={spec.get('tokenizer_revision')}")
            if not revision_match:
                blockers.append("exact model revision does not match the locked registry")
            if not tokenizer_match:
                blockers.append("exact tokenizer revision does not match the locked registry")

            cache = read_family_status(root, model_family, "cache")
            smoke = read_family_status(root, model_family, "smoke")
            batch = read_family_status(root, model_family, "batch")
            cache_pass = fixture or cache.get("status") == "PASS"
            smoke_pass = fixture or smoke.get("status") == "PASS"
            batch_pass = fixture or (batch.get("status") == "PASS" and batch.get("frozen") is True and batch.get("successful_batch") is not None)
            _check(checks, "phase15_family_cache_pass", cache_pass, detail=f"family={model_family}; status={cache.get('status')}")
            _check(checks, "offline_smoke_report_pass", smoke_pass, detail=f"family={model_family}; status={smoke.get('status')}")
            _check(checks, "physical_batch_frozen", batch_pass, detail=f"family={model_family}; status={batch.get('status')}; batch={batch.get('successful_batch')}")
            if not fixture:
                if not cache_pass:
                    blockers.append(f"Phase 15 family cache is not PASS for {model_family}")
                if not smoke_pass:
                    blockers.append(f"offline smoke report is not PASS for {model_family}")
                if not batch_pass:
                    blockers.append(f"physical batch probe is not frozen for {model_family}")

            packages = list(entry.raw.get("required_python_packages", []))
            if spec.get("quantization") == "nf4":
                packages.extend(["peft", "bitsandbytes"])
            package_failures = [name for name in packages if importlib.util.find_spec(str(name)) is None]
            package_ok = fixture or not package_failures
            _check(checks, "required_python_packages", package_ok, detail="missing: " + ", ".join(package_failures) if package_failures else "all required packages are installed", required=not fixture)
            if package_failures and not fixture:
                blockers.append("required Python packages are unavailable: " + ", ".join(package_failures))

            hardware = validate_hardware(root) if not fixture else {"status": "PASS", "checks": {"fixture_cpu_path": True}, "blockers": []}
            gpu_ok = fixture or hardware.get("status") == "PASS"
            _check(checks, "required_gpu_and_precision", gpu_ok, detail="fixture CPU path" if fixture else json.dumps(hardware, sort_keys=True), required=not fixture)
            if not gpu_ok:
                blockers.append("required GPU/precision runtime is unavailable: " + ", ".join(hardware.get("blockers", [])))

            rationale_required = "rationale" in (entry.task + ";" + entry.dependencies).casefold()
            rationale_path = root / "data/processed/rationales/azure_rationale_input_train.jsonl"
            rationale_ok = not rationale_required or fixture or rationale_path.exists()
            _check(checks, "rationale_dependency", rationale_ok, detail="not applicable" if not rationale_required else str(rationale_path), required=rationale_required and not fixture)
            if not rationale_ok:
                blockers.append("required rationale dependency is missing")

    q3_path, q3_hash = _mask_hash(root, entry.budget if entry.research_question == "Q3" else None)
    if entry.research_question == "Q3":
        expected_mask = entry.q3_mask_hash or entry.raw.get("mask_hash") or entry.raw.get("q3_mask_hash")
        mask_match = fixture or (q3_hash is not None and (expected_mask in (None, "") or expected_mask == q3_hash))
        _check(checks, "q3_mask_present_and_exact", mask_match, detail=f"path={q3_path}; expected={expected_mask}; actual={q3_hash}")
        if not mask_match:
            blockers.append("Q3 budget/mask hash is missing or does not match")
        if not fixture and (root / "data/processed/vipragsent").exists():
            try:
                from ..data.loaders import load_vipragsent
                from ..data.masks import validate_q3_masks

                train = load_vipragsent(root / "data/processed/vipragsent").train
                q3_report = validate_q3_masks(root / "data/processed/q3_low_resource_sarcasm", {item.sample_id: item for item in train}, strict_frozen=True)
                _check(checks, "q3_mask_semantics_deep", True, detail=json.dumps({"counts": q3_report["selected_positive_counts"], "fixed_negative_count": q3_report["fixed_negative_count"], "nested": q3_report["nested"]}, sort_keys=True))
            except Exception as exc:
                _check(checks, "q3_mask_semantics_deep", False, detail=str(exc))
                blockers.append("Q3 mask semantic validation failed: " + str(exc))
    else:
        _check(checks, "q3_mask_present_and_exact", True, detail="not applicable", required=False)

    if entry.research_question == "Q4":
        source = entry.source_checkpoint_id or entry.raw.get("source_checkpoint_id")
        source_ok = fixture or bool(source)
        _check(checks, "q4_approved_source_checkpoint", source_ok, detail=str(source) if source else "source checkpoint is not specified")
        _check(checks, "q4_source_seed", fixture or entry.seed in (20260521, 20260522, 20260523), detail=f"seed={entry.seed}")
        _check(checks, "q4_raw_pragmatic_probability_source", fixture or bool(entry.raw.get("raw_probability_source") or source), detail="raw pragmatic probabilities are required")
        if not fixture and not source_ok:
            blockers.append("Q4 approved source checkpoint is missing")

    if entry.research_question == "Q1b":
        external_manifest = root / "data/manifests/external_datasets.json"
        external_ok = fixture or external_manifest.exists()
        external_finetuning = bool(entry.raw.get("external_finetuning", False))
        _check(checks, "external_official_test_manifests", external_ok, detail=str(external_manifest))
        _check(checks, "external_finetuning_false", not external_finetuning, detail=f"external_finetuning={external_finetuning}")
        if not external_ok:
            blockers.append("official external test manifests are missing")
        if external_finetuning:
            blockers.append("Q1b requires external_finetuning=false")

    run_root = (root / "runs/fixture/results/runs" / entry.run_id) if fixture else (root / "results/runs" / entry.run_id)
    existing_state = run_root / "state.json"
    overwrite_ok = True
    if existing_state.exists():
        try:
            existing = json.loads(existing_state.read_text(encoding="utf-8"))
            overwrite_ok = str(existing.get("run_id")) == entry.run_id
        except (OSError, json.JSONDecodeError):
            overwrite_ok = False
    _check(checks, "no_different_completed_run_overwrite", overwrite_ok, detail=str(run_root))
    if not overwrite_ok:
        blockers.append("a different completed run occupies the canonical run directory")

    writable = True
    try:
        run_root.mkdir(parents=True, exist_ok=True)
        writable = os.access(run_root, os.W_OK)
    except OSError:
        writable = False
    _check(checks, "output_path_writable", writable, detail=str(run_root))
    if not writable:
        blockers.append("canonical output path is not writable")
    disk_evidence = derive_minimum_free_disk_bytes(root, str(entry.backbone)) if not entry.is_azure else {"minimum_free_bytes": 0, "available_free_bytes": shutil.disk_usage(root).free, "passed": True}
    disk_ok = fixture or bool(disk_evidence.get("passed"))
    _check(checks, "sufficient_disk_space", disk_ok, detail=json.dumps(disk_evidence, sort_keys=True), required=not fixture)
    if not disk_ok:
        blockers.append("insufficient disk space for the requested run")

    commit = git_commit(root)
    _check(checks, "git_commit_recorded", commit not in {"", "unknown"}, detail=commit)
    _check(checks, "git_worktree_recorded", True, detail="clean" if git_worktree_clean(root) else "dirty changes recorded; no files are discarded", required=False)
    if commit == "unknown" and not fixture:
        blockers.append("Git code commit could not be recorded")

    if dry_run:
        warnings.append("dry-run: no model, GPU, Azure, network, or real experiment was accessed")
    if fixture:
        warnings.append("fixture: runtime checks use fake assets and CPU-only synthetic data")

    return {
        "schema_version": 2,
        "kind": kind,
        "run_id": entry.run_id,
        "execution_kind": entry.execution_kind,
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "warnings": warnings,
        "checks": checks,
        "scientific_protocol_conflicts": protocol_conflicts,
        "code_commit": git_commit(root),
        "run_path": run_root.relative_to(root).as_posix(),
        "preflight_hash": sha256_json(checks),
    }
