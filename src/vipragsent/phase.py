from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PHASE15_SMOKE_TESTS = (
    "offline tokenizer load",
    "offline model load",
    "forward",
    "backward",
    "finite loss",
    "gradient checks",
)

_PHASE15_SMOKE_CHECKS = {
    "tokenizer_load": "offline tokenizer load",
    "smoke_tokenizer_load": "offline tokenizer load",
    "model_load": "offline model load",
    "smoke_model_load": "offline model load",
    "forward": "forward",
    "smoke_forward": "forward",
    "backward": "backward",
    "smoke_backward": "backward",
    "finite_loss": "finite loss",
    "smoke_finite_loss": "finite loss",
    "gradient_presence": "gradient checks",
    "smoke_gradient_presence": "gradient checks",
}

_PHASE15_STATUS_DIRS = {
    "cache": Path("data/model_cache_status"),
    "smoke": Path("data/model_smoke_status"),
    "batch": Path("data/batch_probe_status"),
}


@dataclass
class PhaseHandoff:
    phase: str
    status: str
    inputs_read: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    tests_passed: bool = False
    blockers: list[str] = field(default_factory=list)
    next_phase_ready: bool = False
    phase15_evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "BLOCKED", "FAIL"}:
            raise ValueError(f"Invalid phase status: {self.status}")

    def write(self, report_root: str | Path = "reports/phases") -> tuple[Path, Path]:
        root = Path(report_root)
        root.mkdir(parents=True, exist_ok=True)
        stem = root / f"phase_{self.phase}"
        payload = asdict(self)
        if not self.phase15_evidence:
            payload.pop("phase15_evidence", None)
        payload["generated_at_utc"] = datetime.now(UTC).isoformat()
        handoff_path = stem.with_name(stem.name + "_handoff.json")
        status_path = stem.with_name(stem.name + "_status.md")
        handoff_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        lines = [
            f"# Phase {self.phase} status",
            "",
            f"- Status: `{self.status}`",
            f"- Tests passed: `{self.tests_passed}`",
            f"- Next phase ready: `{self.next_phase_ready}`",
            *([f"- Approval basis: `{self.phase15_evidence['approval_basis']}`"] if self.phase15_evidence.get("approval_basis") else []),
            "",
            "## Inputs read",
            *[f"- `{item}`" for item in self.inputs_read],
            "",
            "## Files created",
            *[f"- `{item}`" for item in self.files_created],
            "",
            "## Tests run",
            *[f"- `{item}`" for item in self.tests_run],
            "",
            "## Blockers",
            *([f"- {item}" for item in self.blockers] or ["- None"]),
            "",
        ]
        status_path.write_text("\n".join(lines), encoding="utf-8")
        return status_path, handoff_path


def _read_json(path: Path) -> tuple[bool, dict[str, Any]]:
    if not path.exists():
        return False, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return True, {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return True, {"status": "FAIL", "error": "status artifact is not a JSON object"}
    return True, payload


def _project_root(report_root: str | Path) -> Path:
    path = Path(report_root)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.parent.parent


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        value = str(item)
        if value and value not in result:
            result.append(value)
    return result


def _phase15_family(root: Path, report_root: Path, requested: str | None) -> str | None:
    if requested:
        return requested
    manifest_exists, manifest = _read_json(root / "data/model_cache_manifest.json")
    if manifest_exists:
        for key in ("requested_model_family", "selected_model_family"):
            value = manifest.get(key)
            if isinstance(value, str) and value:
                return value
    _, previous = _read_json(report_root / "phase_15_handoff.json")
    evidence = previous.get("phase15_evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("model_family"), str):
        return str(evidence["model_family"])
    for key in ("model_family", "selected_model_family"):
        value = previous.get(key)
        if isinstance(value, str) and value:
            return value
    families = {
        path.stem
        for directory in _PHASE15_STATUS_DIRS.values()
        for path in (root / directory).glob("*.json")
    }
    return next(iter(families)) if len(families) == 1 else None


def _phase15_artifact(root: Path, family: str, category: str) -> tuple[bool, dict[str, Any], Path]:
    path = root / _PHASE15_STATUS_DIRS[category] / f"{family}.json"
    exists, payload = _read_json(path)
    return exists, payload, path


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _cache_valid(cache: dict[str, Any]) -> bool:
    return (
        cache.get("status") == "PASS"
        and _has_text(cache.get("repo_id"))
        and _has_text(cache.get("revision"))
        and _has_text(cache.get("tokenizer_revision"))
    )


def _smoke_valid(smoke: dict[str, Any]) -> bool:
    checks = smoke.get("checks")
    return (
        smoke.get("status") == "PASS"
        and smoke.get("actual_local_loads") is True
        and not smoke.get("blockers")
        and isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values())
    )


def _batch_valid(batch: dict[str, Any]) -> bool:
    successful_batch = batch.get("successful_batch")
    return (
        batch.get("status") == "PASS"
        and batch.get("frozen") is True
        and batch.get("fixture_probe") is not True
        and isinstance(successful_batch, int)
        and not isinstance(successful_batch, bool)
        and successful_batch > 0
    )


def _smoke_tests_run(smoke: dict[str, Any]) -> list[str]:
    if smoke.get("status") == "PASS":
        return list(PHASE15_SMOKE_TESTS)
    checks = smoke.get("checks")
    if isinstance(checks, dict):
        names = _dedupe(_PHASE15_SMOKE_CHECKS[key] for key in checks if key in _PHASE15_SMOKE_CHECKS)
        return names or ["offline smoke validation"]
    return ["offline smoke validation"]


def _artifact_blockers(family: str, category: str, exists: bool, artifact: dict[str, Any], valid: bool) -> list[str]:
    label = {"cache": "cache", "smoke": "offline smoke", "batch": "physical batch"}[category]
    if not exists:
        return [f"{family}: {label} status artifact is missing"]
    blockers: list[str] = []
    raw_blockers = artifact.get("blockers")
    if isinstance(raw_blockers, list):
        blockers.extend(str(item) for item in raw_blockers if item)
    if _has_text(artifact.get("error")):
        blockers.append(f"{family}: {artifact['error']}")
    if not valid:
        status = artifact.get("status", "UNKNOWN")
        if category == "cache" and status == "PASS":
            for field_name in ("repo_id", "revision", "tokenizer_revision"):
                if not _has_text(artifact.get(field_name)):
                    blockers.append(f"{family}: cache evidence is missing {field_name}")
        elif category == "smoke" and status == "PASS":
            if artifact.get("actual_local_loads") is not True:
                blockers.append(f"{family}: smoke evidence is not an actual local load")
            checks = artifact.get("checks")
            if isinstance(checks, dict):
                blockers.extend(f"{family}: smoke check failed: {key}" for key, value in checks.items() if value is not True)
        elif category == "batch" and status == "PASS":
            if artifact.get("fixture_probe") is True:
                blockers.append(f"{family}: physical batch evidence is fixture-only")
            if artifact.get("frozen") is not True:
                blockers.append(f"{family}: physical batch evidence is not frozen")
            if artifact.get("successful_batch") is None:
                blockers.append(f"{family}: physical batch evidence has no successful batch")
        if not blockers:
            blockers.append(f"{family}: {label} status is {status}")
    if category == "batch" and artifact.get("status") != "PASS":
        failures = artifact.get("failed_candidates")
        if isinstance(failures, list):
            blockers.extend(
                f"{family}: batch {item.get('batch')} failed: {item.get('reason')}"
                for item in failures
                if isinstance(item, dict) and item.get("reason")
            )
    return blockers


def _phase15_handoff_values(
    root: Path,
    report_root: Path,
    *,
    family: str | None,
    incoming_status: str,
    incoming_blockers: Iterable[str],
    incoming_files: Iterable[str],
    incoming_inputs: Iterable[str],
    explicit_family: bool,
    approval_basis: str | None,
) -> dict[str, Any]:
    if not family:
        return {
            "status": "FAIL" if incoming_status == "FAIL" else "BLOCKED",
            "inputs_read": _dedupe(incoming_inputs),
            "files_created": _dedupe(incoming_files),
            "tests_run": [],
            "tests_passed": False,
            "blockers": _dedupe(incoming_blockers) or ["Phase 15 model family is not selected"],
            "next_phase_ready": False,
            "phase15_evidence": {},
        }

    artifacts = {
        category: _phase15_artifact(root, family, category)
        for category in _PHASE15_STATUS_DIRS
    }
    cache_exists, cache, cache_path = artifacts["cache"]
    smoke_exists, smoke, smoke_path = artifacts["smoke"]
    batch_exists, batch, batch_path = artifacts["batch"]
    cache_valid = _cache_valid(cache)
    smoke_valid = _smoke_valid(smoke)
    batch_valid = _batch_valid(batch)
    blockers: list[str] = []
    for category, (exists, artifact, _) in artifacts.items():
        valid = {"cache": cache_valid, "smoke": smoke_valid, "batch": batch_valid}[category]
        blockers.extend(_artifact_blockers(family, category, exists, artifact, valid))

    for field_name, label in (("repo_id", "repository"), ("revision", "model revision"), ("tokenizer_revision", "tokenizer revision")):
        cache_value = cache.get(field_name)
        smoke_value = smoke.get(field_name)
        if _has_text(cache_value) and _has_text(smoke_value) and cache_value != smoke_value:
            blockers.append(f"{family}: {label} differs between cache and smoke evidence")

    manifest_exists, manifest = _read_json(root / "data/model_cache_manifest.json")
    if manifest_exists:
        model_records = manifest.get("models")
        selected = next((item for item in model_records if isinstance(item, dict) and item.get("name") == family), None) if isinstance(model_records, list) else None
        if isinstance(selected, dict):
            for field_name, label in (("revision", "model revision"), ("tokenizer_revision", "tokenizer revision")):
                selected_value = selected.get(field_name)
                cache_value = cache.get(field_name)
                if _has_text(selected_value) and _has_text(cache_value) and selected_value != cache_value:
                    blockers.append(f"{family}: {label} differs between cache and manifest evidence")

    if explicit_family:
        blockers.extend(str(item) for item in incoming_blockers if item)
    blockers = _dedupe(blockers)
    statuses = [artifact.get("status") for exists, artifact, _ in artifacts.values() if exists]
    if "FAIL" in statuses:
        status = "FAIL"
    elif not (cache_valid and smoke_valid and batch_valid) or blockers:
        status = "BLOCKED"
    else:
        status = "PASS"
    tests_run: list[str] = []
    if cache_exists:
        tests_run.append("locked cache/revision validation")
    if smoke_exists:
        tests_run.extend(_smoke_tests_run(smoke))
    if batch_exists:
        tests_run.append("physical batch probe")
    tests_run = _dedupe(tests_run)

    relative_paths = {
        "cache_manifest": root / "data/model_cache_manifest.json",
        "cache_status": cache_path,
        "smoke_status": smoke_path,
        "batch_status": batch_path,
    }
    smoke_report = root / "data/model_smoke_report.json"
    smoke_report_exists, smoke_report_payload = _read_json(smoke_report)
    if smoke_report_exists and smoke_report_payload.get("selected_model_family", smoke_report_payload.get("model_family")) == family:
        relative_paths["smoke_report"] = smoke_report
    files_created = list(incoming_files)
    files_created.extend(_path_string(path, root) for path in relative_paths.values() if path.exists())
    inputs_read = list(incoming_inputs)
    if manifest_exists:
        inputs_read.append("data/model_cache_manifest.json")
    inputs_read.append("configs/models/model_registry.yaml")
    evidence = {
        "model_family": family,
        "repo_id": cache.get("repo_id") or smoke.get("repo_id"),
        "model_revision": cache.get("revision") or smoke.get("revision"),
        "tokenizer_revision": cache.get("tokenizer_revision") or smoke.get("tokenizer_revision"),
        "physical_batch": batch.get("successful_batch"),
        "effective_batch_size": batch.get("effective_batch_size"),
        "gradient_accumulation_steps": batch.get("gradient_accumulation_steps"),
        "hardware_identity": batch.get("hardware_identity"),
        "local_snapshot": _path_string(Path(cache.get("local_path")), root) if cache.get("local_path") else None,
        "hashes": {
            "cache_manifest_hash": cache.get("manifest_hash"),
            "smoke_verification_hash": smoke.get("verification_hash"),
            "batch_probe_hash": batch.get("probe_hash"),
        },
        "status_artifacts": {key: _path_string(path, root) for key, path in relative_paths.items()},
        "validation": {"cache": cache_valid, "smoke": smoke_valid, "physical_batch": batch_valid},
    }
    previous_exists, previous = _read_json(report_root / "phase_15_handoff.json")
    previous_evidence = previous.get("phase15_evidence") if previous_exists else None
    preserved_basis = previous_evidence.get("approval_basis") if isinstance(previous_evidence, dict) and previous_evidence.get("model_family") == family else None
    if status == "PASS" and (approval_basis or preserved_basis):
        evidence["approval_basis"] = approval_basis or preserved_basis
    return {
        "status": status,
        "inputs_read": _dedupe(inputs_read),
        "files_created": _dedupe(files_created),
        "tests_run": tests_run,
        "tests_passed": status == "PASS" and cache_valid and smoke_valid and batch_valid and not blockers,
        "blockers": blockers,
        "next_phase_ready": status == "PASS" and cache_valid and smoke_valid and batch_valid and not blockers,
        "phase15_evidence": evidence,
    }


def _path_string(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def write_phase_handoff(
    phase: str,
    status: str,
    inputs_read: Iterable[str] = (),
    files_created: Iterable[str] = (),
    tests_run: Iterable[str] = (),
    tests_passed: bool = False,
    blockers: Iterable[str] = (),
    next_phase_ready: bool = False,
    report_root: str | Path = "reports/phases",
    model_family: str | None = None,
    approval_basis: str | None = None,
) -> PhaseHandoff:
    phase_name = f"{int(phase):02d}"
    input_values = list(inputs_read)
    file_values = list(files_created)
    test_values = list(tests_run)
    blocker_values = list(blockers)
    evidence: dict[str, Any] = {}
    if phase_name == "15":
        report_path = Path(report_root)
        if not report_path.is_absolute():
            report_path = Path.cwd() / report_path
        family = _phase15_family(_project_root(report_root), report_path, model_family)
        resolved = _phase15_handoff_values(
            _project_root(report_root),
            report_path,
            family=family,
            incoming_status=status,
            incoming_blockers=blocker_values,
            incoming_files=file_values,
            incoming_inputs=input_values,
            explicit_family=model_family is not None,
            approval_basis=approval_basis,
        )
        status = resolved["status"]
        input_values = resolved["inputs_read"]
        file_values = resolved["files_created"]
        test_values = resolved["tests_run"]
        tests_passed = resolved["tests_passed"]
        blocker_values = resolved["blockers"]
        next_phase_ready = resolved["next_phase_ready"]
        evidence = resolved["phase15_evidence"]
    handoff = PhaseHandoff(
        phase=phase_name,
        status=status,
        inputs_read=input_values,
        files_created=file_values,
        tests_run=test_values,
        tests_passed=tests_passed,
        blockers=blocker_values,
        next_phase_ready=next_phase_ready,
        phase15_evidence=evidence,
    )
    handoff.write(report_root)
    return handoff


def inspect_phase15_handoff(root: str | Path, model_family: str | None = None) -> dict[str, Any]:
    """Reconcile Phase 15 evidence without rewriting the authoritative handoff.

    Status writers may be invoked independently, so a Phase 15 state reader must
    validate the per-family cache, smoke, and physical-batch artifacts together.
    A prior FAIL/BLOCKED handoff is retained until an explicit status writer
    finalizes a new PASS; a downloader or setup refresh cannot promote it.
    """
    project_root = Path(root).resolve()
    report_root = project_root / "reports/phases"
    handoff_exists, previous = _read_json(report_root / "phase_15_handoff.json")
    family = _phase15_family(project_root, report_root, model_family)
    if not family:
        return {
            "status": "BLOCKED",
            "tests_passed": False,
            "blockers": ["Phase 15 model family is not selected"],
            "next_phase_ready": False,
            "phase15_evidence": {},
            "handoff_exists": handoff_exists,
            "previous_status": previous.get("status"),
        }

    resolved = _phase15_handoff_values(
        project_root,
        report_root,
        family=family,
        incoming_status=str(previous.get("status", "BLOCKED")),
        incoming_blockers=previous.get("blockers", []),
        incoming_files=previous.get("files_created", []),
        incoming_inputs=previous.get("inputs_read", []),
        explicit_family=False,
        approval_basis=None,
    )
    blockers = _dedupe(resolved["blockers"])
    previous_evidence = previous.get("phase15_evidence")
    previous_family = previous_evidence.get("model_family") if isinstance(previous_evidence, dict) else None
    if not handoff_exists:
        blockers.append("Phase 15 authoritative handoff is missing")
    if previous_family and previous_family != family:
        blockers.append("Phase 15 handoff family does not match the selected model family")

    previous_status = previous.get("status")
    if previous_status in {"FAIL", "BLOCKED"}:
        blockers.extend(str(item) for item in previous.get("blockers", []) if item)
        if resolved["status"] == "PASS":
            blockers.append("Phase 15 terminal status requires explicit finalization after the failed attempt")
        status = "FAIL" if previous_status == "FAIL" or resolved["status"] == "FAIL" else "BLOCKED"
    elif previous_status == "PASS":
        if previous.get("tests_passed") is not True or previous.get("next_phase_ready") is not True:
            blockers.append("Phase 15 PASS handoff metadata is incomplete")
        if previous.get("blockers"):
            blockers.extend(str(item) for item in previous["blockers"] if item)
        status = resolved["status"]
    else:
        status = "BLOCKED"
        blockers.append("Phase 15 handoff has not been finalized")

    blockers = _dedupe(blockers)
    if status != "PASS" or blockers:
        status = "FAIL" if status == "FAIL" else "BLOCKED"
    return {
        "status": status,
        "tests_passed": status == "PASS" and not blockers,
        "blockers": blockers,
        "next_phase_ready": status == "PASS" and not blockers,
        "phase15_evidence": resolved.get("phase15_evidence", {}),
        "handoff_exists": handoff_exists,
        "previous_status": previous_status,
    }
