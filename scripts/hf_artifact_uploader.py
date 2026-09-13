"""Independent, resumable uploader for local experiment artifacts.

The scientific runner only appends queue records.  This process watches the
local run directories, waits for a file to be unchanged for one poll, uploads
it to an existing Hub repository, and verifies the remote path and byte size
before recording success.  It never creates a repository.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import signal
import time
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, get_token

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.hashing import sha256_file

CAMPAIGN_ID = "vipragsent-v8-local-mig2g20gb"
ARTIFACT_REPOSITORIES = (
    # Keep new campaigns away from the full private repository.  These are
    # existing repositories; the allocator spreads whole runs across them.
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-021",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-018",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-025",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-017",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-019",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-016",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-015",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-014",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-013",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-012",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-011",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-010",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-009",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-008",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-007",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-006",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-005",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-004",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-003",
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-002",
    "Thundergod2007/vipragsent-experiment-artifacts",
)
INITIAL_ENTRY_COUNTS = {
    "Thundergod2007/vipragsent-experiment-artifacts": 21156,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-002": 24670,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-003": 20018,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-004": 19944,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-005": 28665,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-006": 28693,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-007": 19947,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-008": 19944,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-009": 19924,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-010": 19937,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-011": 19900,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-012": 19809,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-013": 19921,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-014": 19902,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-015": 19962,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-016": 19914,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-017": 11801,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-018": 720,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-019": 22893,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-020": 20089,
    # This repository is already at the Hub's file-count ceiling despite the
    # old allocation snapshot reporting zero entries.  Keep it in the legacy
    # list for resumability, but never allocate new runs to it.
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-021": 20000,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-022": 471,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-023": 14694,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-024": 23470,
    "Thundergod2007/vipragsent-experiment-artifacts-overflow-025": 2661,
}
CHECKPOINT_REPOSITORIES = {
    "phobert_base": "Thundergod2007/vipragsent-phobert-checkpoints",
    "sailor_7b": "Thundergod2007/vipragsent-sailor7b-checkpoints",
    "vistral_7b": "Thundergod2007/vipragsent-vistral7b-checkpoints",
    "xlmr_large": "Thundergod2007/vipragsent-xlmr-checkpoints",
}
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".ckpt")
DEFAULT_QUEUE = ROOT / "runtime/hf_upload_queue.jsonl"
DEFAULT_STATUS = ROOT / "runtime/hf_uploader_status.json"
DEFAULT_STOP = ROOT / "runtime/STOP_UPLOADER"
# The Hub rejects commits once a model repository would contain 20,000 files.
# Keep a small reserve because one run uploads several files in separate
# commits and the historical counts below are only allocation hints.
REPOSITORY_ENTRY_CAP = 20000
REPOSITORY_ENTRY_RESERVE = 128
RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS = 3600.0
RATE_LIMIT_MIN_COOLDOWN_SECONDS = 60.0
_STOP_REQUESTED = False


def _signal_handler(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError):
        return default


def _write_status(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.time()
    atomic_write_json(path, state)


def _queue_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not payload.get("run_id") or not payload.get("source_root"):
            continue
        if payload.get("status", "queued") in {"cancelled", "disabled"}:
            continue
        entries[str(payload["run_id"])] = payload
    return list(entries.values())


def _local_run_root(item: dict[str, Any]) -> Path:
    candidate = (ROOT / str(item["source_root"])).resolve()
    results_root = (ROOT / "results/runs").resolve()
    candidate.relative_to(results_root)
    return candidate


def _is_checkpoint(relative: str) -> bool:
    return relative.startswith("checkpoints/") and relative.endswith(CHECKPOINT_SUFFIXES)


def _iter_files(run_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in run_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(run_root).as_posix()
        parts = set(path.relative_to(run_root).parts)
        # The engine copies these files for local resume; canonical checkpoints
        # and exported artifacts below are the public reproducible record.
        if (
            path.name.endswith(".lock")
            or "__pycache__" in parts
            or relative.startswith("runtime/")
            # The production engine has used both spellings over time. These
            # are resume copies, not the canonical checkpoints/artifacts that
            # the experiment queue promises to persist.
            or parts.intersection({"engine_checkpoints", "_engine_checkpoints", "engine_output", "_engine_output"})
        ):
            continue
        paths.append(path)
    return sorted(paths, key=lambda path: (_is_checkpoint(path.relative_to(run_root).as_posix()), path.relative_to(run_root).as_posix()))


def _ensure_repository(api: HfApi, repo_id: str, cache: dict[str, bool]) -> None:
    if cache.get(repo_id):
        return
    api.model_info(repo_id=repo_id)
    cache[repo_id] = True


def _is_file_cap_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "too many files" in text or "over the limit of 20000 files" in text or "file-count" in text


def _allocate_artifact_repository(api: HfApi, run_state: dict[str, Any], run_id: str, cache: dict[str, bool]) -> str:
    assignments = run_state.setdefault("artifact_repositories", {})
    unusable = {str(item) for item in run_state.get("unusable_artifact_repositories", [])}
    assignment = assignments.get(run_id)
    # A prior uploader process may have recorded the Hub file-cap error before
    # this resumable state gained an explicit unusable-repository marker.
    if assignment and any(_is_file_cap_error(item.get("error", "")) for item in run_state.get("errors", [])):
        unusable.add(str(assignment))
        run_state["unusable_artifact_repositories"] = sorted(unusable)
    if assignment and str(assignment) not in unusable:
        _ensure_repository(api, str(assignment), cache)
        return str(assignment)
    if assignment and str(assignment) in unusable:
        assignments.pop(run_id, None)
    assigned_counts = {
        repo_id: sum(1 for value in assignments.values() if str(value) == repo_id)
        for repo_id in ARTIFACT_REPOSITORIES
    }
    candidates: list[tuple[int, int, str]] = []
    for order, repo_id in enumerate(ARTIFACT_REPOSITORIES):
        if repo_id in unusable:
            continue
        recorded_count = INITIAL_ENTRY_COUNTS.get(repo_id, REPOSITORY_ENTRY_CAP)
        if recorded_count + REPOSITORY_ENTRY_RESERVE >= REPOSITORY_ENTRY_CAP:
            continue
        try:
            _ensure_repository(api, repo_id, cache)
        except Exception:
            continue
        candidates.append((assigned_counts.get(repo_id, 0), order, repo_id))
    if candidates:
        _, _, repo_id = min(candidates)
        assignments[run_id] = repo_id
        run_state.setdefault("allocation_evidence", {})[run_id] = {
            "repository": repo_id,
            "initial_entry_count": INITIAL_ENTRY_COUNTS.get(repo_id),
            "selection_policy": "existing_repository_first; least_assigned_run; no_create_repo",
        }
        return repo_id
    raise RuntimeError("no existing artifact repository with available recorded capacity")


def _checkpoint_repository(backbone: str) -> str | None:
    return CHECKPOINT_REPOSITORIES.get(backbone)


def _clear_path_errors(run_state: dict[str, Any], path: str) -> None:
    run_state["errors"] = [
        item for item in run_state.get("errors", []) if str(item.get("path")) != path
    ]


def _rate_limit_retry_after(exc: BaseException) -> float | None:
    """Return a bounded cooldown for a Hub commit rate-limit response."""
    text = str(exc)
    lowered = text.lower()
    if "429" not in lowered and "rate limit" not in lowered:
        return None

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            return max(RATE_LIMIT_MIN_COOLDOWN_SECONDS, min(float(retry_after), RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS))
        except (TypeError, ValueError):
            pass

    match = re.search(r"retry this action in\s+(\d+)\s+minutes?", text, flags=re.IGNORECASE)
    if match:
        return max(RATE_LIMIT_MIN_COOLDOWN_SECONDS, min(float(match.group(1)) * 60.0, RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS))
    match = re.search(r"retry this action in\s+(\d+)\s+seconds?", text, flags=re.IGNORECASE)
    if match:
        return max(RATE_LIMIT_MIN_COOLDOWN_SECONDS, min(float(match.group(1)), RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS))
    return RATE_LIMIT_DEFAULT_COOLDOWN_SECONDS


def _set_rate_limit_cooldown(state: dict[str, Any], run_state: dict[str, Any], exc: BaseException, retry_after: float) -> None:
    until = time.time() + retry_after
    state["hub_cooldown_until"] = max(float(state.get("hub_cooldown_until", 0.0) or 0.0), until)
    run_state["hub_rate_limit"] = {
        "status": "WAITING",
        "retry_after_seconds": retry_after,
        "retry_at": state["hub_cooldown_until"],
        "error": f"{type(exc).__name__}: {exc}",
        "recorded_at": time.time(),
    }


def _remote_info(api: HfApi, repo_id: str, remote_path: str) -> dict[str, Any]:
    rows = api.get_paths_info(repo_id=repo_id, paths=[remote_path], repo_type="model")
    if not rows:
        raise RuntimeError(f"Hub path is not visible after upload: {repo_id}/{remote_path}")
    row = rows[0]
    return {
        "path": getattr(row, "path", None),
        "size": getattr(row, "size", None),
        "oid": getattr(row, "oid", None),
    }


def _upload_one(api: HfApi, repo_id: str, remote_path: str, local_path: Path, run_id: str) -> dict[str, Any]:
    size = local_path.stat().st_size
    digest = sha256_file(local_path)
    before = local_path.stat()
    commit = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"upload verified artifacts for {run_id}",
    )
    after = local_path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"local file changed during upload: {local_path}")
    remote = _remote_info(api, repo_id, remote_path)
    if remote.get("size") != size:
        raise RuntimeError(f"remote size mismatch for {repo_id}/{remote_path}: {remote.get('size')} != {size}")
    return {
        "status": "PASS",
        "verified": True,
        "repo": repo_id,
        "path": remote_path,
        "size": size,
        "sha256": digest,
        "commit_hash": getattr(commit, "commit_hash", None),
        "remote": remote,
        "verified_at": time.time(),
    }


def _upload_batch(
    api: HfApi,
    repo_id: str,
    entries: list[dict[str, Any]],
    run_id: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Upload stable files for one target repository in one Hub commit."""
    prepared: list[dict[str, Any]] = []
    operations: list[CommitOperationAdd] = []
    for entry in entries:
        local_path = Path(entry["local_path"])
        try:
            before = local_path.stat()
            if before.st_size != entry["size"] or before.st_mtime_ns != entry["mtime_ns"]:
                continue
            digest = sha256_file(local_path)
            after_hash = local_path.stat()
        except OSError:
            continue
        if before.st_size != after_hash.st_size or before.st_mtime_ns != after_hash.st_mtime_ns:
            continue
        prepared.append(
            {
                "relative": entry["relative"],
                "local_path": local_path,
                "size": before.st_size,
                "mtime_ns": before.st_mtime_ns,
                "sha256": digest,
                "remote_path": entry["remote_path"],
            }
        )
        operations.append(
            CommitOperationAdd(
                path_in_repo=entry["remote_path"],
                path_or_fileobj=str(local_path),
            )
        )
    if not operations:
        return []

    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"batch upload verified artifacts for {run_id}",
    )
    commit_hash = getattr(commit, "commit_hash", None)
    uploaded: list[tuple[str, dict[str, Any]]] = []
    for entry in prepared:
        local_path = entry["local_path"]
        after = local_path.stat()
        if after.st_size != entry["size"] or after.st_mtime_ns != entry["mtime_ns"]:
            continue
        remote = _remote_info(api, repo_id, entry["remote_path"])
        if remote.get("size") != entry["size"]:
            raise RuntimeError(
                f"remote size mismatch for {repo_id}/{entry['remote_path']}: "
                f"{remote.get('size')} != {entry['size']}"
            )
        uploaded.append(
            (
                str(entry["relative"]),
                {
                    "status": "PASS",
                    "verified": True,
                    "repo": repo_id,
                    "path": entry["remote_path"],
                    "size": entry["size"],
                    "sha256": entry["sha256"],
                    "commit_hash": commit_hash,
                    "remote": remote,
                    "verified_at": time.time(),
                },
            )
        )
    return uploaded


def _upload_receipt(api: HfApi, repo_id: str, run_id: str, remote_root: str, uploaded: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "run_id": run_id,
        "uploader": "scripts/hf_artifact_uploader.py",
        "artifact_repository": repo_id,
        "remote_root": remote_root,
        "verified_file_count": len(uploaded),
        "verified_files": uploaded[-100:],
        "created_at": time.time(),
    }
    remote_path = f"{remote_root}/uploader/receipt.json"
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    commit = api.upload_file(
        path_or_fileobj=io.BytesIO(data),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"record verified upload receipt for {run_id}",
    )
    remote = _remote_info(api, repo_id, remote_path)
    if remote.get("size") != len(data):
        raise RuntimeError(f"receipt size mismatch for {repo_id}/{remote_path}")
    return {"status": "PASS", "repo": repo_id, "path": remote_path, "size": len(data), "commit_hash": getattr(commit, "commit_hash", None), "remote": remote}


def _process_run(api: HfApi, item: dict[str, Any], state: dict[str, Any], repository_cache: dict[str, bool], campaign_id: str) -> dict[str, Any]:
    run_id = str(item["run_id"])
    run_root = _local_run_root(item)
    if not run_root.exists():
        return {"status": "WAITING", "run_id": run_id, "reason": "local run directory does not exist yet"}
    run_state = state.setdefault("runs", {}).setdefault(run_id, {"uploaded": {}, "observed": {}, "errors": []})
    cooldown_until = float(state.get("hub_cooldown_until", 0.0) or 0.0)
    assignment = state.setdefault("artifact_repositories", {}).get(run_id)
    if assignment and any(_is_file_cap_error(item.get("error", "")) for item in run_state.get("errors", [])):
        unusable = run_state.setdefault("unusable_artifact_repositories", [])
        if str(assignment) not in {str(item) for item in unusable}:
            unusable.append(str(assignment))
        # Reallocation is local state only; defer the Hub call until the
        # global cooldown expires so the uploader does not amplify 429s.
        state["artifact_repositories"].pop(run_id, None)
    if cooldown_until > time.time():
        return {
            "status": "WAITING",
            "run_id": run_id,
            "reason": "Hugging Face commit rate limit cooldown",
            "retry_at": cooldown_until,
        }
    state.pop("hub_cooldown_until", None)
    artifact_repo = _allocate_artifact_repository(api, state, run_id, repository_cache)
    checkpoint_repo = _checkpoint_repository(str(item.get("backbone", "")))
    remote_root = f"campaigns/{campaign_id}/{run_id}"
    uploaded_now: list[dict[str, Any]] = []
    pending = 0
    files = _iter_files(run_root)
    batches: dict[str, list[dict[str, Any]]] = {}
    for local_path in files:
        relative = local_path.relative_to(run_root).as_posix()
        key = relative
        try:
            stat = local_path.stat()
        except OSError:
            continue
        signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        previous = run_state.setdefault("observed", {}).get(key)
        if previous != signature:
            run_state["observed"][key] = signature
            pending += 1
            continue
        target_repo = checkpoint_repo if _is_checkpoint(relative) and checkpoint_repo else artifact_repo
        remote_path = f"{remote_root}/{relative}"
        previous_upload = run_state.setdefault("uploaded", {}).get(key)
        if previous_upload and previous_upload.get("sha256") and previous_upload.get("size") == stat.st_size and previous_upload.get("verified"):
            continue
        batches.setdefault(target_repo, []).append(
            {
                "relative": relative,
                "local_path": local_path,
                "remote_path": remote_path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    batch_items = list(batches.items())
    for batch_index, (target_repo, entries) in enumerate(batch_items):
        if _STOP_REQUESTED:
            pending += sum(len(batch) for _, batch in batch_items[batch_index:])
            break
        try:
            uploaded = _upload_batch(api, target_repo, entries, run_id)
            uploaded_keys = {relative for relative, _ in uploaded}
            for relative, result in uploaded:
                run_state["uploaded"][relative] = result
                _clear_path_errors(run_state, relative)
                uploaded_now.append(result)
            pending += len(entries) - len(uploaded_keys)
        except Exception as exc:
            retry_after = _rate_limit_retry_after(exc)
            if retry_after is not None:
                _set_rate_limit_cooldown(state, run_state, exc, retry_after)
                pending += sum(len(batch) for _, batch in batch_items[batch_index:])
                break
            if _is_file_cap_error(exc) and target_repo == artifact_repo:
                unusable = run_state.setdefault("unusable_artifact_repositories", [])
                if target_repo not in unusable:
                    unusable.append(target_repo)
                # Reallocate on the next poll; the current batch remains
                # pending and no partial receipt is emitted for this repo.
                pending += len(entries)
                continue
            pending += len(entries)
            for entry in entries:
                error = {
                    "path": entry["relative"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "at": time.time(),
                }
                _clear_path_errors(run_state, entry["relative"])
                run_state.setdefault("errors", []).append(error)
    for error in list(run_state.get("errors", [])):
        relative = str(error.get("path", ""))
        verified = run_state.get("uploaded", {}).get(relative)
        local_path = run_root / relative
        if (
            verified
            and verified.get("verified")
            and local_path.is_file()
            and verified.get("size") == local_path.stat().st_size
        ):
            _clear_path_errors(run_state, relative)
    receipt = None
    if uploaded_now and pending == 0 and not run_state.get("receipt"):
        try:
            receipt = _upload_receipt(api, artifact_repo, run_id, remote_root, list(run_state["uploaded"].values()))
            run_state["receipt"] = receipt
            _clear_path_errors(run_state, "uploader/receipt.json")
        except Exception as exc:
            retry_after = _rate_limit_retry_after(exc)
            if retry_after is not None:
                _set_rate_limit_cooldown(state, run_state, exc, retry_after)
            else:
                run_state.setdefault("errors", []).append({"path": "uploader/receipt.json", "error": f"{type(exc).__name__}: {exc}", "at": time.time()})
    return {
        "status": "PASS" if run_state.get("uploaded") else ("WAITING" if pending else "NOOP"),
        "run_id": run_id,
        "artifact_repository": artifact_repo,
        "checkpoint_repository": checkpoint_repo,
        "remote_root": remote_root,
        "local_file_count": len(files),
        "verified_file_count": sum(1 for value in run_state.get("uploaded", {}).values() if value.get("verified")),
        "uploaded_now": len(uploaded_now),
        "pending": pending,
        "receipt": receipt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload stable experiment artifacts independently of the scientific runner")
    parser.add_argument("--queue-file", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--stop-file", type=Path, default=DEFAULT_STOP)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--campaign-id", default=CAMPAIGN_ID)
    parser.add_argument("--once", action="store_true", help="Process one queue scan and exit")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN or an authenticated Hugging Face credential is required for the uploader")
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    api = HfApi(token=token)
    state = _load_json(args.status_file, {})
    if not isinstance(state, dict):
        state = {}
    state.setdefault("schema_version", 1)
    state["pid"] = os.getpid()
    state["status"] = "RUNNING"
    state["campaign_id"] = args.campaign_id
    repository_cache: dict[str, bool] = {}
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    while True:
        if args.stop_file.exists():
            state["status"] = "STOPPED_BY_FILE"
            _write_status(args.status_file, state)
            return 0
        results: list[dict[str, Any]] = []
        for item in _queue_entries(args.queue_file):
            if _STOP_REQUESTED:
                break
            try:
                results.append(_process_run(api, item, state, repository_cache, args.campaign_id))
            except Exception as exc:
                results.append({"status": "ERROR", "run_id": item.get("run_id"), "error": f"{type(exc).__name__}: {exc}"})
        state["last_scan"] = {"at": time.time(), "results": results}
        state["status"] = "STOPPING" if _STOP_REQUESTED else "RUNNING"
        _write_status(args.status_file, state)
        if args.once or _STOP_REQUESTED:
            return 0
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
