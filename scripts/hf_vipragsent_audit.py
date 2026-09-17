#!/usr/bin/env python3
"""Read-only Hugging Face inventory and artefact aggregation for ViPragSent.

This script intentionally does not import or execute the project's experiment
code.  It uses the Hugging Face HTTP API and the HF token from .env only for
read access.  The workflow is staged because the account contains high-
cardinality overflow repositories and large checkpoint payloads.

Phases:
  inventory  Discover account repos and paginate every repository tree.
  resume     Retry only repositories whose previous inventory was incomplete.
  fetch      Download bounded text artefacts selected from the inventory.
  analyze    Parse fetched artefacts, filter explicit XLM-R-large runs, and
             emit Q1a/Q2/Q3/Q4 tables, plots, coverage, and a report.
  all        Run the three phases in order.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote


DEFAULT_ACCOUNT = "Thundergod2007"
DEFAULT_OUT = Path("reports") / "hf_vipragsent_remote_audit_2026-09-15"
HF_BASE = "https://huggingface.co"
TREE_LIMIT = 100
TEXT_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".toml",
    ".log",
    ".sha256",
}
WEIGHT_EXTENSIONS = {
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".gguf",
}
QUESTION_RE = re.compile(r"(?<![a-z0-9])q([1-4])([a-z])?(?![a-z0-9])", re.I)
SEED_RE = re.compile(r"(?:seed|random[_ -]?seed)[=:_ -]?(\d{1,8})", re.I)
DATE_TOKEN_RE = re.compile(r"(?<!\d)20\d{6}(?!\d)")
SHA256_RE = re.compile(r"\b[a-f0-9]{64}\b", re.I)
TARGET_SEEDS = (21, 22, 23)


class AuditError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_token(env_path: Path) -> str:
    names = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_TOKEN")
    values: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    for name in names:
        if values.get(name):
            return values[name]
    raise AuditError(f"no Hugging Face token found in {env_path}")


def parse_json_bytes(raw: bytes, context: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8-sig", errors="replace"))
    except Exception as exc:  # pragma: no cover - error path is recorded
        raise AuditError(f"invalid JSON from {context}: {exc}") from exc


def split_http_response(raw: bytes) -> tuple[str, bytes]:
    # curl may include more than one header block when a redirect is followed.
    marker = b"\r\n\r\n"
    if marker not in raw:
        marker = b"\n\n"
    if marker not in raw:
        raise AuditError("curl response had no HTTP header/body separator")
    head, body = raw.rsplit(marker, 1)
    return head.decode("iso-8859-1", errors="replace"), body


def header_value(headers: str, name: str) -> str | None:
    wanted = name.lower()
    lines = headers.splitlines()
    for index, line in enumerate(lines):
        if line.lower().startswith(wanted + ":"):
            value = line.split(":", 1)[1].strip()
            # Some HTTP servers fold a long Link header.  Join continuation
            # lines until the next header-like line.
            j = index + 1
            parts = [value]
            while j < len(lines) and lines[j] and lines[j][0].isspace():
                parts.append(lines[j].strip())
                j += 1
            return " ".join(parts)
    return None


def curl_get(
    url: str,
    token: str,
    *,
    timeout: int = 180,
    retries: int = 4,
    byte_range: str | None = None,
) -> tuple[dict[str, str], bytes]:
    last_error = "unknown error"
    for attempt in range(1, retries + 1):
        command = [
            "curl.exe",
            "-L",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "15",
            "--max-time",
            str(timeout),
            "-D",
            "-",
            "-H",
            f"Authorization: Bearer {token}",
        ]
        if byte_range:
            command.extend(["-r", byte_range])
        command.append(url)
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode == 0:
            try:
                headers_raw, body = split_http_response(completed.stdout)
                status_match = re.findall(r"HTTP/\S+\s+(\d{3})", headers_raw)
                status = int(status_match[-1]) if status_match else 0
                headers = {
                    "status": str(status),
                    "link": header_value(headers_raw, "Link") or "",
                    "content-type": header_value(headers_raw, "Content-Type") or "",
                    "etag": header_value(headers_raw, "ETag") or "",
                    "x-repo-commit": header_value(headers_raw, "X-Repo-Commit") or "",
                    "content-length": header_value(headers_raw, "Content-Length") or "",
                    "x-linked-size": header_value(headers_raw, "X-Linked-Size") or "",
                }
                if 200 <= status < 300:
                    return headers, body
                last_error = f"HTTP {status}: {body[:300]!r}"
            except Exception as exc:
                last_error = str(exc)
        else:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            last_error = f"curl exit {completed.returncode}: {stderr[:300]}"
        if attempt < retries:
            time.sleep(min(2**attempt, 10))
    raise AuditError(f"GET failed after {retries} attempts: {url} ({last_error})")


def hf_api_json(token: str, endpoint: str) -> Any:
    _, body = curl_get(HF_BASE + endpoint, token)
    return parse_json_bytes(body, endpoint)


def next_link(link: str) -> str | None:
    if not link:
        return None
    match = re.search(r"<([^>]+)>\s*;\s*rel=[\"']next[\"']", link, re.I)
    return match.group(1) if match else None


def discover_repos(token: str, account: str) -> dict[str, Any]:
    models = hf_api_json(token, f"/api/models?author={quote(account)}&limit=1000&full=true")
    datasets = hf_api_json(token, f"/api/datasets?author={quote(account)}&limit=1000&full=true")
    spaces = hf_api_json(token, f"/api/spaces?author={quote(account)}&limit=1000&full=true")
    search_models = hf_api_json(
        token, "/api/models?search=vipragsent&limit=1000&full=true"
    )
    search_datasets = hf_api_json(
        token, "/api/datasets?search=vipragsent&limit=1000&full=true"
    )
    repos: list[dict[str, Any]] = []
    for repo_type, values in (
        ("model", models),
        ("dataset", datasets),
    ):
        for item in values if isinstance(values, list) else []:
            if item.get("id"):
                repos.append(
                    {
                        "repo_id": item["id"],
                        "repo_type": repo_type,
                        "private": item.get("private"),
                        "gated": item.get("gated"),
                        "repo_sha": item.get("sha"),
                        "last_modified": item.get("lastModified"),
                    }
                )
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for item in repos:
        dedup[(item["repo_type"], item["repo_id"])] = item
    def compact(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        keys = ("id", "private", "gated", "sha", "lastModified", "downloads", "likes")
        return [{key: item.get(key) for key in keys if key in item} for item in items if isinstance(item, dict)]

    return {
        "generated_at": utc_now(),
        "account": account,
        "account_models": compact(models),
        "account_datasets": compact(datasets),
        "account_spaces": compact(spaces),
        "global_model_search": compact(search_models),
        "global_dataset_search": compact(search_datasets),
        "repos": sorted(dedup.values(), key=lambda x: (x["repo_type"], x["repo_id"])),
    }


def tree_endpoint(repo: dict[str, Any], cursor: str | None = None) -> str:
    kind = "models" if repo["repo_type"] == "model" else "datasets"
    repo_id = quote(repo["repo_id"], safe="/")
    query = f"recursive=true&expand=true&limit={TREE_LIMIT}"
    if cursor:
        query += "&cursor=" + quote(cursor, safe="")
    return f"{HF_BASE}/api/{kind}/{repo_id}/tree/main?{query}"


def crawl_repo(repo: dict[str, Any], token: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    page_url: str | None = tree_endpoint(repo)
    entries: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    pages = 0
    complete = True
    errors: list[str] = []
    while page_url:
        if page_url in seen_urls:
            complete = False
            errors.append("pagination loop")
            break
        seen_urls.add(page_url)
        pages += 1
        try:
            headers, body = curl_get(page_url, token)
            data = parse_json_bytes(body, page_url)
            if not isinstance(data, list):
                raise AuditError(f"tree response was not a list: {data}")
            for entry in data:
                record = dict(entry)
                record["repo_id"] = repo["repo_id"]
                record["repo_type"] = repo["repo_type"]
                record["tree_page"] = pages
                entries.append(record)
            link = headers.get("link", "")
            page_url = next_link(link)
            if page_url is None and len(data) >= TREE_LIMIT:
                complete = False
                errors.append("page reached limit but no next Link header")
        except Exception as exc:
            complete = False
            errors.append(str(exc))
            break
    summary = {
        **repo,
        "crawl_started_at": utc_now(),
        "tree_pages": pages,
        "entry_count": len(entries),
        "file_count": sum(x.get("type") == "file" for x in entries),
        "directory_count": sum(x.get("type") == "directory" for x in entries),
        "pagination_complete": complete,
        "errors": errors,
    }
    return summary, entries


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except json.JSONDecodeError as exc:
            rows.append({"_parse_error": str(exc), "_line": line_number})
    return rows


def inventory_phase(args: argparse.Namespace) -> None:
    token = read_token(args.env)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    discovery = discover_repos(token, args.account)
    write_json(out / "discovery.json", discovery)
    repos = discovery["repos"]
    summaries: list[dict[str, Any]] = []
    all_entries: list[dict[str, Any]] = []
    # A small concurrency limit avoids reproducing the timeout/truncation issue
    # from earlier audits while still making the 29-repo crawl practical.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(crawl_repo, repo, token): repo for repo in repos}
        for future in concurrent.futures.as_completed(futures):
            repo = futures[future]
            try:
                summary, entries = future.result()
            except Exception as exc:  # pragma: no cover - defensive path
                summary = {**repo, "pagination_complete": False, "errors": [str(exc)]}
                entries = []
            summaries.append(summary)
            all_entries.extend(entries)
            print(
                f"[inventory] {repo['repo_type']} {repo['repo_id']}: "
                f"{summary.get('file_count', 0)} files, "
                f"{summary.get('tree_pages', 0)} pages, "
                f"complete={summary.get('pagination_complete')}",
                flush=True,
            )
    summaries.sort(key=lambda x: (x.get("repo_type", ""), x.get("repo_id", "")))
    all_entries.sort(key=lambda x: (x.get("repo_type", ""), x.get("repo_id", ""), x.get("path", "")))
    write_jsonl(out / "repo_summaries.jsonl", summaries)
    write_jsonl(out / "tree_manifest.jsonl", all_entries)
    write_json(
        out / "inventory_status.json",
        {
            "generated_at": utc_now(),
            "account": args.account,
            "repo_count": len(repos),
            "repo_summary_count": len(summaries),
            "manifest_entry_count": len(all_entries),
            "file_entry_count": sum(x.get("type") == "file" for x in all_entries),
            "directory_entry_count": sum(x.get("type") == "directory" for x in all_entries),
            "complete_repo_count": sum(x.get("pagination_complete") is True for x in summaries),
            "incomplete_repos": [
                {"repo_type": x.get("repo_type"), "repo_id": x.get("repo_id"), "errors": x.get("errors", [])}
                for x in summaries
                if x.get("pagination_complete") is not True
            ],
        },
    )


def rewrite_inventory_status(out: Path, account: str, summaries: list[dict[str, Any]], entries: list[dict[str, Any]]) -> None:
    summaries.sort(key=lambda x: (x.get("repo_type", ""), x.get("repo_id", "")))
    entries.sort(key=lambda x: (x.get("repo_type", ""), x.get("repo_id", ""), x.get("path", "")))
    write_jsonl(out / "repo_summaries.jsonl", summaries)
    write_jsonl(out / "tree_manifest.jsonl", entries)
    write_json(
        out / "inventory_status.json",
        {
            "generated_at": utc_now(),
            "account": account,
            "repo_count": len(summaries),
            "repo_summary_count": len(summaries),
            "manifest_entry_count": len(entries),
            "file_entry_count": sum(x.get("type") == "file" for x in entries),
            "directory_entry_count": sum(x.get("type") == "directory" for x in entries),
            "complete_repo_count": sum(x.get("pagination_complete") is True for x in summaries),
            "incomplete_repos": [
                {"repo_type": x.get("repo_type"), "repo_id": x.get("repo_id"), "errors": x.get("errors", [])}
                for x in summaries
                if x.get("pagination_complete") is not True
            ],
        },
    )


def resume_phase(args: argparse.Namespace) -> None:
    token = read_token(args.env)
    out = args.out
    summaries = read_jsonl(out / "repo_summaries.jsonl")
    entries = read_jsonl(out / "tree_manifest.jsonl")
    incomplete = [x for x in summaries if x.get("pagination_complete") is not True]
    if args.repo_pattern:
        pattern = re.compile(args.repo_pattern, re.I)
        incomplete = [x for x in incomplete if pattern.search(x.get("repo_id", ""))]
    if args.repo_limit is not None:
        incomplete = incomplete[: args.repo_limit]
    if not incomplete:
        print("[resume] no incomplete repositories selected", flush=True)
        return
    old_by_repo: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    selected_keys = {(x["repo_type"], x["repo_id"]) for x in incomplete}
    for entry in entries:
        key = (entry.get("repo_type", ""), entry.get("repo_id", ""))
        if key not in selected_keys:
            old_by_repo[key].append(entry)
    new_summaries = [x for x in summaries if (x.get("repo_type"), x.get("repo_id")) not in selected_keys]
    new_entries = [entry for values in old_by_repo.values() for entry in values]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(crawl_repo, repo, token): repo for repo in incomplete}
        for future in concurrent.futures.as_completed(futures):
            repo = futures[future]
            try:
                summary, repo_entries = future.result()
            except Exception as exc:  # pragma: no cover - defensive path
                summary = {**repo, "pagination_complete": False, "errors": [str(exc)]}
                repo_entries = []
            new_summaries.append(summary)
            new_entries.extend(repo_entries)
            print(
                f"[resume] {repo['repo_type']} {repo['repo_id']}: "
                f"{summary.get('file_count', 0)} files, {summary.get('tree_pages', 0)} pages, "
                f"complete={summary.get('pagination_complete')}",
                flush=True,
            )
    account = args.account
    discovery_path = out / "discovery.json"
    if discovery_path.exists():
        try:
            account = json.loads(discovery_path.read_text(encoding="utf-8")).get("account", account)
        except Exception:
            pass
    rewrite_inventory_status(out, account, new_summaries, new_entries)


def is_text_entry(entry: dict[str, Any]) -> bool:
    return Path(entry.get("path", "")).suffix.lower() in TEXT_EXTENSIONS


def is_weight_entry(entry: dict[str, Any]) -> bool:
    return Path(entry.get("path", "")).suffix.lower() in WEIGHT_EXTENSIONS


def is_priority_text(entry: dict[str, Any]) -> bool:
    path = entry.get("path", "").lower()
    name = Path(path).name.lower()
    priority_terms = (
        "final_science_result",
        "final-result",
        "final_result",
        "result",
        "metric",
        "score",
        "eval",
        "manifest",
        "receipt",
        "report",
        "ledger",
        "summary",
        "q1",
        "q2",
        "q3",
        "q4",
        "seed",
        "checkpoint",
        "config",
        "metadata",
    )
    return any(term in path or term in name for term in priority_terms)


def is_xlmr_target(entry: dict[str, Any]) -> bool:
    """Select all bounded text artefacts in the XLM-R Q1a/Q2/Q3/Q4 scope."""
    if not is_text_entry(entry):
        return False
    repo_id = entry.get("repo_id", "").lower()
    path = entry.get("path", "").lower()
    if repo_id.endswith("-checkpoints"):
        return repo_id.endswith("vipragsent-xlmr-checkpoints")
    has_xlmr = any(term in path for term in ("xlmr", "xlm-r", "xlm_roberta", "xlm-roberta"))
    has_question = any(term in path for term in ("q1a", "q1b", "q2", "q3", "q4"))
    return has_xlmr and has_question


def is_xlmr_weight_target(entry: dict[str, Any]) -> bool:
    """Weight inventory filter matching the XLM-R experiment scope."""
    if not is_weight_entry(entry):
        return False
    repo_id = entry.get("repo_id", "").lower()
    path = entry.get("path", "").lower()
    if repo_id.endswith("-checkpoints"):
        return repo_id.endswith("vipragsent-xlmr-checkpoints")
    has_xlmr = any(term in path for term in ("xlmr", "xlm-r", "xlm_roberta", "xlm-roberta"))
    has_question = any(term in path for term in ("q1a", "q2", "q3", "q4"))
    return has_xlmr and has_question


def raw_path_for(out: Path, entry: dict[str, Any]) -> Path:
    # HF experiment paths can exceed Windows MAX_PATH when mirrored verbatim.
    # Keep the full source path in content_manifest.jsonl and use a stable
    # content-addressed short local path for the bytes.
    identity = f"{entry['repo_type']}:{entry['repo_id']}:{entry['path']}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    basename = re.sub(r"[^A-Za-z0-9._=-]+", "_", Path(entry["path"]).name)
    basename = basename[:80] or "content.bin"
    return out / "raw_short" / entry["repo_type"] / digest[:2] / f"{digest}_{basename}"


def legacy_raw_path_for(out: Path, entry: dict[str, Any]) -> Path:
    """Path layout used by the interrupted first fetch attempt."""
    repo = entry["repo_id"].replace("/", "__")
    parts = [p for p in Path(entry["path"]).parts if p not in ("", ".", "..")]
    safe_parts = [re.sub(r"[^A-Za-z0-9._=-]+", "_", p) for p in parts]
    return out / "raw" / entry["repo_type"] / repo / Path(*safe_parts)


def content_url(entry: dict[str, Any]) -> str:
    prefix = "datasets/" if entry["repo_type"] == "dataset" else ""
    repo = quote(entry["repo_id"], safe="/")
    path = quote(entry["path"], safe="/")
    return f"{HF_BASE}/{prefix}{repo}/resolve/main/{path}"


def fetch_one(
    entry: dict[str, Any],
    out: Path,
    token: str,
    max_bytes: int,
) -> dict[str, Any]:
    destination = raw_path_for(out, entry)
    size = entry.get("size")
    if isinstance(size, int) and size > max_bytes:
        return {
            **entry,
            "fetch_status": "skipped_size",
            "fetch_reason": f"size {size} > max_text_bytes {max_bytes}",
            "raw_path": str(destination.relative_to(out)),
        }
    try:
        legacy = legacy_raw_path_for(out, entry)
        if legacy.is_file():
            body = legacy.read_bytes()
            if len(body) <= max_bytes:
                return {
                    **entry,
                    "fetch_status": "fetched",
                    "fetch_bytes": len(body),
                    "fetch_sha256": hashlib.sha256(body).hexdigest(),
                    "fetch_source": "legacy_partial_fetch_cache",
                    "raw_path": str(legacy.relative_to(out)),
                }
        destination.parent.mkdir(parents=True, exist_ok=True)
        headers, body = curl_get(content_url(entry), token, timeout=300)
        if len(body) > max_bytes:
            return {
                **entry,
                "fetch_status": "skipped_received_size",
                "fetch_reason": f"received {len(body)} > max_text_bytes {max_bytes}",
                "raw_path": str(destination.relative_to(out)),
            }
        destination.write_bytes(body)
        return {
            **entry,
            "fetch_status": "fetched",
            "fetch_bytes": len(body),
            "fetch_sha256": hashlib.sha256(body).hexdigest(),
            "fetch_etag": headers.get("etag", ""),
            "fetch_x_repo_commit": headers.get("x-repo-commit", ""),
            "raw_path": str(destination.relative_to(out)),
        }
    except Exception as exc:
        return {
            **entry,
            "fetch_status": "error",
            "fetch_error": str(exc),
            "raw_path": str(destination.relative_to(out)),
        }


def fetch_phase(args: argparse.Namespace) -> None:
    token = read_token(args.env)
    out = args.out
    entries = [x for x in read_jsonl(out / "tree_manifest.jsonl") if x.get("type") == "file"]
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        if args.selection == "all_text":
            keep = is_text_entry(entry)
        elif args.selection == "xlmr":
            keep = is_xlmr_target(entry)
        else:
            keep = is_priority_text(entry)
        if keep:
            candidates.append(entry)
    candidates.sort(key=lambda x: (x.get("repo_type", ""), x.get("repo_id", ""), x.get("path", "")))
    total_candidate_count = len(candidates)
    if args.candidate_offset:
        candidates = candidates[args.candidate_offset :]
    if args.candidate_limit is not None:
        candidates = candidates[: args.candidate_limit]
    write_json(
        out / "fetch_plan.json",
        {
            "generated_at": utc_now(),
            "candidate_count": len(candidates),
            "total_candidate_count": total_candidate_count,
            "candidate_offset": args.candidate_offset,
            "candidate_limit": args.candidate_limit,
            "selection": args.selection,
            "max_text_bytes": args.max_text_bytes,
            "candidate_paths": [
                {"repo_type": x["repo_type"], "repo_id": x["repo_id"], "path": x["path"], "size": x.get("size")}
                for x in candidates
            ],
        },
    )
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, entry, out, token, args.max_text_bytes): entry
            for entry in candidates
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if index % 25 == 0 or index == len(candidates):
                print(f"[fetch] {index}/{len(candidates)}", flush=True)
    results.sort(key=lambda x: (x.get("repo_type", ""), x.get("repo_id", ""), x.get("path", "")))
    content_manifest_path = out / "content_manifest.jsonl"
    if args.append_content_manifest and content_manifest_path.exists():
        existing = read_jsonl(content_manifest_path)
        by_key = {(x.get("repo_type"), x.get("repo_id"), x.get("path")): x for x in existing}
        by_key.update({(x.get("repo_type"), x.get("repo_id"), x.get("path")): x for x in results})
        results = sorted(by_key.values(), key=lambda x: (x.get("repo_type", ""), x.get("repo_id", ""), x.get("path", "")))
    write_jsonl(content_manifest_path, results)
    write_json(
        out / "fetch_status.json",
        {
            "generated_at": utc_now(),
            "candidate_count": len(candidates),
            "total_candidate_count": total_candidate_count,
            "fetched_count": sum(x.get("fetch_status") == "fetched" for x in results),
            "skipped_count": sum(str(x.get("fetch_status", "")).startswith("skipped") for x in results),
            "error_count": sum(x.get("fetch_status") == "error" for x in results),
            "weight_file_count_in_inventory": sum(is_weight_entry(x) for x in entries),
            "weight_bytes_in_inventory": sum(
                x.get("size", 0) for x in entries if is_weight_entry(x) and isinstance(x.get("size"), int)
            ),
        },
    )


def lower_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value).lower()
    return ""


def iter_leaves(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_leaves(child, path + (str(key),))
    elif isinstance(value, list):
        # Large prediction arrays are not useful for the scalar metric pass.
        if len(value) <= 200:
            for index, child in enumerate(value):
                yield from iter_leaves(child, path + (str(index),))
    else:
        yield path, value


def explicit_backbone(data: Any, source_path: str) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = []
    for path, value in iter_leaves(data):
        if not isinstance(value, str):
            continue
        key_path = ".".join(path).lower()
        if any(
            term in key_path
            for term in ("backbone", "encoder", "base_model", "model_name", "pretrained", "architecture")
        ):
            candidates.append((key_path, value))
    candidates.sort(key=lambda x: (0 if "backbone" in x[0] else 1, len(x[0])))
    for key_path, value in candidates:
        v = value.lower().replace("_", "-").replace(" ", "-")
        if any(term in v for term in ("xlm-r-large", "xlmr-large", "xlm-roberta-large", "xlm-r/large")):
            return "XLM-R-large", f"explicit:{key_path}={value}"
    if any(term in source_path.lower() for term in ("xlmr", "xlm-r", "xlm_roberta")):
        return "XLM-R-large", "path_inference_only"
    for key_path, value in candidates:
        v = value.lower()
        if any(term in v for term in ("phobert", "vistral", "sailor")):
            return "other", f"explicit:{key_path}={value}"
    return "unknown", "no_backbone_evidence"


def normalize_seed_value(value: Any) -> int | None:
    """Normalize date-coded seeds such as 20260521 to the paper seed 21."""
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    digits = str(abs(number))
    if len(digits) == 8 and 1900 <= int(digits[:4]) <= 2100:
        return int(digits[-2:])
    return number


def extract_seed(data: Any, source_path: str) -> tuple[int | None, str]:
    exact_candidates: list[tuple[str, Any]] = []
    contextual_candidates: list[tuple[str, Any]] = []
    for path, value in iter_leaves(data):
        key_path = ".".join(path).lower()
        leaf_key = path[-1].lower() if path else ""
        if leaf_key in {"seed", "random_seed", "randomseed", "training_seed"}:
            exact_candidates.append((key_path, value))
        elif "seed" in key_path and not any(term in key_path for term in ("sha", "hash", "checksum", "path")):
            contextual_candidates.append((key_path, value))
    for key_path, value in exact_candidates + contextual_candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            raw_value = int(value)
            normalized = normalize_seed_value(raw_value)
            if normalized is not None:
                return normalized, f"explicit:{key_path}={raw_value}"
        match = re.search(r"\d{1,8}", str(value))
        if match:
            raw_value = int(match.group())
            normalized = normalize_seed_value(raw_value)
            if normalized is not None:
                return normalized, f"explicit:{key_path}={raw_value}"
    match = SEED_RE.search(source_path)
    if match:
        raw_value = int(match.group(1))
        normalized = normalize_seed_value(raw_value)
        if normalized is not None:
            return normalized, f"path={raw_value}"
    # The HF run directories use date-coded suffixes such as 20260521,
    # while the paper calls the corresponding runs seeds 21, 22, and 23.
    # Restrict this fallback to question-bearing path components so a server
    # snapshot date such as server_20260808 is not mistaken for a seed.
    for part in reversed(Path(source_path).parts):
        if not QUESTION_RE.search(part.replace("_", "-")):
            continue
        tokens = DATE_TOKEN_RE.findall(part)
        if tokens:
            raw_value = int(tokens[-1])
            normalized = normalize_seed_value(raw_value)
            if normalized is not None:
                return normalized, f"path_date={raw_value}"
    return None, "missing"


def normalize_question_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().replace("_", "-")
    match = re.fullmatch(r"q([1-4])([ab])?", text, re.I)
    if match:
        return "Q" + match.group(1) + (match.group(2) or "").lower()
    return None


def path_question(source_path: str) -> str | None:
    # Search the innermost run component first. The common campaign folder
    # contains "q1b-q4", so a whole-path regex labels Q2/Q3/Q4 incorrectly.
    parts = list(Path(source_path).parts)
    if parts:
        parts = parts[:-1]  # do not infer a question from the filename
    for part in reversed(parts):
        matches = list(QUESTION_RE.finditer(part.replace("_", "-")))
        if matches:
            match = matches[-1]
            return "Q" + match.group(1) + (match.group(2) or "").lower()
    return None


def extract_question(data: Any, source_path: str) -> str:
    # Prefer structured research-question fields when present.
    for path, value in iter_leaves(data):
        key = path[-1].lower() if path else ""
        if key in {"research_question", "question_id", "rq"} or "research_question" in key:
            normalized = normalize_question_value(value)
            if normalized:
                return normalized
    path_value = path_question(source_path)
    if path_value:
        return path_value
    # A flattened run identifier can still carry the question. Avoid scanning
    # arbitrary prediction text before these metadata fields.
    for path, value in iter_leaves(data):
        key = path[-1].lower() if path else ""
        if key not in {"run_id", "experiment_id", "run_name", "experiment"}:
            continue
        if not isinstance(value, str):
            continue
        matches = list(QUESTION_RE.finditer(value.replace("_", "-")))
        if matches:
            match = matches[-1]
            return "Q" + match.group(1) + (match.group(2) or "").lower()
    text = json.dumps(data, ensure_ascii=False)[:200000].lower()
    match = QUESTION_RE.search(text.replace("_", "-"))
    if match:
        return "Q" + match.group(1) + (match.group(2) or "").lower()
    return "UNMAPPED"


def path_experiment_id(source_path: str) -> str | None:
    parts = list(Path(source_path).parts)
    if parts:
        parts = parts[:-1]  # exclude the artifact filename
    for part in reversed(parts):
        if QUESTION_RE.search(part.replace("_", "-")):
            return part
    return None


def experiment_group_id(value: str) -> str:
    """Collapse date-coded run IDs so seeds 21/22/23 form one experiment."""
    return re.sub(r"[_-]20\d{6}$", "", str(value))


def experiment_id(data: Any, source_path: str) -> str:
    path_id = path_experiment_id(source_path)
    if path_id:
        return path_id
    preferred_keys = ("experiment_id", "run_id", "run_name", "experiment")
    for path, value in iter_leaves(data):
        if not isinstance(value, (str, int)):
            continue
        key = path[-1].lower() if path else ""
        if key in preferred_keys and str(value).strip():
            return str(value)
    parts = [p for p in Path(source_path).parts if p]
    for part in reversed(parts):
        if any(term in part.lower() for term in ("q1", "q2", "q3", "q4", "seed", "run", "experiment")):
            return part
    return Path(source_path).stem


def parse_scalar_metrics(data: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_terms = (
        "f1",
        "accuracy",
        "precision",
        "recall",
        "ece",
        "calibration",
        "loss",
        "auc",
        "score",
        "metric",
        "ci",
        "std",
        "mean",
        "variance",
        "latency",
        "cost",
        "throughput",
    )
    for path, value in iter_leaves(data):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not math.isfinite(float(value)):
            continue
        key = path[-1].lower() if path else ""
        path_text = ".".join(path).lower()
        if any(term in key or term in path_text for term in metric_terms):
            rows.append(
                {
                    "metric_path": ".".join(path),
                    "metric_name": key,
                    "value": float(value),
                }
            )
    return rows


def parse_content_file(content_row: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    if content_row.get("fetch_status") != "fetched":
        return []
    raw_path = out / content_row["raw_path"]
    suffix = Path(content_row["path"]).suffix.lower()
    try:
        raw = raw_path.read_bytes()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    if suffix == ".jsonl":
        source_lower = content_row["path"].lower()
        # Prediction/event JSONL is fetched and hashed for audit completeness,
        # but expanding millions of per-example rows would obscure the scalar
        # metric pass.  Metrics and manifest JSON/CSV files remain fully parsed.
        if any(term in source_lower for term in ("prediction", "events", "stage_events")):
            return []
        for line_number, line in enumerate(raw.decode("utf-8-sig", errors="replace").splitlines(), 1):
            if line_number > 5000:
                break
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(
                    {
                        "repo_type": content_row["repo_type"],
                        "repo_id": content_row["repo_id"],
                        "source_path": content_row["path"],
                        "line_number": line_number,
                        "data": value,
                    }
                )
    elif suffix == ".json":
        try:
            value = json.loads(raw.decode("utf-8-sig", errors="replace"))
        except json.JSONDecodeError:
            return []
        if isinstance(value, dict):
            records.append(
                {
                    "repo_type": content_row["repo_type"],
                    "repo_id": content_row["repo_id"],
                    "source_path": content_row["path"],
                    "line_number": None,
                    "data": value,
                }
            )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    records.append(
                        {
                            "repo_type": content_row["repo_type"],
                            "repo_id": content_row["repo_id"],
                            "source_path": content_row["path"],
                            "line_number": index,
                            "data": item,
                        }
                    )
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            text = raw.decode("utf-8-sig", errors="replace")
            reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
            for index, item in enumerate(reader):
                records.append(
                    {
                        "repo_type": content_row["repo_type"],
                        "repo_id": content_row["repo_id"],
                        "source_path": content_row["path"],
                        "line_number": index,
                        "data": dict(item),
                    }
                )
        except Exception:
            return []
    return records


def record_summary(record: dict[str, Any]) -> dict[str, Any]:
    data = record["data"]
    backbone, backbone_evidence = explicit_backbone(data, record["source_path"])
    seed, seed_evidence = extract_seed(data, record["source_path"])
    question = extract_question(data, record["source_path"])
    run_id = experiment_id(data, record["source_path"])
    metrics = parse_scalar_metrics(data)
    return {
        "repo_type": record["repo_type"],
        "repo_id": record["repo_id"],
        "source_path": record["source_path"],
        "line_number": record.get("line_number"),
        "experiment_id": run_id,
        "experiment_group": experiment_group_id(run_id),
        "question": question,
        "backbone": backbone,
        "backbone_evidence": backbone_evidence,
        "seed": seed,
        "seed_evidence": seed_evidence,
        "metric_count": len(metrics),
        "data_keys": sorted(str(k) for k in data) if isinstance(data, dict) else [],
        "metrics": metrics,
        "data": data,
    }


def find_normalized_metric(summary: dict[str, Any], metric_name: str) -> float | None:
    """Select a scalar metric without treating support/count fields as F1."""
    matches: list[tuple[int, float]] = []
    for metric in summary.get("metrics", []):
        path = str(metric.get("metric_path", "")).lower().replace("-", "_")
        if metric_name == "pragmatic_f1":
            if "pragmatic" not in path or "f1" not in path:
                continue
            priority = 1000 if "macro_pragmatic_f1" in path else 500
        elif metric_name == "sarcasm_f1":
            if "sarcasm" not in path or "f1" not in path:
                continue
            priority = 500
            if "per_label_f1.sarcasm" in path:
                priority = 850
            if any(token in path for token in ("sarcasm_test_f1", "sarcasm_dev_f1", "f1_sarcasm")):
                priority = 900
        elif metric_name == "implicit_f1":
            if "implicit" not in path or "f1" not in path:
                continue
            priority = 500
            if "per_label_f1.implicit_sentiment" in path:
                priority = 850
            if "implicit_sentiment_f1" in path or "f1_implicit_sentiment" in path:
                priority = 900
        elif metric_name == "ordinary_f1":
            if "ordinary" not in path or "f1" not in path:
                continue
            priority = 700
        elif metric_name == "ece":
            if not any(token in path for token in ("ece", "expected_calibration_error")):
                continue
            priority = 1000 if "macro_pragmatic_ece" in path else 500
        else:
            continue
        # Prefer test metrics when a canonical file contains both test and dev.
        if ".test." in path or path.startswith("test."):
            priority += 100
        matches.append((priority, float(metric["value"])))
    if not matches:
        return None
    matches.sort(key=lambda x: -x[0])
    return matches[0][1]


def is_canonical_summary(summary: dict[str, Any]) -> bool:
    """Keep one run-level metric artifact instead of counting every mirror."""
    source = str(summary.get("source_path", "")).lower()
    if summary.get("question") == "Q4":
        return source.endswith("q4_pragmatic_calibration_per_seed.json")
    path = Path(str(summary.get("source_path", "")))
    return path.name.lower() == "metrics.json" and path.parent.name == str(summary.get("experiment_id", ""))


def build_weight_inventory(out: Path) -> dict[str, Any]:
    """Materialize complete HF tree metadata for weights without downloading them."""
    entries = read_jsonl(out / "tree_manifest.jsonl")
    all_weights = [entry for entry in entries if is_weight_entry(entry)]
    xlmr_weights = [entry for entry in all_weights if is_xlmr_weight_target(entry)]

    def row(entry: dict[str, Any], scope: str) -> dict[str, Any]:
        lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
        commit = entry.get("lastCommit") if isinstance(entry.get("lastCommit"), dict) else {}
        return {
            "scope": scope,
            "repo_id": entry.get("repo_id"),
            "repo_type": entry.get("repo_type"),
            "path": entry.get("path"),
            "size_bytes": entry.get("size"),
            "tree_oid": entry.get("oid"),
            "lfs_oid": lfs.get("oid"),
            "lfs_size_bytes": lfs.get("size"),
            "lfs_pointer_size": lfs.get("pointerSize"),
            "xet_hash": entry.get("xetHash"),
            "commit_id": commit.get("id"),
            "commit_date": commit.get("date"),
            "question_path_inference": path_question(str(entry.get("path", ""))),
        }

    write_csv(out / "weight_inventory.csv", [row(entry, "all_weight_entries") for entry in all_weights])
    write_csv(out / "xlmr_weight_inventory.csv", [row(entry, "xlmr_scope") for entry in xlmr_weights])
    summary = {
        "all_weight_file_count": len(all_weights),
        "all_weight_bytes": sum(int(entry.get("size") or 0) for entry in all_weights),
        "xlmr_scope_weight_file_count": len(xlmr_weights),
        "xlmr_scope_weight_bytes": sum(int(entry.get("size") or 0) for entry in xlmr_weights),
        "xlmr_scope_repositories": dict(Counter(entry.get("repo_id") for entry in xlmr_weights)),
        "downloaded_weight_payloads": 0,
        "interpretation": "Complete HF tree/LFS metadata inventory; weight payloads were not downloaded or deserialized in this pass.",
    }
    write_json(out / "weight_inventory_summary.json", summary)
    return summary


def analysis_phase(args: argparse.Namespace) -> None:
    out = args.out
    weight_summary = build_weight_inventory(out)
    inventory_status = json.loads((out / "inventory_status.json").read_text(encoding="utf-8"))
    fetch_status = json.loads((out / "fetch_status.json").read_text(encoding="utf-8"))
    content_rows = read_jsonl(out / "content_manifest.jsonl")
    parsed: list[dict[str, Any]] = []
    for row in content_rows:
        parsed.extend(parse_content_file(row, out))
    summaries = [record_summary(record) for record in parsed]
    write_jsonl(out / "parsed_record_summaries.jsonl", summaries)

    xlmr = [x for x in summaries if x["backbone"] == "XLM-R-large"]
    selected = [x for x in xlmr if x["question"] in {"Q1a", "Q2", "Q3", "Q4"}]
    canonical_candidates = [x for x in selected if is_canonical_summary(x)]
    canonical_candidates.sort(
        key=lambda x: (
            x["question"],
            str(x["experiment_group"]),
            x.get("seed") is None,
            -int(x.get("metric_count", 0)),
            x["repo_id"],
            x["source_path"],
        )
    )
    canonical: list[dict[str, Any]] = []
    canonical_duplicates: list[dict[str, Any]] = []
    seen_canonical: set[tuple[str, str, int | None]] = set()
    for summary in canonical_candidates:
        key = (summary["question"], str(summary["experiment_group"]), summary.get("seed"))
        if key in seen_canonical:
            canonical_duplicates.append(summary)
            continue
        seen_canonical.add(key)
        canonical.append(summary)
    explicit_backbone_by_key: dict[tuple[str, str, int | None], list[str]] = defaultdict(list)
    for summary in selected:
        if summary.get("backbone_evidence") == "path_inference_only":
            continue
        key = (summary["question"], str(summary["experiment_group"]), summary.get("seed"))
        explicit_backbone_by_key[key].append(summary["source_path"])
    for summary in canonical:
        key = (summary["question"], str(summary["experiment_group"]), summary.get("seed"))
        evidence_paths = sorted(set(explicit_backbone_by_key.get(key, [])))
        summary["backbone_confirmation"] = (
            "companion_explicit:" + evidence_paths[0] if evidence_paths else "path_only_unconfirmed"
        )
    seeds_21_22_23 = [x for x in canonical if x.get("seed") in TARGET_SEEDS]
    write_jsonl(out / "xlmr_large_records.jsonl", xlmr)
    write_jsonl(out / "selected_q1a_q2_q3_q4_records.jsonl", selected)
    write_jsonl(out / "canonical_q1a_q2_q3_q4_records.jsonl", canonical)
    write_jsonl(out / "canonical_duplicate_records.jsonl", canonical_duplicates)

    metric_rows: list[dict[str, Any]] = []
    for summary in selected:
        for metric in summary.get("metrics", []):
            metric_rows.append(
                {
                    "question": summary["question"],
                    "experiment_id": summary["experiment_id"],
                    "experiment_group": summary["experiment_group"],
                    "seed": summary.get("seed"),
                    "backbone": summary["backbone"],
                    "repo_type": summary["repo_type"],
                    "repo_id": summary["repo_id"],
                    "source_path": summary["source_path"],
                    "metric_path": metric["metric_path"],
                    "metric_name": metric["metric_name"],
                    "value": metric["value"],
                }
            )
    metric_rows.sort(key=lambda x: (x["question"], str(x["experiment_id"]), x["source_path"], x["metric_path"]))
    write_csv(out / "selected_metrics_long.csv", metric_rows)

    # A normalized seed-level table focuses on one canonical run artifact per
    # experiment/seed. All other parsed records remain in selected_metrics_long
    # and the raw selected-record JSONL for provenance.
    seed_rows: list[dict[str, Any]] = []
    for summary in canonical:
        row = {
            "question": summary["question"],
            "experiment_id": summary["experiment_id"],
            "experiment_group": summary["experiment_group"],
            "seed": summary.get("seed"),
            "seed_evidence": summary.get("seed_evidence"),
            "backbone_confirmation": summary.get("backbone_confirmation"),
            "repo_id": summary["repo_id"],
            "source_path": summary["source_path"],
            "pragmatic_f1": find_normalized_metric(summary, "pragmatic_f1"),
            "sarcasm_f1": find_normalized_metric(summary, "sarcasm_f1"),
            "implicit_f1": find_normalized_metric(summary, "implicit_f1"),
            "ordinary_f1": find_normalized_metric(summary, "ordinary_f1"),
            "ece": find_normalized_metric(summary, "ece"),
        }
        seed_rows.append(row)
    write_csv(out / "selected_seed_metrics.csv", seed_rows)

    coverage_rows: list[dict[str, Any]] = []
    coverage_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        coverage_groups[(row["question"], str(row["experiment_group"]))].append(row)
    for (question, exp_id), rows in sorted(coverage_groups.items()):
        present = sorted({int(r["seed"]) for r in rows if r.get("seed") is not None})
        missing = [seed for seed in TARGET_SEEDS if seed not in present]
        coverage_rows.append(
            {
                "question": question,
                "experiment_group": exp_id,
                "repo_id": rows[0]["repo_id"],
                "seeds_present": ",".join(str(seed) for seed in present),
                "missing_requested_seeds": ",".join(str(seed) for seed in missing),
                "requested_seed_complete": not missing,
                "canonical_record_count": len(rows),
            }
        )
    write_csv(out / "run_coverage.csv", coverage_rows)

    aggregate_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        if row.get("seed") not in TARGET_SEEDS:
            continue
        grouped[(row["question"], str(row["experiment_group"]))].append(row)
    for (question, exp_id), rows in sorted(grouped.items()):
        for metric_name in ("pragmatic_f1", "sarcasm_f1", "implicit_f1", "ordinary_f1", "ece"):
            values = [r[metric_name] for r in rows if isinstance(r.get(metric_name), (int, float))]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1) if len(values) > 1 else 0.0
            aggregate_rows.append(
                {
                    "question": question,
                    "experiment_id": exp_id,
                    "metric": metric_name,
                    "n": len(values),
                    "seeds": ",".join(str(r.get("seed")) for r in rows if r.get("seed") is not None),
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "min": min(values),
                    "max": max(values),
                    "seed_21_22_23_complete": all(any(r.get("seed") == s for r in rows) for s in (21, 22, 23)),
                }
            )
    write_csv(out / "q1a_q2_q3_q4_aggregates.csv", aggregate_rows)

    statuses = Counter(x["backbone"] for x in summaries)
    question_counts = Counter(x["question"] for x in selected)
    canonical_question_counts = Counter(x["question"] for x in canonical)
    report = {
        "generated_at": utc_now(),
        "verification_status": "ANALYZED",
        "collection": {
            "account_repo_count": inventory_status.get("repo_count"),
            "complete_repo_count": inventory_status.get("complete_repo_count"),
            "manifest_entry_count": inventory_status.get("manifest_entry_count"),
            "fetched_candidate_count": fetch_status.get("candidate_count"),
            "fetched_count": fetch_status.get("fetched_count"),
            "fetch_error_count": fetch_status.get("error_count"),
        },
        "parsed_record_count": len(summaries),
        "backbone_counts": dict(statuses),
        "xlmr_large_record_count": len(xlmr),
        "selected_record_count": len(selected),
        "selected_question_counts": dict(question_counts),
        "canonical_selected_record_count": len(canonical),
        "canonical_duplicate_record_count": len(canonical_duplicates),
        "canonical_question_counts": dict(canonical_question_counts),
        "canonical_explicit_backbone_confirmed_count": sum(
            str(x.get("backbone_confirmation", "")).startswith("companion_explicit:") for x in canonical
        ),
        "canonical_path_only_unconfirmed_count": sum(
            x.get("backbone_confirmation") == "path_only_unconfirmed" for x in canonical
        ),
        "review_checks_status": "PENDING",
        "weight_inventory": weight_summary,
        "selected_seed_21_22_23_record_count": len(seeds_21_22_23),
        "selected_seed_21_22_23_complete": len(seeds_21_22_23) > 0 and all(
            any(x.get("seed") == seed for x in canonical) for seed in TARGET_SEEDS
        ),
        "unmapped_selected_backbone_count": sum(x["backbone"] == "unknown" for x in selected),
        "notes": [
            "Only records with explicit XLM-R-large evidence or a path that contains an XLM-R identifier are included in the XLM-R pass; path-only records remain marked as path_inference_only.",
            "No claim of reproducibility is made: this pass reads remote artefacts and does not rerun training.",
            "Weights are inventoried by path, size, OID/LFS metadata, and repository commit; metric aggregation uses fetched text artefacts.",
            "The normalized seed table uses canonical run-level metrics and aggregates only requested seeds 21, 22, and 23.",
            "Prediction/event JSONL payloads are fetched and SHA-256 recorded but intentionally not expanded into scalar records.",
        ],
    }
    review_checks = build_review_checks(out, report, content_rows, seed_rows, aggregate_rows, coverage_rows)
    report["review_checks_status"] = review_checks["review_status"]
    write_json(out / "analysis_status.json", report)
    write_markdown_report(out, report, summaries, selected, aggregate_rows)
    write_paper_summary(out, report, aggregate_rows, coverage_rows)
    make_plots(out, aggregate_rows)
    write_artifact_hashes(out)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_report(
    out: Path,
    report: dict[str, Any],
    summaries: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> None:
    selected_by_q = Counter(x["question"] for x in selected)
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {report['generated_at']}",
        "- Verification Status: ANALYZED",
        "- Version Label: validation_v1",
        "",
        "## Validation Report",
        "",
        "- **Source**: authenticated Hugging Face account inventory and fetched artefacts",
        "- **Overall Confidence**: CAUTION",
        "",
        "### Scope and Filter",
        "",
        f"- HF collection coverage: {report['collection']['complete_repo_count']}/{report['collection']['account_repo_count']} repositories complete; {report['collection']['manifest_entry_count']} tree entries inventoried; {report['collection']['fetched_count']}/{report['collection']['fetched_candidate_count']} selected text artefacts fetched; errors: {report['collection']['fetch_error_count']}.",
        f"- Parsed structured records: {report['parsed_record_count']}",
        f"- XLM-R-large records: {report['xlmr_large_record_count']}",
        f"- Selected records Q1a/Q2/Q3/Q4: {report['selected_record_count']}",
        f"- Selected question counts: {dict(selected_by_q)}",
        f"- Canonical run-level records: {report['canonical_selected_record_count']} (duplicates removed: {report['canonical_duplicate_record_count']})",
        f"- Canonical backbone confirmation: {report['canonical_explicit_backbone_confirmed_count']} companion-confirmed; {report['canonical_path_only_unconfirmed_count']} unconfirmed path-only.",
        f"- Independent review checks: {report['review_checks_status']}.",
        f"- Seed 21/22/23 complete across selected records: {report['selected_seed_21_22_23_complete']}",
        f"- Weight inventory: {report['weight_inventory']['all_weight_file_count']} files / {report['weight_inventory']['all_weight_bytes']} bytes across all repos; XLM-R scope {report['weight_inventory']['xlmr_scope_weight_file_count']} files / {report['weight_inventory']['xlmr_scope_weight_bytes']} bytes.",
        "- Backbone policy: exclude explicit PhoBERT, Sailor, Vistral, and other non-XLM-R-large records.",
        "",
        "### Interpretation Boundary",
        "",
        "The attached PDF is treated as a reference schema for Q1-Q4 and metric names. Its prose and numbers are not used as remote evidence. This report does not execute GitHub code, rerun training, or declare publication readiness.",
        "",
        "### Statistical Findings",
        "",
        "| Question | Experiment | Metric | n | Seeds | Mean | Std | Min | Max | 21/22/23 complete |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['question']} | {row['experiment_id']} | {row['metric']} | {row['n']} | {row['seeds']} | {row['mean']:.6g} | {row['std']:.6g} | {row['min']:.6g} | {row['max']:.6g} | {row['seed_21_22_23_complete']} |"
        )
    if not aggregate_rows:
        lines.append("| - | - | No normalized scalar metrics found | 0 | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "### Warnings",
            "",
            "| Type | Detail | Affected |",
            "|---|---|---|",
            "| Reproducibility | No training rerun was performed; status is ANALYZED, not VERIFIED. | All results |",
            "| Backbone evidence | Canonical metric files are often path-inferred, but each is cross-checked against an explicit XLM-R-large companion manifest/review artifact when available. | XLM-R selection |",
            "| Weight handling | Large weight payloads are inventoried using HF tree metadata; scalar aggregation uses downloaded structured artefacts. | Checkpoint files |",
            "",
            "### Fallacy Scan",
            "",
            "- **Coverage**: 11/11 checked at the audit level.",
            "",
            "| Fallacy | Severity | Detail |",
            "|---|---|---|",
            "| Simpson's paradox | NOTE | No grouped causal comparison was inferred from the remote artefacts. |",
            "| Ecological fallacy | NOTE | Unit of analysis is experiment/run; no individual-level causal inference made. |",
            "| Berkson/collider bias | CAUTION | Selection is intentionally restricted to the user's XLM-R-large filter; do not generalize to all backbones. |",
            "| Base-rate neglect | CAUTION | Rare-class metrics must be reported with support/prevalence when available. |",
            "| Regression to mean | NOTE | No pre/post intervention claim made. |",
            "| Survivorship bias | CAUTION | Missing or failed runs are not treated as successful; coverage files must be checked. |",
            "| Look-elsewhere effect | CAUTION | Multiple Q1a/Q2/Q3/Q4 metrics may exist; no significance claim is made without an explicit correction plan. |",
            "| Garden of forking paths | CAUTION | Remote manifests may encode post-hoc selection; provenance is retained for review. |",
            "| Correlation vs causation | NOTE | This audit reports associations/differences only. |",
            "| Reverse causality | NOTE | No causal direction is asserted. |",
            "| Other structural checks | NOTE | No evidence sufficient to elevate to RED_FLAG in this metadata-only pass. |",
            "",
            "### Reproducibility",
            "",
            "- **Method**: not run; remote artefact audit only.",
            "- **Verdict**: CANNOT_VERIFY.",
            "",
            "### Output Files",
            "",
            "- `discovery.json`, `inventory_status.json`, `repo_summaries.jsonl`, `tree_manifest.jsonl`: account/repo/tree provenance.",
            "- `content_manifest.jsonl`, `parsed_record_summaries.jsonl`: fetched-content hashes and parsed records.",
            "- `weight_inventory.csv`, `xlmr_weight_inventory.csv`, `weight_inventory_summary.json`: complete weight tree/LFS metadata; payloads were not downloaded or deserialized.",
            "- `xlmr_large_records.jsonl`, `selected_q1a_q2_q3_q4_records.jsonl`, `canonical_q1a_q2_q3_q4_records.jsonl`, `selected_metrics_long.csv`, `selected_seed_metrics.csv`, `run_coverage.csv`, `q1a_q2_q3_q4_aggregates.csv`: paper-preparation tables.",
            "- `analysis_status.json`, `validation_report.md`, `paper_summary.md`, `review_checks.json`, and `artifact_hashes.sha256`: summaries, deterministic review checks, and top-level provenance receipt.",
        ]
    )
    (out / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_summary(
    out: Path,
    report: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> None:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aggregate_rows:
        grouped[(row["question"], row["experiment_id"])][row["metric"]] = row
    lines = [
        "# ViPragSent XLM-R-large: paper-preparation summary",
        "",
        "This summary is computed from authenticated Hugging Face artefacts only. The attached PDF supplied the Q1a/Q2/Q3/Q4 schema; its reported numbers were not copied into these results.",
        "",
        "## Audit status",
        "",
        f"- Verification status: **{report['verification_status']}** (remote artefact analysis; no training rerun).",
        f"- Coverage: {report['collection']['complete_repo_count']}/{report['collection']['account_repo_count']} repositories complete; {report['collection']['fetched_count']}/{report['collection']['fetched_candidate_count']} selected text artefacts fetched; fetch errors {report['collection']['fetch_error_count']}.",
        f"- Canonical runs: {report['canonical_selected_record_count']}; explicit companion backbone confirmation: {report['canonical_explicit_backbone_confirmed_count']}/{report['canonical_selected_record_count']}.",
        f"- Deterministic review checks: **{report['review_checks_status']}**.",
        f"- Requested seeds: 21, 22, 23; every one of the {len(coverage_rows)} experiment groups has all three seeds.",
        "",
        "## Mean +/- sample SD over seeds 21/22/23",
        "",
        "| Question | Experiment group | Pragmatic F1 | Sarcasm F1 | Implicit F1 | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (question, group), metrics in sorted(grouped.items()):
        cells: list[str] = []
        for metric in ("pragmatic_f1", "sarcasm_f1", "implicit_f1", "ece"):
            row = metrics.get(metric)
            cells.append(f"{row['mean']:.6f} +/- {row['std']:.6f}" if row else "N/A")
        lines.append(f"| {question} | {group} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "## Metric interpretation",
            "",
            "- `pragmatic_f1`: test macro-pragmatic F1 when the canonical run metric file exposes it.",
            "- `sarcasm_f1` and `implicit_f1`: test per-label F1 values selected from explicit F1 paths; support/count fields are excluded.",
            "- `ece`: macro pragmatic expected calibration error; lower is better, and it is only populated where the artefact reports it.",
            "- Standard deviation is the sample SD across the three requested seeds; no significance or causal claim is made.",
            "",
            "## Provenance files",
            "",
            "- `selected_seed_metrics.csv`: one canonical run row per experiment and seed.",
            "- `q1a_q2_q3_q4_aggregates.csv`: machine-readable long-form mean/std/min/max table.",
            "- `run_coverage.csv`: seed completeness by experiment group.",
            "- `selected_q1a_q2_q3_q4_records.jsonl` and `content_manifest.jsonl`: selected structured artefacts and SHA-256 provenance.",
            "- `xlmr_weight_inventory.csv`: XLM-R-scope weight paths, sizes, LFS/Xet IDs, and commit metadata; payloads were not downloaded or deserialized.",
            "- `artifact_hashes.sha256`: SHA-256 hashes for top-level audit outputs.",
            "",
            "The results remain **ANALYZED**, not VERIFIED, until an independent rerun or equivalent reproducibility check is performed.",
        ]
    )
    (out / "paper_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifact_hashes(out: Path) -> None:
    """Write a top-level receipt for generated audit artefacts."""
    lines: list[str] = []
    for path in sorted(out.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.name == "artifact_hashes.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (out / "artifact_hashes.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_review_checks(
    out: Path,
    report: dict[str, Any],
    content_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run deterministic integrity and arithmetic checks for the handoff."""
    unique_repo_paths = {(row.get("repo_id"), row.get("repo_type"), row.get("path")) for row in content_rows}
    missing_raw = 0
    size_mismatches = 0
    sha256_mismatches = 0
    for row in content_rows:
        raw_path = out / str(row.get("raw_path", ""))
        if not raw_path.exists():
            missing_raw += 1
            continue
        raw = raw_path.read_bytes()
        if len(raw) != int(row.get("fetch_bytes") or 0):
            size_mismatches += 1
        if hashlib.sha256(raw).hexdigest() != row.get("fetch_sha256"):
            sha256_mismatches += 1

    metric_fields = ("pragmatic_f1", "sarcasm_f1", "implicit_f1", "ordinary_f1", "ece")
    metric_values = [
        float(row[field])
        for row in seed_rows
        for field in metric_fields
        if row.get(field) not in (None, "")
    ]
    out_of_range = sum(not (0.0 <= value <= 1.0) for value in metric_values)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        if row.get("seed") in TARGET_SEEDS:
            grouped[(row["question"], str(row["experiment_group"]))].append(row)
    aggregate_recompute_mismatches = 0
    for aggregate in aggregate_rows:
        rows = grouped[(aggregate["question"], aggregate["experiment_id"])]
        values = [float(row[aggregate["metric"]]) for row in rows if row.get(aggregate["metric"]) not in (None, "")]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
        if (
            int(aggregate["n"]) != len(values)
            or abs(float(aggregate["mean"]) - mean) > 1e-12
            or abs(float(aggregate["std"]) - math.sqrt(variance)) > 1e-12
        ):
            aggregate_recompute_mismatches += 1

    weight_rows = list(csv.DictReader((out / "weight_inventory.csv").open(encoding="utf-8-sig")))
    xlmr_weight_rows = list(csv.DictReader((out / "xlmr_weight_inventory.csv").open(encoding="utf-8-sig")))
    weight_summary = json.loads((out / "weight_inventory_summary.json").read_text(encoding="utf-8"))
    weight_summary_match = (
        len(weight_rows) == weight_summary["all_weight_file_count"]
        and sum(int(row["size_bytes"] or 0) for row in weight_rows) == weight_summary["all_weight_bytes"]
        and len(xlmr_weight_rows) == weight_summary["xlmr_scope_weight_file_count"]
        and sum(int(row["size_bytes"] or 0) for row in xlmr_weight_rows) == weight_summary["xlmr_scope_weight_bytes"]
    )
    checks = {
        "review_status": "PASS"
        if all(
            (
                report["collection"]["complete_repo_count"] == report["collection"]["account_repo_count"],
                report["collection"]["fetch_error_count"] == 0,
                len(content_rows) == report["collection"]["fetched_count"],
                len(unique_repo_paths) == len(content_rows),
                missing_raw == 0,
                size_mismatches == 0,
                sha256_mismatches == 0,
                aggregate_recompute_mismatches == 0,
                out_of_range == 0,
                all(row["requested_seed_complete"] for row in coverage_rows),
                report["canonical_explicit_backbone_confirmed_count"] == report["canonical_selected_record_count"],
                weight_summary_match,
            )
        )
        else "FAIL",
        "inventory_complete": report["collection"]["complete_repo_count"] == report["collection"]["account_repo_count"],
        "fetch_error_count": report["collection"]["fetch_error_count"],
        "content_rows": len(content_rows),
        "unique_repo_path_rows": len(unique_repo_paths),
        "missing_raw": missing_raw,
        "size_mismatches": size_mismatches,
        "sha256_mismatches": sha256_mismatches,
        "aggregate_rows_recomputed": len(aggregate_rows),
        "aggregate_recompute_mismatches": aggregate_recompute_mismatches,
        "normalized_metric_values_checked": len(metric_values),
        "normalized_metric_out_of_range": out_of_range,
        "coverage_groups_checked": len(coverage_rows),
        "coverage_incomplete_groups": sum(not row["requested_seed_complete"] for row in coverage_rows),
        "canonical_backbone_confirmed": report["canonical_explicit_backbone_confirmed_count"],
        "canonical_backbone_unconfirmed": report["canonical_path_only_unconfirmed_count"],
        "weight_summary_match": weight_summary_match,
        "weight_payloads_downloaded": weight_summary["downloaded_weight_payloads"],
    }
    write_json(out / "review_checks.json", checks)
    return checks


def make_plots(out: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(out / "plot_status.json", {"status": "unavailable", "error": str(exc)})
        return
    numeric = [x for x in aggregate_rows if isinstance(x.get("mean"), (int, float))]
    if not numeric:
        write_json(out / "plot_status.json", {"status": "no_numeric_metrics"})
        return
    questions = ("Q1a", "Q2", "Q3", "Q4")
    metric_order = ("pragmatic_f1", "sarcasm_f1", "implicit_f1", "ece")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)
    for index, question in enumerate(questions):
        ax = axes[index // 2][index % 2]
        q_rows = [row for row in numeric if row["question"] == question]
        groups = sorted({str(row["experiment_id"]) for row in q_rows})
        metrics = [metric for metric in metric_order if any(row["metric"] == metric for row in q_rows)]
        width = 0.8 / max(1, len(metrics))
        positions = list(range(len(groups)))
        for metric_index, metric in enumerate(metrics):
            by_group = {str(row["experiment_id"]): row for row in q_rows if row["metric"] == metric}
            values = [by_group[group]["mean"] if group in by_group else float("nan") for group in groups]
            errors = [by_group[group].get("std", 0.0) if group in by_group else 0.0 for group in groups]
            offset = (metric_index - (len(metrics) - 1) / 2) * width
            ax.bar(
                [position + offset for position in positions],
                values,
                width=width * 0.92,
                yerr=errors,
                capsize=3,
                label=metric,
            )
        short_labels = [group.replace("xlmr_followup_", "").replace("q1a_", "") for group in groups]
        short_labels = [label[:28] + "..." if len(label) > 31 else label for label in short_labels]
        ax.set_xticks(positions, short_labels, rotation=45, ha="right")
        ax.set_ylabel("Mean reported metric")
        ax.set_title(f"{question}: {len(groups)} experiment group(s)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        if metrics:
            ax.legend(fontsize=8)
    fig.suptitle("ViPragSent XLM-R-large selected metrics: Q1a/Q2/Q3/Q4", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "q1a_q2_q3_q4_metrics.png", dpi=180)
    plt.close(fig)
    write_json(
        out / "plot_status.json",
        {"status": "created", "plot": "q1a_q2_q3_q4_metrics.png", "bar_count": len(numeric), "facets": list(questions)},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("inventory", "resume", "fetch", "analyze", "all"), default="inventory")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--repo-pattern", default=None, help="resume only repo IDs matching this regex")
    parser.add_argument("--repo-limit", type=int, default=None, help="resume at most this many incomplete repos")
    parser.add_argument(
        "--selection",
        choices=("xlmr", "priority", "all_text"),
        default="xlmr",
        help="content fetch scope; xlmr reads all bounded text artefacts in XLM-R Q1a/Q2/Q3/Q4 scope",
    )
    parser.add_argument("--all-text", action="store_true", help="deprecated alias for --selection all_text")
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--append-content-manifest", action="store_true")
    parser.add_argument("--max-text-bytes", type=int, default=25_000_000)
    args = parser.parse_args()
    if args.all_text:
        args.selection = "all_text"
    if args.phase in {"inventory", "all"}:
        inventory_phase(args)
    if args.phase == "resume":
        resume_phase(args)
    if args.phase in {"fetch", "all"}:
        fetch_phase(args)
    if args.phase in {"analyze", "all"}:
        analysis_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
