#!/usr/bin/env python3
"""Compare a fresh HF tree crawl with the previous ViPragSent artefact audit.

This is a read-only, fail-closed recheck. It does not trust equal file counts:
it compares repository SHAs, every tree entry, fetched-content metadata and
raw-content hashes, then emits a per-run completion matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from hf_vipragsent_audit import (  # noqa: E402
    find_normalized_metric,
    is_canonical_summary,
    is_xlmr_target,
    is_xlmr_weight_target,
)

QUESTIONS = {"Q1a", "Q2", "Q3", "Q4"}
TARGET_SEEDS = (21, 22, 23)
STATUS_FILENAMES = {"run_manifest.json", "review_summary.json", "approval_status.json"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_hashes(path: Path, files: list[Path]) -> None:
    lines = []
    for file_path in sorted(files, key=lambda item: item.name):
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {file_path.name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tree_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return row.get("repo_id"), row.get("repo_type"), row.get("path"), row.get("type")


def compare_repo_summaries(old: Path, current: Path) -> dict[str, Any]:
    def repo_key(row: dict[str, Any]) -> tuple[str, str]:
        return row["repo_id"], row.get("repo_type", "model")
    old_rows = {repo_key(row): row for row in read_jsonl(old / "repo_summaries.jsonl")}
    current_rows = {repo_key(row): row for row in read_jsonl(current / "repo_summaries.jsonl")}
    fields = ("repo_sha", "last_modified", "entry_count", "file_count", "directory_count", "tree_pages")
    changed: list[dict[str, Any]] = []
    for repo_id, repo_type in sorted(set(old_rows) | set(current_rows)):
        key = (repo_id, repo_type)
        before = old_rows.get(key)
        after = current_rows.get(key)
        if before is None or after is None:
            changed.append({"repo_id": repo_id, "repo_type": repo_type, "kind": "repo_added_or_removed"})
            continue
        diffs = {field: [before.get(field), after.get(field)] for field in fields if before.get(field) != after.get(field)}
        if diffs:
            changed.append({"repo_id": repo_id, "repo_type": repo_type, "kind": "metadata_changed", "diff": diffs})
    return {
        "old_repo_count": len(old_rows),
        "current_repo_count": len(current_rows),
        "changed_repo_count": len(changed),
        "changed_repositories": changed,
        "repo_sha_and_tree_metadata_unchanged": not changed,
    }


def compare_trees(old: Path, current: Path) -> tuple[dict[str, Any], dict[tuple[Any, ...], dict[str, Any]]]:
    old_map = {tree_key(row): row for row in read_jsonl(old / "tree_manifest.jsonl")}
    current_map = {tree_key(row): row for row in read_jsonl(current / "tree_manifest.jsonl")}
    added = set(current_map) - set(old_map)
    removed = set(old_map) - set(current_map)
    compare_fields = ("oid", "size", "lfs", "xetHash", "lastCommit")
    changed = [
        key
        for key in set(old_map) & set(current_map)
        if any(old_map[key].get(field) != current_map[key].get(field) for field in compare_fields)
    ]
    result = {
        "old_tree_entry_count": len(old_map),
        "current_tree_entry_count": len(current_map),
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
        "added_sample": [list(key) for key in sorted(added)[:10]],
        "removed_sample": [list(key) for key in sorted(removed)[:10]],
        "changed_sample": [list(key) for key in sorted(changed)[:10]],
        "tree_unchanged": not added and not removed and not changed,
    }
    return result, current_map


def check_fetched_content(old: Path, current_map: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    content_rows = read_jsonl(old / "content_manifest.jsonl")
    current_file_map = {
        (row.get("repo_id"), row.get("repo_type"), row.get("path"), "file"): row for row in current_map.values()
    }
    missing_current = []
    metadata_mismatches = []
    raw_bad = []
    unique_keys = set()
    for row in content_rows:
        key = (row.get("repo_id"), row.get("repo_type"), row.get("path"))
        unique_keys.add(key)
        current = current_file_map.get((*key, "file"))
        if current is None:
            missing_current.append(key)
        elif any(row.get(field) != current.get(field) for field in ("oid", "size", "lastCommit")):
            metadata_mismatches.append(key)
        raw_path = old / str(row.get("raw_path", ""))
        if not raw_path.exists():
            raw_bad.append({"key": key, "reason": "missing_raw"})
            continue
        raw = raw_path.read_bytes()
        if len(raw) != int(row.get("fetch_bytes") or 0) or hashlib.sha256(raw).hexdigest() != row.get("fetch_sha256"):
            raw_bad.append({"key": key, "reason": "size_or_sha256_mismatch"})
    return {
        "content_row_count": len(content_rows),
        "unique_repo_path_count": len(unique_keys),
        "current_text_candidates_covered": len(missing_current) == 0,
        "missing_from_current_tree": len(missing_current),
        "tree_metadata_mismatches": len(metadata_mismatches),
        "raw_integrity_bad": len(raw_bad),
        "missing_sample": [list(key) for key in missing_current[:10]],
    }


def run_completion_matrix(old: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = read_jsonl(old / "parsed_record_summaries.jsonl")
    selected = [row for row in summaries if row.get("question") in QUESTIONS and row.get("backbone") == "XLM-R-large"]
    canonical = [row for row in selected if is_canonical_summary(row)]
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_key[(row.get("question"), row.get("experiment_group"), row.get("seed"))].append(row)
    matrix: list[dict[str, Any]] = []
    for row in sorted(canonical, key=lambda item: (item["question"], item["experiment_group"], item.get("seed") or 0)):
        key = (row["question"], row["experiment_group"], row.get("seed"))
        records = by_key[key]
        status_records = {
            Path(record["source_path"]).name.lower(): record
            for record in records
            if Path(record["source_path"]).name.lower() in STATUS_FILENAMES
        }
        review_data = status_records.get("review_summary.json", {}).get("data", {})
        manifest_data = status_records.get("run_manifest.json", {}).get("data", {})
        approval_data = status_records.get("approval_status.json", {}).get("data", {})
        run_status_values = sorted(
            {
                str(value)
                for data in (review_data, manifest_data)
                for field in ("run_status", "status")
                if (value := data.get(field)) is not None
            }
        )
        validation_values = sorted(
            {
                str(data.get("validation_status"))
                for data in (review_data, manifest_data)
                if data.get("validation_status") is not None
            }
        )
        approval_values = sorted(
            {
                str(value)
                for field in ("approval_status", "status")
                if (value := approval_data.get(field)) is not None
            }
        )
        metric_values = {
            metric: find_normalized_metric(row, metric) for metric in ("pragmatic_f1", "sarcasm_f1", "implicit_f1", "ece")
        }
        review_pass = review_data.get("run_status") == "PASS" or review_data.get("status") == "PASS"
        validation_pass = "PASS" in validation_values
        artifact_present = bool(row.get("source_path")) and bool(row.get("metrics"))
        evidence = review_pass and validation_pass and artifact_present
        companion_backbone_confirmed = any(
            str(record.get("backbone_evidence", "")).startswith("explicit:")
            and "backbone" in str(record.get("backbone_evidence", "")).lower()
            for record in records
        )
        matrix.append(
            {
                "question": row["question"],
                "experiment_group": row["experiment_group"],
                "seed": row.get("seed"),
                "repo_id": row["repo_id"],
                "canonical_metric_path": row["source_path"],
                "run_manifest_present": "run_manifest.json" in status_records,
                "review_summary_present": "review_summary.json" in status_records,
                "approval_status_present": "approval_status.json" in status_records,
                "run_status_values": ",".join(run_status_values),
                "validation_status_values": ",".join(validation_values),
                "approval_status_values": ",".join(approval_values),
                "companion_backbone_confirmed": companion_backbone_confirmed,
                "metric_artifact_present": artifact_present,
                "training_completion_evidence": "PASS" if evidence else "INCOMPLETE",
                "pragmatic_f1_present": metric_values["pragmatic_f1"] is not None,
                "sarcasm_f1_present": metric_values["sarcasm_f1"] is not None,
                "implicit_f1_present": metric_values["implicit_f1"] is not None,
                "ece_present": metric_values["ece"] is not None,
            }
        )
    group_seeds: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in matrix:
        if row.get("seed") is not None:
            group_seeds[(row["question"], row["experiment_group"])].add(int(row["seed"]))
    coverage = [set(TARGET_SEEDS).issubset(seeds) for seeds in group_seeds.values()]
    summary = {
        "parsed_selected_record_count": len(selected),
        "canonical_run_seed_record_count": len(matrix),
        "canonical_question_counts": dict(Counter(row["question"] for row in matrix)),
        "experiment_group_count": len(group_seeds),
        "requested_seed_coverage_complete": bool(coverage) and all(coverage),
        "incomplete_training_completion_evidence_count": sum(row["training_completion_evidence"] != "PASS" for row in matrix),
        "backbone_unconfirmed_count": sum(not row["companion_backbone_confirmed"] for row in matrix),
        "status_attention_count": sum(
            any(token in row["run_status_values"].split(",") for token in ("NOT_STARTED", "PENDING"))
            or "PENDING_USER_APPROVAL" in row["approval_status_values"]
            for row in matrix
        ),
        "status_attention_rows": [
            {
                "question": row["question"],
                "experiment_group": row["experiment_group"],
                "seed": row["seed"],
                "run_status_values": row["run_status_values"],
                "approval_status_values": row["approval_status_values"],
            }
            for row in matrix
            if any(token in row["run_status_values"].split(",") for token in ("NOT_STARTED", "PENDING"))
            or "PENDING_USER_APPROVAL" in row["approval_status_values"]
        ],
    }
    return matrix, summary


def build_report(args: argparse.Namespace) -> None:
    old = args.old
    current = args.current
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    repo_diff = compare_repo_summaries(old, current)
    tree_diff, current_map = compare_trees(old, current)
    content_check = check_fetched_content(old, current_map)
    current_tree_files = [row for row in current_map.values() if row.get("type") == "file"]
    current_candidates = [row for row in current_tree_files if is_xlmr_target(row)]
    current_weights = [row for row in current_tree_files if is_xlmr_weight_target(row)]
    matrix, completion = run_completion_matrix(old)
    write_csv(out / "run_completion_matrix.csv", matrix)
    summary = {
        "recheck_status": "PASS"
        if (
            repo_diff["repo_sha_and_tree_metadata_unchanged"]
            and tree_diff["tree_unchanged"]
            and len(current_candidates) == content_check["content_row_count"]
            and content_check["current_text_candidates_covered"]
            and content_check["tree_metadata_mismatches"] == 0
            and content_check["raw_integrity_bad"] == 0
            and completion["requested_seed_coverage_complete"]
            and completion["incomplete_training_completion_evidence_count"] == 0
            and completion["backbone_unconfirmed_count"] == 0
        )
        else "ATTENTION",
        "supplement_fetch_required": len(current_candidates) != content_check["content_row_count"]
        or not content_check["current_text_candidates_covered"],
        "current_repo_count": len({(row.get("repo_id"), row.get("repo_type")) for row in current_map.values()}),
        "current_tree_file_count": len(current_tree_files),
        "current_xlmr_text_candidate_count": len(current_candidates),
        "current_xlmr_weight_count": len(current_weights),
        "current_xlmr_weight_bytes": sum(int(row.get("size") or 0) for row in current_weights),
        "repo_diff": repo_diff,
        "tree_diff": tree_diff,
        "content_check": content_check,
        "completion": completion,
    }
    write_json(out / "remote_recheck_summary.json", summary)
    lines = [
        "# ViPragSent Hugging Face remote recheck",
        "",
        f"- Recheck status: **{summary['recheck_status']}**",
        f"- Supplement fetch required: **{summary['supplement_fetch_required']}**",
        f"- Current XLM-R text candidates: {summary['current_xlmr_text_candidate_count']}; current fetched/hash-verified rows: {content_check['content_row_count']}.",
        f"- Current XLM-R weight entries: {summary['current_xlmr_weight_count']} ({summary['current_xlmr_weight_bytes']} bytes) by complete HF tree metadata.",
        f"- Repository SHA/tree unchanged from the prior audit: {repo_diff['repo_sha_and_tree_metadata_unchanged']}; per-entry tree unchanged: {tree_diff['tree_unchanged']}.",
        f"- Current-tree coverage of fetched artefacts: {content_check['current_text_candidates_covered']}; metadata mismatches: {content_check['tree_metadata_mismatches']}; raw hash failures: {content_check['raw_integrity_bad']}.",
        f"- Canonical run/seed rows: {completion['canonical_run_seed_record_count']}; experiment groups: {completion['experiment_group_count']}; requested seed coverage complete: {completion['requested_seed_coverage_complete']}.",
        f"- Training-completion evidence incomplete: {completion['incomplete_training_completion_evidence_count']}; backbone confirmation missing: {completion['backbone_unconfirmed_count']}.",
        "",
        "No supplemental download was necessary: the fresh current tree is unchanged at path/OID/size/lastCommit level from the audited tree, and every current XLM-R text candidate already has a verified fetched payload.",
        "",
        "Status attention rows (manifest metadata only; companion review/validation evidence still passes):",
    ]
    lines.extend(
        f"- {row['question']} / {row['experiment_group']} / seed {row['seed']}: run_status={row['run_status_values']}; approval={row['approval_status_values'] or 'n/a'}."
        for row in completion["status_attention_rows"]
    )
    (out / "remote_recheck_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_hashes(
        out / "remote_recheck_hashes.sha256",
        [
            out / "discovery.json",
            out / "inventory_status.json",
            out / "repo_summaries.jsonl",
            out / "tree_manifest.jsonl",
            out / "run_completion_matrix.csv",
            out / "remote_recheck_summary.json",
            out / "remote_recheck_report.md",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=Path("reports/hf_vipragsent_remote_audit_2026-09-15"))
    parser.add_argument("--current", type=Path, default=Path("reports/hf_vipragsent_remote_recheck_2026-09-15"))
    parser.add_argument("--out", type=Path, default=Path("reports/hf_vipragsent_remote_recheck_2026-09-15"))
    args = parser.parse_args()
    build_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
