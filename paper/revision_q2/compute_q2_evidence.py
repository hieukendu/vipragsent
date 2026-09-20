"""Audit and recompute the canonical Q2 table from live Hugging Face artifacts.

This script is intentionally fail-closed for the no-polarity variant.  It does
not synthesize a polarity distribution when the saved row has no polarity head
or polarity probabilities.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "revision_q2"
HF = "https://huggingface.co"
CAMPAIGN = "campaigns/vipragsent-xlmr-q1b-q4-followup-v1"
SEEDS = (20260521, 20260522, 20260523)
VARIANT_ORDER = (
    "full",
    "no_emotion_auxiliary",
    "no_polarity_auxiliary",
    "no_rationale",
    "no_multitask",
    "no_uncertainty_weighting",
)
# Complete root artifact trees.  Alternate/partial candidates are inventoried
# separately from the local remote manifest and are never silently promoted.
ARTIFACT_REPOS = {
    "full": {20260521: "019", 20260522: "016", 20260523: "015"},
    "no_emotion_auxiliary": {20260521: "014", 20260522: "013", 20260523: "012"},
    "no_polarity_auxiliary": {20260521: "011", 20260522: "010", 20260523: "009"},
    "no_rationale": {20260521: "008", 20260522: "007", 20260523: "018"},
    "no_multitask": {20260521: "018", 20260522: "025", 20260523: "012"},
    "no_uncertainty_weighting": {20260521: "025", 20260522: "017", 20260523: "012"},
}
REPO_PREFIX = "Thundergod2007/vipragsent-experiment-artifacts-overflow-"
CHECKPOINT_REPO = "Thundergod2007/vipragsent-xlmr-checkpoints"
ECE_TOLERANCE = 1e-12
POLARITY_LABELS = ("negative", "neutral", "positive")
RAW_FILES = (
    "config_snapshot.yaml",
    "training/resolved_training_config.json",
    "run_manifest.json",
    "review_summary.json",
    "metrics/dev_metrics.json",
    "metrics/test_metrics.json",
    "predictions/dev_predictions.jsonl",
    "predictions/test_predictions.jsonl",
    "selection/best_checkpoint.json",
    "selection/freeze_manifest.json",
    "selection/selection_metric.json",
)
Q2_REQUIRED_RAW_FILES = (
    "run_manifest.json",
    "review_summary.json",
    "metrics/dev_metrics.json",
    "metrics/test_metrics.json",
    "predictions/dev_predictions.jsonl",
    "predictions/test_predictions.jsonl",
)


def get(url: str) -> tuple[int, bytes, dict[str, str]]:
    request = Request(url, headers={"User-Agent": "ViPragSent-Q2-evidence-audit/1.0"})
    retryable = {0, 429, 500, 502, 503, 504}
    last_status = 0
    last_body = b""
    last_headers: dict[str, str] = {}
    for attempt in range(3):
        try:
            with urlopen(request, timeout=90) as response:
                status = int(response.status)
                body = response.read()
                headers = {str(k): str(v) for k, v in response.headers.items()}
                if status in retryable and attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return status, body, headers
        except HTTPError as exc:
            last_status = int(exc.code)
            last_body = exc.read()
            last_headers = {str(k): str(v) for k, v in exc.headers.items()}
            if last_status not in retryable or attempt == 2:
                return last_status, last_body, last_headers
        except Exception as exc:  # preserve status-like evidence without hiding failure
            last_status = 0
            last_body = f"ERROR: {type(exc).__name__}: {exc}".encode("utf-8")
            last_headers = {}
            if attempt == 2:
                return last_status, last_body, last_headers
        time.sleep(1.0 * (attempt + 1))
    return last_status, last_body, last_headers


def raw_url(repo: str, path: str, revision: str) -> str:
    return f"{HF}/{repo}/resolve/{revision}/{quote(path, safe='/')}?download=true"


def tree_url(repo: str, path: str, revision: str, cursor: str | None = None) -> str:
    # Keep path separators for the Hub tree endpoint; only unsafe characters
    # within each path component are escaped.
    encoded = quote(path, safe="/")
    suffix = "&cursor=" + quote(cursor, safe="") if cursor else ""
    return f"{HF}/api/models/{repo}/tree/{revision}/{encoded}?recursive=true&expand=false&limit=1000{suffix}"


def fetch_json(repo: str, path: str, revision: str) -> tuple[int, bytes, Any]:
    status, body, _ = get(raw_url(repo, path, revision))
    if status != 200:
        return status, body, None
    try:
        return status, body, json.loads(body)
    except json.JSONDecodeError:
        return status, body, None


def repo_revision(repo: str) -> tuple[int, bytes, str | None]:
    url = f"{HF}/api/models/{quote(repo, safe='/')}"
    status, body, _ = get(url)
    if status != 200:
        return status, body, None
    try:
        return status, body, str(json.loads(body).get("sha"))
    except (json.JSONDecodeError, AttributeError):
        return status, body, None


def fetch_tree(repo: str, path: str, revision: str) -> tuple[int, bytes, list[dict[str, Any]]]:
    """Fetch every tree page through the API, following the RFC5988 next link."""
    cursor: str | None = None
    pages: list[bytes] = []
    entries: list[dict[str, Any]] = []
    first_status = 0
    while True:
        url = tree_url(repo, path, revision, cursor)
        status, body, headers = get(url)
        if first_status == 0:
            first_status = status
        pages.append(body)
        if status != 200:
            break
        try:
            page = json.loads(body)
        except json.JSONDecodeError:
            break
        if not isinstance(page, list):
            break
        entries.extend(item for item in page if isinstance(item, dict))
        link = headers.get("Link", "")
        match = re.search(r"<([^>]+)>;\s*rel=\"next\"", link)
        if not match:
            break
        next_url = match.group(1)
        cursor_match = re.search(r"[?&]cursor=([^&]+)", next_url)
        if not cursor_match:
            break
        cursor = cursor_match.group(1)
    return first_status, b"\n".join(pages), entries


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def ece_from_rows(rows: list[dict[str, Any]], task: str = "polarity") -> float | None:
    values: list[tuple[float, int]] = []
    for row in rows:
        probabilities = row.get("probabilities", {}).get(task)
        gold = row.get("gold", {}).get(task)
        if not isinstance(probabilities, list) or gold is None or len(probabilities) != 3:
            return None
        confidence = float(max(probabilities))
        prediction = max(range(len(probabilities)), key=lambda i: float(probabilities[i]))
        if isinstance(gold, bool):
            return None
        if isinstance(gold, (int, float)):
            gold_index = int(gold)
        elif isinstance(gold, str) and gold in POLARITY_LABELS:
            # The saved exporter uses the canonical label order from
            # src/vipragsent/constants.py and configs/labels.json.
            gold_index = POLARITY_LABELS.index(gold)
        else:
            return None
        if gold_index < 0 or gold_index >= len(probabilities):
            return None
        values.append((confidence, int(prediction == gold_index)))
    if not values:
        return None
    total = 0.0
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        bucket = [item for item in values if lower <= item[0] < upper or (index == 9 and item[0] == upper)]
        if bucket:
            accuracy = sum(item[1] for item in bucket) / len(bucket)
            confidence = sum(item[0] for item in bucket) / len(bucket)
            total += len(bucket) / len(values) * abs(accuracy - confidence)
    return total


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def discover_local_candidates() -> dict[str, list[dict[str, Any]]]:
    manifest = ROOT / "reports" / "hf_vipragsent_remote_audit_2026-09-15" / "tree_manifest.jsonl"
    result: dict[str, list[dict[str, Any]]] = {}
    if not manifest.exists():
        return result
    pattern = re.compile(r"xlmr_followup_q2_(.+)_202605(21|22|23)(?:/|$)")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        path = str(item.get("path", ""))
        match = pattern.search(path)
        if not match:
            continue
        run = f"xlmr_followup_q2_{match.group(1)}_202605{match.group(2)}"
        key = run
        result.setdefault(key, []).append(
            {
                "repo": item.get("repo_id"),
                "path": path,
                "size": item.get("size"),
                "type": item.get("type"),
            }
        )
    return result


def persist_checkpoint(state: dict[str, Any], inventory: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["inventory_count"] = len(inventory)
    state["evidence_count"] = len(evidence)
    (OUT / "q2_checkpoint.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "source_inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "run_evidence.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in evidence) + ("\n" if evidence else ""), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / "configs" / "experiments" / "q2" / "protocol.yaml"
    protocol_text = protocol_path.read_text(encoding="utf-8")
    if not re.search(r"ece:\s*\n\s+split:\s+vipragsent_dev", protocol_text):
        raise RuntimeError("Q2 protocol ECE split is not explicitly vipragsent_dev")

    inventory: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol_ece_split": "vipragsent_dev",
        "completed_runs": [],
        "repo_revisions": {},
    }
    persist_checkpoint(state, inventory, evidence)
    alternate_candidates = discover_local_candidates()

    smoke = "--smoke" in sys.argv[1:]
    run_specs = (
        [("no_polarity_auxiliary", 20260521), ("full", 20260521)]
        if smoke
        else [(variant, seed) for variant in VARIANT_ORDER for seed in SEEDS]
    )

    all_repos = {REPO_PREFIX + suffix for mapping in ARTIFACT_REPOS.values() for suffix in mapping.values()}
    all_repos.add(CHECKPOINT_REPO)
    for repo in sorted(all_repos):
        status, body, revision = repo_revision(repo)
        if status != 200 or not revision:
            state.update({"status": "blocked", "error": f"HF revision lookup failed for {repo}: HTTP {status}"})
            persist_checkpoint(state, inventory, evidence)
            raise RuntimeError(state["error"])
        state["repo_revisions"][repo] = {"revision": revision, "metadata_url": f"{HF}/api/models/{repo}", "metadata_sha256": sha256(body)}
        persist_checkpoint(state, inventory, evidence)

    for variant, seed in run_specs:
            run = f"xlmr_followup_q2_{variant}_{seed}"
            repo = REPO_PREFIX + ARTIFACT_REPOS[variant][seed]
            revision = state["repo_revisions"][repo]["revision"]
            checkpoint_revision = state["repo_revisions"][CHECKPOINT_REPO]["revision"]
            run_path = f"{CAMPAIGN}/{run}"
            tree_status, tree_body, tree_files = fetch_tree(repo, run_path, revision)
            inventory.append(
                {
                    "run": run,
                    "repo": repo,
                    "kind": "artifact",
                    "revision": revision,
                    "tree_url": tree_url(repo, run_path, revision),
                    "tree_status": tree_status,
                    "tree_file_count": len([x for x in tree_files if x.get("type") == "file"]),
                    "tree_paths": [x.get("path") for x in tree_files if x.get("type") == "file"],
                    "tree_sha256": sha256(tree_body),
                    "alternate_local_candidates": alternate_candidates.get(run, []),
                }
            )
            checkpoint_status, checkpoint_body, checkpoint_tree = fetch_tree(CHECKPOINT_REPO, run_path, checkpoint_revision)
            inventory.append(
                {
                    "run": run,
                    "repo": CHECKPOINT_REPO,
                    "kind": "checkpoint",
                    "revision": checkpoint_revision,
                    "tree_url": tree_url(CHECKPOINT_REPO, run_path, checkpoint_revision),
                    "tree_status": checkpoint_status,
                    "tree_file_count": len([x for x in checkpoint_tree if x.get("type") == "file"]),
                    "tree_paths": [x.get("path") for x in checkpoint_tree if x.get("type") == "file"],
                    "tree_sha256": sha256(checkpoint_body),
                }
            )

            fetched: dict[str, Any] = {}
            raw_meta: dict[str, Any] = {}
            for relative in RAW_FILES:
                path = f"{run_path}/{relative}"
                url = raw_url(repo, path, revision)
                status, body, _ = get(url)
                item: dict[str, Any] = {"url": url, "status": status, "bytes": len(body), "sha256": sha256(body)}
                if status != 200:
                    item["error_preview"] = body.decode("utf-8", errors="replace")[:300]
                raw_meta[relative] = item
                raw_destination = OUT / "raw_sources" / run / relative
                raw_destination.parent.mkdir(parents=True, exist_ok=True)
                raw_destination.write_bytes(body)
                item["local_path"] = str(raw_destination.relative_to(ROOT))
                if status == 200:
                    try:
                        if relative.endswith(".json"):
                            fetched[relative] = json.loads(body)
                        elif relative.endswith(".jsonl"):
                            fetched[relative] = [json.loads(line) for line in body.decode("utf-8").splitlines() if line]
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        item["parse_error"] = f"{type(exc).__name__}: {exc}"

            dev_metrics = fetched.get("metrics/dev_metrics.json", {})
            test_metrics = fetched.get("metrics/test_metrics.json", {})
            manifest_obj = fetched.get("run_manifest.json", {})
            review = fetched.get("review_summary.json", {})
            dev_rows = fetched.get("predictions/dev_predictions.jsonl", [])
            test_rows = fetched.get("predictions/test_predictions.jsonl", [])
            changed = review.get("changed_components", {}) if isinstance(review, dict) else {}
            changed = changed.get("changed_components", {}) if isinstance(changed, dict) else {}
            active_heads = changed.get("active_heads", {}) if isinstance(changed, dict) else {}
            variant_heads = active_heads.get("variant") if isinstance(active_heads, dict) else None
            active_tasks = manifest_obj.get("resolved_training_config", {}).get("active_uncertainty_tasks", []) if isinstance(manifest_obj, dict) else []
            dev_ece_recomputed = ece_from_rows(dev_rows)
            test_ece_recomputed = ece_from_rows(test_rows)
            polarity_presence = {
                "dev_gold": sum("polarity" in row.get("gold", {}) for row in dev_rows),
                "dev_logits": sum("polarity" in row.get("logits", {}) for row in dev_rows),
                "dev_predictions": sum("polarity" in row.get("predictions", {}) for row in dev_rows),
                "dev_probabilities": sum("polarity" in row.get("probabilities", {}) for row in dev_rows),
                "test_gold": sum("polarity" in row.get("gold", {}) for row in test_rows),
                "test_logits": sum("polarity" in row.get("logits", {}) for row in test_rows),
                "test_predictions": sum("polarity" in row.get("predictions", {}) for row in test_rows),
                "test_probabilities": sum("polarity" in row.get("probabilities", {}) for row in test_rows),
            }
            required_raw_success = all(
                raw_meta.get(relative, {}).get("status") == 200
                and "parse_error" not in raw_meta.get(relative, {})
                for relative in Q2_REQUIRED_RAW_FILES
            )
            optional_missing_raw_files = [
                relative
                for relative, item in raw_meta.items()
                if relative not in Q2_REQUIRED_RAW_FILES
                and (item.get("status") != 200 or "parse_error" in item)
            ]
            manifest_status_ok = bool(
                manifest_obj.get("status") == "PASS"
                or (
                    manifest_obj.get("status") == "NOT_STARTED"
                    and manifest_obj.get("execution_kind") == "component_bundle"
                    and manifest_obj.get("direct_classification_outputs_used") is True
                    and manifest_obj.get("synthetic_results") is False
                    and review.get("RUN_STATUS") == "PASS"
                )
            )
            source_ok = bool(
                tree_status == 200
                and required_raw_success
                and manifest_status_ok
                and review.get("RUN_STATUS") == "PASS"
                and len(dev_rows) == 1999
                and len(test_rows) == 2000
                and isinstance(test_metrics.get("macro_pragmatic_f1"), (int, float))
                and isinstance(review.get("successful_gpu_hours"), (int, float))
            )
            head_removed = bool(
                source_ok
                and variant == "no_polarity_auxiliary"
                and isinstance(variant_heads, list)
                and "polarity" not in variant_heads
                and "polarity" not in active_tasks
                and polarity_presence == {key: 0 for key in polarity_presence}
                and dev_metrics.get("polarity_dev_ece") is None
                and test_metrics.get("polarity_dev_ece") is None
            )
            saved_dev = dev_metrics.get("polarity_dev_ece")
            saved_test = test_metrics.get("polarity_dev_ece")
            dev_delta = abs(dev_ece_recomputed - saved_dev) if dev_ece_recomputed is not None and isinstance(saved_dev, (int, float)) else None
            test_delta = abs(test_ece_recomputed - saved_test) if test_ece_recomputed is not None and isinstance(saved_test, (int, float)) else None
            dev_split_match = dev_delta is not None and dev_delta <= ECE_TOLERANCE
            test_split_match = test_delta is not None and test_delta <= ECE_TOLERANCE
            if head_removed:
                diagnosis = "not applicable (polarity head removed)"
                split_verdict = "not_applicable_head_removed"
            elif dev_split_match and test_split_match:
                diagnosis = "recomputable from saved polarity probabilities"
                split_verdict = "dev_and_test_saved_fields_match_respective_split"
            else:
                diagnosis = "unresolved; fail closed"
                split_verdict = "mismatch_or_incomplete"
            row_evidence = {
                "run": run,
                "variant": variant,
                "seed": seed,
                "repo": repo,
                "revision": revision,
                "raw_sources": raw_meta,
                "artifact_tree_status": tree_status,
                "artifact_tree_revision": revision,
                "status": manifest_obj.get("status"),
                "manifest_status_ok": manifest_status_ok,
                "manifest_execution_kind": manifest_obj.get("execution_kind"),
                "review_status": review.get("RUN_STATUS"),
                "source_ok": source_ok,
                "required_raw_files": list(Q2_REQUIRED_RAW_FILES),
                "required_raw_success": required_raw_success,
                "optional_missing_raw_files": optional_missing_raw_files,
                "active_uncertainty_tasks": active_tasks,
                "variant_active_heads_from_review": variant_heads,
                "polarity_presence_counts": polarity_presence,
                "dev_prediction_count": len(dev_rows),
                "test_prediction_count": len(test_rows),
                "dev_macro_pragmatic_f1": dev_metrics.get("macro_pragmatic_f1"),
                "test_macro_pragmatic_f1": test_metrics.get("macro_pragmatic_f1"),
                "saved_dev_metrics_polarity_dev_ece": saved_dev,
                "saved_test_metrics_polarity_dev_ece": saved_test,
                "recomputed_dev_polarity_ece": dev_ece_recomputed,
                "recomputed_test_polarity_ece": test_ece_recomputed,
                "dev_ece_abs_delta": dev_delta,
                "test_ece_abs_delta": test_delta,
                "dev_split_match": dev_split_match,
                "test_split_match": test_split_match,
                "split_match_verdict": split_verdict,
                "successful_gpu_hours": review.get("successful_gpu_hours"),
                "head_diagnosis": diagnosis,
                "head_removed_confirmed": head_removed,
            }
            evidence.append(row_evidence)
            state["completed_runs"].append(run)
            persist_checkpoint(state, inventory, evidence)

    if smoke:
        smoke_proof = {
            "selected_runs": [row["run"] for row in evidence],
            "all_source_ok": all(row["source_ok"] for row in evidence),
            "no_polarity_head_removed_confirmed": next(
                row["head_removed_confirmed"]
                for row in evidence
                if row["variant"] == "no_polarity_auxiliary"
            ),
            "full_dev_split_match": next(
                row["dev_split_match"] for row in evidence if row["variant"] == "full"
            ),
            "full_test_split_match": next(
                row["test_split_match"] for row in evidence if row["variant"] == "full"
            ),
        }
        assert smoke_proof["all_source_ok"], smoke_proof
        assert smoke_proof["no_polarity_head_removed_confirmed"], smoke_proof
        assert smoke_proof["full_dev_split_match"] and smoke_proof["full_test_split_match"], smoke_proof
        state.update({"status": "smoke_complete", "smoke_proof": smoke_proof})
        persist_checkpoint(state, inventory, evidence)
        (OUT / "q2_smoke_report.json").write_text(
            json.dumps({"proof": smoke_proof, "evidence": evidence}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output": str(OUT), "runs": len(evidence), "smoke_proof": smoke_proof}, indent=2, ensure_ascii=False))
        return 0

    evidence_by_variant = {variant: [x for x in evidence if x["variant"] == variant] for variant in VARIANT_ORDER}
    for variant in VARIANT_ORDER:
        rows = evidence_by_variant[variant]
        if len(rows) != 3 or not all(row["source_ok"] for row in rows):
            state.update({"status": "blocked", "error": f"Incomplete or failed source evidence for {variant}"})
            persist_checkpoint(state, inventory, evidence)
            raise RuntimeError(state["error"])
    no_polarity_rows = evidence_by_variant["no_polarity_auxiliary"]
    if not all(row["head_removed_confirmed"] for row in no_polarity_rows):
        state.update({"status": "blocked", "error": "No-polarity head removal was not confirmed for all three seeds"})
        persist_checkpoint(state, inventory, evidence)
        raise RuntimeError(state["error"])

    proof = {
        "all_18_source_ok": len(evidence) == 18 and all(row["source_ok"] for row in evidence),
        "all_3_no_polarity_head_removed_confirmed": len(no_polarity_rows) == 3 and all(
            row["head_removed_confirmed"] for row in no_polarity_rows
        ),
        "all_15_head_present_dev_split_match": all(
            row["dev_split_match"]
            for row in evidence
            if row["variant"] != "no_polarity_auxiliary"
        ),
        "all_15_head_present_test_split_match": all(
            row["test_split_match"]
            for row in evidence
            if row["variant"] != "no_polarity_auxiliary"
        ),
        "ece_tolerance": ECE_TOLERANCE,
    }
    if not all(proof[key] for key in (
        "all_18_source_ok",
        "all_3_no_polarity_head_removed_confirmed",
        "all_15_head_present_dev_split_match",
        "all_15_head_present_test_split_match",
    )):
        state.update({"status": "blocked", "error": f"Runtime proof failed: {proof}"})
        persist_checkpoint(state, inventory, evidence)
        raise RuntimeError(state["error"])

    full_gpu = statistics.mean([x["successful_gpu_hours"] for x in evidence_by_variant["full"]])
    table: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        rows = evidence_by_variant[variant]
        f1_mean, f1_sd = mean_sd([float(x["test_macro_pragmatic_f1"]) * 100 for x in rows])
        dev_eces = [float(x["saved_dev_metrics_polarity_dev_ece"]) * 1000 for x in rows if x["saved_dev_metrics_polarity_dev_ece"] is not None]
        if dev_eces:
            if len(dev_eces) != 3 or not all(row["dev_split_match"] for row in rows):
                state.update({"status": "blocked", "error": f"Dev ECE recomputation did not match all seeds for {variant}"})
                persist_checkpoint(state, inventory, evidence)
                raise RuntimeError(state["error"])
            ece_mean, ece_sd = mean_sd(dev_eces)
            ece_display = f"{ece_mean:.1f} +/- {ece_sd:.1f}"
        else:
            if variant != "no_polarity_auxiliary" or len(rows) != 3 or not all(row["head_removed_confirmed"] for row in rows):
                state.update({"status": "blocked", "error": f"ECE unavailable without confirmed structural inapplicability for {variant}"})
                persist_checkpoint(state, inventory, evidence)
                raise RuntimeError(state["error"])
            ece_mean = ece_sd = None
            ece_display = "not applicable (polarity head removed)"
        table.append(
            {
                "variant": variant,
                "macro_pragmatic_f1_test_pct": f"{f1_mean:.1f} +/- {f1_sd:.1f}",
                "polarity_dev_ece_x1000": ece_display,
                "relative_cost_to_full_q2_xlm_r": f"{statistics.mean([x['successful_gpu_hours'] for x in rows]) / full_gpu:.2f}",
                "seed_count": len(rows),
                "ece_mean_x1000": ece_mean,
                "ece_sd_x1000": ece_sd,
            }
        )

    legacy_test_table = {}
    for variant, rows in evidence_by_variant.items():
        values = [float(x["saved_test_metrics_polarity_dev_ece"]) * 1000 for x in rows if x["saved_test_metrics_polarity_dev_ece"] is not None]
        legacy_test_table[variant] = f"{statistics.mean(values):.1f} +/- {statistics.stdev(values):.1f}" if len(values) == 3 else "not applicable (polarity head removed)"
    paper_results_path = ROOT / "paper" / "sections" / "results.tex"
    paper_results = paper_results_path.read_text(encoding="utf-8")
    paper_q2_label_present = "Saved polarity ECE" in paper_results and "No polarity auxiliary" in paper_results
    if not paper_q2_label_present:
        state.update({"status": "blocked", "error": "Existing paper Q2 table anchor not found"})
        persist_checkpoint(state, inventory, evidence)
        raise RuntimeError(state["error"])

    (OUT / "q2_canonical_table.json").write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "q2_canonical_table.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["variant", "macro_pragmatic_f1_test_pct", "polarity_dev_ece_x1000", "relative_cost_to_full_q2_xlm_r", "seed_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in table)

    labels = {
        "full": "Full follow-up",
        "no_emotion_auxiliary": "No emotion auxiliary",
        "no_polarity_auxiliary": "No polarity auxiliary",
        "no_rationale": "No explanation auxiliary",
        "no_multitask": "No multitask bundle",
        "no_uncertainty_weighting": "No task-uncertainty weighting",
    }
    report = [
        "## Material Passport",
        "",
        "- ID: q2-no-polarity-evidence-2026-09-20",
        "- Type: validation report / reproducibility audit",
        "- Verification Status: ANALYZED (live artifact fetch and deterministic recomputation; no fresh training rerun)",
        "- Scope: Q2 XLM-R follow-up, six variants, three seeds per variant",
        "",
        "## Finding",
        "",
        f"The no-polarity diagnosis is asserted from the completed evidence rows, not hardcoded: proof={json.dumps(proof, sort_keys=True)}. This is structural inapplicability, not an unresolved missing-data case: all three source-complete rows expose no polarity head/task and have zero polarity fields in both saved splits, so no polarity probability source exists to recompute. The canonical cell is therefore `not applicable (polarity head removed)`; no pragmatic ECE, zero, or surrogate value is inserted.",
        "",
        "The canonical table follows the verified Q2 protocol: F1 is from `metrics/test_metrics.json`; ECE is the ten-bin top-label ECE on `metrics/dev_metrics.json` / dev predictions. Relative cost uses successful GPU hours normalized to the mean full Q2 run.",
        "",
        "| Variant | Macro-pragmatic F1, test (%) | Polarity dev ECE x 10^3 | Relative cost to full Q2 XLM-R | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table:
        report.append(f"| {labels[row['variant']]} | {row['macro_pragmatic_f1_test_pct']} | {row['polarity_dev_ece_x1000']} | {row['relative_cost_to_full_q2_xlm_r']} | {row['seed_count']} |")
    report += ["", "## Per-seed split-match evidence", "", "The tolerance is absolute ECE <= %.1e." % ECE_TOLERANCE, "", "| Run | Saved dev field | Recomputed dev | Abs delta | Dev verdict | Saved test field | Recomputed test | Abs delta | Test verdict |", "|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for row in evidence:
        report.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (row["run"], row["saved_dev_metrics_polarity_dev_ece"], row["recomputed_dev_polarity_ece"], row["dev_ece_abs_delta"], row["split_match_verdict"] if row["head_removed_confirmed"] else row["dev_split_match"], row["saved_test_metrics_polarity_dev_ece"], row["recomputed_test_polarity_ece"], row["test_ece_abs_delta"], row["split_match_verdict"] if row["head_removed_confirmed"] else row["test_split_match"]))
    report += [
        "",
        "## Split diagnostic for production",
        "",
        f"The Q2 protocol file `{protocol_path.relative_to(ROOT)}` lines 8-11 explicitly declares `ece.split: vipragsent_dev`, head `intended_polarity_3way`, and ten-bin top-label ECE. Runtime proof={json.dumps(proof, sort_keys=True)}. Although both split files use the legacy key `polarity_dev_ece`, the per-seed recomputation shows the dev-file value matches dev predictions and the test-file value matches test predictions; the key name does not change the split source. The old paper table used the test-file values; those legacy aggregate values are recorded here: `{json.dumps(legacy_test_table, sort_keys=True)}`. Because the production caption says dev ECE and the protocol says dev, the canonical table above uses the dev aggregates instead. This is a split-label/value mismatch, not an imputation.",
        "",
        "## Provenance",
        "",
        "- Live HF tree API inventories are pinned to each repository commit SHA and paginated through the API `Link: rel=next` cursor; raw files use the same pinned revision.",
        "- Machine-readable per-seed URLs, statuses, sizes, SHA-256 values, head evidence, probabilities, and split deltas: `source_inventory.json` and `run_evidence.jsonl`.",
        "- Persisted raw payload mirror for audit/replay: `raw_sources/`; the earlier 12-row pre-normalization and four-row pre-component-bundle checkpoints remain in the `*_before_*` backup files.",
        "- The source gate requires the Q2 evidence core (run manifest, review summary, dev/test metrics, and dev/test predictions); optional config/selection omissions are retained per run in `optional_missing_raw_files` and are not misreported as missing Q2 metrics.",
        "- No-multitask manifests are accepted only under the recorded component-bundle protocol: `status=NOT_STARTED` is paired with `execution_kind=component_bundle`, `direct_classification_outputs_used=true`, `synthetic_results=false`, and `review_summary.RUN_STATUS=PASS`; this is recorded as `manifest_status_ok=true`, not relabeled as a generic PASS.",
        "- Complete no-polarity artifacts: overflow-011 (20260521), overflow-010 (20260522), overflow-009 (20260523).",
        "- Checkpoint-tree candidates are inventoried separately under `Thundergod2007/vipragsent-xlmr-checkpoints`; checkpoint payloads were not downloaded.",
        "- Local alternate-manifest candidates are retained. The partial no-rationale 20260523 overflow-006 tree is not promoted over complete overflow-018 artifacts.",
        "- Structural-only reference PDF inspection: `D:/vipragsent-pr/tmp/pdf_reference/main.pdf`; no numerical claims were extracted from it.",
    ]
    (OUT / "q2_evidence_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    state.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "table": table})
    persist_checkpoint(state, inventory, evidence)
    print(json.dumps({"output": str(OUT), "runs": len(evidence), "table": table}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
