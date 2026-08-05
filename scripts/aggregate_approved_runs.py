from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.orchestration.sequential import load_inventory
from vipragsent.protocol import validate_protocol_resolution


def _scope_rows(root: Path, research_question: str) -> list[dict[str, Any]]:
    rows = load_inventory(root)
    if research_question == "all":
        return rows
    wanted = research_question.casefold()
    return [row for row in rows if str(row.get("research_question", "")).casefold() == wanted]


def validate_approved_scope(root: str | Path, research_question: str) -> dict[str, Any]:
    root = Path(root)
    rows = _scope_rows(root, research_question)
    blockers: list[str] = []
    if not rows:
        blockers.append(f"No inventory entries found for research question {research_question}")
    protocol = validate_protocol_resolution(root)
    if protocol["scientific_protocol_conflicts"]:
        blockers.extend(protocol["scientific_protocol_conflicts"])
    accepted: list[dict[str, Any]] = []
    for row in rows:
        experiment_id = str(row.get("experiment_id", row.get("run_id")))
        run_root = root / "results/runs" / experiment_id
        review_path = run_root / "review_summary.json"
        approval_path = run_root / "approval_status.json"
        metrics_path = run_root / "metrics.json"
        if not review_path.exists() or not approval_path.exists():
            blockers.append(f"{experiment_id}: review summary or approval status is missing")
            continue
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"{experiment_id}: invalid review/approval JSON ({exc})")
            continue
        status = review.get("RUN_STATUS", review.get("run_status"))
        if status != "PASS":
            blockers.append(f"{experiment_id}: RUN_STATUS={status!r}, expected PASS")
        if approval.get("run_id") != experiment_id or approval.get("status") != "APPROVED":
            blockers.append(f"{experiment_id}: approval_status is not APPROVED")
        if not approval.get("approved_by") or not approval.get("approved_at"):
            blockers.append(f"{experiment_id}: approved_by and approved_at are required")
        if not metrics_path.exists():
            blockers.append(f"{experiment_id}: metrics.json is missing")
        else:
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                blockers.append(f"{experiment_id}: metrics.json is invalid ({exc})")
                continue
            if not isinstance(metrics, dict):
                blockers.append(f"{experiment_id}: metrics.json must be an object")
            prediction_files = metrics.get("prediction_files", metrics.get("prediction_file")) if isinstance(metrics, dict) else None
            if not prediction_files:
                blockers.append(f"{experiment_id}: prediction file reference is missing")
            else:
                values = [prediction_files] if isinstance(prediction_files, str) else list(prediction_files)
                if not any((run_root / value).exists() for value in values):
                    blockers.append(f"{experiment_id}: referenced prediction file is missing")
        if status == "PASS" and approval.get("status") == "APPROVED" and metrics_path.exists():
            accepted.append({"experiment_id": experiment_id, "review_summary": str(review_path.relative_to(root).as_posix()), "approval_status": str(approval_path.relative_to(root).as_posix())})

    if research_question in {"Q3", "all"} and not blockers:
        q3_rows = [row for row in rows if row.get("research_question") == "Q3"]
        budgets_by_system: dict[str, set[str]] = defaultdict(set)
        for row in q3_rows:
            budgets_by_system[str(row["system_id"])].add(str(row.get("budget")))
        required_budgets = {"32", "64", "128", "256", "512", "full"}
        for system, budgets in budgets_by_system.items():
            if budgets != required_budgets:
                blockers.append(f"Q3 {system}: incomplete budget set {sorted(budgets)}")
    return {
        "research_question": research_question,
        "required_run_count": len(rows),
        "accepted_run_count": len(accepted),
        "accepted_runs": accepted,
        "blockers": blockers,
        "status": "PASS" if not blockers else "BLOCKED",
        "aggregation_performed": False if blockers else True,
        "policy": "approval_gated; no training or Azure calls",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate only explicitly approved sequential runs")
    parser.add_argument("--research-question", choices=("Q1a", "Q1b", "Q2", "Q3", "Q4", "all"), required=True)
    args = parser.parse_args()
    report = validate_approved_scope(ROOT, args.research_question)
    output = ROOT / "reports" / f"approved_aggregation_{args.research_question.casefold()}.json"
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 3 if any(item.startswith("SCIENTIFIC_PROTOCOL_CONFLICT") for item in report["blockers"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
