from __future__ import annotations

import json

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.stage_plans import validate_stage_plan_registry
from vipragsent.orchestration.system_registry import validate_execution_registry


def main() -> int:
    registry = validate_execution_registry(ROOT)
    stage_plans = validate_stage_plan_registry(ROOT)
    report = {"status": "PASS" if registry["status"] == "PASS" and stage_plans["status"] == "PASS" else "FAIL", "execution_registry": registry, "stage_plan_registry": stage_plans, "inventory_hash": build_expected_runs(ROOT)["inventory_hash"]}
    atomic_write_json(ROOT / "reports/system_execution_registry_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
