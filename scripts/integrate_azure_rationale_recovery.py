from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT
from vipragsent.orchestration.azure_rationale_recovery import (
    integrate_azure_rationale_recovery,
    validate_supplemental_recovery,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrate an already supplied supplemental Azure rationale recovery batch")
    parser.add_argument("--check-only", action="store_true", help="Validate the frozen IDs, hashes, schema, and provenance without writing or approving anything")
    parser.add_argument("--reviewer", default="standing_user_authorization_after_successful_audit")
    args = parser.parse_args()
    try:
        if args.check_only:
            validation = validate_supplemental_recovery(ROOT)
            report = {
                "status": "PASS",
                "frozen_rows": len(validation["frozen_rows"]),
                "original_successful_rows": len(validation["original"]),
                "original_failure_rows": len(validation["failures"]),
                "supplemental_rows": len(validation["submitted"]),
                "ids_exact_match": set(validation["failure_ids"]) == set(validation["submitted_ids"]),
                "source_hashes_exact_match": True,
                "paid_api_request_made": False,
            }
        else:
            report = integrate_azure_rationale_recovery(ROOT, reviewer=args.reviewer)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
