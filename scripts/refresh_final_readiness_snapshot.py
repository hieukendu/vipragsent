from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from _bootstrap import ROOT
    from readiness_utils import build_snapshot, read_json, validate_ci_evidence, write_snapshot
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT
    from scripts.readiness_utils import (
        build_snapshot,
        read_json,
        validate_ci_evidence,
        write_snapshot,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the canonical final readiness snapshot")
    parser.add_argument("--ci-evidence", required=True)
    args = parser.parse_args()
    evidence_path = Path(args.ci_evidence)
    if not evidence_path.is_absolute():
        evidence_path = ROOT / evidence_path
    evidence = read_json(evidence_path, {})
    errors = validate_ci_evidence(evidence)
    if errors:
        raise SystemExit("Invalid CI evidence: " + "; ".join(errors))
    snapshot = build_snapshot(ROOT, evidence)
    if snapshot["ci"]["validation_errors"]:
        raise SystemExit("CI evidence does not bind to current HEAD: " + "; ".join(snapshot["ci"]["validation_errors"]))
    write_snapshot(ROOT, snapshot)
    print(snapshot["audited_code_commit"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
