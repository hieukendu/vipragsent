from __future__ import annotations

import argparse

from vipragsent.phase import write_phase_handoff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase")
    parser.add_argument("status", choices=["PASS", "BLOCKED", "FAIL"])
    parser.add_argument("--tests-passed", action="store_true")
    parser.add_argument("--next-phase-ready", action="store_true")
    parser.add_argument("--blocker", action="append", default=[])
    parser.add_argument("--model-family")
    parser.add_argument("--approval-basis")
    args = parser.parse_args()
    write_phase_handoff(
        args.phase,
        args.status,
        tests_passed=args.tests_passed,
        blockers=args.blocker,
        next_phase_ready=args.next_phase_ready,
        model_family=args.model_family,
        approval_basis=args.approval_basis,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
