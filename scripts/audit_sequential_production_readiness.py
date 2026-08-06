from __future__ import annotations

import json

from audit_final_production_correctness import audit


def main() -> int:
    report = audit()
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
