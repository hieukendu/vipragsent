from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.artifacts.schemas import validate_artifact_tree
from vipragsent.config_validation import validate_config_tree


def main() -> int:
    errors = []
    for path in (ROOT / "configs/schemas").glob("*.schema.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("type") != "object" or "required" not in value:
                errors.append(f"invalid schema shape: {path}")
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON schema {path}: {exc}")
    config_report = validate_config_tree(ROOT)
    errors.extend(config_report["errors"])
    if validate_artifact_tree(ROOT / "experiment_artifacts"):
        errors.append("artifact tree does not satisfy locked columns")
    print("schema validation passed" if not errors else "\n".join(errors))
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
