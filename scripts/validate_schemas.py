from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.artifacts.schemas import validate_artifact_tree
from vipragsent.config_validation import validate_config_tree


def main() -> int:
    errors = []
    schema_paths = list((ROOT / "configs/schemas").glob("*.schema.json")) + list((ROOT / "schemas").glob("*.schema.json"))
    for path in schema_paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("type") != "object" or "required" not in value:
                errors.append(f"invalid schema shape: {path}")
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON schema {path}: {exc}")
    config_report = validate_config_tree(ROOT)
    errors.extend(config_report["errors"])
    artifact_root = ROOT / "experiment_artifacts"
    if artifact_root.exists() and any(path.is_file() for path in artifact_root.rglob("*")) and validate_artifact_tree(artifact_root):
        errors.append("artifact tree does not satisfy locked columns")
    print("schema validation passed" if not errors else "\n".join(errors))
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
