from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.orchestration.sequential import load_execution_policy
from vipragsent.protocol import compare_frozen_hashes, validate_protocol_resolution


TOP_LEVEL_ALLOWED = {
    "PROJECT_STATE.json", "SETUP_READY.md", "SETUP_FREEZE_MANIFEST.json", "SETUP_CHECKSUMS.sha256",
    "FINAL_CHECKSUMS.sha256", "README.md", "KNOWN_LIMITATIONS.md", "REPRODUCIBILITY_REPORT.md",
    "RELEASE_MANIFEST.json", "DATASET_CARD.md", "EXPERIMENT_MODEL_REGISTRY.md", "pyproject.toml",
}
ALLOWED_PREFIXES = ("configs/", "src/", "scripts/", "tests/", "schemas/", "prompts/", "reports/", "docs/adr/")


def _changed_paths() -> list[str]:
    result = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, capture_output=True, text=True, check=True)
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        value = line[3:] if len(line) >= 4 else line
        if " -> " in value:
            value = value.rsplit(" -> ", 1)[-1]
        paths.append(value.replace("\\", "/"))
    return sorted(paths)


def audit(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    protocol = validate_protocol_resolution(root)
    policy = load_execution_policy(root)
    q1a = yaml.safe_load((root / "configs/experiments/q1a/system_roles.yaml").read_text(encoding="utf-8"))["q1a"]["roles"]
    q4 = yaml.safe_load((root / "configs/experiments/q4/protocol.yaml").read_text(encoding="utf-8"))["q4"]
    q4_calibration = yaml.safe_load((root / "configs/experiments/q4/pragmatic_calibration.yaml").read_text(encoding="utf-8"))["q4_pragmatic_calibration"]
    significance = yaml.safe_load((root / "configs/statistics/significance_method.yaml").read_text(encoding="utf-8"))
    paper_roles = yaml.safe_load((root / "configs/paper_roles.yaml").read_text(encoding="utf-8"))["paper_roles"]
    state = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    changed = _changed_paths()
    outside_scope = [path for path in changed if path not in TOP_LEVEL_ALLOWED and not path.startswith(ALLOWED_PREFIXES)]
    data_changes = [path for path in changed if path.startswith("data/")]
    q1a_resolved = (
        q1a["vistral_baseline"]["system_id"] == "vistral_pragmatic_sft"
        and q1a["no_auxiliary"]["system_id"] == "vipragsent_no_auxiliary_vistral"
        and q1a["vistral_baseline"]["system_id"] != q1a["no_auxiliary"]["system_id"]
        and q1a["no_auxiliary"]["active_heads"] == ["implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking"]
        and q1a["no_auxiliary"]["loss_aggregation"] == "homoscedastic_uncertainty"
    )
    q4_systems = [item["system_id"] for item in q4["systems"]]
    q4_resolved = q4_systems == ["phobert_pragmatic_finetune", "vistral_pragmatic_sft", "vipragsent_full_vistral"] and q4["probability_definition"] == "raw_positive_class_probability_sigmoid" and q4["bins"] == 10 and q4["temperature_scaling"] is False
    q4_config_resolved = q4_calibration["systems"] == q4_systems and q4_calibration["calibration"]["probability_pooling_across_seeds"] is False
    significance_resolved = all(significance.get(key) == value for key, value in {
        "resolution_status": "RESOLVED",
        "method_id": "paired_hierarchical_bootstrap_sign_plus_one_v1",
        "difference_direction": "left_minus_right",
        "resamples": 1000,
        "bootstrap_seed": 20260525,
        "confidence_interval": "percentile_95",
        "finite_resample_correction": "plus_one",
        "multiple_comparisons": "holm_within_7_metric_family",
    }.items())
    preserved = {
        "Q1B": paper_roles["table_3_retention"]["backbone"] == "phobert_base" and protocol["resolution_status"].get("Q1B") == "RESOLVED",
        "Q3": protocol["resolution_status"].get("Q3") == "RESOLVED",
    }
    approved = {
        "Q1A_no_auxiliary": q1a_resolved,
        "Q4_pragmatic_calibration": q4_resolved and q4_config_resolved,
        "significance_plus_one": significance_resolved,
        "sequential_review_gate": policy == {
            "execution_policy": "sequential_review_gated",
            "global_full_dag_enabled": False,
            "maximum_concurrent_gpu_jobs": 1,
            "automatic_next_run": False,
            "require_user_approval_after_each_run": True,
        },
    }
    setup_state = {
        "current_phase": state.get("current_phase"),
        "setup_implementation_ready": state.get("setup_implementation_ready"),
        "setup_frozen": state.get("setup_frozen"),
        "weights_downloaded": state.get("weights_downloaded"),
        "full_run_started": state.get("full_run_started"),
    }
    setup_state_checks = {
        "current_phase_is_15": setup_state["current_phase"] == "15",
        "setup_implementation_ready": setup_state["setup_implementation_ready"] is True,
        "setup_frozen": setup_state["setup_frozen"] is True,
        "weights_downloaded_is_false": setup_state["weights_downloaded"] is False,
        "full_run_started_is_false": setup_state["full_run_started"] is False,
    }
    frozen = compare_frozen_hashes(root)
    report = {
        "schema_version": 1,
        "status": "PASS" if not outside_scope and not data_changes and not protocol["scientific_protocol_conflicts"] and frozen["unchanged"] and all(approved.values()) and all(preserved.values()) and all(setup_state_checks.values()) else "FAIL",
        "changed_files": changed,
        "outside_approved_change_scope": outside_scope,
        "data_files_changed": data_changes,
        "approved_protocol_changes": approved,
        "preserved_protocol_contracts": preserved,
        "protocol_resolution_status": protocol["resolution_status"],
        "scientific_protocol_conflicts": protocol["scientific_protocol_conflicts"],
        "frozen_data_hashes": frozen,
        "setup_state": setup_state,
        "setup_state_checks": setup_state_checks,
        "prohibited_execution_state": {"phase15_executed": state.get("weights_downloaded") is True, "full_dag_started": state.get("full_run_started") is True},
        "proof_scope": "No frozen data files changed; only approved protocol/configuration, implementation, orchestration, schema, report, and generated-prompt paths are changed.",
    }
    atomic_write_json(root / "reports/protocol_change_audit.json", report)
    return report


def main() -> int:
    report = audit(ROOT)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
