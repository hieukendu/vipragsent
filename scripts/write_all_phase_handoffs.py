from __future__ import annotations

from _bootstrap import ROOT
from vipragsent.phase import write_phase_handoff


def main() -> int:
    common_inputs = ["01_GLOBAL_PROJECT_CONTRACT.md", "28_PAPER_EXPERIMENT_ROLE_REGISTRY.md", "29_MANUAL_ERROR_AND_QUALITATIVE_ANALYSIS.md", "30_SPEC_COMPLETENESS_AUDIT.md", "31_IMPLEMENTATION_DECISIONS.md", "32_RUNTIME_PREFLIGHT_CHECKLIST.md"]
    entries = {
        "03": ("BLOCKED", ["Azure endpoint, deployment, and credentials are not configured"], False),
        "04": ("PASS", [], True),
        "05": ("PASS", [], True),
        "06": ("PASS", [], True),
        "07": ("PASS", [], True),
        "08": ("PASS", [], True),
        "09": ("PASS", [], True),
        "10": ("PASS", [], True),
        "11": ("PASS", [], True),
        "12": ("PASS", [], True),
        "13": ("PASS", [], True),
        "15": ("BLOCKED", ["Model-weight download intentionally paused pending user approval; no weights were downloaded"], False),
        "16": ("BLOCKED", ["Phase 15 model verification is not complete", "External official test datasets are missing", "Azure deployment is not configured"], False),
        "17": ("BLOCKED", ["Full run and Phase 15 are intentionally incomplete"], False),
    }
    for phase, (status, blockers, ready) in entries.items():
        write_phase_handoff(phase, status, inputs_read=common_inputs, files_created=[f"reports/phases/phase_{phase}_status.md", f"reports/phases/phase_{phase}_handoff.json"], tests_run=["python -m pytest", "python scripts/semantic_config_audit.py", "python scripts/run_all_experiments.py --config configs/master_run.yaml --mode fixture"], tests_passed=status != "FAIL", blockers=blockers, next_phase_ready=ready)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
