from __future__ import annotations

import json

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.sequential import build_azure_job_inventory, load_execution_policy
from vipragsent.protocol import validate_protocol_resolution


def main() -> int:
    inventory = build_expected_runs(ROOT)
    policy = load_execution_policy(ROOT)
    protocol = validate_protocol_resolution(ROOT)
    prompt_manifest = json.loads((ROOT / "reports/sequential_prompt_manifest.json").read_text(encoding="utf-8"))
    prompt_validation = json.loads((ROOT / "reports/sequential_prompt_validation.json").read_text(encoding="utf-8"))
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    setup = {
        "schema_version": 1,
        "status": "PASS" if prompt_validation.get("status") == "PASS" and not protocol["scientific_protocol_conflicts"] else "FAIL",
        "execution_policy": policy,
        "experiment_count": inventory["derived_run_count"],
        "experiment_counts_by_question": inventory["counts_by_question"],
        "azure_job_count": len(build_azure_job_inventory()),
        "phase15_model_count": prompt_manifest.get("phase15_model_count"),
        "aggregation_prompt_count": prompt_manifest.get("aggregation_count"),
        "generated_prompt_count": prompt_manifest.get("prompt_count"),
        "prompt_manifest": "reports/generated_sequential_prompts_manifest.json",
        "prompt_validation": prompt_validation,
        "protocol_resolution": protocol,
        "setup_implementation_ready": state.get("setup_implementation_ready") is True,
        "setup_frozen": state.get("setup_frozen") is True,
        "runtime_environment_ready": state.get("runtime_environment_ready") is True,
        "weights_downloaded": state.get("weights_downloaded") is True,
        "full_run_started": state.get("full_run_started") is True,
        "phase15_executed": False,
        "azure_called": False,
        "real_experiment_executed": False,
        "next_action": "Run the first Phase 15 model-family prompt only after explicit user approval.",
    }
    atomic_write_json(ROOT / "reports/sequential_execution_setup.json", setup)
    atomic_write_text(ROOT / "reports/sequential_execution_setup.md", "\n".join([
        "# Sequential experiment setup",
        "",
        f"- Setup status: `{setup['status']}`",
        f"- Execution policy: `{policy['execution_policy']}`",
        f"- Experiment prompts: `{setup['experiment_count']}`",
        f"- Azure prompts: `{setup['azure_job_count']}`",
        f"- Phase 15 model-family prompts: `{setup['phase15_model_count']}`",
        f"- Aggregation prompts: `{setup['aggregation_prompt_count']}` plus the final aggregation prompt",
        f"- Generated prompt files: `{setup['generated_prompt_count']}`",
        "- Global full DAG: `DISABLED`",
        "- Approval after every run: `REQUIRED`",
        "- Automatic next run: `DISABLED`",
        "- Phase 15 executed: `false`",
        "- Azure called: `false`",
        "- Real experiments executed: `false`",
        "",
        "## Resolution status",
        *[f"- `{key}`: `{value}`" for key, value in protocol["resolution_status"].items()],
        "",
        "## Next action",
        "Use `prompts/sequential/phase15/phobert_base.md` only after explicit user approval. It stops after the model-family report and does not advance automatically.",
    ]) + "\n")
    atomic_write_text(ROOT / "reports/q1a_no_auxiliary_resolution.md", """# Q1a no-auxiliary resolution

Status: `RESOLVED`

The no-auxiliary system is the distinct `vipragsent_no_auxiliary_vistral` system. It is not a reuse of the `vistral_pragmatic_sft` baseline checkpoint identity.

It exposes exactly the six pragmatic binary heads, excludes polarity, emotion, and rationale heads from the model forward path and optimizer, and uses six independent homoscedastic uncertainty parameters initialized at zero with zero weight decay. The baseline remains equal-weight and has no uncertainty parameters. Selection and threshold protocols are unchanged.
""")
    atomic_write_text(ROOT / "reports/q4_pragmatic_calibration_protocol.md", """# Q4 pragmatic calibration protocol

Status: `RESOLVED`

Q4 is pragmatic calibration and learning dynamics for exactly `phobert_pragmatic_finetune`, `vistral_pragmatic_sft`, and `vipragsent_full_vistral`. Each system exposes the same six pragmatic positive-class sigmoid probabilities.

Calibration uses the frozen ViPragSent test split, ten equal-width bins, no temperature scaling, no thresholding, and no probability pooling across seeds. ECE is computed independently per seed and summarized by arithmetic mean and sample standard deviation (`ddof=1`). Learning curves use only frozen ViPragSent dev histories and dev macro pragmatic F1 by epoch.

Required tables, reliability backing data, learning curves, and PDF/PNG figures are listed in `configs/experiments/q4/pragmatic_calibration.yaml`.
""")
    atomic_write_text(ROOT / "reports/significance_method_resolution.md", """# Significance method resolution

Status: `RESOLVED`

The locked method is `paired_hierarchical_bootstrap_sign_plus_one_v1`, with left-minus-right differences, 1,000 paired hierarchical resamples, bootstrap seed `20260525`, percentile 95% confidence intervals, and Holm correction within each seven-metric family.

The two-sided finite-resample p-value is `min(1, 2 * min(p_lower, p_upper))`, where `p_lower = (1 + count(delta <= 0)) / (B + 1)` and `p_upper = (1 + count(delta >= 0)) / (B + 1)`. Trainable systems share sampled seed and example indices; Azure remains one fixed-prompt prediction vector and never receives fabricated training seeds.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
