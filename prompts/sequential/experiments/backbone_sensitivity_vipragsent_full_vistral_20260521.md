# ViPragSent sequential experiment run: backbone_sensitivity_vipragsent_full_vistral_20260521

This runbook names exactly one inventory entry. Do not start another experiment or Azure job.

## Locked entry

- Experiment ID: `backbone_sensitivity_vipragsent_full_vistral_20260521`
- Research question: `backbone_sensitivity`
- System ID: `vipragsent_full_vistral`
- Display name: Full ViPragSent Vistral
- Variant: `full`
- Backbone: `vistral_7b`
- Seed: `20260521`
- Budget: ``
- Task: `pragmatic;ordinary;polarity_ece;profiling`
- Split: `test`
- Dependencies: `reused_predictions;reused_profiles`
- Required Phase 15 assets: `model_weights;tokenizer;runtime_profile`
- Expected artifacts: `backbone_sensitivity`
- Selection metric: `macro_prag_f1_test`
- Evaluation protocol: `backbone_sensitivity_v1`
- Reusable checkpoint key: `vipragsent_full_vistral:20260521`
- Protocol resolution: `RESOLVED`

## Required command sequence

1. Run `python scripts/run_single_experiment.py --experiment-id backbone_sensitivity_vipragsent_full_vistral_20260521 --stage preflight`.
2. If preflight is BLOCKED or FAIL, paste its report into the Codex chat and stop.
3. After preflight passes, run `python scripts/run_single_experiment.py --experiment-id backbone_sensitivity_vipragsent_full_vistral_20260521 --stage all`.
4. On interruption, resume only this entry with `python scripts/run_single_experiment.py --experiment-id backbone_sensitivity_vipragsent_full_vistral_20260521 --resume`.
5. Do not call the global DAG, change the locked protocol, tune on the test split, or substitute a model, dataset, seed, or checkpoint.

## Required review handoff

The run must complete these stages in order: preflight, train_or_run, evaluate_dev, freeze_selection, evaluate_test, export_artifacts, validate_artifacts, generate_review_summary.

Print the complete review summary with `python scripts/print_run_review_summary.py --run-id backbone_sensitivity_vipragsent_full_vistral_20260521` and paste it into the Codex chat. It must include `RUN_STATUS`, `USER_REVIEW_STATUS`, `NEXT_RUN_ALLOWED`, artifact hashes, and blockers.

The initial approval file must remain `PENDING_USER_APPROVAL`; never fabricate approval. After the summary is pasted, stop and wait for explicit user approval. Do not begin the next run automatically.

RUN_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO
