# ViPragSent sequential experiment run: q2_no_multitask_bundle_20260521

This runbook names exactly one inventory entry. Do not start another experiment or Azure job.

## Locked entry

- Experiment ID: `q2_no_multitask_bundle_20260521`
- Research question: `Q2`
- System ID: `no_multitask`
- Display name: no_multitask
- Variant: `no_multitask`
- Backbone: `phobert_base`
- Seed: `20260521`
- Budget: ``
- Task: `pragmatic;polarity;emotion`
- Split: `dev;external_test`
- Dependencies: `phobert_jobs`
- Required Phase 15 assets: `model_weights;tokenizer;runtime_profile`
- Execution kind: `component_bundle`
- Expected artifacts: `bundle_metrics;predictions`
- Selection metric: `macro_prag_f1_dev`
- Evaluation protocol: `q2_ablation_v1`
- Reusable checkpoint key: `no_multitask_bundle:20260521`
- Protocol resolution: `RESOLVED`
- CLI kind: `experiment`
- Resolved execution stage plan: `component_bundle`


## Required command sequence

1. Run `python scripts/run_single_experiment.py --experiment-id q2_no_multitask_bundle_20260521 --stage preflight`.
2. If preflight is BLOCKED or FAIL, paste its report into the Codex chat and stop.
3. After preflight passes, run `python scripts/run_single_experiment.py --experiment-id q2_no_multitask_bundle_20260521 --stage all`.
4. On interruption, resume only this entry with `python scripts/run_single_experiment.py --experiment-id q2_no_multitask_bundle_20260521 --resume`.
5. Do not call the global DAG, change the locked protocol, tune on the test split, or substitute a model, dataset, seed, or checkpoint.

## Required review handoff

The run must complete these stages in order: `preflight, execute_components, combine_component_predictions, evaluate_dev, freeze_component_selection, evaluate_test, export_artifacts, validate_artifacts, generate_review_summary`.

Print the complete review summary with `python scripts/print_run_review_summary.py --run-id q2_no_multitask_bundle_20260521` and paste it into the Codex chat. It must include `RUN_STATUS`, `USER_REVIEW_STATUS`, `NEXT_RUN_ALLOWED`, artifact hashes, and blockers.

The initial approval file must remain `PENDING_USER_APPROVAL`; never fabricate approval. After the summary is pasted, stop and wait for explicit user approval. Do not begin the next run automatically.

RUN_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO
