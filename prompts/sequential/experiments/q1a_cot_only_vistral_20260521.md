# ViPragSent sequential experiment run: q1a_cot_only_vistral_20260521

This runbook names exactly one inventory entry. Do not start another experiment or Azure job.

## Locked entry

- Experiment ID: `q1a_cot_only_vistral_20260521`
- Research question: `Q1a`
- System ID: `cot_only_vistral`
- Display name: Vistral CoT-only
- Variant: `cot_only`
- Backbone: `vistral_7b`
- Seed: `20260521`
- Budget: ``
- Task: `pragmatic`
- Split: `vipragsent_test`
- Dependencies: `preflight_validation;rationale_generation`
- Required Phase 15 assets: `model_weights;tokenizer;runtime_profile`
- Execution kind: `generation`
- Expected artifacts: `predictions;metrics;history`
- Selection metric: `macro_prag_f1_dev`
- Evaluation protocol: `q1a_frozen_dev_threshold_v1`
- Reusable checkpoint key: `cot_only_vistral:20260521`
- Protocol resolution: `RESOLVED`
- CLI kind: `experiment`
- Resolved execution stage plan: `cot_only_vistral_generation`

## Locked reasoning-generation contract

Use the exact Vietnamese reasoning prompt, causal generation-only objective, greedy decoding, and the shared zero-shot reasoning judge. The judge receives generated reasoning only and emits the strict six-key JSON schema. Select only on the full-split all-zero-fallback dev metric; do not inspect test data before `freeze_selection`. Do not use a direct label parser or generated label target.


## Required command sequence

1. Run `python scripts/run_single_experiment.py --experiment-id q1a_cot_only_vistral_20260521 --stage preflight`.
2. If preflight is BLOCKED or FAIL, paste its report into the Codex chat and stop.
3. After preflight passes, run `python scripts/run_single_experiment.py --experiment-id q1a_cot_only_vistral_20260521 --stage all`.
4. On interruption, resume only this entry with `python scripts/run_single_experiment.py --experiment-id q1a_cot_only_vistral_20260521 --resume`.
5. Do not call the global DAG, change the locked protocol, tune on the test split, or substitute a model, dataset, seed, or checkpoint.

## Required review handoff

The run must complete these stages in order: `preflight, train_generation, generate_dev_reasoning, judge_dev_reasoning, compute_dev_reasoning_metrics, freeze_selection, generate_test_reasoning, judge_test_reasoning, compute_test_reasoning_metrics, export_artifacts, validate_artifacts, generate_review_summary`.

Print the complete review summary with `python scripts/print_run_review_summary.py --run-id q1a_cot_only_vistral_20260521` and paste it into the Codex chat. It must include `RUN_STATUS`, `USER_REVIEW_STATUS`, `NEXT_RUN_ALLOWED`, artifact hashes, and blockers.

The initial approval file must remain `PENDING_USER_APPROVAL`; never fabricate approval. After the summary is pasted, stop and wait for explicit user approval. Do not begin the next run automatically.

RUN_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO
