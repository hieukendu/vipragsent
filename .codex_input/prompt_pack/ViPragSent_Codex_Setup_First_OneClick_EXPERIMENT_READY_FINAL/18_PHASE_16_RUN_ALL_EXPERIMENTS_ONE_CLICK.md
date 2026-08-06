> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 16 — RUN THE COMPLETE EXPERIMENT SUITE WITH ONE COMMAND

Execute the entire experiment DAG and export all artifacts automatically through one command.

Preconditions:

- `SETUP_READY.md` reports PASS.
- All model weights are verified.
- External datasets are complete.
- Azure credentials and deployment are valid.
- Disk, GPU, and API quota are sufficient.

One-click command:

```bash
python scripts/run_all_experiments.py   --config configs/master_run.yaml   --mode full
```

Do not require the user to run Q1, Q2, Q3, or Q4 separately.

Required automatic order:

1. Preflight and checksum validation.
2. Generate all missing rationales with Azure GPT-4.1-mini.
3. Build deterministic CoT targets.
4. Run all Azure zero-shot and 8-shot baselines.
5. Run all PhoBERT jobs.
6. Run all XLM-R jobs.
7. Run all Sailor and Vistral QLoRA jobs.
8. Evaluate Q1a.
9. Evaluate Q1b.
10. Run Q2 ablations.
11. Run Q3 low-resource experiments.
12. Run Q4 calibration and learning curves.
13. Compute confidence intervals and significance tests.
14. Measure cost and latency.
15. Export all tables, figures, and backing CSVs.
16. Create the result provenance index.
17. Create the final experiment manifest and checksums.

Scheduler requirements: one GPU job at a time, safe API/CPU overlap, resume after disconnection, skip completed valid nodes, retry transient failures, stop on non-transient data/protocol errors, periodic progress and ETA, and no manual intervention unless a genuine blocker occurs.

Required final outputs:

```text
results/final/
experiment_artifacts/tables/
experiment_artifacts/figures/
experiment_artifacts/backing_data/
reports/full_run/
FINAL_EXPERIMENT_MANIFEST.json
FINAL_RESULT_CHECKSUMS.sha256
```

Success criteria: every mandatory DAG node passes, every expected model/seed/budget is present, all artifacts come from final results, and the paper is not modified.

# REQUIRED COMPLETENESS CHECKS DURING FULL RUN

Before declaring the run complete, verify:

- Table 2 uses `vipragsent_full_vistral`.
- Table 3 uses `vipragsent_full_phobert`.
- Table 4 uses the PhoBERT controlled-ablation family.
- The exact Table 3 checkpoint matrix is complete.
- Q3 uses frozen nested masks, fixed negatives, per-budget `pos_weight`, and fixed dev/test.
- Q4 reliability diagrams contain:
  - PhoBERT fine-tune;
  - Vistral-7B SFT;
  - full ViPragSent Vistral.
- Required significance comparisons are complete.
- Backbone-sensitivity artifacts exist.
- Error-analysis and qualitative candidate files exist.
- No old Figure 5 artifact exists.

## Required significance families

Run paired comparisons for:

1. full ViPragSent Vistral vs PhoBERT fine-tune;
2. full ViPragSent Vistral vs Azure GPT-4.1-mini 8-shot;
3. full ViPragSent Vistral vs Vistral-7B SFT.

Compute each comparison for:

- six individual pragmatic labels;
- macro-pragmatic F1.

Use two-sided paired bootstrap tests and apply Holm correction within each seven-metric comparison family.


# FINAL COMPLETION STATES

The one-click run must always generate dataset-summary artifacts before model result tables.

If human-reviewed error analysis and qualitative approvals are not yet available, write:

```text
CORE_EXPERIMENTS_READY=true
MANUAL_PAPER_ANALYSIS_PENDING=true
```

Do not mark manual analysis complete from automatically selected candidates alone.
