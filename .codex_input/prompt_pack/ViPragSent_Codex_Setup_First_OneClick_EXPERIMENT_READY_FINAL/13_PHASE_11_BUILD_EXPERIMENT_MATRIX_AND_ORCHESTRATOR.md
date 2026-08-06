> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 11 — BUILD THE MASTER EXPERIMENT MATRIX AND ORCHESTRATOR

Represent the complete Q1–Q4 workflow as one automated dependency-aware DAG without executing it.

Create:

```text
configs/master_run.yaml
configs/experiments/master_matrix.yaml
scripts/plan_all_experiments.py
scripts/run_all_experiments.py
src/vipragsent/orchestration/
```

Required DAG nodes:

1. Preflight validation.
2. Rationale generation.
3. Azure prompted baselines.
4. PhoBERT jobs.
5. XLM-R jobs.
6. Sailor jobs.
7. Vistral jobs.
8. Q1a evaluation.
9. Q1b external-retention evaluation.
10. Q2 ablations.
11. Q3 low-resource experiments.
12. Q4 calibration and learning curves.
13. Statistics.
14. Cost and latency.
15. Artifact export.
16. Final manifest.

Scheduling rules:

- One GPU job at a time on an A100 20 GB.
- API and CPU work may overlap only when safe.
- Never launch concurrent 7B jobs.
- Respect dependencies.
- Resume completed valid nodes.
- Retry transient failures.
- Fail fast on data, protocol, or checksum corruption.
- Never rerun a successful expensive node unless `--force` is explicitly supplied.

Dry-run must print every job, dependencies, seeds, total run count, expected outputs, model downloads required later, Azure request counts, disk estimate, and time estimate.

Do not download model weights or execute the real DAG.

# REQUIRED ADDITIONAL DAG NODES

Add these explicit nodes:

```text
table3_checkpoint_training
backbone_sensitivity
error_analysis_candidate_export
qualitative_candidate_export
paper_artifact_schema_validation
```

## Backbone sensitivity node

Run the complete PhoBERT and Vistral ViPragSent systems with identical data, targets, seeds, and reporting
definitions. Export:

- macro-pragmatic F1;
- external ordinary F1;
- intended-polarity ECE;
- GPU-hours;
- peak VRAM;
- batch-1 latency;
- batch-32 throughput;
- relative cost.

## Manual-analysis behavior

The one-click run must automatically export the error-analysis and qualitative candidate files.
It must never invent human coding. The core numerical experiment DAG may finish with:

```text
CORE_EXPERIMENTS_READY=true
MANUAL_PAPER_ANALYSIS_PENDING=true
```

After reviewed manual files are supplied, rerunning with `--resume` must generate the final manual-analysis artifacts.
