# ViPragSent

This repository implements the setup-first ViPragSent experiment contract. The project is intentionally
dependency-light during setup and keeps private datasets, model weights, credentials, and generated results
out of Git.

## Workflow

1. Put `ViPragSent_Experiment_Dataset_FINAL_V8.zip` at the repository root.
2. Run `python scripts/ingest_vipragsent.py`.
3. Supply license-compliant external test files through `data/external/manual_drop/` or the download script.
4. Configure Azure using `.env.example` and verify the deployment.
5. Run `python scripts/run_all_experiments.py --config configs/master_run.yaml --mode fixture`.
6. Generate and validate the sequential Codex runbooks with `python scripts/generate_sequential_prompts.py` and `python scripts/validate_sequential_prompts.py`.
7. Run exactly one approved inventory entry with `scripts/run_single_experiment.py` or exactly one Azure job with `scripts/run_single_azure_job.py`.

The generated runbooks are grouped under `prompts/sequential/phase15/`,
`prompts/sequential/experiments/`, `prompts/sequential/azure/`, and
`prompts/sequential/aggregation/`. Every future run stops at a review handoff with
`PENDING_USER_APPROVAL`; setup generation itself never downloads weights or starts a run.

The former global full-run entry point is disabled by policy:

```text
python scripts/run_all_experiments.py --config configs/master_run.yaml --mode full  # BLOCKED by policy
```

Use `python scripts/aggregate_approved_runs.py --research-question <Q1a|Q1b|Q2|Q3|Q4|all>` only after every required run has PASS status and explicit user approval. The fixture DAG remains available for setup validation and writes phase handoffs under `reports/phases/`. The repository reports a blocker instead of substituting an unofficial dataset, model revision, or Azure endpoint.

## Current input boundary

The supplied V8 archive contains ViPragSent and the bundled `AIVIVN-human-derived-3way` split. UIT-VSFC,
UIT-VSMEC, Azure credentials, and large model weights are external prerequisites and are never committed.

## Development

```text
python -m pytest
python scripts/validate_project_layout.py
python scripts/check_environment.py
```

Target runtime: Python 3.11. Python 3.12 is supported for setup and fixture validation when installed
dependencies are compatible.
