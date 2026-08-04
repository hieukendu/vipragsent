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
6. After the setup freeze and model-access preflight pass, run the same command with `--mode full`.

The one-click full-run entry point is:

```text
python scripts/run_all_experiments.py --config configs/master_run.yaml --mode full
```

The orchestrator is resume-safe and writes phase handoffs under `reports/phases/`. It will report a blocker
instead of substituting an unofficial dataset, model revision, or Azure endpoint.

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
