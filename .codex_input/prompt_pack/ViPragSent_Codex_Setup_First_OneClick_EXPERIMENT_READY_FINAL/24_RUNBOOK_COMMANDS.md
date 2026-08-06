# Runbook Commands

## Setup

```bash
make setup
make doctor
make test

python scripts/ingest_vipragsent.py
python scripts/download_external_datasets.py --all
python scripts/verify_azure_deployment.py
python scripts/plan_all_experiments.py --config configs/master_run.yaml
python scripts/run_all_experiments.py --config configs/master_run.yaml --mode fixture
```

## Final setup step: download models

```bash
python scripts/download_all_models.py   --manifest configs/models/download_manifest.yaml
```

## One-click full execution

```bash
python scripts/run_all_experiments.py   --config configs/master_run.yaml   --mode full
```

## Resume

```bash
python scripts/run_all_experiments.py   --config configs/master_run.yaml   --mode full   --resume
```
