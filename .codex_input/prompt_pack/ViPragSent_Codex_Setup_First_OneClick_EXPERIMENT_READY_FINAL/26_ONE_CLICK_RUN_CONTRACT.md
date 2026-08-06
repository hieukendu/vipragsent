# One-Click Run Contract

The complete project must run through:

```bash
python scripts/run_all_experiments.py   --config configs/master_run.yaml   --mode full
```

## Required guarantees

- Dependency-aware DAG.
- One GPU job at a time.
- Safe API/CPU overlap.
- Idempotency.
- Resume support.
- Cache validation.
- Retries for transient failures.
- No duplicate expensive runs.
- Automatic rationale generation.
- Automatic Azure baselines.
- Automatic Q1–Q4 execution.
- Automatic statistics, cost, and latency.
- Automatic tables, figures, and artifacts.
- Final manifest and checksums.

## Exit codes

- `0`: all mandatory nodes passed.
- `2`: credentials, quota, data, or model-access blocker.
- `3`: protocol or data-validation failure.
- `4`: non-recoverable training failure.
- `5`: artifact or audit failure.
