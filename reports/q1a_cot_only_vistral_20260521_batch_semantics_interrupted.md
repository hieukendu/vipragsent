# q1a CoT-only Vistral interrupted batch-semantics attempt

Status: `INTERRUPTED_FOR_REPAIR`

The exact q1a process was stopped after preflight passed but before any epoch
history, checkpoint, or metric artifact was emitted. The production generation
executor was performing one optimizer update per record, while the locked
Vistral contract requires physical batch `2`, gradient accumulation `8`, and
effective batch `16`.

The original run state, stage events, and CUDA device report remain preserved
under `results/runs/q1a_cot_only_vistral_20260521/`. No scientific output was
promoted. This is classified as `A_REPAIRABLE_IMPLEMENTATION_DEFECT` with no
protocol change.

Safe resume after repair:

```text
PYTHONPATH=src ./.venv/bin/python scripts/run_single_experiment.py --experiment-id q1a_cot_only_vistral_20260521 --stage all --resume
```
