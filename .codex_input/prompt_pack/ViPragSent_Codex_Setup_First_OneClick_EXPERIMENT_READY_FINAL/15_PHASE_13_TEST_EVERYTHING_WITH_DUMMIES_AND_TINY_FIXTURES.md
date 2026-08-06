> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 13 — TEST THE COMPLETE SETUP WITH DUMMIES AND TINY FIXTURES

Rehearse the entire workflow without downloading real model weights.

Required checks:

- all unit tests;
- all integration tests;
- dummy-model training;
- tiny-batch overfitting;
- mocked full Azure workflow;
- optional tiny Azure smoke;
- Q3 masks;
- external-evaluation fixtures;
- statistics;
- artifact export;
- orchestrator dry-run;
- interruption and resume simulation;
- failure injection;
- secret scan.

Required fixture run:

```bash
python scripts/run_all_experiments.py   --config configs/master_run.yaml   --mode fixture
```

The command must traverse the complete fixture DAG and produce every required artifact type.

Acceptance criteria: zero failing tests, one-click fixture run passes, resume passes, and no real model weights were downloaded.
