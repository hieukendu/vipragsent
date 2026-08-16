# LUNA_NAACL_PROFILE

Status: policy-only artifact, ready after `WAVE0_ACCEPTED`.

This profile records the NAACL-balanced slice for scientific traceability. It is not a benchmark, experiment, or execution plan. Activation is default-off, requires explicit opt-in after Wave 0 acceptance, and real execution is prohibited.

## Retained scope

- Q3 retains the audited local systems PhoBERT, Vistral, and full ViPragSent.
- Each retained system has budgets `32`, `128`, `512`, and `full`, across seeds `20260521`, `20260522`, and `20260523` (36 cells).
- Q2 retains its three audited seeds: `20260521`, `20260522`, and `20260523`.
- Q1b remains evaluation-only: official external tests use approved upstream checkpoints, with no training and zero optimizer steps.

## Exclusions and invariants

XLM-R Q3, Q3 budgets `64` and `256`, and the non-local Azure Q3 system are excluded. The original profile and all audited source files remain untouched. Fixture, synthetic, and run-data rows cannot enter aggregation.

Aggregation is profile-aware and fail-closed: it consumes only the declared Cartesian cells, rejects missing cells, and does not mix excluded budgets or systems. Q1b aggregation first resolves an exact approved producer for the consumer, requires the same seed and matching checkpoint key, and blocks if that dependency is absent; training metrics are never substituted for Q1b evaluation metrics.

Source references and machine-readable invariants are recorded in `naacl_balanced_profile.json`; the declarative activation policy is in `configs/experiments/naacl_balanced_runtime_profile.yaml`.
