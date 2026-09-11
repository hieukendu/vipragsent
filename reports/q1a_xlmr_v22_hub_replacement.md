# V22 seed 21/23 Hub replacement and stop audit

Audit date: 2026-09-11 UTC.

## Scope

The requested replacement used only the exact V22 replay sources below. The
canonical seed-22 V22 run was retained and was not overwritten.

| Seed | Exact local source | Exact replacement root |
|---|---|---|
| 21 | `results/runs/q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_v22_replay_20260909_20260521` | `campaigns/vipragsent-v8-local-mig2g20gb/q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_v22_replay_20260909_20260521` |
| 23 | `results/runs/q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_v22_replay_20260909_20260523` | `campaigns/vipragsent-v8-local-mig2g20gb/q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_v22_replay_20260909_20260523` |

The checkpoint repository is `Thundergod2007/vipragsent-xlmr-checkpoints`.
The artifact repository is
`Thundergod2007/vipragsent-experiment-artifacts-overflow-018`.

## Upload verification

The uploader completed both runs with 31/31 verified files and zero errors;
each receipt reports `PASS`. An independent Hub tree audit then checked exact
paths and sizes, compared every checkpoint LFS SHA-256, and downloaded and
rehashed every artifact file.

| Seed | Checkpoint files / bytes | Artifact files / bytes | Result |
|---|---:|---:|---|
| 21 | 12 / 85,186,021,304 | 20 / 5,071,836 | PASS |
| 23 | 12 / 85,186,022,648 | 20 / 5,074,608 | PASS |

The 20 artifact files are the 19 canonical run artifacts plus the uploader
receipt. The uploader was then stopped cleanly with status
`STOPPED_BY_FILE`.

## Replaced roots removed

Only the following previously existing roots were deleted, after the upload
audit passed, in both repositories:

- seed 21 old V9 root: `.../q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_loss_focus_cosine_v9_20260521`
- seed 23 old V32 root: `.../q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup010_implicit105_v32_20260523`

The deletion was performed from an explicitly enumerated list of 24
checkpoint files and 40 artifact files. The two Hub deletion commits were:

- checkpoint repo: `81758116727257c5ed2876087f0cfb6b539ddb32`
- artifact repo: `40f45b1c1ea3412a55c7bcfaca6880e0a15c2b32`

Post-delete tree queries confirmed both old roots are empty in both
repositories. The two new V22 roots and the canonical seed-22 V22 root remain
untouched. Local V22 source directories were retained.

## Experiment and acceptance interpretation

All training/optimization runs were stopped. The retained common V22 trio is
the closest audited comparison, but its three-seed mean passes 6/7 strict
gates; only code-switching misses (`94.22` versus the `94.27` baseline).
This upload is therefore a user-requested archival/reproducibility
replacement, not a claim of a seven-gate passing model. No further experiment
was started or authorized.

## Git scope

Optimization rationale, audit records, and the stop/replacement decision are
tracked on branch
`codex/q1a-xlmr-optimization-pr-20260908` and in PR #14. Model/checkpoint
files, `.env`, runtime queue/state, caches, and secrets are excluded from the
Git change.
