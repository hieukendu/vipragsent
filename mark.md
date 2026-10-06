## Reviewer-gap run: setup

- ViSoBERT `uitnlp/visobert` pinned at `196a62afad9cbe4f52a54aabad828b13f0eec59a`.
- Cache, GPU smoke, and physical-batch probe PASS; selected batch 32 / effective batch 32.
- Reviewer-gap HF uploader is running for campaign `reviewer-gaps-20261005`.

## ViSoBERT baseline

- Seeds 20260521/20260522/20260523 PASS; test macro pragmatic F1: 0.881298 / 0.886117 / 0.884542.
- Dev-only thresholds and frozen test evaluation validated; all three prediction sets have the expected 7,998 / 1,999 / 2,000 rows.
- Each seed queued 22 files for HF upload.

## XLM-R all multipliers = 1.0

- Seed 20260521 PASS; test macro pragmatic F1 `0.928607`, best epoch `10`, dev-only checkpoint/threshold selection and post-freeze test evaluation validated.
- HF upload PASS: 22/22 files; artefact repo `Thundergod2007/vipragsent-experiment-artifacts-overflow-018`, checkpoint repo `Thundergod2007/vipragsent-xlmr-checkpoints`.
- Seed 20260522 PASS; test macro pragmatic F1 `0.927266`, best epoch `10`, dev-only checkpoint/threshold selection and post-freeze test evaluation validated.
- HF upload PASS: 22/22 files; artefact repo `Thundergod2007/vipragsent-experiment-artifacts-overflow-025`, checkpoint repo `Thundergod2007/vipragsent-xlmr-checkpoints`.
- Seed 20260523 PASS; test macro pragmatic F1 `0.933567`, best epoch `10`, dev-only checkpoint/threshold selection and post-freeze test evaluation validated.
- HF upload PASS: 22/22 files; artefact repo `Thundergod2007/vipragsent-experiment-artifacts-overflow-017`, checkpoint repo `Thundergod2007/vipragsent-xlmr-checkpoints`.
- All-multipliers=1 mean macro pragmatic F1 across seeds: `0.929814 ± 0.003319` (sample SD).

## XLM-R rationale beta = 0.1

- Seed 20260521 PASS; test macro pragmatic F1 `0.928155`, best epoch `10`, dev-only checkpoint/threshold selection and post-freeze test evaluation validated.
- HF upload PASS: 22/22 files; artefact repo `Thundergod2007/vipragsent-experiment-artifacts-overflow-018`, checkpoint repo `Thundergod2007/vipragsent-xlmr-checkpoints`.
- Seed 20260522 PASS; test macro pragmatic F1 `0.935364`, best epoch `10`, dev-only checkpoint/threshold selection and post-freeze test evaluation validated.
- HF upload PASS: 22/22 files; artefact repo `Thundergod2007/vipragsent-experiment-artifacts-overflow-025`, checkpoint repo `Thundergod2007/vipragsent-xlmr-checkpoints`.
- Seed 20260523 PASS; test macro pragmatic F1 `0.932902`, best epoch `10`, dev-only checkpoint/threshold selection and post-freeze test evaluation validated.
- HF upload PASS: 22/22 files; artefact repo `Thundergod2007/vipragsent-experiment-artifacts-overflow-017`, checkpoint repo `Thundergod2007/vipragsent-xlmr-checkpoints`.
- Beta=0.1 mean macro pragmatic F1 across seeds: `0.932140 ± 0.003664` (sample SD).

## XLM-R rationale beta = 0.5
- Seed 20260521 PASS; test macro pragmatic F1 `0.920973`, best epoch `10`, peak VRAM `13.519 GiB`, dev-only checkpoint/threshold selection and post-freeze test evaluation validated.
- Seed 20260521 HF upload PASS: 22/22 verified; artefact repo `Thundergod2007/vipragsent-experiment-artifacts-overflow-018`, checkpoint repo `Thundergod2007/vipragsent-xlmr-checkpoints`.
- Seed 20260522 PASS; test macro pragmatic F1 `0.925744`, best epoch `10`, peak VRAM `13.519 GiB`, dev-only checkpoint/threshold selection and post-freeze test evaluation validated.
- Seed 20260522 HF upload PASS: 22/22 verified; artefact repo `Thundergod2007/vipragsent-experiment-artifacts-overflow-025`, checkpoint repo `Thundergod2007/vipragsent-xlmr-checkpoints`.
- Seed 20260523 PASS; test macro pragmatic F1 `0.933908`, best epoch `10`, peak VRAM `13.519 GiB`, dev-only checkpoint/threshold selection and post-freeze test evaluation validated.
- Seed 20260523 HF upload PASS: 22/22 verified; artefact repo `Thundergod2007/vipragsent-experiment-artifacts-overflow-017`, checkpoint repo `Thundergod2007/vipragsent-xlmr-checkpoints`.
- Beta=0.5 mean macro pragmatic F1 across seeds: `0.926875 ± 0.006542` (sample SD); all training jobs in this campaign are now complete.

## Reviewer-gap post-hoc analysis

- Analyzer PASS with all six experiment groups, exact overlap, PR/calibration, code-switching heuristic, and 15/15 learned log-variance entries.
- Analysis artefact upload PASS: 8/8 verified in `Thundergod2007/vipragsent-experiment-artifacts-overflow-018` under `campaigns/reviewer-gaps-20261005/reviewer_gap_analyses_20261005`.
