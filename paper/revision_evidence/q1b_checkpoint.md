# Q1b remote inventory checkpoint

Checkpoint date: 2026-09-20. This is intentionally incremental; it records discovery and receipt inspection before metric extraction.

## Inspected live

Live `FINAL_SCIENCE_RESULT.json` fetches returned HTTP 200 for 16 Q1b candidate directories in `Thundergod2007/vipragsent-experiment-artifacts-overflow-002`:

- `q1b_phobert_pol_single_20260521`, `20260522`, `20260523`
- `q1b_phobert_emo_single_20260521`, `20260522`, `20260523`
- `q1b_phobert_multitask_8head_20260521`, `20260522`, `20260523`
- `q1b_xlmr_multitask_8head_20260521`, `20260522`, `20260523`
- `q1b_vipragsent_full_phobert_20260521`, `20260522`, `20260523`
- `q1b_azure_gpt41_mini`

The 15 non-Azure training/evaluation receipts reported `APPROVED`, `PASS`, `PASS`, `APPROVED` for run, finalization, validation, and approval respectively. Their packs contain the external metric files. The Azure receipt's `macro_pragmatic_f1: NOT_APPLICABLE` is the Q1a-style pragmatic metric and is expected to be absent for Q1b; its pack separately contains `metrics/external_retention_metrics.json` with all three external F1 values and ordinary-F1.

Exact source prefix:

`https://huggingface.co/Thundergod2007/vipragsent-experiment-artifacts-overflow-002/blob/main/campaigns/vipragsent-v7-compact-20260820-190bb0f24a7e75a8/SHARD_H100_MIG20_COMPACT/nb-170d4c47-8a39-47ad-9425-d231de44fc92-857677f648-2qn8f/science/final/`

Append `q1b_<family>/FINAL_SCIENCE_RESULT.json` to inspect each receipt, or append the LFS tar filename recorded in the saved tree manifest for the pack.

## Pending / not promoted

- PhoBERT fine-tune Q1b reference row: the saved actual recipe is `phobert_multitask_8head`; retain that label and do not silently rewrite it as another historical family name. Its three complete packs are extracted in `q1b_canonical_table.csv`.
- Sailor and Vistral Q1b: completed external packs are now closed and extracted for all three seeds. Sailor is pinned to overflow-014/010/004; Vistral to overflow-011/012/011. Their exact per-seed metric-file/checkpoint hashes are in `q1b_provenance.csv`.
- GPT-4o-mini zero/8-shot: Q1a artifacts use `gpt41_mini` and are `NOT_STARTED`; the Q1b pack's request manifest explicitly says configured deployment `gpt-4.1-mini`, dedicated prompts, temperature 0, strict schema, 5,468 successful requests, and 6 retries. No response-side model version is recorded, so the evidence label is “Azure deployment gpt-4.1-mini (version unrecorded),” not an independently resolved backend and not the reference GPT-4o-mini identity.
- Q1b 8-shot GPT row: no completed comparable pack confirmed.
- For all promoted rows, `q1b_provenance.csv` records UIT-VSFC, VSMEC, and AIVIVN macro-F1, ordinary-F1 aggregate, split/gold identifiers, protocol, seed, source revisions, and full available hashes. Model revision/code commit remain `NOT_RECORDED` for the Sailor/Vistral external manifests and are not inferred.

## Local evidence already retained

The existing local Q1a package covers six rows (primary XLM-R-large plus five classification baselines) with 3 seeds each, 2,000 common IDs/gold, deterministic recomputation, and row-level hashes in `paper/evidence/q1a_fairness_run_table.csv`. In the structural crosswalk, `xlmr_baseline` is the reference XLM-R-large row; the primary XLM-R-large ViPragSent row is tracked separately as the current replacement for the Vistral-backed reference “ours” row. Q1b primary XLM-R follow-up packs are separately recorded in `q1b_provenance.csv`.

## Process handle

The 16-receipt live fetch completed (exit code 0; 17.2 seconds). No active fetch process or session handle remains. No training process was started.
