# Q1a CoT seed-22 live Hugging Face audit

Last direct recheck: 2026-09-20T18:05:08.4786925Z (2026-09-21 Asia/Ho_Chi_Minh).

The exact run `q1a_cot_only_vistral_clean_rerun_003__seed_20260522` does exist
on current Hugging Face trees. Direct exact-path checks across all 29 current
artifact repositories found it in `overflow-006`, `overflow-021`, and the
Vistral checkpoint repository.

The remote run is not a completed Q1a test run. The current artifact set contains
epoch-1 development predictions and development reasoning metrics in
`overflow-006`, a preflight/NOT_STARTED state and pack receipts in `overflow-021`,
and checkpoint/receipt artifacts in the checkpoint repository.
The development metric is `PASS` for 1,999/1,999 rows, but its primary metric is
explicitly `full_split_macro_pragmatic_f1_all_zero_fallback` with truncation rate
1.0, and it is a development value. The exhaustive paginated `overflow-006` tree
audit found zero `test_predictions`, zero `test_reasoning_metrics`, and zero
`test_metrics` files for this exact run ID; `overflow-021` also reports all test
stages as `NOT_STARTED`.

Therefore seed 20260522 is now recorded as **present but dev/checkpoint-only**.
It remains in the seed coverage ledger, while the Q1a COT test aggregate remains
`PROVISIONAL_2_OF_3_MISSING_TEST_SCORE`; no test value is imputed.

The direct recheck returned HTTP 200 for the recorded development prediction and
development-metric paths, HTTP 404 for the exact test-prediction and
test-reasoning-metric paths, and HTTP 200 for the `overflow-021` state/metric
manifests. That state still reports `RUNNING`, with test generation and test
metrics `NOT_STARTED`. The checkpoint repository still exposes epoch-1 and
epoch-2 model files. This confirms that seed 22 was not omitted from the
remote search; it is simply not a completed test-scoring run.
