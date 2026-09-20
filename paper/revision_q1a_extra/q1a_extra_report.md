# Q1a extra compact audit

- Compact source: `reports/q1a_best_f1_table.json` generated 2026-09-06T08:49:12.196119+00:00; SHA-256 `96b93fe16c768f125d0c5340bf6f1e35d855379c2f6882a581e60aad379e59bc`.
- Metric: six binary per-head macro-F1 values; macro is their arithmetic mean.
- Aggregates in `q1a_extra_canonical_rows.csv` use arithmetic mean and sample SD across available per-seed rows. The source report bootstrap CI `half_width` is retained separately and is not SD.
- `ANALYZED_RECOMPUTED`: local six-head prediction JSONL was loaded and recomputed; direct no-auxiliary/explanation cache rows are compared to compact per-seed values.
- `ANALYZED_REPORTED_REMOTE`: persisted remote per-seed metric/prediction evidence is preserved with source ID/path/hash, without pretending this worker independently downloaded every source.
- CoT-only is `PROVISIONAL_2_OF_3_MISSING_TEST_SCORE`; seed 20260522 is `ANALYZED_REMOTE_DEV_ONLY`: the exact run exists on current HF trees with development/checkpoint evidence, but no test prediction or test metric exists. The local and remote dev-only numbers are excluded.
- GPT rows retain exact identity `GPT-4.1-mini`; no GPT-4o-mini relabeling.
- `raw_sources/` is generated local audit cache only; full-Vistral record download was stopped at the requested checkpoint (~1.7k files/~85 MB) and is not required by this compact handoff.
- Validation command: `python paper/revision_q1a_extra/validate_q1a_extra_compact.py`.
