# Q1a extra compact handoff

Use `q1a_extra_canonical_rows.csv/json`, `q1a_extra_per_seed_rows.csv`, `q1a_extra_coverage_ledger.csv`, and `q1a_extra_source_proof.json`. The exact source IDs/remote paths come from `reports/q1a_best_f1_table.json`; `q1a_extra_canonical_rows.csv` reports mean + sample SD, not bootstrap CI half-width. Raw `raw_sources/` is generated local audit cache only. CoT is provisional 2/3: seed 20260522 exists on current HF trees with dev/checkpoint evidence but no test score, so no test value is imputed. GPT identity is GPT-4.1-mini.
