# Q1a expanded coverage ledger

> **WORKING COVERAGE ARTIFACT — NOT ACCEPTED FINAL**

This ledger records the requested 11-row Q1a inventory without importing
unverified values into the manuscript. The files
`reports/q1a_best_f1_table.md`, `.csv`, `.json`, and `.tex` are held pending
the Nash canonical source-path, hash, cohort, and protocol audit. Their
reported values are not substituted for the current paper/raw-fairness
artifact values: for example, the report's Sailor and Vistral values differ
from the current canonical subset, and its uncertainty is documented as a
bootstrap percentile interval rather than the manuscript's required sample
SD.

## Requested row inventory

| Requested exact row | Working handling | Evidence status | Reason / next gate |
|---|---|---|---|
| PhoBERT (single-task) | Retained in working plot with current artifact mean ± sample SD | ANALYZED; current artifact subset | Keep only if Nash maps the same source paths and cohort. |
| PhoBERT fine-tune | Retained in working plot with current artifact mean ± sample SD | ANALYZED; current artifact subset | Do not replace with report CI values. |
| XLM-R-large fine-tune/standard baseline | Retained as `XLM-R-large (standard baseline)` | ANALYZED; current artifact subset | Exact artifact identity is standard baseline; do not map it to the primary row. |
| Sailor-7B SFT | Retained in working plot with current artifact mean ± sample SD | ANALYZED; current artifact subset | Report value differs; source/hash/cohort mapping pending. |
| Vistral-7B SFT | Retained in working plot with current artifact mean ± sample SD | ANALYZED; current artifact subset | Report value differs; source/hash/cohort mapping pending. |
| ViPragSent - no auxiliary loss | Placeholder only | UNVERIFIED | Vistral-backed variant; add only after per-seed metric evidence and identity mapping. |
| Vistral CoT-only clean rerun 003 (2/3 if missing test) | Placeholder only | UNVERIFIED / partial candidate | Keep the missing test cell as `—` or status, never infer it from a CI report. |
| ViPragSent - explanation only | Placeholder only | UNVERIFIED | Requires canonical per-seed evidence. |
| ViPragSent (ours, Vistral) | Placeholder only in Q1a working inventory | UNVERIFIED for requested expanded row | Distinguish from the sole primary ViPragSent XLM-R-large row. |
| GPT-4.1-mini zero-shot | Placeholder only | UNVERIFIED; identity/protocol audit pending | Do not relabel GPT-4o-mini artifacts as GPT-4.1-mini. |
| GPT-4.1-mini 8-shot | Placeholder only | UNVERIFIED; identity/protocol audit pending | Record `n=1` if canonical evidence confirms it; no sample SD for n=1. |

The plot input is `figures/source_data/q1a_full_row_inventory_working.csv`.
Blank cells are explicit unverified placeholders. The companion preview is a
layout check only; the manuscript Table 1 remains on verified current rows
until Nash accepts the canonical mapping and uncertainty convention.

## Integration rule

Only rows with source-path/hash/cohort/protocol evidence and mean ± sample SD
may enter the final Table 1. Rows that are analyzed but not verified stay in
this ledger or a clearly marked companion coverage ledger. No historical
image/report value is copied merely because it fills a requested row.
