## Q1b coverage ledger after canonical artifact audit

| Historical/reference row | Draft handling | Value status | Reason |
|---|---|---|---|
| Q1b-01 PhoBERT single-task | Included as same-seed routed composite | `ANALYZED`, numeric | AIVIVN/VSFC polarity and VSMEC emotion component packs are combined by matching seed; not presented as one historical run. |
| Q1b-02 PhoBERT fine-tune | Included under actual recipe name | `ANALYZED`, numeric | Saved recipe is `phobert_multitask_8head`; it is not silently relabelled as the historical fine-tune family. |
| Q1b-03 XLM-R-large | Included as standard classification baseline | `ANALYZED_SEMANTIC_COHORT_VERIFIED`, numeric | Distinct from the single primary ViPragSent XLM-R-large row; per-seed ID/gold/raw-text/NFC parity passes. |
| Q1b-04 Sailor-7B SFT | Included | `ANALYZED_SEMANTIC_COHORT_VERIFIED`, numeric | Actual `sailor_multitask_8head`, completed external packs for seeds 21/22/23; model revision/code commit unavailable in manifest. |
| Q1b-05 Vistral-7B SFT | Included | `ANALYZED_SEMANTIC_COHORT_VERIFIED`, numeric | Actual `vistral_multitask_8head`, completed external packs for seeds 21/22/23; model revision/code commit unavailable in manifest. |
| Q1b-06 GPT-4o-mini zero-shot | Included as actual deployment, not reference identity | `ANALYZED_IDENTITY_MISMATCH`, numeric | Azure deployment is `gpt-4.1-mini` with response-side version unrecorded; do not relabel as GPT-4o-mini. |
| Q1b-07 GPT-4o-mini 8-shot | Explicit unresolved row | `UNRESOLVED`, `—` | No completed comparable 8-shot pack confirmed; gpt-4.1-mini zero-shot is not substituted. |
| Q1b-08 ViPragSent (ours) | Included as the sole primary row | `ANALYZED_SEMANTIC_COHORT_VERIFIED`, numeric | Primary is XLM-R-large; AIVIVN manifest hash mismatch is retained as provenance limitation after full observed-record parity passes. |
| Contextual ViPragSent full PhoBERT | Inventory only, outside primary scope | `ANALYZED_CONTEXTUAL_VARIANT`, numeric in canonical evidence | Preserved for audit completeness; never promoted to the primary row. |

Source hashes, exact per-seed URLs/revisions, and full available provenance are recorded in `revision_evidence/q1b_provenance.csv`, `q1b_http_payload_manifest.csv`, and `q1b_split_gold_comparability.md`. This artifact remains for layout review only and is not a scientific approval or final manuscript source.
