# Q1b expanded working draft

> **WORKING DRAFT - ANALYZED / PENDING CROSS-COHORT VERIFICATION - NOT ACCEPTED FINAL**
>
> This layout combines the extracted canonical rows with the existing local Sailor/Vistral subset; the current canonical primary XLM-R row is used once when available, with the local ours row retained as the fallback. It does not modify the manuscript or accepted PDF. The contextual PhoBERT-backed ViPragSent inventory row is omitted from the main table by scope decision.

| Model | Actual recipe | n | UIT-VSFC | UIT-VSMEC | AIVIVN | Ordinary F1 (display only) | Working status |
|---|---|---:|---:|---:|---:|---:|---|
| ViPragSent (ours, XLM-R-large) | xlmr_followup_full | 3 | 37.9 +/- 2.8 | 40.0 +/- 0.7 | 47.9 +/- 2.2 | 42.0 +/- 1.8 | ANALYZED; PENDING_CROSS_COHORT_VERIFICATION |
| PhoBERT single-task routed | phobert_pol_single for AIVIVN/VSFC + phobert_emo_single for VSMEC | 3 | 29.7 +/- 0.5 | 38.8 +/- 1.5 | 45.4 +/- 2.6 | 38.0 +/- 0.3 | ANALYZED; PENDING_CROSS_COHORT_VERIFICATION |
| PhoBERT multitask 8-head | phobert_multitask_8head | 3 | 33.6 +/- 3.0 | 36.4 +/- 0.7 | 44.7 +/- 1.3 | 38.2 +/- 1.5 | ANALYZED; PENDING_CROSS_COHORT_VERIFICATION |
| XLM-R-large multitask 8-head | xlmr_multitask_8head | 3 | 39.1 +/- 1.1 | 40.2 +/- 0.2 | 48.3 +/- 1.2 | 42.5 +/- 0.5 | ANALYZED; PENDING_CROSS_COHORT_VERIFICATION |
| Sailor-7B SFT | Current Sailor record; recipe/result closure pending | 3 | 34.4 +/- 3.2 | 40.7 +/- 0.7 | 48.7 +/- 1.4 | 41.3 +/- 1.2 | PROVISIONAL_LOCAL_SUBSET; PENDING_CROSS_COHORT_VERIFICATION |
| Vistral-7B SFT | Current Vistral record; recipe/result closure pending | 3 | 30.7 +/- 6.4 | 39.4 +/- 1.5 | 49.0 +/- 0.8 | 39.7 +/- 2.5 | PROVISIONAL_LOCAL_SUBSET; PENDING_CROSS_COHORT_VERIFICATION |
| Azure deployment (gpt-4.1-mini) | azure_gpt41_mini dedicated_prompts | 1 | 62.7 (n=1; SD unavailable) | 53.1 (n=1; SD unavailable) | 63.4 (n=1; SD unavailable) | 59.7 (n=1; SD unavailable) | ANALYZED_IDENTITY_MISMATCH; PENDING_CROSS_COHORT_VERIFICATION |

Values are displayed in percentage points. `+/-` denotes the supplied sample SD; the Azure deployment (gpt-4.1-mini) row has n=1 and therefore no SD estimate; the deployed version is unrecorded. All rows remain pending cross-cohort verification for common task definition, split/gold alignment, and recipe comparability.
**Cohort caveat — direct ordinary-F1 ranking is withheld.** Comparisons are per-dataset cross-group checks, not a claim that one hash is shared across AIVIVN, VSFC, and VSMEC. The supplied checkpoint reports primary AIVIVN normalized-test SHA-256 prefix `27DA59A4...` and baseline prefix `F6F6A78E...`; the aggregate is shown only for layout review until per-dataset matched ID/gold proof or an explicit cohort separation is available.

## Coverage ledger for unresolved historical rows

| Historical/reference row | Draft handling | Value status | Reason |
|---|---|---|---|
| Q1b-02 PhoBERT fine-tune | Not added as a duplicate historical label | No value in this historical slot | The extracted actual recipe is retained as `PhoBERT multitask 8-head`; do not silently rewrite it as fine-tune. |
| Q1b-06 GPT-4o-mini zero-shot | Not added | No value | The extracted artifact is GPT-4.1-mini dedicated prompts, identity-mismatched with the GPT-4o reference row. |
| Q1b-07 GPT-4o-mini 8-shot | Not added | No value | No completed comparable 8-shot pack is available in the supplied checkpoint. |
| Q1b-08 ViPragSent full PhoBERT | Omitted from main table | No value in main-table scope | Contextual PhoBERT-backed ViPragSent row is excluded to retain one primary ours row; provenance remains outside this draft. |

Source hashes are recorded in `q1b_working_draft_manifest.json`. This artifact is for layout review only and is not a scientific approval or final manuscript source.
