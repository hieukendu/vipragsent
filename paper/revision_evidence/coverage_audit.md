# Q1a/Q1b reference coverage audit — partial checkpoint

**Audit state:** PARTIAL / IN PROGRESS (checkpoint written 2026-09-20, Asia/Ho_Chi_Minh)

**Scope and safety boundary.** This is an evidence-worker handoff. It does not edit `paper/manuscript.md`, `paper/sections/`, the accepted PDF, or production-owned files. Artifact status is reported as `ANALYZED` unless an independent rerun exists; a remote `APPROVED` receipt is not silently converted into metric comparability.

## Immediate findings

- The production inventory in `paper/revision_plan_initial.md` expands the structural reference to **11 Q1a rows and 8 Q1b rows**. The prior three-row Q1b subset is incomplete.
- For **Q1a**, the available Azure zero/8-shot artifacts use `azure_gpt41_mini` / `gpt41_mini`, have `NOT_STARTED` manifests, and do not record a deployed-model field; the Q1a GPT rows therefore remain identity/completion unresolved. For **Q1b**, the completed Azure pack records deployment `gpt-4.1-mini` in `azure/request_manifest.json`; it is metrically extracted but is not silently relabelled as the reference GPT-4o-mini row.
- A live/remote scan found a complete-receipt family for Q1b that was absent from the prior three-row paper subset: **PhoBERT polarity single-task, PhoBERT emotion single-task, PhoBERT multitask 8-head, XLM-R multitask 8-head, and ViPragSent full PhoBERT**, each with seeds 21/22/23. The external metric packs have now been inspected. The two single-task packs are partial by design (polarity: AIVIVN/VSFC; emotion: VSMEC) and are combined by seed for the routed PhoBERT single-task row; the multitask packs contain all three external datasets and ordinary-F1.
- The same scan found completed Q1b Sailor and Vistral external packs beyond the earlier selected subset. Sailor uses overflow-014/010/004 for seeds 20260521/22/23; Vistral uses overflow-011/012/011. Their external manifests and metrics are now extracted as analyzed comparison rows; no row is promoted from a train receipt alone.
- Q1a remote variant families are distinct: `vipragsent_no_auxiliary_vistral`, `cot_only_vistral`, `explanation_only_vistral`, and `vipragsent_full_vistral`. A Vistral-backed ViPragSent variant is not the primary XLM-R-large ViPragSent row.

## Row-by-row structural crosswalk

The reference values are structural targets only and are not imported as current results. Current status/provenance is maintained in `q1a_q1b_crosswalk.csv` and the Q1b checkpoint table.

### Q1a (reference Table 2)

| Ref row | Reference row | Current evidence status | Classification |
|---|---|---|---|
| Q1a-01 | PhoBERT single-task | 3 local prediction runs; same 2,000 IDs/gold; deterministic recompute | comparable current baseline, `ANALYZED` |
| Q1a-02 | PhoBERT fine-tune | 3 local prediction runs; same 2,000 IDs/gold; deterministic recompute | comparable current baseline, `ANALYZED` |
| Q1a-03 | XLM-R-large | 3 local `xlmr_baseline` prediction runs; same 2,000 IDs/gold; deterministic recompute | standard classification baseline, `ANALYZED` |
| Q1a-04 | Sailor-7B SFT | 3 local prediction runs; same 2,000 IDs/gold | comparable current baseline, `ANALYZED` |
| Q1a-05 | Vistral-7B SFT | 3 local prediction runs; same 2,000 IDs/gold | comparable current baseline, `ANALYZED` |
| Q1a-06 | GPT-4o-mini zero-shot | Azure `gpt41_mini` artifacts; manifest says `NOT_STARTED`; model identity unresolved | excluded pending identity/completion |
| Q1a-07 | GPT-4o-mini 8-shot | Azure `gpt41_mini` artifacts; manifest says `NOT_STARTED`; model identity unresolved | excluded pending identity/completion |
| Q1a-08 | ViPragSent without auxiliary loss | Vistral-backed remote variant family found; final metric/protocol extraction pending | contextual variant, not primary |
| Q1a-09 | ViPragSent CoT-only | Vistral-backed remote family found, including clean-rerun attempts; seed/final closure audit pending | contextual variant, not primary |
| Q1a-10 | ViPragSent explanation-only | Vistral-backed remote family found in overflow artifacts; final metric/protocol extraction pending | contextual variant, not primary |
| Q1a-11 | ViPragSent (ours, Vistral) | Structural reference row is Vistral-backed; current primary replacement is XLM-R-large and is tracked separately from Q1a-03; Vistral full-family artifacts remain pending | primary replacement plus contextual Vistral candidate; not conflated |

### Q1b (reference Table 3)

| Ref row | Reference row | Newly inspected evidence | Current disposition |
|---|---|---|---|
| Q1b-01 | PhoBERT single-task | Polarity and emotion single-task packs provide AIVIVN/VSFC and VSMEC respectively; combined by seed with common gold hashes | artifact-derived routed baseline, `ANALYZED`, extracted |
| Q1b-02 | PhoBERT fine-tune | Actual saved recipe is `phobert_multitask_8head` (not silently renamed); 3 packs contain all three external datasets and ordinary-F1 | actual recipe retained, `ANALYZED`, extracted; not asserted equivalent to historical name |
| Q1b-03 | XLM-R-large | `q1b_xlmr_multitask_8head_20260521/22/23`; this is the classification baseline, separate from the primary ViPragSent XLM-R Q1a row | classification baseline, `ANALYZED` |
| Q1b-04 | Sailor-7B SFT | Actual `sailor_multitask_8head` external packs, seeds 20260521/22/23; explicit polarity/emotion routing; common F6F6A78E/C27B61C7/1D2E604B hashes | `ANALYZED_SEMANTIC_COHORT_VERIFIED`; comparison row |
| Q1b-05 | Vistral-7B SFT | Actual `vistral_multitask_8head` external packs, seeds 20260521/22/23; explicit polarity/emotion routing; common F6F6A78E/C27B61C7/1D2E604B hashes | `ANALYZED_SEMANTIC_COHORT_VERIFIED`; comparison row |
| Q1b-06 | GPT-4o-mini zero-shot | Q1b Azure final pack has external AIVIVN/VSFC/VSMEC F1 and ordinary-F1; request manifest pins deployment `gpt-4.1-mini`, dedicated prompts, temperature 0, strict schema; reference identity is GPT-4o-mini | metric-ready artifact, but identity-mismatch exclusion from historical GPT-4o row |
| Q1b-07 | GPT-4o-mini 8-shot | No completed comparable Q1b 8-shot evidence confirmed in this checkpoint | unresolved |
| Q1b-08 | ViPragSent (ours) | `xlmr_followup_q1b_vipragsent_full_20260521/22/23` in overflow-018/025/017; all three external packs are `PASS/APPROVED`, XLM-R-large, and carry complete external metrics; record-level parity against standard XLM-R passes for all 3 seeds and 3 datasets | primary XLM-R-large, `ANALYZED_SEMANTIC_COHORT_VERIFIED`; AIVIVN manifest hash retained as provenance limitation |

## Newly discovered Q1b remote receipts

All paths below are under the same public Hub repository and were found in the saved remote tree manifest, then the `FINAL_SCIENCE_RESULT.json` file was fetched live. The exact tree revision for the overflow repository must be recorded from the current API response in the completed audit; the saved manifest identifies the corresponding per-file commit OIDs.

Repository: [`Thundergod2007/vipragsent-experiment-artifacts-overflow-002`](https://huggingface.co/Thundergod2007/vipragsent-experiment-artifacts-overflow-002)

Base path (all rows):

`campaigns/vipragsent-v7-compact-20260820-190bb0f24a7e75a8/SHARD_H100_MIG20_COMPACT/nb-170d4c47-8a39-47ad-9425-d231de44fc92-857677f648-2qn8f/science/final/`

Discovered final-result directories:

- `q1b_phobert_pol_single_20260521`, `q1b_phobert_pol_single_20260522`, `q1b_phobert_pol_single_20260523`
- `q1b_phobert_emo_single_20260521`, `q1b_phobert_emo_single_20260522`, `q1b_phobert_emo_single_20260523`
- `q1b_phobert_multitask_8head_20260521`, `q1b_phobert_multitask_8head_20260522`, `q1b_phobert_multitask_8head_20260523`
- `q1b_xlmr_multitask_8head_20260521`, `q1b_xlmr_multitask_8head_20260522`, `q1b_xlmr_multitask_8head_20260523`
- `q1b_vipragsent_full_phobert_20260521`, `q1b_vipragsent_full_phobert_20260522`, `q1b_vipragsent_full_phobert_20260523`
- `q1b_azure_gpt41_mini`

For each non-Azure family above, the final receipt reported `run_status=APPROVED`, `finalization_status=PASS`, `validation_status=PASS`, and `approval_status=APPROVED` in the live fetch. This confirms saved final packs, not yet comparable external-retention numbers. The receipt metric keys include `macro_pragmatic_f1`, but the value and its dataset scope must be extracted from each pack before publication.

## Comparability gate

The current local Q1a rows have identical 2,000-example sample-ID and pragmatic-gold sequence hashes, and deterministic metric recomputation matches the recorded values. The extracted Q1b packs pin the three external split/gold hashes, explicit label routing, preprocessing, source checkpoint, prediction filenames, and metric files. The standard XLM-R versus primary parity script checks expected counts 1,609/3,166/693, nonempty unique IDs, explicit label vocabularies, ID/gold/raw-text/NFC equality, and recomputes every per-seed macro-F1 within `5e-11` of the stored values. The single-task PhoBERT row is an explicit same-seed composition of its polarity and emotion artifacts; it is not a fabricated single run. Independent reruns have not been performed, so these are `ANALYZED` artifact summaries rather than `VERIFIED` independent reruns.

The primary AIVIVN normalized-test manifest hash (`27DA59...`) differs from the standard-baseline/common external hash (`F6F6A7...`). The complete record-level comparison passes for every seed and dataset, so descriptive ordinary-F1 and dataset-wise comparisons are permitted on the verified same observed cohort. The unexplained manifest-byte difference remains a provenance limitation; it is not relabelled as formatting-only and does not block that descriptive comparison. No causal or statistical-significance claim follows from these rows.

The exact Sailor/Vistral closure, six source revisions, checkpoint hashes, metric-file hashes, per-seed values, and recomputed means/SDs are in `q1b_external_baseline_closure.md`.

## Process state at checkpoint

The live fetch completed successfully in 17.2 seconds (PowerShell command returned exit code 0; no active fetch process/session remains). No training was started. The next audit pass is read-only pack inspection and metric/provenance extraction.
