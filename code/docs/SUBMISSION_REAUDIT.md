# Submission re-audit resolution — 2026-10-10

The current numerical reference is the supplied 19-page PDF version 7, SHA-256
`cf9c2b7a8f328edbfdc4d5cb486584d3bf7b2426294438b0a16f5a0bd23c0dfe`.
`AVAILABILITY.json`, `TABLE_BINDINGS.json` and the generated verification receipts
refer to this PDF. Numerical table targets are unchanged; the old PDF is used
only for an explicitly historical comparison in `PDF_NUMERIC_REFERENCE_CHECK.json`.
The incoming audit's separate PDF and its layout findings are not certified by
this metadata correction. The PDF and manuscript remain read-only.

## Resolved findings

| Finding | Resolution and evidence |
| --- | --- |
| Seven absent frozen payloads | Restored exact locally available train/full data, both annotation CSVs, both rationale JSONLs and aggregate results. All 11,997 gold/text rows and split assignments validate. |
| Q3 rationale policy | Restored original corrected mask CSVs. At each budget, all 545 positive rows have rationale loss and all 7,453 negative rows have zero rationale loss. Out-of-budget positives retain rationale supervision. Six LF-serialized mask hashes match all 18 historical corrected XLM-R run manifests. Code, preflight defaults, input preparation and verifier enforce this policy. |
| Q2 cohort | Removed the obsolete follow-up result cohort. Eight primary variants, three seeds each, use the primary cosine/warmup 0.20/patience 10 recipe. Table 3 ECE is pragmatic test ECE, averaged across six heads. The dedicated runner queues all 24 runs. |
| Sailor/Vistral binding | Recovered the original three-seed Table 1 cohorts: 85.56 ± 0.10 and 87.67 ± 0.06. Removed the later mismatched cohorts. |
| GPT output replication | Recovered three inference outputs per prompt: zero-shot 54.16 ± 0.26 and eight-shot 53.92 ± 0.16. Recomputed missing metrics from records. The original output has no recorded API seed; output indices are not training seeds. The runner supports three separate output/cache directories. |
| Other table rows | Retained/recovered ViSoBERT, historical XLM-R comparison, all six external systems and all three budgeted families. Tables 1–4 pass saved-count mean/sample-SD validation. |
| Availability | Both first-pass CSVs and final gold are present; zero historical posthoc notes are available for release. The user confirmed both PDF 7 historical descriptions and reported losing the supporting information/material. `AVAILABILITY.json` and `POSTHOC_EVIDENCE_SCOPE.md` record current availability and the recovery waiver without asserting nonexistence. The PDF wording is retained; the historical claim is user-reported and not independently verified from available artifacts. |
| Layout and packaging | Default resolution supports this workspace and extracted sibling `code/` / `ARR_clean/` folders. Software and dataset export separately, with CRC, exact-member, SHA-256 and anonymity readback checks. |

## Evidence and validation boundaries

Read-only authenticated Hub audit covered all 31 owned repositories (30 model,
one dataset), 418,362 recursively paginated file names and 563 bounded unique
payloads at pinned revisions. Small payload sizes and Git blob hashes were
checked. No checkpoint weights were downloaded. This is a complete file
inventory with relevant content reads; it is not a download of every tensor.
Anonymous inventory receipts are in `replay/HUB_AUDIT.json`; exact source
locations are excluded from the submission packages.

`validate_saved_results.py` checks 257 aggregate run/split outputs, 36 table
groups plus six Table 2 external rows, frozen gold marginals, F1 from confusion
counts and primary ECE from saved bins. Seven saved bootstrap intervals agree
with their reference. The original Vistral budget-32 threshold grid reselects
six frozen development thresholds independently. It does not reconstruct
checkpoint tensors, annotation decisions, original extraction receipts or
individual paired predictions. Historical comparison names are numerical table
bindings; model architecture lineage is not newly certified by count arithmetic.

First-pass agreement was recomputed from both annotation CSVs: pragmatic nominal
alpha mean 0.81724856, polarity kappa 0.85448058 and emotion kappa 0.75927297.
All published rounding checks pass. Gold labels, split membership and required
training rationales were preserved. Two previously flagged mention candidates
remain in frozen dev/test text and are documented in `PRIVACY_REVIEW.json`;
the release does not claim all user identifiers were removed.

Validation status is **ANALYZED / PASS_AGGREGATE_VALIDATION**. Training, live API
inference and bootstrap resampling were not rerun. Administrative submission
requirements and direct PDF/manuscript edits are outside the authorized edit
scope. Release metadata now refers to PDF version 7 and distinguishes its
user-confirmed historical descriptions from current release availability.
No change to the two posthoc passages is requested after the user's clarification.
The recovery waiver removes the missing-note recovery task from this scope.

## Clean release structure

The active supplement contains frozen inputs, six masks, annotation/data
documentation, manifest/checksums, read-only validators and compact aggregate
evidence. Removed 264 obsolete or redundant replay files; numerical targets
are consolidated in `TABLE_BINDINGS.json`. Downloaded records, inventory scratch
files and test extraction trees are deleted after validation. Prepared `code/data`
contains necessary execution inputs and is excluded from the software archive.
Private recovery snapshots are excluded from both archives.

To validate from either supported layout:

```bash
cd code
python -B prepare_inputs.py
python -B scripts/analyze_annotation_agreement.py
python -B -m pytest tests -p no:cacheprovider --basetemp=.work/pytest_tmp
python -B build_submission.py
```

The package report records archive/member checksums and result-verification
limits. `build_submission.py` verifies the actual software/data ZIPs before
writing all three package receipts. `verify_submission.py` checks receipt hashes,
exact current source/data bytes, availability, all six Q3 masks and Q2 Full
93.67 against extracted payloads; stale receipts or hash-consistent incorrect
masks fail. Independent full test and authorized-write-boundary evidence is
retained beside the archives.

The canonical workspace dataset is `paper/ARR_clean/`. A superseded
`data/ARR_clean/` tree remains outside the authorized edit scope and is no longer
an automatic fallback. Use the data ZIP under the canonical folder's `submission/`.
Authenticated repository discovery was refreshed: all 31 revisions remain
unchanged, so their complete pinned inventories of 418,362 files remain applicable.
