> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 01 — INGEST AND VALIDATE VIPRAGSENT

Extract, validate, and freeze the V8 dataset package.

Required work:

1. Verify the ZIP checksum and all internal checksums.
2. Extract to `data/raw/vipragsent_package/`.
3. Place model-ready data under `data/processed/vipragsent/`.
4. Verify total count, frozen split sizes, unique IDs, no missing text, label schemas, unchanged split manifest, Q3 nested masks, and actual train-set sarcasm positives.
5. Recompute pre-adjudication human agreement: raw agreement, Cohen's kappa, and nominal Krippendorff's alpha.
6. Compare recomputed values with bundled reports.
7. Create an immutable data manifest and fingerprint.
8. Write schema, split, and IAA tests.

Locked duplicate policy: do not deduplicate, do not remove near-overlaps, preserve emoji/punctuation variants, and never use raw reviewer notes as model features.

Do not download model weights, train models, or perform full rationale generation.


# RATIONALE TEMPLATE SANITIZATION

Inspect the bundled rationale-generation JSONL and prompt templates for legacy generator placeholders.
Create a new active input manifest under:

```text
data/processed/rationales/azure_rationale_input_train.jsonl
```

Preserve sample IDs, comments, and gold labels, but remove placeholder generator metadata.
Do not edit the original archived files. Record the transformation hash and source checksum.
The active file must use the canonical label keys defined in the global contract.
