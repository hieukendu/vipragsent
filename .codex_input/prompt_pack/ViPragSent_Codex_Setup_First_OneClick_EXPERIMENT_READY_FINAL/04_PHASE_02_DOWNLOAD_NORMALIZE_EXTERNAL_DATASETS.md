> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 02 — DOWNLOAD AND NORMALIZE EXTERNAL DATASETS

Complete all external-dataset setup in this phase.

## UIT-VSFC

- Prefer the official UIT source or official UIT Hugging Face organization.
- Verify the official test split, fields, and negative/neutral/positive mapping.
- Normalize to `sample_id,text,polarity`.
- Record source, revision, access terms, license note, and checksum.

## UIT-VSMEC

- Prefer the official UIT source.
- Verify the official test split and seven emotion labels.
- Do not create a random split from an unsplit mirror and call it official.
- Normalize to `sample_id,text,emotion`.
- Record provenance and checksum.

## AIVIVN original

Download Kaggle slug `mcocoz/aivivn-2019`. Preserve original binary files for provenance. Do not replace the bundled `AIVIVN-human-derived-3way`; Q1b uses the bundled three-way test split.

Create idempotent download, normalization, and validation scripts with `--dry-run`, retry handling, checksum verification, and safe overwrite behavior. Create `data/manifests/external_datasets.json`.

Expected normalized outputs:

```text
data/processed/external/uit_vsfc/test.csv
data/processed/external/uit_vsmec/test.csv
data/processed/external/aivivn_human_derived_3way/test.csv
```

Do not download model weights or train models.


# LICENSE-AWARE MANUAL-DROP FALLBACK

If an official dataset requires a user agreement, authentication, or manual download:

1. Do not substitute an unofficial split silently.
2. Create `data/external/manual_drop/<dataset>/README.md` with exact placement instructions.
3. Mark the phase `BLOCKED` until the required files are supplied.
4. After placement, verify checksums, schema, label mapping, and official split metadata.
5. Do not include restricted original text in a public release ZIP unless redistribution is explicitly permitted.

Kaggle AIVIVN download must support credentials through the standard Kaggle environment or configuration file,
without storing credentials in the repository.
