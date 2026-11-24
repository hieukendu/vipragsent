# ViPragSent dataset and validated results

11,997 frozen comments: 7,998 train, 1,999 development and 2,000 test. Gold labels,
split membership and training rationales are unchanged. `annotations/` contains
two complete first-pass annotation CSVs. `q3_masks/` contains the six historical
paper-cohort masks: rationale loss on all 545 positives at every budget, zero on
7,453 negatives; only direct sarcasm supervision follows the positive budget.

`replay/AGGREGATE_RESULTS.json` contains 257 aggregate run/split outputs, including
confusion counts and calibration bins. `TABLE_BINDINGS.json` binds the supplied
PDF to every row of Tables 1-4. The former follow-up Q2 cohort, old Sailor/Vistral
cohort and old XLM-R Q3 cohort are removed. No downloaded prediction records or
checkpoint weights are retained. Training rationales are required runtime inputs.

```bash
python -B verify_release.py
python -B validate_saved_results.py
python -B package_release.py
```

Validation is read-only and checks release hashes, content and aggregate
arithmetic. It does not rerun training, API inference or bootstrap. After an
intentional release edit, refresh hashes before verification using
`python -B refresh_manifest.py`. `submission/` contains generated data archives
and is excluded from the release manifest.

`AVAILABILITY.json` is bound to PDF version 7 and records zero historical posthoc
notes available in this release. The user confirmed the PDF's historical posthoc
descriptions and reported losing the supporting material. Zero available notes
does not mean notes never existed; the historical description is user-reported
and cannot be independently verified from the released artifacts. The paper
wording is retained. See `code/docs/POSTHOC_EVIDENCE_SCOPE.md`. The operational
annotation rubric and reconstructed source receipt
do not replace historical evidence. `DATASET_CARD.md` and `REPRODUCE.md` describe
the data and verification limits. See `LICENSE` for reuse terms.
