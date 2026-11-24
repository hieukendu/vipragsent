# ViPragSent: dataset and experimental results

The original text and gold labels are retained unchanged: 11,997 examples split into
7,998 train / 1,999 dev / 2,000 test, annotations, training rationales,
label schema, loss weights and paper-reported Q3 masks. Out-of-budget positive
sarcasm targets and their rationale losses are masked; negatives remain active.

The experimental supplement now retains aggregate results only:
per-seed F1 and confusion counts, mean/sample SD, calibration bins,
saved CI/p-values, thresholds, run settings and compute measurements.
Added per-example predictions and external gold-only CSVs were removed
by request. Recovered Q3 dev records were processed only in RAM; no additional
per-example prediction files were saved or retained.

```bash
python -B verify_release.py
python -B validate_saved_results.py
python -B package_release.py
```

See `REPRODUCE.md` for validation scope. Aggregate validation is not a
fresh training or bootstrap run. Original Vistral Q3 budget-32 dev threshold
selection is independently validated from retained grid counts.

Private Excel workbooks, legacy duplicates and recovery/source plans are not
part of this folder. `submission/` contains only generated anonymous exports;
it is excluded from the manifest to avoid recursively packaging the ZIP.
After intentional release edits, regenerate current hashes with
`python -B refresh_manifest.py`, then run both validators before exporting.
License: see `LICENSE`.
