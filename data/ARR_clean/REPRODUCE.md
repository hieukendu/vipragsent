# Validate the retained experimental results

Keep the original dataset; no per-example experimental predictions are delivered.
Python and NumPy are needed for result validation. Dataset/hash verification uses
the standard library.

```bash
python -B verify_release.py
python -B validate_saved_results.py
python -B package_release.py
```

## Retained numbers

- `replay/AGGREGATE_RESULTS.json`: statistics derived from 235
  validated run/split outputs, with no sample IDs, text, gold arrays or
  per-example probabilities. Confusion matrices permit recalculating F1.
  Ten-bin counts, mean probabilities and positive rates permit recalculating ECE.
- Run manifests retain the paper's mean/SD targets and per-seed results.
  Mean uses three seeds; SD is sample SD (ddof=1), except single-run GPT rows.
- `replay/REPLAY_REPORT.json` and `reference/paired_bootstrap_ci.csv` retain
  the completed matched-pair analysis: 10,000 resamples, RNG seed 20260525.
  The aggregate gap is +0.630464 percentage points, nominal 95% CI
  [0.068308, 1.195883], p=0.0289971. All six head-specific CIs cross zero.
  These are saved results; the bootstrap is not reexecuted without paired records.
- `RESOURCE_USAGE.json` files retain measured compute and relative-cost results.
  Hardware measurements are validated numerically, not repeated.
- Threshold and recipe files retain frozen settings needed to compare future runs.

## Coverage and remaining verification boundary

Q1a's eight rows, all six Q1b rows, Q2's six variants, Q3's three models at six
budgets and Q4's supplied calibration results have paper-matched saved numbers.

All six Q1b rows have aggregate confusion matrices for every seed and external
test dataset. Their F1, mean and sample SD are independently recalculated and
match the paper. The Sailor/Vistral counts were checked against the fixed
external test IDs, text and gold before the temporary records were removed.

For Vistral Q3 budget 32, seed 20260521,
`Q3_reported/VISTRAL32_SELECTION_RESULTS.json` adds frozen thresholds and the
three-epoch history. Epoch 2 has the largest saved dev sarcasm F1
(0.7312263685361996). `VISTRAL32_THRESHOLD_SEARCH.json` retains confusion counts
at all 91 grid thresholds from the original selection cohort. The dev input
matches its frozen hash and released gold/IDs; independent threshold search
reproduces all six frozen thresholds, including sarcasm 0.20 and mocking 0.84.

Only aggregate counts and experimental settings are retained. No downloaded
per-example records, model checkpoints, author repository links or credentials
are included.

## Q3 input policy

Canonical `q3_masks/` now contains the paper-reported nested masks. Both sarcasm
and rationale losses are masked for out-of-budget positives; negatives retain
both losses. Other task targets stay active. Code preparation copies these
exact bytes to `data/processed/q3_low_resource_sarcasm/`.
Current input hashes lock the delivered files, not an unverifiable historical
source. Changing mask policy did not change text, gold labels or rationales.

The dataset card describes the current release. The supplied annotation rubric
does not prove a separate historical manual existed. The source receipt is
explicitly reconstructed, not original extraction proof. No unavailable
posthoc-note collection is claimed.

## Submission boundary

Only manifest-listed files enter the anonymous ZIP. Legacy/private directories
and Excel host metadata were removed from this folder. No training, API rerun, human annotation replay or
full-paper reproduction is claimed. Use `validate_saved_results.py` directly;
the obsolete prediction-replay compatibility entrypoint has been removed.
