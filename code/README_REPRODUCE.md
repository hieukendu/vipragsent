# Reproduce and validate ViPragSent

Use Python 3.11/3.12. GPU runners require Linux/WSL2 and CUDA; offline validators
and CPU tests run on Windows. Install `.[dev,data,viz,core,ml,azure,server]` and
`requirements-paper.txt` as appropriate for the execution host.

## Dataset and layout

Workspace: `code/` with `paper/ARR_clean/`. Extracted archives: sibling `code/`
and `ARR_clean/`. Commands resolve these layouts automatically; `--release`
can override the location. The older `data/ARR_clean/` tree is not an automatic
fallback: it may contain a superseded cohort or incompatible Q3 masks.

```bash
cd code
python -B prepare_inputs.py
python -B scripts/analyze_annotation_agreement.py
python -B -m pytest tests -p no:cacheprovider --basetemp=.work/pytest_tmp
```

Preparation checks all frozen payloads and current manifest hashes, then runs
both release validators. It copies labels, splits, rationales and masks unchanged
into local runtime `data/`. Conflicting existing runtime files are not overwritten.
Use `--output-root <checkout>` for a separate execution tree. `--external-root`
imports separately obtained, authorized external test splits; they are excluded
from the anonymous archives.

## Paper cohorts

Repeat GPU commands for seeds 20260521, 20260522 and 20260523. Table 1 Full and
Table 3 share the primary recipe: cosine schedule, warmup 0.20, patience 10.

```bash
python scripts/run_priority12_xlmr.py --experiment priority1 --variant full --seed 20260521
python scripts/run_matched_xlmr_ft_q1a.py --seed 20260521
# Table 3: all eight variants over all three seeds.
python scripts/run_priority12_xlmr.py --run-all
# Or select no_emotion_auxiliary, no_polarity_auxiliary, no_rationale,
# no_uncertainty_weighting, all_multipliers_1, beta_01 or beta_05 explicitly.
```

Table 3 ECE is six-head pragmatic test ECE on raw positive probabilities, in
[0,1]. The obsolete six-variant follow-up cohort and polarity-dev ECE targets
are removed. Table 4 uses linear schedule, warmup 0.10, patience 2 and best dev
sarcasm binary macro-F1 selection. At every budget, rationale loss is active on
all 545 positive rows and zero on 7,453 negatives. Out-of-budget positives mask
only direct sarcasm supervision; all other tasks remain active. The six released
mask hashes match all 18 historical corrected XLM-R run manifests.

```bash
python scripts/run_xlmr_followup.py --run-id xlmr_followup_q3_full_32_20260521
python scripts/run_xlmr_followup.py --list
python scripts/run_single_experiment.py --experiment-id <EXPERIMENT_ID> --dry-run
```

Generic baseline runners use their family-specific recipes. Models and execution
host preflight reports must be prepared separately. PhoBERT requires Java and
VnCoreNLP. Azure credentials and service configuration stay local. The GPT table
contains three inference outputs per prompt; the original run has no recorded
API seed. Run indices must not be interpreted as three training seeds.

```bash
python scripts/run_single_azure_job.py --job-id azure_pragmatic_zero_shot --inference-run-index 1 --dry-run
python scripts/run_single_azure_job.py --job-id azure_pragmatic_zero_shot --inference-run-index 2 --dry-run
python scripts/run_single_azure_job.py --job-id azure_pragmatic_zero_shot --inference-run-index 3 --dry-run
# Repeat for --job-id azure_gpt41_mini_8shot.
```

Each index has a separate output directory and API cache. The runner currently
does not request an API seed, and records this explicitly. Historical second
and third output seed metadata remain in the saved aggregates. Remove `--dry-run`
only when actually executing on a configured Azure account.

## Offline validation and exports

```bash
python -B ../ARR_clean/verify_release.py
python -B ../ARR_clean/validate_saved_results.py
python -B build_submission.py
python -B verify_submission.py
```

In this workspace replace `../ARR_clean` with `../paper/ARR_clean`, or invoke
`prepare_inputs.py`, which resolves the location. The result supplement retains
257 aggregate run/split outputs, confusion counts, calibration bins, table
bindings and saved confidence intervals. It validates all rows of Tables 1-4.
Seven saved paired CIs are checked against their reference; bootstrap is not
resampled. Model tensors and per-example predictions are not shipped.

Packaging verifies the actual ZIP payloads and rewrites `FINAL_VERIFICATION.json`
from their current hashes and extracted validator output. `verify_submission.py`
independently checks both ZIPs against that receipt and the current source tree,
including availability, every Q3 mask and the primary Q2 cohort. A changed ZIP
fails verification rather than inheriting an earlier PASS receipt.

The operational rubric is `docs/annotation_rubric.md`; two first-pass annotation
CSVs are included. It is not an original annotation replay. The reference is PDF
version 7: Appendix B claims 17 notes and section 2.3 mentions existing posthoc
notes. The user confirmed both historical descriptions and reported losing the
supporting information/material. The release includes zero available notes;
this is an availability limitation, not evidence that notes never existed.
Recovery remains waived and no replacement notes or statistics are invented.
See `AVAILABILITY.json` and `docs/POSTHOC_EVIDENCE_SCOPE.md`. The paper wording
is retained; the historical claim is not independently verified from artifacts.
Source extraction receipts remain explicitly reconstructed. Archive validation
also requires availability and numerical table bindings to refer to the same PDF.
