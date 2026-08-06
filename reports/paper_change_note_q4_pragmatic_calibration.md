# Paper change note: Q4 pragmatic calibration

This note records a protocol change for a later manuscript update. The manuscript itself is not edited by this task.

- The old Q4 claim concerned calibration of the intended-polarity head.
- The new Q4 concerns calibration of the six pragmatic predictions and their dev-set learning dynamics.
- Q4 no longer supports a polarity-calibration claim.
- Table 3 polarity performance remains a separate cross-domain retention result.
- Any result claim remains conditional until the real sequential experiments have run and received explicit user approval.

## Later manuscript updates

Update the Q4 research-question paragraph, the calibration-methods subsection, the Q4 results subsection, the figure captions, and the limitations/discussion text that referred to polarity calibration. Preserve the separate Table 3 polarity-performance section.

## Artifact references

Use the following frozen artifact paths after Phase 16:

- `experiment_artifacts/tables/q4_pragmatic_calibration_per_seed.csv`
- `experiment_artifacts/tables/q4_pragmatic_calibration_summary.csv`
- `experiment_artifacts/backing_data/q4_pragmatic_reliability_bins.csv`
- `experiment_artifacts/backing_data/q4_learning_curves.csv`
- `experiment_artifacts/figures/q4_pragmatic_ece_heatmap.pdf`
- `experiment_artifacts/figures/q4_pragmatic_reliability_by_label.pdf`
- `experiment_artifacts/figures/q4_learning_curves.pdf`

The human-readable report must describe raw positive-class probabilities, ten equal-width bins, no temperature scaling, independent per-seed computation, and arithmetic mean/sample standard deviation aggregation.
