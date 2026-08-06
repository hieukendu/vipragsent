> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 08 — IMPLEMENT EVALUATION AND STATISTICS

Implement every metric and statistical procedure before real predictions exist.

Required metrics:

- binary macro-F1 for each pragmatic label;
- macro-pragmatic F1 as the arithmetic mean of the six label-level F1 values;
- polarity macro-F1;
- emotion macro-F1;
- `Ord.F1` as the arithmetic mean of the three external benchmark macro-F1 scores;
- per-label threshold selection;
- top-label ECE;
- reliability-bin export;
- learning-curve aggregation.

Required statistics:

- paired hierarchical bootstrap for trainable models;
- bootstrap over examples only for Azure outputs;
- 1,000 resamples;
- paired model comparisons;
- missing-prediction and failed-request accounting.

Use synthetic fixtures with exact expected values for F1, threshold selection, ECE, bootstrap reproducibility, paired alignment, and missing-prediction handling.

Do not run real models and do not hard-code draft results.


# METRIC EDGE-CASE AND BOOTSTRAP CONTRACT

Binary macro-F1 is the arithmetic mean of class-0 F1 and class-1 F1 with `zero_division=0`.
Preserve a fixed sample-ID ordering across systems before paired computations.

For trainable-system confidence intervals and differences:

1. Sample training-seed indices with replacement.
2. Sample test-example indices with replacement.
3. Apply the same sampled example indices to every compared system.
4. Compute the metric for each sampled seed run, then average over sampled seed runs.
5. Use percentile 2.5% and 97.5% bounds.

For Azure outputs, perform step 2 only.
If a bootstrap resample contains only one class for a binary label, compute using the fixed two-class label set and
`zero_division=0`; do not drop the resample.
