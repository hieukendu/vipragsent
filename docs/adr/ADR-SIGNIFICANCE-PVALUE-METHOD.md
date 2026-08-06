# ADR-SIGNIFICANCE-PVALUE-METHOD

## Status

RESOLVED by the sequential experiment protocol approval.

## Decision

Use `paired_hierarchical_bootstrap_sign_plus_one_v1` with left-minus-right differences,
1,000 resamples, bootstrap seed `20260525`, percentile 95% intervals, and plus-one
finite-resample correction. The raw two-sided p-value is `min(1, 2 * min(p_lower,
p_upper))`, where `p_lower = (1 + count(delta <= 0)) / (B + 1)` and
`p_upper = (1 + count(delta >= 0)) / (B + 1)`. Apply Holm correction within each
seven-metric family.
