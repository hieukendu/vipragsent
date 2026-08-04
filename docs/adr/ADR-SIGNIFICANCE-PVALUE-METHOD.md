# ADR-SIGNIFICANCE-PVALUE-METHOD

## Status

CONFLICT, awaiting an explicit statistical protocol decision.

## Decision

The implementation exposes a versioned p-value strategy interface and refuses to label
significance results PASS while the exact two-sided empirical p-value definition and finite
resample correction are unspecified by the locked protocol.

The paired hierarchical resampling, percentile confidence intervals, fixed label spaces, and
Holm correction are implemented independently of this unresolved method.
