# Significance method resolution

Status: `RESOLVED`

The locked method is `paired_hierarchical_bootstrap_sign_plus_one_v1`, with left-minus-right differences, 1,000 paired hierarchical resamples, bootstrap seed `20260525`, percentile 95% confidence intervals, and Holm correction within each seven-metric family.

The two-sided finite-resample p-value is `min(1, 2 * min(p_lower, p_upper))`, where `p_lower = (1 + count(delta <= 0)) / (B + 1)` and `p_upper = (1 + count(delta >= 0)) / (B + 1)`. Trainable systems share sampled seed and example indices; Azure remains one fixed-prompt prediction vector and never receives fabricated training seeds.
