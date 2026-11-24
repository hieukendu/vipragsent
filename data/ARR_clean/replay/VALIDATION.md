## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-09
- Verification Status: ANALYZED
- Version Label: aggregate_results_v2

## Validation report

Dataset text, gold and rationales are unchanged. Experimental records were converted to aggregate
statistics and removed by request. See `VALIDATION_REPORT.json` for fresh
aggregate checks and `REPRODUCE.md` for precise scope.

F1 is recalculated from aggregate confusion counts; ECE is recalculated from
calibration bins. Paper mean/sample SD and compute ratios are checked.
Saved CI values agree with the retained reference, but bootstrap resampling is
not repeated. No training, API rerun or full reproduction is claimed.

All six Q1b systems have independently validated confusion counts for three
seeds and three external tests. Vistral Q3 budget 32, seed 20260521 has independent
threshold reselection from original dev records, checked against their freeze
hash and released dev gold. Search counts retain all 91 thresholds per head.

## Statistical fallacy scan

Coverage: 11/11 checked; aggregate evidence limits some assessments.

| Check | Finding |
| --- | --- |
| Simpson's paradox | Per-seed/head summaries retained; no unreported subgroup reversal established. |
| Ecological fallacy | Aggregate metrics are not individual-level intent claims. |
| Berkson's paradox | Original extraction/sampling not replayed. |
| Collider bias | No causal adjustment claim evaluated. |
| Base rate neglect | Dataset prevalence remains available; both-class macro-F1 used. |
| Regression to the mean | Not a pre/post intervention comparison. |
| Survivorship bias | Three-seed cohorts required; no incomplete cohort replaces a paper row. |
| Look-elsewhere effect | Head-level CIs are nominal, without multiplicity-adjusted claims. |
| Garden of forking paths | Selection remains conditional; the original Vistral32 dev search is independently validated. |
| Correlation versus causation | Ablations do not establish unrestricted causal attribution. |
| Reverse causality | No population causal direction inferred. |
