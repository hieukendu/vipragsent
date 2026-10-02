# Priority 3 — Positive-class performance analysis

Status: COMPLETE on frozen test predictions; no model was retrained and no seed was outcome-selected.

## Scope and protocol

This analysis addresses the rare-positive limitation by reporting positive precision, positive recall, positive F1, and TP/FP/FN/TN for all six pragmatic labels. The comparison is the frozen XLM-R-large multi-task Q1a system minus the frozen matched XLM-R-large pragmatic-only system.

Seeds: 20260521, 20260522, 20260523. Test rows per seed: 2000. Thresholds were already frozen from dev and were not re-tuned on test. Paired bootstrap: 10,000 resamples with seed 20260525; intervals are 95% percentage-point intervals.

Expected sarcasm/mocking gains from the limitation plan are interpretation targets only. All three scientifically valid seeds are retained regardless of whether a target is met.

## Frozen-source validation

- Proposed source: `reports/q1a_matched_xlmr_ft/proposed_source/.../q1a_vipragsent_full_XLM_R_large_optimization_...`.
- Baseline source: `reports/q1a_matched_xlmr_ft/seed_<seed>/` matched pragmatic-only runs.
- Every seed has 1,999 dev rows and 2,000 test rows; IDs and six-label gold values are paired across systems and seeds.
- Every recorded test prediction matches its frozen dev-selected threshold; this script never changes a prediction.

## Per-seed positive metrics

Values are fractions for precision/recall/F1 and exact counts for confusion fields. The complete six-label table is in `per_seed_metrics.csv`.

| Label | Model | Seed | Precision+ | Recall+ | F1+ | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| implicit_sentiment | proposed_multitask | 20260521 | 0.877030 | 0.900000 | 0.888367 | 378 | 53 | 42 | 1527 |
| implicit_sentiment | proposed_multitask | 20260522 | 0.884615 | 0.876190 | 0.880383 | 368 | 48 | 52 | 1532 |
| implicit_sentiment | proposed_multitask | 20260523 | 0.889157 | 0.878571 | 0.883832 | 369 | 46 | 51 | 1534 |
| implicit_sentiment | baseline_pragmatic_only | 20260521 | 0.855204 | 0.900000 | 0.877030 | 378 | 64 | 42 | 1516 |
| implicit_sentiment | baseline_pragmatic_only | 20260522 | 0.865741 | 0.890476 | 0.877934 | 374 | 58 | 46 | 1522 |
| implicit_sentiment | baseline_pragmatic_only | 20260523 | 0.884337 | 0.873810 | 0.879042 | 367 | 48 | 53 | 1532 |
| sarcasm | proposed_multitask | 20260521 | 0.746667 | 0.823529 | 0.783217 | 112 | 38 | 24 | 1826 |
| sarcasm | proposed_multitask | 20260522 | 0.782313 | 0.845588 | 0.812721 | 115 | 32 | 21 | 1832 |
| sarcasm | proposed_multitask | 20260523 | 0.773333 | 0.852941 | 0.811189 | 116 | 34 | 20 | 1830 |
| sarcasm | baseline_pragmatic_only | 20260521 | 0.729560 | 0.852941 | 0.786441 | 116 | 43 | 20 | 1821 |
| sarcasm | baseline_pragmatic_only | 20260522 | 0.765517 | 0.816176 | 0.790036 | 111 | 34 | 25 | 1830 |
| sarcasm | baseline_pragmatic_only | 20260523 | 0.773723 | 0.779412 | 0.776557 | 106 | 31 | 30 | 1833 |
| irony | proposed_multitask | 20260521 | 0.977941 | 0.956835 | 0.967273 | 133 | 3 | 6 | 1858 |
| irony | proposed_multitask | 20260522 | 0.992481 | 0.949640 | 0.970588 | 132 | 1 | 7 | 1860 |
| irony | proposed_multitask | 20260523 | 0.985294 | 0.964029 | 0.974545 | 134 | 2 | 5 | 1859 |
| irony | baseline_pragmatic_only | 20260521 | 0.964029 | 0.964029 | 0.964029 | 134 | 5 | 5 | 1856 |
| irony | baseline_pragmatic_only | 20260522 | 0.970803 | 0.956835 | 0.963768 | 133 | 4 | 6 | 1857 |
| irony | baseline_pragmatic_only | 20260523 | 0.985185 | 0.956835 | 0.970803 | 133 | 2 | 6 | 1859 |
| idiom_figurative | proposed_multitask | 20260521 | 0.909774 | 0.858156 | 0.883212 | 121 | 12 | 20 | 1847 |
| idiom_figurative | proposed_multitask | 20260522 | 0.909774 | 0.858156 | 0.883212 | 121 | 12 | 20 | 1847 |
| idiom_figurative | proposed_multitask | 20260523 | 0.859155 | 0.865248 | 0.862191 | 122 | 20 | 19 | 1839 |
| idiom_figurative | baseline_pragmatic_only | 20260521 | 0.870504 | 0.858156 | 0.864286 | 121 | 18 | 20 | 1841 |
| idiom_figurative | baseline_pragmatic_only | 20260522 | 0.900763 | 0.836879 | 0.867647 | 118 | 13 | 23 | 1846 |
| idiom_figurative | baseline_pragmatic_only | 20260523 | 0.857143 | 0.851064 | 0.854093 | 120 | 20 | 21 | 1839 |
| code_switching | proposed_multitask | 20260521 | 0.931034 | 0.894040 | 0.912162 | 270 | 20 | 32 | 1678 |
| code_switching | proposed_multitask | 20260522 | 0.926421 | 0.917219 | 0.921797 | 277 | 22 | 25 | 1676 |
| code_switching | proposed_multitask | 20260523 | 0.898361 | 0.907285 | 0.902801 | 274 | 31 | 28 | 1667 |
| code_switching | baseline_pragmatic_only | 20260521 | 0.896104 | 0.913907 | 0.904918 | 276 | 32 | 26 | 1666 |
| code_switching | baseline_pragmatic_only | 20260522 | 0.903333 | 0.897351 | 0.900332 | 271 | 29 | 31 | 1669 |
| code_switching | baseline_pragmatic_only | 20260523 | 0.915541 | 0.897351 | 0.906355 | 271 | 25 | 31 | 1673 |
| mocking | proposed_multitask | 20260521 | 0.918919 | 0.875536 | 0.896703 | 204 | 18 | 29 | 1749 |
| mocking | proposed_multitask | 20260522 | 0.896861 | 0.858369 | 0.877193 | 200 | 23 | 33 | 1744 |
| mocking | proposed_multitask | 20260523 | 0.861925 | 0.884120 | 0.872881 | 206 | 33 | 27 | 1734 |
| mocking | baseline_pragmatic_only | 20260521 | 0.902778 | 0.836910 | 0.868597 | 195 | 21 | 38 | 1746 |
| mocking | baseline_pragmatic_only | 20260522 | 0.885321 | 0.828326 | 0.855876 | 193 | 25 | 40 | 1742 |
| mocking | baseline_pragmatic_only | 20260523 | 0.885463 | 0.862661 | 0.873913 | 201 | 26 | 32 | 1741 |

## Mean ± SD across seeds

| Label | Model | Precision+ | Recall+ | F1+ | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| implicit_sentiment | proposed_multitask | 0.883601 ± 0.006127 | 0.884921 ± 0.013113 | 0.884194 ± 0.004004 | 371.67 ± 5.51 | 49.00 ± 3.61 | 48.33 ± 5.51 | 1531.00 ± 3.61 |
| implicit_sentiment | baseline_pragmatic_only | 0.868427 ± 0.014751 | 0.888095 ± 0.013257 | 0.878002 ± 0.001008 | 373.00 ± 5.57 | 56.67 ± 8.08 | 47.00 ± 5.57 | 1523.33 ± 8.08 |
| sarcasm | proposed_multitask | 0.767438 ± 0.018540 | 0.840686 ± 0.015306 | 0.802375 ± 0.016610 | 114.33 ± 2.08 | 34.67 ± 3.06 | 21.67 ± 2.08 | 1829.33 ± 3.06 |
| sarcasm | baseline_pragmatic_only | 0.756267 ± 0.023490 | 0.816176 ± 0.036765 | 0.784344 ± 0.006980 | 111.00 ± 5.00 | 36.00 ± 6.24 | 25.00 ± 5.00 | 1828.00 ± 6.24 |
| irony | proposed_multitask | 0.985239 ± 0.007270 | 0.956835 ± 0.007194 | 0.970802 ± 0.003641 | 133.00 ± 1.00 | 2.00 ± 1.00 | 6.00 ± 1.00 | 1859.00 ± 1.00 |
| irony | baseline_pragmatic_only | 0.973339 ± 0.010804 | 0.959233 ± 0.004154 | 0.966200 ± 0.003988 | 133.33 ± 0.58 | 3.67 ± 1.53 | 5.67 ± 0.58 | 1857.33 ± 1.53 |
| idiom_figurative | proposed_multitask | 0.892901 ± 0.029225 | 0.860520 ± 0.004095 | 0.876205 ± 0.012136 | 121.33 ± 0.58 | 14.67 ± 4.62 | 19.67 ± 0.58 | 1844.33 ± 4.62 |
| idiom_figurative | baseline_pragmatic_only | 0.876137 ± 0.022349 | 0.848700 ± 0.010834 | 0.862008 ± 0.007058 | 119.67 ± 1.53 | 17.00 ± 3.61 | 21.33 ± 1.53 | 1842.00 ± 3.61 |
| code_switching | proposed_multitask | 0.918606 ± 0.017684 | 0.906181 ± 0.011629 | 0.912253 ± 0.009499 | 273.67 ± 3.51 | 24.33 ± 5.86 | 28.33 ± 3.51 | 1673.67 ± 5.86 |
| code_switching | baseline_pragmatic_only | 0.904993 ± 0.009824 | 0.902870 ± 0.009559 | 0.903868 ± 0.003145 | 272.67 ± 2.89 | 28.67 ± 3.51 | 29.33 ± 2.89 | 1669.33 ± 3.51 |
| mocking | proposed_multitask | 0.892568 ± 0.028739 | 0.872675 ± 0.013112 | 0.882259 ± 0.012693 | 203.33 ± 3.06 | 24.67 ± 7.64 | 29.67 ± 3.06 | 1742.33 ± 7.64 |
| mocking | baseline_pragmatic_only | 0.891187 ± 0.010038 | 0.842632 ± 0.017868 | 0.866129 ± 0.009268 | 196.33 ± 4.16 | 24.00 ± 2.65 | 36.67 ± 4.16 | 1743.00 ± 2.65 |

## Paired proposed-minus-baseline confidence intervals

Positive metrics are shown in percentage points. `CI excludes zero` is the requested paired 95% evidence flag; it is descriptive and was not used to remove or rerun a seed.

| Label | Metric | Seed deltas (pp) | Mean delta (pp) | 95% CI (pp) | CI excludes zero | Seeds favoring proposed |
| --- | --- | --- | ---: | --- | --- | ---: |
| implicit_sentiment | positive_precision | +2.183, +1.887, +0.482 | +1.517349 | [-0.672462, +3.729328] | False | 3/3 |
| implicit_sentiment | positive_recall | +0.000, -1.429, +0.476 | -0.317460 | [-2.722812, +1.938924] | False | 1/3 |
| implicit_sentiment | positive_f1 | +1.134, +0.245, +0.479 | +0.619180 | [-1.043547, +2.233160] | False | 3/3 |
| sarcasm | positive_precision | +1.711, +1.680, -0.039 | +1.117110 | [-2.720893, +5.081740] | False | 2/3 |
| sarcasm | positive_recall | -2.941, +2.941, +7.353 | +2.450980 | [-4.098466, +9.160305] | False | 2/3 |
| sarcasm | positive_f1 | -0.322, +2.269, +3.463 | +1.803113 | [-2.027997, +5.606050] | False | 2/3 |
| irony | positive_precision | +1.391, +2.168, +0.011 | +1.189987 | [-0.763270, +3.589560] | False | 3/3 |
| irony | positive_recall | -0.719, -0.719, +0.719 | -0.239808 | [-1.587302, +0.938967] | False | 1/3 |
| irony | positive_f1 | +0.324, +0.682, +0.374 | +0.460220 | [-0.558367, +1.623189] | False | 3/3 |
| idiom_figurative | positive_precision | +3.927, +0.901, +0.201 | +1.676466 | [-1.958282, +5.554824] | False | 3/3 |
| idiom_figurative | positive_recall | +0.000, +2.128, +1.418 | +1.182033 | [-1.822917, +4.197531] | False | 2/3 |
| idiom_figurative | positive_f1 | +1.893, +1.556, +0.810 | +1.419629 | [-0.875556, +3.768841] | False | 3/3 |
| code_switching | positive_precision | +3.493, +2.309, -1.718 | +1.361292 | [-2.022284, +4.613848] | False | 2/3 |
| code_switching | positive_recall | -1.987, +1.987, +0.993 | +0.331126 | [-2.222222, +3.000000] | False | 2/3 |
| code_switching | positive_f1 | +0.724, +2.146, -0.355 | +0.838502 | [-0.920196, +2.767610] | False | 2/3 |
| mocking | positive_precision | +1.614, +1.154, -2.354 | +0.138105 | [-3.608456, +3.780241] | False | 2/3 |
| mocking | positive_recall | +3.863, +3.004, +2.146 | +3.004292 | [+0.129604, +5.987148] | True | 3/3 |
| mocking | positive_f1 | +2.811, +2.132, -0.103 | +1.613063 | [-0.944505, +4.151079] | False | 2/3 |

## Sarcasm and mocking interpretation checks

- **sarcasm**: positive-F1 change +1.803 pp; 95% CI [-2.028, +5.606] pp; recall change +2.451 pp; precision change +1.117 pp; proposed wins 2/3 seeds.
- **mocking**: positive-F1 change +1.613 pp; 95% CI [-0.945, +4.151] pp; recall change +3.004 pp; precision change +0.138 pp; proposed wins 2/3 seeds.

## Reproducibility

- `per_seed_metrics.csv`: all requested per-seed metrics and confusion counts.
- `aggregate_metrics.csv`: mean and sample SD across the three seeds.
- `paired_confidence_intervals.csv`: paired proposed-minus-baseline intervals for positive precision, recall, and F1 for every label.
- `seed_validation.json`: per-seed source, split, pairing, and threshold checks.
- `analysis_manifest.json`: source hashes, protocol, and report hashes.

Code commit used for the analysis: `ba8f36961e86fd140f5c2f9160948a984b391545`.
Source manifest hash: `050482CA114C4E1B96DA72B33C5CBF09CE05636496BCC146EC3F9F1BA042A2C5`.
