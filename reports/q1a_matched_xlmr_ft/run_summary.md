# Q1a matched XLM-R-large run summary

Status: COMPLETE for the three controlled seeds; paper files were not modified.

## Baseline and recipe

The implementation preserves `xlmr_pragmatic_finetune`: a fully fine-tuned `FacebookAI/xlm-roberta-large` encoder, first-attended-token pooling, dropout 0.1, and exactly six binary pragmatic linear heads (`implicit_sentiment`, `sarcasm`, `irony`, `idiom_figurative`, `code_switching`, `mocking`). It has no polarity/emotion/rationale/auxiliary/proposed-only component and uses equal-weight pragmatic BCE-with-logits with train-only negative/positive positive weights.

Matched recipe: max length 128; AdamW; learning rate 2e-5; weight decay 0.01; cosine schedule; warmup ratio 0.20; ten maximum epochs; patience 10; bf16; physical batch 8; accumulation 4; effective batch 32; gradient clipping 1.0; highest dev macro-pragmatic F1 checkpoint; dev-only 0.05–0.95 threshold grid with 0.01 step and nearest-0.5 tie break.

## Seeds, checkpoints, thresholds, and metrics

| Seed | Best epoch | Best dev macro F1 | Test macro F1 | Checkpoint SHA256 |
| ---: | ---: | ---: | ---: | --- |
| 20260521 | 10 | 0.921714203 | 0.930657563 | `B19587D8E1D35B2B09532165A7323D57FC7449A3DB703849DF3E8A4AE8D1F600` |
|  | thresholds | implicit_sentiment=0.90, sarcasm=0.95, irony=0.90, idiom_figurative=0.95, code_switching=0.88, mocking=0.94 |  |  |
| 20260522 | 10 | 0.929564556 | 0.929864780 | `47CB07C8D613C5E830FF5DB0F4DCE6C83E82CE2DD6E33A37F2F270388EAD39BE` |
|  | thresholds | implicit_sentiment=0.89, sarcasm=0.95, irony=0.73, idiom_figurative=0.85, code_switching=0.95, mocking=0.89 |  |  |
| 20260523 | 10 | 0.928111475 | 0.930538874 | `1E1701D1897031EBFDEEE9EA894EFC377E2E364E27398E271E4F9E247F4A8C5B` |
|  | thresholds | implicit_sentiment=0.94, sarcasm=0.95, irony=0.95, idiom_figurative=0.86, code_switching=0.90, mocking=0.78 |  |  |

The required per-seed artifacts are in `reports/q1a_matched_xlmr_ft/seed_<seed>/`; each seed has 7,998 train, 1,999 dev, and 2,000 test rows. Validation confirmed unique frozen IDs, gold alignment, paired proposed IDs/gold, bounded finite probabilities, metric reproducibility, and dev-only selection.

Matched TEST macro mean = 93.035374%; proposed existing Q1a TEST macro mean = 93.665838%; proposed-minus-matched observed macro delta = 0.630464 pp.

Aggregate metrics: [aggregate_metrics.csv](aggregate_metrics.csv) and [aggregate_metrics.txt](aggregate_metrics.txt).

## Paired bootstrap

[q1a_matched_paired_bootstrap_ci.csv](q1a_matched_paired_bootstrap_ci.csv) and [q1a_matched_paired_bootstrap_ci.txt](q1a_matched_paired_bootstrap_ci.txt) use the locked paired hierarchical seed-then-test-example protocol with 10000 resamples and bootstrap seed 20260525. The comparison is proposed existing Q1a minus matched baseline and all intervals are percentage points.

Primary macro: observed 0.630464 pp, 95% CI [0.068308, 1.195883], crosses zero=False.

| Head | Observed delta pp | 95% CI pp | Crosses zero |
| --- | ---: | --- | --- |
| implicit_sentiment | 0.414694 | [-0.645424, 1.443092] | True |
| sarcasm | 0.963950 | [-1.082517, 2.995165] | True |
| irony | 0.248074 | [-0.301112, 0.870785] | True |
| idiom_figurative | 0.763602 | [-0.468052, 2.026740] | True |
| code_switching | 0.498467 | [-0.542317, 1.632932] | True |
| mocking | 0.893997 | [-0.554983, 2.325025] | True |
| macro_pragmatic | 0.630464 | [0.068308, 1.195883] | False |

## Error analysis

[q1a_matched_error_analysis.csv](q1a_matched_error_analysis.csv) contains paired TP/TN/FP/FN, ours-correct/baseline-wrong, baseline-correct/ours-wrong, and corrected/reversed >=2-label example counts. `ours` is the existing proposed model; `baseline` is the matched six-head baseline.

Largest observed head F1 gains: sarcasm (0.963950 pp) and mocking (0.893997 pp).
Sarcasm remains one of the two lowest proposed mean-F1 heads=True; mocking remains one of the two lowest=False. Their deltas are 0.963950 pp and 0.893997 pp respectively.
Corrected >=2-label examples across seeds: 42; reverse >=2-label regressions: 27.

## Narrative and cooccurrence flags

The proposed-minus-old-current-baseline macro effect is 0.731587 pp; proposed-minus-matched effect is 0.630464 pp. Direction changed=False. Numeric narrative review flag=True because the matched comparator differs from the current Table-3-style baseline macro row (92.934251%) by 0.101123 pp. This is a review flag only; no paper text was changed.
Label cooccurrence was not recomputed; existing cooccurrence artifacts and paper inputs were left untouched.

## Hugging Face status

| Seed | Status | Checkpoint repo | Artifact repo | Remote root | Verified files |
| ---: | --- | --- | --- | --- | ---: |
| 20260521 | PASS | Thundergod2007/vipragsent-xlmr-checkpoints | Thundergod2007/vipragsent-experiment-artifacts-overflow-018 | `campaigns/matched_xlmr_ft_q1a/xlmr_large_ft_q1a_matched_20260521` | 24 |
| 20260522 | PASS | Thundergod2007/vipragsent-xlmr-checkpoints | Thundergod2007/vipragsent-experiment-artifacts-overflow-025 | `campaigns/matched_xlmr_ft_q1a/xlmr_large_ft_q1a_matched_20260522` | 24 |
| 20260523 | PASS | Thundergod2007/vipragsent-xlmr-checkpoints | Thundergod2007/vipragsent-experiment-artifacts-overflow-017 | `campaigns/matched_xlmr_ft_q1a/xlmr_large_ft_q1a_matched_20260523` | 24 |

Aggregate artifact upload: PASS — Thundergod2007/vipragsent-experiment-artifacts-overflow-012 at `campaigns/matched_xlmr_ft_q1a/xlmr_large_ft_q1a_matched_aggregate`; verified files=9; receipt=campaigns/matched_xlmr_ft_q1a/xlmr_large_ft_q1a_matched_aggregate/uploader/receipt.json.

Final reports are under `reports/q1a_matched_xlmr_ft/`. The paper files (`main.tex`, tables, Abstract, Discussion, Conclusion) were not modified.
