# ViPragSent reviewer-gap experiment report

Generated from canonical run manifests, validation files, test predictions, and HF uploader status.
All checkpoint/threshold selection is dev-only; test metrics are reported only after the frozen selection protocol.

## Protocol and parameters

| Experiment | Model/revision | Batch (physical → effective; accumulation) | Optimizer / LR / schedule | Epochs / precision | Rationale | Uncertainty | Multipliers | Selection / threshold | Data |
|---|---|---|---|---|---|---|---|---|---|
| ViSoBERT baseline | `uitnlp/visobert` @ `196a62afad9c` | 32 → 32; 1 | AdamW / 0.000020 / cosine, warmup 0.20 | 10 / bf16 | decoder=no, beta=0.00 | learned | code_switching=1.04, idiom_figurative=1.00, implicit_sentiment=1.01, irony=1.05, mocking=1.00, sarcasm=1.01, polarity=1.00, emotion=1.00 | `best_checkpoint_selected_on_development_macro_pragmatic_f1`; dev; test-after-freeze | train 7,998 / dev 1,999 / test 2,000; max len 128 |
| XLM-R all multipliers = 1 | `FacebookAI/xlm-roberta-large` @ `c23d21b0620b` | 8 → 32; 4 | AdamW / 0.000020 / cosine, warmup 0.20 | 10 / bf16 | decoder=yes, beta=0.30; all task multipliers=1 | learned | code_switching=1.00, idiom_figurative=1.00, implicit_sentiment=1.00, irony=1.00, mocking=1.00, sarcasm=1.00, polarity=1.00, emotion=1.00 | `best_checkpoint_selected_on_development_macro_pragmatic_f1`; dev; test-after-freeze | train 7,998 / dev 1,999 / test 2,000; max len 128 |
| XLM-R rationale beta = 0.1 | `FacebookAI/xlm-roberta-large` @ `c23d21b0620b` | 8 → 32; 4 | AdamW / 0.000020 / cosine, warmup 0.20 | 10 / bf16 | decoder=yes, beta=0.10 | learned | code_switching=1.04, idiom_figurative=1.00, implicit_sentiment=1.01, irony=1.05, mocking=1.00, sarcasm=1.01, polarity=1.00, emotion=1.00 | `best_checkpoint_selected_on_development_macro_pragmatic_f1`; dev; test-after-freeze | train 7,998 / dev 1,999 / test 2,000; max len 128 |
| XLM-R rationale beta = 0.5 | `FacebookAI/xlm-roberta-large` @ `c23d21b0620b` | 8 → 32; 4 | AdamW / 0.000020 / cosine, warmup 0.20 | 10 / bf16 | decoder=yes, beta=0.50 | learned | code_switching=1.04, idiom_figurative=1.00, implicit_sentiment=1.01, irony=1.05, mocking=1.00, sarcasm=1.01, polarity=1.00, emotion=1.00 | `best_checkpoint_selected_on_development_macro_pragmatic_f1`; dev; test-after-freeze | train 7,998 / dev 1,999 / test 2,000; max len 128 |

## Per-seed overview

Macro pragmatic F1 is the mean of the six pragmatic binary macro-F1 values. `F1+` is the positive-class F1.

| Experiment | Seed | Run status | Best epoch | Macro pragmatic F1 | Macro ECE | Peak VRAM GiB | HF upload | Verified | Artifact repo |
|---|---:|---|---:|---:|---:|---:|---|---:|---|
| ViSoBERT baseline | 20260521 | PASS | 9 | 0.8813 | 0.0294 | 2.93 | PASS | 22/22 | `Thundergod2007/vipragsent-experiment-artifacts-overflow-018` |
| ViSoBERT baseline | 20260522 | PASS | 7 | 0.8861 | 0.0372 | 2.92 | PASS | 22/22 | `Thundergod2007/vipragsent-experiment-artifacts-overflow-025` |
| ViSoBERT baseline | 20260523 | PASS | 8 | 0.8845 | 0.0297 | 2.92 | PASS | 22/22 | `Thundergod2007/vipragsent-experiment-artifacts-overflow-017` |
| XLM-R all multipliers = 1 | 20260521 | PASS | 10 | 0.9286 | 0.0347 | 13.52 | PASS | 22/22 | `Thundergod2007/vipragsent-experiment-artifacts-overflow-018` |
| XLM-R all multipliers = 1 | 20260522 | PASS | 10 | 0.9273 | 0.0345 | 13.52 | PASS | 22/22 | `Thundergod2007/vipragsent-experiment-artifacts-overflow-025` |
| XLM-R all multipliers = 1 | 20260523 | PASS | 10 | 0.9336 | 0.0355 | 13.52 | PASS | 22/22 | `Thundergod2007/vipragsent-experiment-artifacts-overflow-017` |
| XLM-R rationale beta = 0.1 | 20260521 | PASS | 10 | 0.9282 | 0.0358 | 13.52 | PASS | 22/22 | `Thundergod2007/vipragsent-experiment-artifacts-overflow-018` |
| XLM-R rationale beta = 0.1 | 20260522 | PASS | 10 | 0.9354 | 0.0313 | 13.52 | PASS | 22/22 | `Thundergod2007/vipragsent-experiment-artifacts-overflow-025` |
| XLM-R rationale beta = 0.1 | 20260523 | PASS | 10 | 0.9329 | 0.0325 | 13.52 | PASS | 22/22 | `Thundergod2007/vipragsent-experiment-artifacts-overflow-017` |
| XLM-R rationale beta = 0.5 | 20260521 | PASS | 10 | 0.9210 | 0.0327 | 13.52 | PASS | 22/22 | `Thundergod2007/vipragsent-experiment-artifacts-overflow-018` |
| XLM-R rationale beta = 0.5 | 20260522 | PASS | 10 | 0.9257 | 0.0455 | 13.52 | PASS | 22/22 | `Thundergod2007/vipragsent-experiment-artifacts-overflow-025` |
| XLM-R rationale beta = 0.5 | 20260523 | PASS | 10 | 0.9339 | 0.0368 | 13.52 | PASS | 22/22 | `Thundergod2007/vipragsent-experiment-artifacts-overflow-017` |

## Per-seed six-pragmatic metrics

Each row contains macro-F1 for the label, then positive-class precision / recall / F1 and confusion counts.

| Experiment | Seed | Pragmatic label | Macro-F1 | P+ | R+ | F1+ | TP | FP | FN | TN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ViSoBERT baseline | 20260521 | code_switching | 0.9162 | 0.8344 | 0.8841 | 0.8585 | 267 | 53 | 35 | 1645 |
| ViSoBERT baseline | 20260521 | idiom_figurative | 0.9046 | 0.8227 | 0.8227 | 0.8227 | 116 | 25 | 25 | 1834 |
| ViSoBERT baseline | 20260521 | implicit_sentiment | 0.8914 | 0.8608 | 0.7952 | 0.8267 | 334 | 54 | 86 | 1526 |
| ViSoBERT baseline | 20260521 | irony | 0.9801 | 0.9924 | 0.9353 | 0.9630 | 130 | 1 | 9 | 1860 |
| ViSoBERT baseline | 20260521 | mocking | 0.8316 | 0.6901 | 0.7167 | 0.7032 | 167 | 75 | 66 | 1692 |
| ViSoBERT baseline | 20260521 | sarcasm | 0.7638 | 0.6381 | 0.4926 | 0.5560 | 67 | 38 | 69 | 1826 |
| ViSoBERT baseline | 20260522 | code_switching | 0.9158 | 0.8386 | 0.8775 | 0.8576 | 265 | 51 | 37 | 1647 |
| ViSoBERT baseline | 20260522 | idiom_figurative | 0.9101 | 0.8357 | 0.8298 | 0.8327 | 117 | 23 | 24 | 1836 |
| ViSoBERT baseline | 20260522 | implicit_sentiment | 0.8864 | 0.8329 | 0.8071 | 0.8198 | 339 | 68 | 81 | 1512 |
| ViSoBERT baseline | 20260522 | irony | 0.9763 | 0.9774 | 0.9353 | 0.9559 | 130 | 3 | 9 | 1858 |
| ViSoBERT baseline | 20260522 | mocking | 0.8339 | 0.7168 | 0.6953 | 0.7059 | 162 | 64 | 71 | 1703 |
| ViSoBERT baseline | 20260522 | sarcasm | 0.7942 | 0.5855 | 0.6544 | 0.6181 | 89 | 63 | 47 | 1801 |
| ViSoBERT baseline | 20260523 | code_switching | 0.9127 | 0.8826 | 0.8212 | 0.8508 | 248 | 33 | 54 | 1665 |
| ViSoBERT baseline | 20260523 | idiom_figurative | 0.9037 | 0.8485 | 0.7943 | 0.8205 | 112 | 20 | 29 | 1839 |
| ViSoBERT baseline | 20260523 | implicit_sentiment | 0.8818 | 0.8124 | 0.8143 | 0.8133 | 342 | 79 | 78 | 1501 |
| ViSoBERT baseline | 20260523 | irony | 0.9801 | 0.9924 | 0.9353 | 0.9630 | 130 | 1 | 9 | 1860 |
| ViSoBERT baseline | 20260523 | mocking | 0.8349 | 0.6971 | 0.7210 | 0.7089 | 168 | 73 | 65 | 1694 |
| ViSoBERT baseline | 20260523 | sarcasm | 0.7941 | 0.5906 | 0.6471 | 0.6175 | 88 | 61 | 48 | 1803 |
| XLM-R all multipliers = 1 | 20260521 | code_switching | 0.9475 | 0.9278 | 0.8940 | 0.9106 | 270 | 21 | 32 | 1677 |
| XLM-R all multipliers = 1 | 20260521 | idiom_figurative | 0.9360 | 0.8971 | 0.8652 | 0.8809 | 122 | 14 | 19 | 1845 |
| XLM-R all multipliers = 1 | 20260521 | implicit_sentiment | 0.9264 | 0.8797 | 0.8881 | 0.8839 | 373 | 51 | 47 | 1529 |
| XLM-R all multipliers = 1 | 20260521 | irony | 0.9823 | 0.9851 | 0.9496 | 0.9670 | 132 | 2 | 7 | 1859 |
| XLM-R all multipliers = 1 | 20260521 | mocking | 0.9182 | 0.8609 | 0.8498 | 0.8553 | 198 | 32 | 35 | 1735 |
| XLM-R all multipliers = 1 | 20260521 | sarcasm | 0.8612 | 0.7143 | 0.7721 | 0.7420 | 105 | 42 | 31 | 1822 |
| XLM-R all multipliers = 1 | 20260522 | code_switching | 0.9472 | 0.9133 | 0.9073 | 0.9103 | 274 | 26 | 28 | 1672 |
| XLM-R all multipliers = 1 | 20260522 | idiom_figurative | 0.9210 | 0.8623 | 0.8440 | 0.8530 | 119 | 19 | 22 | 1840 |
| XLM-R all multipliers = 1 | 20260522 | implicit_sentiment | 0.9189 | 0.9136 | 0.8310 | 0.8703 | 349 | 33 | 71 | 1547 |
| XLM-R all multipliers = 1 | 20260522 | irony | 0.9770 | 0.9504 | 0.9640 | 0.9571 | 134 | 7 | 5 | 1854 |
| XLM-R all multipliers = 1 | 20260522 | mocking | 0.9299 | 0.8894 | 0.8627 | 0.8758 | 201 | 25 | 32 | 1742 |
| XLM-R all multipliers = 1 | 20260522 | sarcasm | 0.8697 | 0.7248 | 0.7941 | 0.7579 | 108 | 41 | 28 | 1823 |
| XLM-R all multipliers = 1 | 20260523 | code_switching | 0.9460 | 0.9158 | 0.9007 | 0.9082 | 272 | 25 | 30 | 1673 |
| XLM-R all multipliers = 1 | 20260523 | idiom_figurative | 0.9377 | 0.9037 | 0.8652 | 0.8841 | 122 | 13 | 19 | 1846 |
| XLM-R all multipliers = 1 | 20260523 | implicit_sentiment | 0.9245 | 0.8943 | 0.8667 | 0.8803 | 364 | 43 | 56 | 1537 |
| XLM-R all multipliers = 1 | 20260523 | irony | 0.9805 | 0.9708 | 0.9568 | 0.9638 | 133 | 4 | 6 | 1857 |
| XLM-R all multipliers = 1 | 20260523 | mocking | 0.9293 | 0.8964 | 0.8541 | 0.8747 | 199 | 23 | 34 | 1744 |
| XLM-R all multipliers = 1 | 20260523 | sarcasm | 0.8833 | 0.7586 | 0.8088 | 0.7829 | 110 | 35 | 26 | 1829 |
| XLM-R rationale beta = 0.1 | 20260521 | code_switching | 0.9462 | 0.9130 | 0.9040 | 0.9085 | 273 | 26 | 29 | 1672 |
| XLM-R rationale beta = 0.1 | 20260521 | idiom_figurative | 0.9317 | 0.8955 | 0.8511 | 0.8727 | 120 | 14 | 21 | 1845 |
| XLM-R rationale beta = 0.1 | 20260521 | implicit_sentiment | 0.9272 | 0.8693 | 0.9024 | 0.8855 | 379 | 57 | 41 | 1523 |
| XLM-R rationale beta = 0.1 | 20260521 | irony | 0.9862 | 0.9925 | 0.9568 | 0.9744 | 133 | 1 | 6 | 1860 |
| XLM-R rationale beta = 0.1 | 20260521 | mocking | 0.9266 | 0.8777 | 0.8627 | 0.8701 | 201 | 28 | 32 | 1739 |
| XLM-R rationale beta = 0.1 | 20260521 | sarcasm | 0.8511 | 0.7174 | 0.7279 | 0.7226 | 99 | 39 | 37 | 1825 |
| XLM-R rationale beta = 0.1 | 20260522 | code_switching | 0.9439 | 0.9153 | 0.8940 | 0.9045 | 270 | 25 | 32 | 1673 |
| XLM-R rationale beta = 0.1 | 20260522 | idiom_figurative | 0.9407 | 0.8929 | 0.8865 | 0.8897 | 125 | 15 | 16 | 1844 |
| XLM-R rationale beta = 0.1 | 20260522 | implicit_sentiment | 0.9279 | 0.8821 | 0.8905 | 0.8863 | 374 | 50 | 46 | 1530 |
| XLM-R rationale beta = 0.1 | 20260522 | irony | 0.9803 | 0.9850 | 0.9424 | 0.9632 | 131 | 2 | 8 | 1859 |
| XLM-R rationale beta = 0.1 | 20260522 | mocking | 0.9302 | 0.9041 | 0.8498 | 0.8761 | 198 | 21 | 35 | 1746 |
| XLM-R rationale beta = 0.1 | 20260522 | sarcasm | 0.8893 | 0.7550 | 0.8382 | 0.7944 | 114 | 37 | 22 | 1827 |
| XLM-R rationale beta = 0.1 | 20260523 | code_switching | 0.9456 | 0.9244 | 0.8907 | 0.9073 | 269 | 22 | 33 | 1676 |
| XLM-R rationale beta = 0.1 | 20260523 | idiom_figurative | 0.9317 | 0.8955 | 0.8511 | 0.8727 | 120 | 14 | 21 | 1845 |
| XLM-R rationale beta = 0.1 | 20260523 | implicit_sentiment | 0.9313 | 0.9211 | 0.8619 | 0.8905 | 362 | 31 | 58 | 1549 |
| XLM-R rationale beta = 0.1 | 20260523 | irony | 0.9785 | 0.9706 | 0.9496 | 0.9600 | 132 | 4 | 7 | 1857 |
| XLM-R rationale beta = 0.1 | 20260523 | mocking | 0.9317 | 0.8831 | 0.8755 | 0.8793 | 204 | 27 | 29 | 1740 |
| XLM-R rationale beta = 0.1 | 20260523 | sarcasm | 0.8786 | 0.7552 | 0.7941 | 0.7742 | 108 | 35 | 28 | 1829 |
| XLM-R rationale beta = 0.5 | 20260521 | code_switching | 0.9448 | 0.9184 | 0.8940 | 0.9060 | 270 | 24 | 32 | 1674 |
| XLM-R rationale beta = 0.5 | 20260521 | idiom_figurative | 0.9377 | 0.9037 | 0.8652 | 0.8841 | 122 | 13 | 19 | 1846 |
| XLM-R rationale beta = 0.5 | 20260521 | implicit_sentiment | 0.9147 | 0.8775 | 0.8524 | 0.8647 | 358 | 50 | 62 | 1530 |
| XLM-R rationale beta = 0.5 | 20260521 | irony | 0.9765 | 0.9704 | 0.9424 | 0.9562 | 131 | 4 | 8 | 1857 |
| XLM-R rationale beta = 0.5 | 20260521 | mocking | 0.9110 | 0.8900 | 0.7983 | 0.8416 | 186 | 23 | 47 | 1744 |
| XLM-R rationale beta = 0.5 | 20260521 | sarcasm | 0.8411 | 0.7090 | 0.6985 | 0.7037 | 95 | 39 | 41 | 1825 |
| XLM-R rationale beta = 0.5 | 20260522 | code_switching | 0.9443 | 0.9070 | 0.9040 | 0.9055 | 273 | 28 | 29 | 1670 |
| XLM-R rationale beta = 0.5 | 20260522 | idiom_figurative | 0.9227 | 0.8686 | 0.8440 | 0.8561 | 119 | 18 | 22 | 1841 |
| XLM-R rationale beta = 0.5 | 20260522 | implicit_sentiment | 0.9117 | 0.8822 | 0.8381 | 0.8596 | 352 | 47 | 68 | 1533 |
| XLM-R rationale beta = 0.5 | 20260522 | irony | 0.9823 | 0.9851 | 0.9496 | 0.9670 | 132 | 2 | 7 | 1859 |
| XLM-R rationale beta = 0.5 | 20260522 | mocking | 0.9087 | 0.8171 | 0.8627 | 0.8392 | 201 | 45 | 32 | 1722 |
| XLM-R rationale beta = 0.5 | 20260522 | sarcasm | 0.8847 | 0.7296 | 0.8529 | 0.7864 | 116 | 43 | 20 | 1821 |
| XLM-R rationale beta = 0.5 | 20260523 | code_switching | 0.9495 | 0.9313 | 0.8974 | 0.9140 | 271 | 20 | 31 | 1678 |
| XLM-R rationale beta = 0.5 | 20260523 | idiom_figurative | 0.9317 | 0.8955 | 0.8511 | 0.8727 | 120 | 14 | 21 | 1845 |
| XLM-R rationale beta = 0.5 | 20260523 | implicit_sentiment | 0.9224 | 0.8555 | 0.9024 | 0.8783 | 379 | 64 | 41 | 1516 |
| XLM-R rationale beta = 0.5 | 20260523 | irony | 0.9862 | 0.9925 | 0.9568 | 0.9744 | 133 | 1 | 6 | 1860 |
| XLM-R rationale beta = 0.5 | 20260523 | mocking | 0.9355 | 0.8879 | 0.8841 | 0.8860 | 206 | 26 | 27 | 1741 |
| XLM-R rationale beta = 0.5 | 20260523 | sarcasm | 0.8781 | 0.7244 | 0.8309 | 0.7740 | 113 | 43 | 23 | 1821 |

## Mean ± sample SD across seeds

| Experiment | Macro pragmatic F1 | Macro ECE |
|---|---:|---:|
| ViSoBERT baseline | 0.8840 ± 0.0025 | 0.0321 ± 0.0045 |
| XLM-R all multipliers = 1 | 0.9298 ± 0.0033 | 0.0349 ± 0.0005 |
| XLM-R rationale beta = 0.1 | 0.9321 ± 0.0037 | 0.0332 ± 0.0024 |
| XLM-R rationale beta = 0.5 | 0.9269 ± 0.0065 | 0.0383 ± 0.0065 |

| Experiment | Pragmatic label | Macro-F1 mean ± SD | P+ mean ± SD | R+ mean ± SD | F1+ mean ± SD |
|---|---|---:|---:|---:|---:|
| ViSoBERT baseline | code_switching | 0.9149 ± 0.0020 | 0.8518 ± 0.0267 | 0.8609 ± 0.0346 | 0.8556 ± 0.0042 |
| ViSoBERT baseline | idiom_figurative | 0.9061 ± 0.0034 | 0.8356 ± 0.0129 | 0.8156 ± 0.0188 | 0.8253 ± 0.0065 |
| ViSoBERT baseline | implicit_sentiment | 0.8866 ± 0.0048 | 0.8354 ± 0.0243 | 0.8056 ± 0.0096 | 0.8200 ± 0.0067 |
| ViSoBERT baseline | irony | 0.9789 ± 0.0022 | 0.9874 ± 0.0086 | 0.9353 ± 0.0000 | 0.9606 ± 0.0041 |
| ViSoBERT baseline | mocking | 0.8334 ± 0.0017 | 0.7013 ± 0.0139 | 0.7110 ± 0.0138 | 0.7060 ± 0.0029 |
| ViSoBERT baseline | sarcasm | 0.7840 ± 0.0175 | 0.6047 ± 0.0290 | 0.5980 ± 0.0913 | 0.5972 ± 0.0357 |
| XLM-R all multipliers = 1 | code_switching | 0.9469 ± 0.0008 | 0.9190 ± 0.0078 | 0.9007 ± 0.0066 | 0.9097 ± 0.0013 |
| XLM-R all multipliers = 1 | idiom_figurative | 0.9316 ± 0.0092 | 0.8877 ± 0.0222 | 0.8582 ± 0.0123 | 0.8727 ± 0.0171 |
| XLM-R all multipliers = 1 | implicit_sentiment | 0.9233 ± 0.0039 | 0.8959 ± 0.0170 | 0.8619 ± 0.0289 | 0.8782 ± 0.0070 |
| XLM-R all multipliers = 1 | irony | 0.9799 ± 0.0027 | 0.9687 ± 0.0175 | 0.9568 ± 0.0072 | 0.9626 ± 0.0050 |
| XLM-R all multipliers = 1 | mocking | 0.9258 ± 0.0066 | 0.8822 ± 0.0188 | 0.8555 ± 0.0066 | 0.8686 ± 0.0115 |
| XLM-R all multipliers = 1 | sarcasm | 0.8714 ± 0.0111 | 0.7326 ± 0.0232 | 0.7917 ± 0.0185 | 0.7610 ± 0.0206 |
| XLM-R rationale beta = 0.1 | code_switching | 0.9452 ± 0.0012 | 0.9176 ± 0.0060 | 0.8962 ± 0.0069 | 0.9068 ± 0.0020 |
| XLM-R rationale beta = 0.1 | idiom_figurative | 0.9347 ± 0.0052 | 0.8946 ± 0.0015 | 0.8629 ± 0.0205 | 0.8784 ± 0.0098 |
| XLM-R rationale beta = 0.1 | implicit_sentiment | 0.9288 ± 0.0022 | 0.8908 ± 0.0270 | 0.8849 ± 0.0208 | 0.8874 ± 0.0027 |
| XLM-R rationale beta = 0.1 | irony | 0.9817 ± 0.0040 | 0.9827 ± 0.0111 | 0.9496 ± 0.0072 | 0.9659 ± 0.0075 |
| XLM-R rationale beta = 0.1 | mocking | 0.9295 ± 0.0026 | 0.8883 ± 0.0139 | 0.8627 ± 0.0129 | 0.8752 ± 0.0047 |
| XLM-R rationale beta = 0.1 | sarcasm | 0.8730 ± 0.0197 | 0.7425 ± 0.0218 | 0.7868 ± 0.0555 | 0.7637 ± 0.0370 |
| XLM-R rationale beta = 0.5 | code_switching | 0.9462 ± 0.0029 | 0.9189 ± 0.0122 | 0.8985 ± 0.0051 | 0.9085 ± 0.0048 |
| XLM-R rationale beta = 0.5 | idiom_figurative | 0.9307 ± 0.0076 | 0.8893 ± 0.0184 | 0.8534 ± 0.0108 | 0.8710 ± 0.0141 |
| XLM-R rationale beta = 0.5 | implicit_sentiment | 0.9163 ± 0.0055 | 0.8717 ± 0.0142 | 0.8643 ± 0.0338 | 0.8676 ± 0.0097 |
| XLM-R rationale beta = 0.5 | irony | 0.9817 ± 0.0049 | 0.9827 ± 0.0113 | 0.9496 ± 0.0072 | 0.9659 ± 0.0091 |
| XLM-R rationale beta = 0.5 | mocking | 0.9184 ± 0.0149 | 0.8650 ± 0.0415 | 0.8484 ± 0.0447 | 0.8556 ± 0.0263 |
| XLM-R rationale beta = 0.5 | sarcasm | 0.8680 ± 0.0235 | 0.7210 ± 0.0107 | 0.7941 ± 0.0835 | 0.7547 ± 0.0446 |

## Reviewer-gap post-hoc artifacts

- Analysis status: `PASS`; experiments present: `visobert_baseline, xlmr_all_multipliers_1, xlmr_beta_01, xlmr_beta_05, xlmr_full_existing, xlmr_pragmatic_only_existing`.
- Analysis HF upload: `PASS`, 8/8 verified; repository `Thundergod2007/vipragsent-experiment-artifacts-overflow-018`; remote root `campaigns/reviewer-gaps-20261005/reviewer_gap_analyses_20261005`.
- Files: `per_seed_metrics.json`, `aggregate_metrics.json`, `pr_curves.json`, `calibration.json`, `dataset_overlap.json`, `code_switching_composition.json`, `learned_log_variance.json`.
- Exact normalized overlap with AIVIVN: 0 rows; by ViPragSent split: {'dev': 0, 'test': 0, 'train': 0}.
- Exact normalized overlap with UIT-VSFC: 7 rows; by ViPragSent split: {'dev': 1, 'test': 2, 'train': 4}.
- Exact normalized overlap with UIT-VSMEC: 15 rows; by ViPragSent split: {'dev': 2, 'test': 2, 'train': 11}.
- Code-switching heuristic on test positives: 302 rows; shares `{'latin_with_vietnamese_marks': 0.8940397350993378, 'latin_without_marks': 0.08940397350993377, 'no_latin': 0.0, 'non_latin': 0.016556291390728478}`.
- Learned log-variance extraction: `PASS` (15 entries). See `learned_log_variance.json` for exact values.

## Reproducibility and resource audit

- Dataset fingerprint: `B906C090400BAE115C9C5E3C35E32FA410AC519AE09209EBA741F198087C24F9`; split sizes 7,998 / 1,999 / 2,000.
- Selected device: NVIDIA H100 80GB HBM3 MIG 2g.20gb, visible as `cuda:0`, 19.625 GiB; CPU host has 128 cores.
- Existing relevant uploader/monitor/training processes were preserved. No unrelated ViPragSent-external GPU training process was found to terminate.
- Original Q1a/Q2 artifacts were preserved; all new artifacts are under `reviewer-gaps-20261005`.
