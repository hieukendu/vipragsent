# Q1a Best-F1 test table

All values are test-set binary macro-F1 percentages. +/- is the 95% percentile confidence-interval half-width computed with the locked paired hierarchical bootstrap protocol.

| System | Implicit | Sarcasm | Irony | Idiom | Code-sw. | Mocking | Macro-prag | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PhoBERT (single-task) | 88.0 +/- 1.9 | 70.9 +/- 3.4 | 97.1 +/- 1.7 | 91.2 +/- 2.6 | 91.1 +/- 2.1 | 79.2 +/- 3.9 | 86.2 +/- 1.1 | 3/3 complete |
| PhoBERT fine-tune | 85.8 +/- 1.8 | 72.6 +/- 3.7 | 97.5 +/- 1.3 | 86.6 +/- 3.0 | 91.7 +/- 1.6 | 78.5 +/- 2.9 | 85.5 +/- 1.2 | 3/3 complete |
| XLM-R-large fine-tune | 92.2 +/- 1.3 | 88.1 +/- 2.8 | 97.9 +/- 1.2 | 92.9 +/- 2.3 | 94.3 +/- 1.3 | 92.3 +/- 1.7 | 92.9 +/- 0.8 | 3/3 complete |
| Sailor-7B SFT | 85.4 +/- 1.9 | 72.1 +/- 4.2 | 95.7 +/- 1.7 | 91.2 +/- 2.8 | 91.0 +/- 1.8 | 77.8 +/- 3.0 | 85.5 +/- 1.3 | 3/3 complete |
| Vistral-7B SFT | 87.4 +/- 1.8 | 74.7 +/- 3.5 | 96.7 +/- 1.6 | 91.5 +/- 2.6 | 92.8 +/- 1.6 | 79.9 +/- 2.4 | 87.2 +/- 1.1 | 3/3 complete |
| ViPragSent - no auxiliary loss | 87.4 +/- 1.8 | 74.8 +/- 3.6 | 97.2 +/- 1.4 | 91.5 +/- 2.5 | 92.9 +/- 1.6 | 79.5 +/- 3.7 | 87.2 +/- 1.2 | 3/3 complete |
| Vistral CoT-only clean rerun 003 | 17.4 +/- 1.1 | 48.2 +/- 0.3 | 48.2 +/- 0.3 | 48.2 +/- 0.3 | 45.9 +/- 0.4 | 46.9 +/- 0.4 | 42.5 +/- 0.2 | 2/3; seed 20260522 test score missing |
| ViPragSent - explanation only | 44.1 +/- 0.5 | 48.2 +/- 0.3 | 48.2 +/- 0.3 | 48.2 +/- 0.3 | 46.0 +/- 0.6 | 46.9 +/- 0.4 | 46.9 +/- 0.2 | 3/3 complete |
| ViPragSent (ours, Vistral) | 84.9 +/- 4.9 | 72.6 +/- 7.4 | 94.8 +/- 3.3 | 79.4 +/- 19.4 | 78.4 +/- 23.1 | 77.1 +/- 8.0 | 81.2 +/- 10.8 | 3/3 complete |
| GPT-4.1-mini zero-shot | 45.4 +/- 2.2 | 56.5 +/- 3.1 | 53.2 +/- 2.7 | 52.5 +/- 2.4 | 62.4 +/- 2.7 | 54.7 +/- 2.6 | 54.1 +/- 1.3 | 1/1 complete |
| GPT-4.1-mini 8-shot | 49.0 +/- 2.3 | 61.1 +/- 3.4 | 52.4 +/- 2.6 | 50.9 +/- 2.5 | 56.9 +/- 2.6 | 54.3 +/- 2.5 | 54.1 +/- 1.3 | 1/1 complete |

Notes:

- The COT row is specifically q1a_cot_only_vistral_clean_rerun_003; it is not the old generic COT baseline.
- COT seed 20260522 has canonical epoch 1-3 checkpoints, but no test prediction or test_reasoning_metrics.json artifact was found. Its training status is therefore complete, while its test score remains unavailable and is not imputed.
- Full ViPragSent uses the complete persisted test gold/probability arrays plus the frozen per-seed thresholds. The reconstructed six-column predictions reproduce the persisted test metrics; the remote prediction JSONL contains partial record shards.
- Azure F1 is computed from the valid 2,000-row prediction JSONL because the persisted metrics.json files have empty test objects.
- Re-audit 2026-09-06: every complete Q1a source was rehashed and recomputed from its authoritative prediction/raw-metric source; no baseline score mismatch was found.
- Comparability: XLM-R is fully fine-tuned (558,525,440 trainable parameters, 10 epochs), whereas full ViPragSent is 4-bit QLoRA (19,658,368 trainable parameters, 3 epochs) with eight auxiliary tasks, rationale training, and uncertainty weighting. This is not a compute-matched backbone ablation.
- The closest Vistral controls are Vistral SFT and ViPragSent without auxiliary loss; both average 87.2 macro-F1. Full ViPragSent has high seed variance because seed 20260521 is a genuine dev/test collapse, not a reconstructed metric error.

## Per-seed audit

| Run | Seed | Implicit | Sarcasm | Irony | Idiom | Code-sw. | Mocking | Macro-prag | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| q1a_phobert_pragmatic_single_task_20260521 | 20260521 | 86.5174 | 69.3838 | 95.9544 | 92.1106 | 90.9670 | 82.2439 | 86.1962 | COMPLETE |
| q1a_phobert_pragmatic_single_task_20260522 | 20260522 | 88.6383 | 71.9669 | 97.4272 | 90.7440 | 89.6357 | 76.2834 | 85.7826 | COMPLETE |
| q1a_phobert_pragmatic_single_task_20260523 | 20260523 | 88.8487 | 71.2711 | 98.0004 | 90.7834 | 92.6495 | 78.9430 | 86.7494 | COMPLETE |
| q1a_phobert_pragmatic_finetune_20260521 | 20260521 | 84.9921 | 72.5647 | 97.6492 | 86.6787 | 91.2279 | 77.3701 | 85.0805 | COMPLETE |
| q1a_phobert_pragmatic_finetune_20260522 | 20260522 | 86.4120 | 72.1821 | 97.4272 | 87.2698 | 92.2834 | 79.2281 | 85.8004 | COMPLETE |
| q1a_phobert_pragmatic_finetune_20260523 | 20260523 | 85.8659 | 73.1760 | 97.2758 | 85.9484 | 91.6859 | 78.9798 | 85.4886 | COMPLETE |
| q1a_xlmr_pragmatic_finetune_20260521 | 20260521 | 92.0147 | 88.6193 | 97.8667 | 93.5144 | 94.3166 | 92.2769 | 93.1014 | COMPLETE |
| q1a_xlmr_pragmatic_finetune_20260522 | 20260522 | 92.2168 | 87.8713 | 98.0410 | 92.8720 | 94.2394 | 91.8608 | 92.8502 | COMPLETE |
| q1a_xlmr_pragmatic_finetune_20260523 | 20260523 | 92.2596 | 87.8786 | 97.8809 | 92.1640 | 94.2394 | 92.6843 | 92.8511 | COMPLETE |
| q1a_sailor_pragmatic_sft_20260521 | 20260521 | 85.5124 | 73.6685 | 95.5073 | 89.6946 | 91.8713 | 78.9345 | 85.8648 | COMPLETE |
| q1a_sailor_pragmatic_sft_20260522 | 20260522 | 85.1204 | 68.7330 | 96.3500 | 92.4867 | 90.7767 | 76.5106 | 84.9962 | COMPLETE |
| q1a_sailor_pragmatic_sft_20260523 | 20260523 | 85.6380 | 74.0329 | 95.3917 | 91.2825 | 90.4508 | 77.8247 | 85.7701 | COMPLETE |
| q1a_vistral_pragmatic_sft_20260521 | 20260521 | 87.3910 | 73.8187 | 96.4254 | 91.9951 | 92.8310 | 80.0020 | 87.0772 | COMPLETE |
| q1a_vistral_pragmatic_sft_20260522 | 20260522 | 87.6432 | 75.6934 | 96.8226 | 92.7035 | 92.1796 | 80.1071 | 87.5249 | COMPLETE |
| q1a_vistral_pragmatic_sft_20260523 | 20260523 | 87.0348 | 74.7038 | 96.8226 | 89.7380 | 93.3833 | 79.6189 | 86.8836 | COMPLETE |
| q1a_vipragsent_no_auxiliary_vistral_20260521 | 20260521 | 88.1366 | 74.2566 | 96.8656 | 91.7161 | 92.3697 | 76.2676 | 86.6020 | COMPLETE |
| q1a_vipragsent_no_auxiliary_vistral_20260522 | 20260522 | 86.9944 | 75.3109 | 97.1810 | 92.3699 | 93.5822 | 81.0980 | 87.7560 | COMPLETE |
| q1a_vipragsent_no_auxiliary_vistral_20260523 | 20260523 | 87.1589 | 74.8070 | 97.4094 | 90.4332 | 92.6774 | 81.1729 | 87.2765 | COMPLETE |
| q1a_cot_only_vistral_clean_rerun_003__seed_20260521 | 20260521 | 17.3554 | 48.2402 | 48.1999 | 48.1731 | 45.9167 | 46.9074 | 42.4654 | COMPLETE |
| q1a_cot_only_vistral_clean_rerun_003__seed_20260522 | 20260522 | - | - | - | - | - | - | - | MISSING_TEST_SCORE |
| q1a_cot_only_vistral_clean_rerun_003__seed_20260523 | 20260523 | 17.3554 | 48.2402 | 48.1999 | 48.1731 | 45.9167 | 46.9074 | 42.4654 | COMPLETE |
| q1a_explanation_only_vistral_20260521 | 20260521 | 44.1341 | 48.2402 | 48.1999 | 48.1731 | 45.9167 | 46.9074 | 46.9286 | COMPLETE |
| q1a_explanation_only_vistral_20260522 | 20260522 | 44.1341 | 48.2402 | 48.1999 | 48.1731 | 46.2277 | 46.9074 | 46.9804 | COMPLETE |
| q1a_explanation_only_vistral_20260523 | 20260523 | 44.1341 | 48.2402 | 48.1999 | 48.1731 | 45.9021 | 46.9074 | 46.9261 | COMPLETE |
| q1a_vipragsent_full_vistral_20260521 | 20260521 | 78.5927 | 63.7562 | 91.1241 | 53.8249 | 47.8390 | 67.3843 | 67.0869 | COMPLETE |
| q1a_vipragsent_full_vistral_20260522 | 20260522 | 88.3053 | 78.1106 | 97.0313 | 91.8274 | 94.3886 | 83.5758 | 88.8732 | COMPLETE |
| q1a_vipragsent_full_vistral_20260523 | 20260523 | 87.8808 | 75.9025 | 96.1083 | 92.4867 | 92.8529 | 80.4809 | 87.6187 | COMPLETE |
| azure_pragmatic_zero_shot | single | 45.4446 | 56.5083 | 53.1723 | 52.4536 | 62.4173 | 54.6859 | 54.1137 | COMPLETE |
| azure_gpt41_mini_8shot | single | 49.0434 | 61.0934 | 52.3735 | 50.8677 | 56.8851 | 54.3239 | 54.0978 | COMPLETE |
