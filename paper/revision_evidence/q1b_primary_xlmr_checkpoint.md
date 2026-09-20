# Q1b primary checkpoint: ViPragSent XLM-R-large

This is the primary ViPragSent row for Q1b. It is separate from the standard `xlmr_multitask_8head` classification baseline and from the PhoBERT-backed contextual inventory row.

## Per-seed external metrics

| Seed | AIVIVN macro-F1 | UIT-VSFC macro-F1 | VSMEC macro-F1 | ordinary-F1 |
|---:|---:|---:|---:|---:|
| 20260521 | 0.4612968626 | 0.3493444224 | 0.3926494800 | 0.4010969217 |
| 20260522 | 0.5032473275 | 0.4046305594 | 0.4020502788 | 0.4366427219 |
| 20260523 | 0.4735489199 | 0.3827196975 | 0.4065142553 | 0.4209276242 |
| mean ± sample SD | 0.4793643700 ± 0.0215713924 | 0.3788982264 ± 0.0278404735 | 0.4004046714 ± 0.0070773594 | 0.4195557559 ± 0.0178125657 |

## Remote pins

- Seed 20260521: [`overflow-018`](https://huggingface.co/Thundergod2007/vipragsent-experiment-artifacts-overflow-018/tree/cd0b0367ae24bfa5fac24316f5e153f01cd67f11/campaigns/vipragsent-xlmr-q1b-q4-followup-v1/xlmr_followup_q1b_vipragsent_full_20260521), Hub revision `cd0b0367ae24bfa5fac24316f5e153f01cd67f11`, source checkpoint `A82FEE6195C375957C54431F9C11D8C6114CF851E20D77487BFC948E1A1E1CBA`, metric-file SHA `7D1A8F491CA076CAC2D45A19BD8050BB71B7A50BE7FAF7C30EC405FDF591284A`.
- Seed 20260522: [`overflow-025`](https://huggingface.co/Thundergod2007/vipragsent-experiment-artifacts-overflow-025/tree/7810865b6c9162fbb01923dfb45656d5d49da60a/campaigns/vipragsent-xlmr-q1b-q4-followup-v1/xlmr_followup_q1b_vipragsent_full_20260522), Hub revision `7810865b6c9162fbb01923dfb45656d5d49da60a`, source checkpoint `989E24F53FB29F3A18267B93BCA0720CCF1FAF6B3F1FF4A5B75103B7EEE7FEC1`, metric-file SHA `5896A6AAC6BA8B6621AC1DD9455C462701CD6933D18303A73FBC859977538BE0`.
- Seed 20260523: [`overflow-017`](https://huggingface.co/Thundergod2007/vipragsent-experiment-artifacts-overflow-017/tree/43a4e35edc969630fe33704a80fffea110f7f0a2/campaigns/vipragsent-xlmr-q1b-q4-followup-v1/xlmr_followup_q1b_vipragsent_full_20260523), Hub revision `43a4e35edc969630fe33704a80fffea110f7f0a2`, source checkpoint `5F9C80EB322ED10014229CC0CFE4978D71C4233CDCBC9BEFA51B8D6C568DFFB5`, metric-file SHA `D9E9D99DB570E1D959E8EC92708BD3BA9B2BF9A14FA9EDA66C6A270417325657`.

All three packs report model `FacebookAI/xlm-roberta-large`, model revision `c23d21b0620b635a76227c604d44e43a9f0ee389`, data fingerprint `A13573E38550ABD55D7F63E983C602BEA13C2765A1FFECA285F718478532AF0D`, `external_preprocessing=locked_normalized_test_only`, `partial=false`, and `PASS/APPROVED` state. The three normalized test hashes are AIVIVN `27DA59A452ACC0FA63E26A7F49325DE7D963353E1179C5FE21BC92DB9394EDA8`, UIT-VSFC `C27B61C70DE1090BD0A12F0F2824148D1691C68FB9CB18A07F23AE5282E75D96`, and VSMEC `1D2E604BD06A4AAAAD33A3B98DE2FB910E9020F4D468EF8BC16DDCDEADB3D955`.

## Interpretation

The extracted ordinary-F1 means are `0.4195557559 ± 0.0178125657` for the primary and `0.4253413954 ± 0.0052422067` for the standard XLM-R classification baseline. The verified same-observed-cohort comparison therefore permits this descriptive statement: the standard baseline mean is higher by approximately 0.005786. This is not a causal or statistical-significance claim. VSFC/VSMEC manifest hashes agree and all observed IDs, golds, raw text, and NFC-normalized text match; the AIVIVN source-manifest hashes differ (`27DA59...` versus `F6F6A7...`) and remain a provenance limitation, not a claim of formatting-only equivalence. The executed parity script, HTTP payload hashes/cache, and log are documented in `q1b_split_gold_comparability.md`. This primary row is `ANALYZED_SEMANTIC_COHORT_VERIFIED` from saved artifacts, not independently rerun.
