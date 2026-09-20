# Q1b split/gold/text comparability checkpoint

Checkpoint scope: standard XLM-R-large classification baseline (`Q1b-03`) versus the one primary ViPragSent XLM-R-large row (`Q1b-08`). This is an artifact audit only; no training or external writes were performed.

## Source artifacts

The comparison used the saved test prediction JSONL files from the pinned Hub packs:

- Standard baseline: `Thundergod2007/vipragsent-experiment-artifacts-overflow-002`, the `q1b_xlmr_multitask_8head_20260521/20260522/20260523` packs, with dataset files `predictions/aivivn_test_predictions.jsonl`, `predictions/uit_vsfc_test_predictions.jsonl`, and `predictions/uit_vsmec_test_predictions.jsonl`.
- Primary: `Thundergod2007/vipragsent-experiment-artifacts-overflow-018` revision `cd0b0367ae24bfa5fac24316f5e153f01cd67f11` (seed 20260521), overflow-025 revision `7810865b6c9162fbb01923dfb45656d5d49da60a` (seed 20260522), and overflow-017 revision `43a4e35edc969630fe33704a80fffea110f7f0a2` (seed 20260523), under `xlmr_followup_q1b_vipragsent_full_20260521/20260522/20260523`.

The reported model/protocol fields are XLM-R-large, test-only external evaluation, `external_preprocessing=locked_normalized_test_only`, and `external_finetuning=false`. The primary and standard rows remain distinct recipes: classification baseline versus ViPragSent primary. The three-primary seed metrics and full per-seed provenance are in `q1b_raw_extraction.csv` and `q1b_provenance.csv`.

## Manifest-level hashes

| Dataset | Standard baseline normalized-test hash | Primary normalized-test hash | Manifest result |
|---|---|---|---|
| AIVIVN | `F6F6A78EBADE3BC1038A3B1CFE05FF44E172AF024BBBC8A31079EFCF8133C6D3` | `27DA59A452ACC0FA63E26A7F49325DE7D963353E1179C5FE21BC92DB9394EDA8` | **MISMATCH; unresolved** |
| UIT-VSFC | `C27B61C70DE1090BD0A12F0F2824148D1691C68FB9CB18A07F23AE5282E75D96` | same | **PASS** |
| UIT-VSMEC | `1D2E604BD06A4AAAAD33A3B98DE2FB910E9020F4D468EF8BC16DDCDEADB3D955` | same | **PASS** |

The AIVIVN hash difference must not be relabelled as formatting-only without the missing manifest-generation input. The saved source manifests do not expose enough generation detail to establish that conclusion.

## Observed record-level semantic comparison

For each of seeds 20260521, 20260522, and 20260523, the standard-baseline and primary prediction records were compared by dataset and `sample_id`. The comparison checked record count, ID set and order, gold labels by ID, raw text by ID, NFC-normalized text by ID, and observed label vocabulary. Results were identical across the corresponding saved records:

| Dataset | Records | ID sequence | Gold by ID | Raw text by ID | NFC text by ID | Observed label vocabulary |
|---|---:|---|---|---|---|---|
| AIVIVN | 1,609 | PASS | PASS (0 differences) | PASS (0 differences) | PASS (0 differences) | `negative`, `neutral`, `positive` |
| UIT-VSFC | 3,166 | PASS | PASS (0 differences) | PASS (0 differences) | PASS (0 differences) | `negative`, `neutral`, `positive` |
| UIT-VSMEC | 693 | PASS | PASS (0 differences) | PASS (0 differences) | PASS (0 differences) | `anger`, `disgust`, `enjoyment`, `fear`, `other`, `sadness`, `surprise` |

The observed record hashes are the same for standard and primary on every corresponding seed because the test cohort/text/gold records are identical:

| Dataset | IDs | Gold labels | Raw text | NFC-normalized text |
|---|---|---|---|---|
| AIVIVN | `FF6FFE8B1067DB99E62433619173E3DA47967956BAB8E056EA8D3893495DDF9A` | `C70B4A5BA6E7435D61AFDAEA384EFAB77E5E7B4B69B0CCA0B060D22FAC17A34E` | `3BFF84AA5A7B766A41CE2D0CB4E7AFE41A98B0DEF61171C6B3790224209136A2` | `AA3314BC076C59CE860ABAD596311C9528C77E61593153E66BE48095D2DEB1D6` |
| UIT-VSFC | `75C9266EA370A758C7945B15E9926A454A420E7EA60A95D91E593C8CFE39B4F2` | `9629B98AD0E62A0F61F425BC398953D5BCFDD641F466A99100B87D13723F3E02` | `908762224F4D7DAEEFD3EF6FE1B5C1BB20CA4437A130E40AF13372263B842E35` | same as raw text |
| UIT-VSMEC | `6845ECDB3B71B428CF63DB463BADB7F61166C3ABD5A6B57DC2D02D52B54663DC` | `41CF45EC92438AA5987BCB6D9F3BF3ADA704AB0225D34EA734A8537651B89D83` | `8DF340880EA8E39A16A7690466F8E71C09D33BB463BD31FDF820A2A625609854` | same as raw text |

Prediction outputs are appropriately different between the models. The number of differing prediction records for standard-versus-primary was AIVIVN 125/129/126, UIT-VSFC 361/393/438, and UIT-VSMEC 110/100/135 for seeds 21/22/23 respectively. Thus the parity check is not comparing duplicated predictions.

## Gate decision

- **Observed-record parity:** PASS for IDs, gold labels, raw text, NFC text, counts, order, and label vocabularies on all three datasets and all three seeds.
- **Manifest parity:** PASS for VSFC and VSMEC; **HOLD** for AIVIVN because the source normalized-test hashes differ.
- **Label-route audit:** observed label vocabularies and dataset/task filenames are consistent with polarity for AIVIVN/VSFC and emotion for VSMEC. The primary follow-up manifest does not independently expose the complete label-routing map, so route equivalence is recorded as observed/consistent rather than independently proven from a shared manifest.
- **Descriptive metric reporting:** the executed record-level and metric-recomputation checks permit descriptive same-observed-cohort comparison. The standard baseline ordinary-F1 mean is `0.4253413954 ± 0.0052422067`; the primary mean is `0.4195557559 ± 0.0178125657`; the descriptive difference is approximately `0.005786` in favor of the standard baseline. This is not a causal or statistical-significance claim.
- **Provenance limitation:** the AIVIVN manifest-byte hash difference remains unexplained. It is not called formatting-only, but it does not block descriptive comparison after all records were shown identical by ID, gold, raw text, and NFC-normalized text. The canonical primary row is `ANALYZED_SEMANTIC_COHORT_VERIFIED`.

The reproducibility script `q1b_reproduce_summaries.py` recomputes the saved means and sample SDs, requires exactly seeds 20260521/22/23 for stochastic recipes, and validates all evidence CSV row widths. The executed `q1b_split_gold_parity.py` additionally discovers the actual baseline tar paths, caches and hashes HTTP payloads, checks expected counts 1,609/3,166/693, nonempty unique IDs, explicit labels (including neutral), compares all records, and recomputes each stored per-seed F1 within `5e-11` (the CSV stores rounded 10-decimal canonical summaries). Its output is in `q1b_split_gold_parity.log`; payload URLs, byte counts, full payload SHA-256 values, and cache paths are in `q1b_http_payload_manifest.csv` and `q1b_http_cache/`. `q1b_provenance.csv` is the per-seed source ledger; all recorded SHA-256 fields are full 64-character values, while unavailable source fields are explicit `NOT_RECORDED`.
