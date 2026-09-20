# Q1b extraction checkpoint: PhoBERT multitask 8-head, seed 20260521

This is a concrete inspectable artifact checkpoint, not a claim that the full Q1b audit is complete.

## Source and pinning

- HF repository: `Thundergod2007/vipragsent-experiment-artifacts-overflow-002`
- Remote commit for the tar path: `31f30118cf9fb1a1379ca12dfc33cf7c9b73e302`
- Tar path: `campaigns/vipragsent-v7-compact-20260820-190bb0f24a7e75a8/SHARD_H100_MIG20_COMPACT/nb-170d4c47-8a39-47ad-9425-d231de44fc92-857677f648-2qn8f/science/final/q1b_phobert_multitask_8head_20260521/FINAL_SCIENCE_RESULT_6D605434FE7661049218F5B83BC59680EFFE1106BF5169302B7EA4481BAF08FC.tar.gz`
- Tar/LFS SHA-256: `6D605434FE7661049218F5B83BC59680EFFE1106BF5169302B7EA4481BAF08FC`
- Run ID: `q1b_phobert_multitask_8head_20260521`; seed: `20260521`
- Source checkpoint SHA-256: `0038949F67B1CD0C2B87B0FD402F8CEA16989877ABAD7F3100A270C4B9A39754`
- Model: `vinai/phobert-base`, revision `01daacda68afe13d83023d16ec647239e344a1e6`
- Code commit: `f163bd9ff1607cebf6f3b008388857a3004c6303`
- Data fingerprint: `A13573E38550ABD55D7F63E983C602BEA13C2765A1FFECA285F718478532AF0D`

## Extracted metrics

From `metrics/external_retention_metrics.json` and matching `metrics/test_metrics.json`:

| External split | Label route | Macro-F1 |
|---|---|---:|
| AIVIVN | polarity | 0.4581070139 |
| UIT-VSFC | polarity | 0.3254557942 |
| VSMEC | emotion | 0.3607523722 |
| ordinary-F1 aggregate (`ord_f1`) | three-dataset ordinary mean in artifact | 0.3814383935 |

The artifact declares `external_preprocessing=locked_normalized_test_only`, `external_finetuning=false`, `partial=false`, and `status=PASS`. The external manifest declares `predictor_factory=disk_backed_q1b_v2`, `matrix.key=phobert_multitask`, polarity output `polarity_head`, emotion output `emotion_head`, and source seed `20260521`.

## Split/gold comparability

The pack records the shared normalized test hashes:

- AIVIVN: `F6F6A78EBADE3BC1038A3B1CFE05FF44E172AF024BBBC8A31079EFCF8133C6D3`
- UIT-VSFC: `C27B61C70DE1090BD0A12F0F2824148D1691C68FB9CB18A07F23AE5282E75D96`
- VSMEC: `1D2E604BD06A4AAAAD33A3B98DE2FB910E9020F4D468EF8BC16DDCDEADB3D955`

The pack contains prediction files for all three splits:

`predictions/aivivn_test_predictions.jsonl`, `predictions/uit_vsfc_test_predictions.jsonl`, and `predictions/uit_vsmec_test_predictions.jsonl`.

Provenance hashes for those files are recorded in `provenance.json`: `245D3AF21B7D23ABF805D037861E2613F6BC5B66CF16B991318FEF3B30994D63`, `5C77AFBB0357855DB6CFED6ABB553E86BCA93FB3FD4C076747C34531701F6C57`, and `9BE147EB22B9C54C15C442CCEB1A26D3A64BAF8820DB089843277CB19F813`, respectively. The external manifest hash is `64449CF1D90579AD85B05CA3F4AE2EC211145045C38EAF192A866DA51A8D3EF2`.

## Disposition

This is a legitimate comparison baseline with an explicit actual recipe (`phobert_multitask_8head`), not a silently relabelled historical family. It is `ANALYZED` from saved artifacts; no independent rerun has been performed, so it is not labelled `VERIFIED` under the evidence workflow. It is not the primary ViPragSent row.
