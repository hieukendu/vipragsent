# ViPragSent safe pause and remote backup

- Backup status: **PASS**
- Pipeline status: **SAFELY_PAUSED**
- Paused run: `q1a_cot_only_vistral_20260521`
- Paused stage: `train_generation`
- Active GPU jobs: **0**
- The latest resume attempt passed preflight and began epoch 2 training, then was gracefully interrupted before any epoch-2 checkpoint or metrics were committed.
- Resume boundary: `results/runs/q1a_cot_only_vistral_20260521/checkpoints/epoch_1/model.pt`
- Resume mode: `checkpoint_boundary`
- Replay required: epoch 2 training, epoch 2 dev generation/judging/metrics, and all downstream stages

## Checkpoint selection

Canonical `best` checkpoints for approved runs and both epoch-1/best aliases for the paused run were selected. Upstream base weights and duplicate `latest`/epoch checkpoints were excluded; no model files were committed to GitHub or the general artifact repository.

Expected canonical entries: **35**; remotely verified: **35**; blocked: **0**.
Incremental result: **33** existing checkpoints reused; **2** previously missing checkpoints uploaded; **0** still missing.
Hugging Face visibility invariant: **PUBLIC → PUBLIC** for all five target repositories.
The changed manifest is preserved at the new timestamped artifact paths `server_20260808/incremental_backup_20260808T174018/reports/vipragsent_safe_pause_backup.json` and `.md`; earlier remote report history was not overwritten.

| Status | Family | Remote repository | Remote path | Bytes | SHA-256 |
|---|---|---|---|---:|---|
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_finetune_20260521/checkpoints/best/model.pt` | 1613241834 | `03f9a5a028578b54b9af792ae2d137693ee015a67db301114b397e9b8bd0cbdc` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_finetune_20260522/checkpoints/best/model.pt` | 1613241962 | `7824eb4f042a14429925ad86f35802c96b58e90c05ee336171044eea27dee9d1` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_finetune_20260523/checkpoints/best/model.pt` | 1613241962 | `3267cc2799fe2bce1adec8632da0a2ae3f1426cc0ff2a0102f203232af53d39a` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260521/components/code_switching/checkpoints/best/model.pt` | 1613111701 | `ed32a57eb03722bc1aaaaf0100db434e43ea1f9ec5dba3601a25b1a804af932b` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260521/components/idiom_figurative/checkpoints/best/model.pt` | 1613111701 | `c9a71d9a6b01a0d4e26ec9747a32d1297b9e8c87f342c192d6dd01891b60834a` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260521/components/implicit_sentiment/checkpoints/best/model.pt` | 1613111701 | `b5783e62daf6b542cc8959426a14e46b17bf8dec3615e930f00385341bae5003` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260521/components/irony/checkpoints/best/model.pt` | 1613111701 | `ccd46fafe7b5adda5f6dcf203e46ac726c636a3422ef849517799dc8dc200ffb` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260521/components/mocking/checkpoints/best/model.pt` | 1613111701 | `736dbf6b11534b8320018b0900c824b2d74e0720533958aa05fed43e2ac6ed56` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260521/components/sarcasm/checkpoints/best/model.pt` | 1613111701 | `7fc537a3d984bd2dc70f8e9b352733420749cf4bd21bd47221b06d040135c6a4` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260522/components/code_switching/checkpoints/best/model.pt` | 1613111701 | `a28c992e05df01548ed912d72e8aecf0c885487f2df1be06ccc02c2df43957bb` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260522/components/idiom_figurative/checkpoints/best/model.pt` | 1613111701 | `e3a18b7f81e39ba8aa366d7d9f5979f96eb378180b61b48c960fc8f9d097f012` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260522/components/implicit_sentiment/checkpoints/best/model.pt` | 1613111701 | `227b3fb8c019fa77c0817249932f4564188e7ae59e91d65b708f1ddfd5c71105` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260522/components/irony/checkpoints/best/model.pt` | 1613111701 | `c3e7a3f4e056ebc768a69a15908089202fd1ea6f45bf68f336ddafbf5ce223ef` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260522/components/mocking/checkpoints/best/model.pt` | 1613111701 | `22a807ab6d5c8a7929c094145f736227c911869f922da2fa8add762b636aaaed` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260522/components/sarcasm/checkpoints/best/model.pt` | 1613111701 | `30cab74daaab9f1fed8856b92553f492cc59905a5bf2f73c092aa80343533d98` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260523/components/code_switching/checkpoints/best/model.pt` | 1613111701 | `ea7f68097a4a916d22410e57eaeca6fafcd93e4b8abc7040c4ae86c81ac4bda4` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260523/components/idiom_figurative/checkpoints/best/model.pt` | 1613111701 | `94d8dbff6c98b9098fd4d4380761815e8f1fc4b61425d097d8f25d337ddb6a95` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260523/components/implicit_sentiment/checkpoints/best/model.pt` | 1613111701 | `cbd120a1d22963a1b06057bf36683d1bf773c4b1501934702406a2ebdba84672` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260523/components/irony/checkpoints/best/model.pt` | 1613111701 | `13a36a164fe6444cb019c9a4914969ee687eff784d8d989a8d93976c8c4291fa` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260523/components/mocking/checkpoints/best/model.pt` | 1613111701 | `ba07ca84617079a4df6a0f7d4b3ee4d6b72682da01e4a5388810ed05d68cc143` |
| REMOTE_VERIFIED | phobert_base | Thundergod2007/vipragsent-phobert-checkpoints | `q1a_phobert_pragmatic_single_task_20260523/components/sarcasm/checkpoints/best/model.pt` | 1613111701 | `8d521d5a02a3466f7604d3befda067adebf3369ba862a5023310dac7314b9125` |
| REMOTE_VERIFIED | sailor_7b | Thundergod2007/vipragsent-sailor7b-checkpoints | `q1a_sailor_pragmatic_sft_20260521/checkpoints/best/model.pt` | 5934727186 | `282d13dea3883b4b483f9845e2e2e675f2c54198f011f92efdfdbcfaee45804b` |
| REMOTE_VERIFIED | sailor_7b | Thundergod2007/vipragsent-sailor7b-checkpoints | `q1a_sailor_pragmatic_sft_20260522/checkpoints/best/model.pt` | 5934727314 | `248363853004584fbf944d60212760d61178299882c7a11c12426d920364e135` |
| REMOTE_VERIFIED | sailor_7b | Thundergod2007/vipragsent-sailor7b-checkpoints | `q1a_sailor_pragmatic_sft_20260523/checkpoints/best/model.pt` | 5934727314 | `1c49ea2a60937d652e3136ab28b07f312b27696f71c786a1788ad7e328f5fdfd` |
| REMOTE_VERIFIED | vistral_7b | Thundergod2007/vipragsent-vistral7b-checkpoints | `q1a_cot_only_vistral_20260521/checkpoints/best/model.pt` | 4942159972 | `27f81249de067555827099334acb999f009d0d5970df333e855be231f36fc7c0` |
| REMOTE_VERIFIED | vistral_7b | Thundergod2007/vipragsent-vistral7b-checkpoints | `q1a_cot_only_vistral_20260521/checkpoints/epoch_1/model.pt` | 4942159972 | `27f81249de067555827099334acb999f009d0d5970df333e855be231f36fc7c0` |
| REMOTE_VERIFIED | vistral_7b | Thundergod2007/vipragsent-vistral7b-checkpoints | `q1a_vipragsent_no_auxiliary_vistral_20260521/checkpoints/best/model.pt` | 4313881224 | `78360dda9efb2a925da616047b25c2dc9852af93436c9bd5c0988e6df39f89ba` |
| REMOTE_VERIFIED | vistral_7b | Thundergod2007/vipragsent-vistral7b-checkpoints | `q1a_vipragsent_no_auxiliary_vistral_20260522/checkpoints/best/model.pt` | 4313881288 | `608e88704983b7c2291e933a7b85ba1a79b457803797828c38bba57466291f74` |
| REMOTE_VERIFIED | vistral_7b | Thundergod2007/vipragsent-vistral7b-checkpoints | `q1a_vipragsent_no_auxiliary_vistral_20260523/checkpoints/best/model.pt` | 4313881352 | `64958cff3a21210dcdb0e3084cbd25541d22dcd56de35b9416bd3735e04f0fac` |
| REMOTE_VERIFIED | vistral_7b | Thundergod2007/vipragsent-vistral7b-checkpoints | `q1a_vistral_pragmatic_sft_20260521/checkpoints/best/model.pt` | 4313875610 | `5129514951b1d137256ef60b09b11c9f283e73cc35f376af0b5271b2c9478a70` |
| REMOTE_VERIFIED | vistral_7b | Thundergod2007/vipragsent-vistral7b-checkpoints | `q1a_vistral_pragmatic_sft_20260522/checkpoints/best/model.pt` | 4313875674 | `c341b4ec539ed4340d3194a88c2aab048e3232805dd74ce51695487bb8affcd4` |
| REMOTE_VERIFIED | vistral_7b | Thundergod2007/vipragsent-vistral7b-checkpoints | `q1a_vistral_pragmatic_sft_20260523/checkpoints/best/model.pt` | 4313875802 | `f528e18fc8da1bf9a3c9b3d35073936bd7d21f39884a53f686f5571a326f5f72` |
| REMOTE_VERIFIED | xlmr_large | Thundergod2007/vipragsent-xlmr-checkpoints | `q1a_xlmr_pragmatic_finetune_20260521/checkpoints/best/model.pt` | 6706720214 | `743d0df325f6313f038fa53ab0f5be03a94f496a2b2d30caa1f7b09087c9aabb` |
| REMOTE_VERIFIED | xlmr_large | Thundergod2007/vipragsent-xlmr-checkpoints | `q1a_xlmr_pragmatic_finetune_20260522/checkpoints/best/model.pt` | 6706720342 | `324f416164096fda9e095c63810153667a480a21499816a7d159d7b6d434be92` |
| REMOTE_VERIFIED | xlmr_large | Thundergod2007/vipragsent-xlmr-checkpoints | `q1a_xlmr_pragmatic_finetune_20260523/checkpoints/best/model.pt` | 6706720342 | `680662efae10721c217d9de12b635ad523e34960c7aa6fe02d41ad8d95e5a5e4` |

## Hugging Face verification

- `phobert_base`: Thundergod2007/vipragsent-phobert-checkpoints, revision `14097061c088f3deefe8ea6d9288d77755f36b5a`, public visibility verified before and after upload, authenticated and anonymous access verified.
- `sailor_7b`: Thundergod2007/vipragsent-sailor7b-checkpoints, revision `6af58b4c754d08f9d6316aef58c239bca60397ed`, public visibility verified before and after upload, authenticated and anonymous access verified.
- `vistral_7b`: Thundergod2007/vipragsent-vistral7b-checkpoints, revision `77df04ee8d69ec8e9eba89136040ec1b3e6152ce`, public visibility verified before and after upload, authenticated and anonymous access verified.
- `xlmr_large`: Thundergod2007/vipragsent-xlmr-checkpoints, revision `d004da4621be813a1f3ce8ef3d171dce6102fbe2`, public visibility verified before and after upload, authenticated and anonymous access verified.
- Experiment artifacts: Thundergod2007/vipragsent-experiment-artifacts, verified revision `98806b8c41aa6d95f7e32f97d39e921e920dd62e`, public visibility verified before and after upload, authenticated and anonymous access verified, prefix `server_20260808/`, 8975 files / 273265548 bytes, model files: 0.

## Historical provider blocker and resolution

An earlier attempt was rejected with: `Private repository storage limit reached, please upgrade your plan to increase your private storage limit`. After the target repository was verified PUBLIC, the two affected checkpoints were uploaded incrementally to the correct PhoBERT repository and verified by size and SHA-256. No cross-family move occurred; the historical provider error is preserved for provenance.

## GitHub/recovery provenance

- Remote: https://github.com/hieukendu/vipragsent.git
- Branch: `agent/phase15-handoff-persistence`
- Source HEAD before this backup refresh: `9ef3c6746df249baebe153eb8e804c0e84db61dd`
- Source tree before this backup refresh: `a6f79fa14bcc300673c2c0ef455d60193fdb25e3`
- Local `.env`, credentials, tokens, base-model weights, and fine-tuned model files remain out of GitHub.
- The future recovery flow is: clone the repository, create `.env`, provide the existing Master Prompt, verify the manifest, restore the public checkpoints, and resume the same paused run.

Scientific protocol: **UNCHANGED_BY_BACKUP**.
