# Authenticated Hugging Face repository inventory

Status: `PASS`

The existing repository `.env` was loaded with `override=false`; only presence was recorded (`HF_TOKEN_STATUS: PRESENT`). All five required repositories and their authoritative backup revisions were inspected with the authenticated Hugging Face Hub API. No token or credential value is recorded here.

| Repository | Expected revision | Current remote SHA | Expected files | Local recovery path | Access/materialization |
|---|---|---|---:|---|---|
| `Thundergod2007/vipragsent-experiment-artifacts` | `98806b8c41aa6d95f7e32f97d39e921e920dd62e` | `7afa8164aa5224688f430ee8e30283e72e462d77` | 8977 | `/root/vipragsent_hf_backup/vipragsent-experiment-artifacts-git` | PASS; 1/1 LFS file real |
| `Thundergod2007/vipragsent-phobert-checkpoints` | `14097061c088f3deefe8ea6d9288d77755f36b5a` | same | 23 | `/root/vipragsent_hf_backup/vipragsent-phobert-checkpoints-git` | PASS; 21/21 required files real |
| `Thundergod2007/vipragsent-vistral7b-checkpoints` | `77df04ee8d69ec8e9eba89136040ec1b3e6152ce` | same | 10 | `/root/vipragsent_hf_backup/vipragsent-vistral7b-checkpoints-git` | PASS; 8/8 required files real |
| `Thundergod2007/vipragsent-xlmr-checkpoints` | `d004da4621be813a1f3ce8ef3d171dce6102fbe2` | same | 5 | `/root/vipragsent_hf_backup/vipragsent-xlmr-checkpoints-git` | PASS; 3/3 required files real |
| `Thundergod2007/vipragsent-sailor7b-checkpoints` | `6af58b4c754d08f9d6316aef58c239bca60397ed` | same | 5 | `/root/vipragsent_hf_backup/vipragsent-sailor7b-checkpoints-git` | PASS; 3/3 required files real |

The artifact repository has a newer tip containing a second verified safe-pause manifest pair. It was inspected and not adopted as the checkpoint source; the pinned revision remains authoritative. A transient incomplete LFS cache residue remains in the PhoBERT archive, but all 21 required worktree files are real binaries and pass the separate manifest-driven size/SHA check.

The complete checkpoint inventory result is recorded in `fresh_server_setup_report.json`: 35 expected, 35 verified, zero missing, zero size mismatches, and zero SHA mismatches.
