# Hugging Face reuse audit

As of `2026-08-16T14:26:19Z`, the five configured Hugging Face repositories were inspected through read-only metadata APIs. No checkpoint blob was downloaded and no repository was uploaded, deleted, rewritten, or retagged.

| Repository | Revision | Files | Relevant checkpoint evidence | Reuse decision |
|---|---:|---:|---|---|
| `vipragsent-experiment-artifacts` | `7afa8164aa5224688f430ee8e30283e72e462d77` | 8,979 | server artifact snapshot; no model-weight blobs | blocked pending identity/approval binding |
| `vipragsent-phobert-checkpoints` | `14097061c088f3deefe8ea6d9288d77755f36b5a` | 23 | 21 best checkpoints with remote LFS SHA-256 recorded in JSON | blocked pending config/data/model/tokenizer binding |
| `vipragsent-vistral7b-checkpoints` | `1aadf557a77853cf19946a747f9fe5c40a8288b0` | 11 | 9 best/epoch checkpoints; epoch-2 SHA-256 matches the local observed file | blocked because live code identity is uncertain |
| `vipragsent-xlmr-checkpoints` | `d004da4621be813a1f3ce8ef3d171dce6102fbe2` | 5 | 3 best checkpoints with remote LFS SHA-256 recorded in JSON | blocked pending config/data/model/tokenizer binding |
| `vipragsent-sailor7b-checkpoints` | `6af58b4c754d08f9d6316aef58c239bca60397ed` | 5 | 3 best checkpoints with remote LFS SHA-256 recorded in JSON | blocked pending config/data/model/tokenizer binding |

The complete checkpoint path/size/SHA-256 table is in [`hf_reuse_audit.json`](hf_reuse_audit.json). Regular JSON artifacts in the artifact repository expose an HF blob ID rather than a SHA-256 through the metadata response; those entries explicitly retain `sha256: null` and are not treated as verified checkpoint identity.

The epoch-2 Vistral checkpoint remains candidate evidence only. The authoritative live snapshot has a dirty production worktree, no active PID, and conflicting run code identities, so this PR makes no automatic `REUSE` or `RESUME` decision.
