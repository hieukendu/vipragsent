# Wave-0 live state snapshot

Audit mode: read-only. Prompt V26 SHA256: `457da887b325625b395b2dc63576bed95c02e95c601be68c62f61f97f53a8ed0`.

Run `q1a_cot_only_vistral_20260521` is safely paused after epoch 2. Active PID: none. Recorded state: `RUNNING_STALE`; approval: `PENDING_USER_APPROVAL`; next run allowed: `NO`. Local epoch-2 checkpoint and its exact HF remote path were already verified read-only. No process control, source edit, production/Azure/HF write, model/data benchmark, or evaluation was performed.

Identity is `LIVE_CODE_IDENTITY_UNCERTAIN` because the paused process is absent. Production worktree `/root/vipragsent` is dirty and must not be changed. The optimization worktree has intentional Wave-0 report files (and is therefore not literally clean); its clean source base is commit `fb40c91a7c39ac575db2bd71d9957f0e89069b3e`. Future Builders must materialize a fresh source worktree at that commit and keep report artifacts separate from source-clean assertions.

Checkpoint provenance observed from the authoritative run state: local `results/runs/q1a_cot_only_vistral_20260521/checkpoints/epoch_2/model.pt`, 4,942,818,023 bytes, SHA256 `d3fe7e99bb0758e527c4967fe2a8502c722fce8d86cd7664ea776440a3b41a77`; HF repo `Thundergod2007/vipragsent-vistral7b-checkpoints`, path `q1a_cot_only_vistral_20260521/checkpoints/epoch_2/model.pt`, revision `1aadf557a77853cf19946a747f9fe5c40a8288b0`, `remote_verified: true`. This remains a candidate only: the local state records code commit `fb40c91a7c39ac575db2bd71d9957f0e89069b3e`, tree `a670b1ca9af0a6921b2f0d7f194bfa29fe568c6d`, source fingerprint `2daf51d98fa18b076a4020dada95dcbf8320304abcc0440b684a99291ec6500e`, while `run_manifest.json` records a different code commit `a765b2bca625ff66cf97dc608eacb3a3c63553b5`; this conflict prevents identity-bound reuse.

The run also records model/tokenizer revision `d331b64e61b935cc43c2b3010ae9fb4fde599b45`, training config hash `3F6AF966D078AF57CC8989269380B2DD1C44D2402540AD21E9B5C419A8743642`, checkpoint data hash `B906C090400BAE115C9C5E3C35E32FA410AC519AE09209EBA741F198087C24F9`, and processed-data fingerprint `7C39BEEBC462D1F9076F5DF1565E924BD884263D5503F23398BD83A6BF4205EB`.

Runtime: H100 MIG `2g.20gb`; telemetry `PARTIAL`. Environment: Python `3.11.0rc1`, torch `2.4.0+cu124`, transformers `4.46.3`. Evidence quality is mixed: pause/checkpoint/path facts are existing read-only evidence; loaded-code identity and live telemetry are unresolved.
