# Requirement traceability

| Requirement | Evidence/status |
|---|---|
| Exact master prompt | V26 SHA256 `457da887b325625b395b2dc63576bed95c02e95c601be68c62f61f97f53a8ed0` recorded in artifacts. |
| Safely paused production run | `q1a_cot_only_vistral_20260521`, after epoch 2; active PID none. |
| State/approval gate | `RUNNING_STALE` / `PENDING_USER_APPROVAL` / `NO`. |
| Checkpoint evidence | Local `results/runs/q1a_cot_only_vistral_20260521/checkpoints/epoch_2/model.pt`, 4,942,818,023 bytes, SHA256 `d3fe7e99bb0758e527c4967fe2a8502c722fce8d86cd7664ea776440a3b41a77`; HF repo/path/revision recorded in inventory and verified read-only. |
| Identity warning | `LIVE_CODE_IDENTITY_UNCERTAIN`; paused process absent. |
| Worktree safety | `/root/vipragsent` dirty; no changes made. |
| Isolated base | Clean source commit `/root/vipragsent-runtime-opt` at `fb40c91a7c39ac575db2bd71d9957f0e89069b3e`; current worktree intentionally contains Wave-0 reports and is not used as a literal source-clean assertion. |
| Source/run binding | State: code commit `fb40c91...`, tree `a670b1...`, source fingerprint `2daf51...`; run manifest: `a765b2...`; identity remains uncertain and reuse is blocked. |
| Model/config/data binding | Model/tokenizer `d331b64e61b935cc43c2b3010ae9fb4fde599b45`; config `3F6AF966D078AF57CC8989269380B2DD1C44D2402540AD21E9B5C419A8743642`; data `B906C090400BAE115C9C5E3C35E32FA410AC519AE09209EBA741F198087C24F9`. |
| Environment | Python `3.11.0rc1`; torch `2.4.0+cu124`; transformers `4.46.3`. |
| Hardware/telemetry | H100 MIG `2g.20gb`; telemetry `PARTIAL`. |
| Forbidden actions | No source edits, production/Azure/HF writes, benchmarks, or process control. |
| Wave-0 DAG | Audit Builder → independent Sentinel → Manager gate → serialized generation/checkpoint work and disjoint scheduling/Azure/reuse work; package contracts are recorded in the ledger. |
