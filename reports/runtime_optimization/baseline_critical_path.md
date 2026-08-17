# Baseline critical path

Static audit baseline only; no model/data benchmark or timing measurement was run.

The known production boundary is a paused run after epoch 2, with epoch-2 checkpoint locally present and the exact HF remote path already verified read-only. The authoritative state records the checkpoint as 4,942,818,023 bytes with SHA256 `d3fe7e99bb0758e527c4967fe2a8502c722fce8d86cd7664ea776440a3b41a77`. The source-visible critical path is: Vistral train-generation (GPU/optimizer), DEV generation (GPU), DEV judge/persistence (serial dependency and possible Azure), checkpoint/storage persistence, then downstream freeze/TEST/export; exact remaining order must be reconciled against the absent paused process and dirty production source.

Static code inspection identifies per-record generation and per-row judging/persistence as implementation-serial boundaries, while train→DEV feedback→selection→TEST is a scientific dependency that must remain ordered. Existing committed epoch-2 artifacts are credited once and are not re-counted as future work. No wall-clock contribution is claimable from these facts.

Evidence constraints: production code is dirty, the paused process is absent, and GPU telemetry is partial. Therefore no throughput, latency, cost, quality, or speedup baseline is claimable. Future review should establish checkpoint/resume, generation, persistence, scheduling, and artifact-reuse boundaries using fixtures/dry runs before any authorized benchmark.
