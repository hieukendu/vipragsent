# V28 runtime estimate after the repair wave

The integrated source head is `acc6467864bcea299862f5b0e29c7247cef7afde`. Numeric campaign ETA and speedup credit remain intentionally unset: no real-model throughput/concurrency profile or live Azure usage was authorized, and the paused historical run cannot be reused until its source identity is reconciled.

The code now contains bounded checkpoint/resume provenance, profile-gated generation batching, atomic generation chunks, bounded async judging, synchronous Azure safety ledgers, retry/cache recovery, and profile exclusion parity. Local CPU/mock evidence is 346 passing tests plus clean compilation, Ruff, and diff checks.

Readiness remains `PROJECTED_GATE_CONDITIONAL`. Any future numeric ETA must be produced from an authorized DEV-only measurement and a separately reconciled source/checkpoint identity; no unmeasured runtime credit is included here.
