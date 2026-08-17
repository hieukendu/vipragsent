# V28 runtime estimate after the repair wave

The integrated source head is `168254eb5df094924a49f0363d2403af4c87b35c`. Exact code-head CI run `31988858252` is green. Numeric campaign ETA and speedup credit remain intentionally unset: no real-model throughput/concurrency profile or live Azure usage was authorized, and the paused historical run cannot be reused until its source identity is reconciled.

The code now contains bounded checkpoint/resume provenance, profile-gated generation batching, atomic generation chunks, bounded async judging, synchronous Azure safety ledgers, finite ceiling validation, cache identity validation, retry/cache recovery, and profile exclusion parity. Local CPU/mock evidence is 389 passing tests plus 65 Azure/cache/ceiling regressions and clean compilation, Ruff, and diff checks.

Readiness remains `PROJECTED_GATE_CONDITIONAL`. Any future numeric ETA must be produced from an authorized DEV-only measurement and a separately reconciled source/checkpoint identity; no unmeasured runtime credit is included here.
