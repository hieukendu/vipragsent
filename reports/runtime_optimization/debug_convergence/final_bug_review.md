# V28 finding-driven convergence checkpoint

The integrated source repair head is `168254eb5df094924a49f0363d2403af4c87b35c`. This checkpoint supersedes the stale V27 final-pass claim. GitHub `cpu-ci` run `31988858252` (job `95268598734`) completed successfully for that code head; a fresh independent Sentinel review of the live PR head remains pending.

The five findings from the prior exact-head review were addressed in disjoint scopes:

- `SENTINEL-001`: exact-head report/PR evidence is being refreshed.
- `SENTINEL-002`: generation and explanation resume now require the canonical full identity contract.
- `SENTINEL-003`: Azure paths now enforce finite global safety ceilings and actual-usage checks.
- `SENTINEL-004`: retryable Azure failures are excluded from reusable cache entries.
- `SENTINEL-005`: the NAACL report exclusion set is checked against the retained YAML inventory.

Local evidence is green: 346 CPU/mock-only tests, impacted regression suites, compilation, Ruff, and diff check. The final remote CI result and exact-head Sentinel verdict are intentionally left open until the evidence commit is pushed. No production, Azure, Hugging Face, model-download, benchmark, TEST, process-control, or merge action occurred.
