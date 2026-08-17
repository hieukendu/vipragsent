# V28 Sentinel review dossier

## V30 descendant-closure draft

The historical V28 failure remains preserved. The V30 descendant source head is
f438a61a078e713cfa94c5624b6b0e19b719651e, but its exact-head CI and
independent affected-scope Sentinel are both PENDING. Consequently this
report keeps final_pass: false; it must be closed only after both checks
complete against the final exact head.

The prior exact-head Sentinel review found `SENTINEL-001` through `SENTINEL-005`, followed by F-002/F-003 Azure findings. The complete repair wave is integrated at source head `168254eb5df094924a49f0363d2403af4c87b35c`.

The evidence refresh records the exact successful `cpu-ci` run `31988858252` (job `95268598734`) for that source head. A fresh independent Sentinel must still review the resulting live PR head; this dossier is not yet a final pass. Local evidence is green: 389 CPU/mock-only tests, 65 Azure/cache/ceiling regressions, compilation, Ruff, and diff check.
