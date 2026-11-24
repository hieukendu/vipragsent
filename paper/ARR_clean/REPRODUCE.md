# Verification scope

The canonical source is this folder's `FILE_MANIFEST.json` and `CHECKSUMS.sha256`.
`verify_release.py` verifies actual file existence before checksum comparison,
split disjointness, exact gold/text alignment, annotation and rationale coverage,
train-only weights and all six Q3 masks. Mask bytes match the original corrected
XLM-R run hashes after CRLF-to-LF normalization; no labels or positive subsets
were regenerated.

`validate_saved_results.py` recomputes F1 from 257 aggregate run/split outputs,
mean/sample SD for the ten Table 1 rows, six external Table 2 rows, eight Table 3
rows and eighteen Table 4 groups. Calibration bins independently validate primary
pragmatic test ECE; Q2 uses the primary recipe, not the removed follow-up cohort.
All six external systems have three-seed confusion counts. The original Vistral
budget-32 grid independently reselects all six dev thresholds from retained counts.
These counts do not reverify unavailable checkpoint tensors.

Seven paired bootstrap intervals match a saved reference. They were not freshly
resampled. Dataset statistics are retained in `PAPER_STATISTICS.json`; pair reports
have an explicit verification boundary. No fresh training/API call, annotation
replay, original extraction receipt or full-paper reproduction is claimed.

The numerical reference is PDF version 7. Its Appendix B claims 17 posthoc
notes and section 2.3 mentions existing notes. No historical
posthoc notes or original case-level adjudication records are included in this
release. The user confirmed both historical posthoc descriptions and reported
losing the supporting information/material; recovery remains waived.
`AVAILABILITY.json` distinguishes the user-reported historical description from
current release availability, without treating unavailable evidence as proof of
nonexistence. The operational rubric does not replace those records. See
`code/docs/POSTHOC_EVIDENCE_SCOPE.md`. The PDF wording is retained and the
historical claim is not independently verified from the available artifacts.

`replay/HUB_AUDIT.json` records complete paginated inventories and bounded content
checks without owner identities. Exact source locations are retained privately
outside the anonymous packages. Prediction records fetched for calculations are
deleted after aggregation. Software and data are exported as separate ZIPs.
