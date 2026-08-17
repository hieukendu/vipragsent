# V28 closed by V30 descendant repair

The historical V28 Sentinel failure and all listed findings remain preserved
as historical evidence. The V30 descendant closes the active repair state at
source head `64945bc0fb4f154f59062647b1c91cb9c4dfa003`, with reviewed report
evidence head `3de2d5c1c9a34dbe7030f86a65cf22978751048`.

Source-head CI `32001664211`/job `95303031902` and reviewed report-head CI
`32002103939`/job `95304262130` are green. The independent exact-head
Sentinel is PASS with no blockers. R5 hash-boundary validation and the
corrected bottom-up ETA are included; the accepted protocol and safety
boundary are unchanged.
