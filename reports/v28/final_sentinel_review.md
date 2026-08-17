# V28 Sentinel review dossier

## Historical review preserved

The historical V28 Sentinel failure and all listed findings remain preserved.
The parent repair wave passed its historical CPU/mock and CI validation.

## V30 auxiliary descendant state

V30 auxiliary R5 checkpoint-hash deduplication is implemented at source head
`64945bc0fb4f154f59062647b1c91cb9c4dfa003` and has 11 focused regression tests. Exact-head CI and a
fresh independent Sentinel are pending for this new source head, so descendant
final_pass remains false until revalidation. The accepted protocol and safety
boundary are unchanged.
