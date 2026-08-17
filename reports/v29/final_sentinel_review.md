# ViPragSent V29 runtime convergence evidence

## Historical evidence

The historical V29 findings P0-1 through P1-4 remain PASS. The V29 P2
checkpoint-copy decision remains preserved as historically DEFERRED; its V30
R3 descendant repair was implemented on the parent source line.

## V30 auxiliary descendant state

V30 auxiliary R5 checkpoint-hash deduplication is implemented at source head
`64945bc0fb4f154f59062647b1c91cb9c4dfa003` and has 11 focused regression tests. The final descendant
closure is pending exact-head CI and a fresh independent Sentinel for that
source head. Parent-head evidence is historical until revalidation.

Q3/Q2/Q1b protocol, seeds, budgets, DEV selection, TEST access, and Q1b
evaluation-only semantics remain unchanged. No production or external action
was taken.
