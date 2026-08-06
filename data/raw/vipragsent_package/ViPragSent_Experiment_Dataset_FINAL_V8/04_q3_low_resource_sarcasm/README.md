# Q3 low-resource sarcasm setup

- Train split sarcasm positives: 545
- Train split negatives: 7453
- Subset seed: 20260524
- Positive subsets are nested.
- Every budget retains the full negative pool.
- Positive samples outside the current budget have:
  - `sarcasm_target_mask = 0`
  - `rationale_loss_mask = 0`
- Other task labels remain available unless the training code explicitly masks them.
- GPT-4o-mini 8-shot demonstrations must be selected only from the current budget's eligible pool.
