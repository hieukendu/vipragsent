# Corrected Q3 low-resource sarcasm masks

Derived from the frozen nested positive subsets and fixed negative pool.
For every out-of-budget sarcasm-positive row, only `sarcasm_target_mask` is zero.
`rationale_loss_mask` remains one for every sarcasm-positive row, including
out-of-budget positives; fixed negative rows retain the locked zero mask.
Polarity, emotion, and all non-sarcasm pragmatic targets remain active.
