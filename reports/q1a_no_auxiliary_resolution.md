# Q1a no-auxiliary resolution

Status: `RESOLVED`

The no-auxiliary system is the distinct `vipragsent_no_auxiliary_vistral` system. It is not a reuse of the `vistral_pragmatic_sft` baseline checkpoint identity.

It exposes exactly the six pragmatic binary heads, excludes polarity, emotion, and rationale heads from the model forward path and optimizer, and uses six independent homoscedastic uncertainty parameters initialized at zero with zero weight decay. The baseline remains equal-weight and has no uncertainty parameters. Selection and threshold protocols are unchanged.
