# Generation baseline protocol resolution

Status: `RESOLVED`

The former generation-baseline ambiguity is resolved by the explicit user-approved protocol. Both systems use the same frozen reasoning-only judge; CoT-only trains a causal generation checkpoint, while explanation-only reuses the approved same-seed full Vistral checkpoint and uses only its rationale decoder.

No new experiment row was added and no real model, Azure request, training run, or test inference was executed.
