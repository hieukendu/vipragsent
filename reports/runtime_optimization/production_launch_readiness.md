# Production launch readiness

Status: **NOT READY / FUTURE-ONLY**.

The PR is an implementation and audit artifact, not a campaign authorization. The new resource-aware scheduler is opt-in and default-off; the legacy `sequential_review_gated` policy remains available. No active campaign authorization, production inventory token, runtime profile, or launch command is committed.

Before any later production launch, a user-authorized gate must establish all of the following:

1. reconcile the dirty production worktree and prove the loaded code identity;
2. validate model, tokenizer, config, dataset/mask, protocol, checkpoint, and environment hashes for every REUSE/RESUME row;
3. freeze and review the exact NAACL-balanced inventory, including Q3 retained rows and explicit exclusions;
4. collect the permitted real DEV-only Vistral generation profile and a dedicated non-TEST training-shaped PhoBERT profile if concurrency two is proposed;
5. bind the approved runtime profile, scheduler policy, inventory, source tree, and environment into one campaign authorization;
6. configure finite Azure logical-request, transport-attempt, token, and monetary safety ceilings where pricing is known;
7. run dry-run storage, lease, dependency, and rollback checks on the authorized host;
8. keep canonical TEST sealed until checkpoint, DEV selection, engine identity, and protocol are frozen.

No single config file, checkout, merge, import, dry-run, or PR review may activate the campaign.
