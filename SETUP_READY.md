# Setup readiness

SETUP_READY=false

The complete setup is intentionally not marked ready until all runtime preflight prerequisites pass.

## Blockers
- UIT-VSFC and/or UIT-VSMEC official test files are missing; use the manual-drop fallback
- Azure credentials/deployment are not configured
- Model weights have not passed Phase 15 offline verification
