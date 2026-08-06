# Luna Max Subagent Manifest

Requested profile for all eight roles: `gpt-5.6-luna`, maximum available
reasoning effort (`max`), priority service tier, role name `luna_max`.

The runtime did not expose an independently verifiable resolved model or
reasoning-effort record. Every role is therefore recorded as
`RESOLVED_MODEL_PROFILE=NOT_VERIFIED`; this is a routing limitation, not a
claim that Luna Max execution was confirmed.

| Role | Isolated worktree | Patch/commit | Actual profile |
|---|---|---|---|
| LUNA_MAX_01_GENERATION | `D:/vipragsent_luna_worktrees/luna_max_01_generation` | `4e8cad7fc192d09fbc95e5d3491183801eb0ee65` | `NOT_VERIFIED` |
| LUNA_MAX_02_CHECKPOINT_DEVICE | `D:/vipragsent-luna-max-02` | `3ee7e724b12271aec6707aea5bc6b6c2ff47f802` | `NOT_VERIFIED` |
| LUNA_MAX_03_COMPONENT_BUNDLES | `D:/vipragsent-luna-max-03` | `00b24a3d01aed303b7685677b03ec7c4fdf5c7b3` | `NOT_VERIFIED` |
| LUNA_MAX_04_Q1B_DEPENDENCIES | `D:/vipragsent-q1b-luna-max-04` | `b6278d0a6f30f6f2ed03c6feb99911b33081e002` | `NOT_VERIFIED` |
| LUNA_MAX_05_STATISTICS | `D:/vipragsent-luna-max-05-statistics` | `271df64c32d5ef19412d236e0e5c0ac281e16bd2` | `NOT_VERIFIED` |
| LUNA_MAX_06_AZURE_JUDGE | `D:/vipragsent_luna_max_06_azure_judge` | `5de00e92ab36a4ddd068abee2c9ac4882586829a` | `NOT_VERIFIED` |
| LUNA_MAX_07_PROVENANCE_ARTIFACTS | `D:/vipragsent_luna_worktrees/luna_max_07_provenance_artifacts` | `8ec5b332c6a576b5d66b44aa971dd951cc3a3c6e` | `NOT_VERIFIED` |
| LUNA_MAX_08_RED_TEAM_TESTS | `D:/vipragsent_luna_worktrees/luna_max_08_red_team_tests` | `4f434aae04438f682e6fce908b21a437b945e1b4` | `NOT_VERIFIED` |

The parent inspected each patch, integrated in the required order, and ran
targeted tests after each integration. Red-team testing found two additional
Q1b defects; the parent repaired both and added production-path coverage.
