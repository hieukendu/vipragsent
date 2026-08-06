# Sequential experiment setup

- Setup status: `PASS`
- Execution policy: `sequential_review_gated`
- Experiment prompts: `162`
- Azure prompts: `11`
- Phase 15 model-family prompts: `4`
- Aggregation prompts: `5` plus the final aggregation prompt
- Generated prompt files: `183`
- Global full DAG: `DISABLED`
- Approval after every run: `REQUIRED`
- Automatic next run: `DISABLED`
- Phase 15 executed: `false`
- Azure called: `false`
- Real experiments executed: `false`

## Resolution status
- `Q1A`: `RESOLVED`
- `Q1B`: `RESOLVED`
- `Q3`: `RESOLVED`
- `Q4`: `RESOLVED`
- `SIGNIFICANCE_PVALUE`: `RESOLVED`

## Next action
Use `prompts/sequential/phase15/phobert_base.md` only after explicit user approval. It stops after the model-family report and does not advance automatically.
