# No-Hallucination Checklist

Before marking a phase complete, verify:

- Did I read the actual files?
- Did I verify counts, splits, and schemas?
- Did I verify external source, split, license/access terms, and checksum?
- Did I avoid inventing model IDs or revisions?
- Did I avoid downloading model weights before Phase 15?
- Did I avoid running the full suite before Phase 16?
- Did I verify the Azure deployment and model version?
- Did I avoid the direct OpenAI endpoint?
- Did I avoid logging secrets?
- Did I avoid using test data for tuning or model selection?
- Did I avoid external fine-tuning?
- Are Q3 masks and demonstrations valid?
- Is rationale leakage prevented?
- Did the one-click fixture run pass?
- Is the master DAG complete?
- Can every metric be traced to predictions?
- Does every figure have a backing CSV?
- Did I avoid modifying the paper or manuscript?
- Did I report blockers instead of silently substituting components?
