# Supplemental Azure rationale recovery

Status: **PASS**

The original Azure history is retained: 7,933 successful responses and 65 content-policy failures. The 65 later successful recovery outputs are recorded as a supplemental Azure GPT-4.1-mini recovery batch.

## Coverage

- Frozen input rows: **7998**
- Original successful Azure rows: **7933**
- Original content-policy failures retained: **65**
- Supplemental successful Azure rows: **65**
- Complete candidate rows: **7998**
- Unresolved rows: **0**

## Provenance

- Generation source: `azure_gpt_4_1_mini`
- Supplemental phase: `supplemental_azure_recovery`
- Recovery reason: `original_request_content_policy_blocked`
- Provider/model/deployment: `Azure OpenAI` / `gpt-4.1-mini` / `gpt-4.1-mini`
- Configured model version: `2025-04-14`
- Supplemental response IDs and provider usage are unavailable in the submitted artifact and were not fabricated.
- The original failure report remains unchanged.

## Cost

The original verified Azure cost remains **$1.36426 USD**. Supplemental provider usage is unavailable, so this is an estimated upper bound, not an exact billed amount.

- Supplemental input upper bound: `8448` tokens
- Supplemental cached input upper bound: `0` tokens
- Supplemental output upper bound: `16640` tokens
- Supplemental estimated cost upper bound: **$0.0300032 USD**
- Combined conservative upper bound: **$1.3942632 USD**
- Exact combined billed cost: unavailable because supplemental provider usage was not supplied.

## Artifacts

- Recovery records: `results/runs/azure_rationale_generation/azure/supplemental_azure_recovery.jsonl`
- Complete candidate: `results/runs/azure_rationale_generation/azure/complete_rationale_candidate.jsonl`
- Generation history: `results/runs/azure_rationale_generation/azure/rationale_generation_history.json`
- The candidate is not canonical until the official approval and promotion mechanism succeeds.

Canonical promotion: **PASS**
Approval basis: `standing_user_authorization_after_successful_audit`
