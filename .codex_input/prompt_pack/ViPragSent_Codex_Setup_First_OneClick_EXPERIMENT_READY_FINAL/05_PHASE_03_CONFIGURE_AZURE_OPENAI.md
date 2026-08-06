> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 03 — CONFIGURE AZURE OPENAI

Configure and verify Azure OpenAI without running full workloads.

Required environment variables:

```dotenv
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_BASE_URL=https://<resource>.openai.azure.com/openai/v1/
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_BATCH_DEPLOYMENT=
AZURE_OPENAI_AUTH_MODE=api_key
AZURE_OPENAI_API_KEY=
```

Implement a shared Azure client factory, API-key and Entra ID authentication, deployment verification, one small Responses API smoke request, one strict Structured Output smoke request, mocked 429/retry behavior, deployment metadata recording, and secret scanning.

Verify that the deployment points to GPT-4.1-mini version `2025-04-14`. Detect whether a separate Global-Batch deployment exists.

Do not generate the full rationale dataset and do not run full Azure baselines.
