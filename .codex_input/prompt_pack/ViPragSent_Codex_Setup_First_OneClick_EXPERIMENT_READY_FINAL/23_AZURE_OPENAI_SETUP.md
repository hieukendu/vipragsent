# Azure OpenAI Setup

Required environment variables:

```dotenv
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_BASE_URL=https://<resource>.openai.azure.com/openai/v1/
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_BATCH_DEPLOYMENT=
AZURE_OPENAI_AUTH_MODE=api_key
AZURE_OPENAI_API_KEY=
```

Rules:

- Never hard-code the deployment name.
- Never call the direct OpenAI endpoint.
- Support API-key and Microsoft Entra ID authentication.
- Use strict JSON Schema.
- Log deployment, version, request ID, token usage, and content-filter results.
- Never log secrets.
- Use Global Batch only when a separate verified batch deployment exists.
- Full Azure jobs run only in Phase 16.


# ACTIVE METADATA OVERRIDES

Ignore any legacy GPT-4o-mini placeholders in the dataset package's rationale templates.
Active request metadata must come from the verified Azure GPT-4.1-mini deployment.

Structured label outputs use the canonical keys from the global contract.


# REQUEST DEFAULTS

Use:

```yaml
temperature: 0
rationale_max_output_tokens: 256
pragmatic_label_max_output_tokens: 128
polarity_label_max_output_tokens: 32
emotion_label_max_output_tokens: 32
```

Do not claim bitwise determinism. Record the Azure response ID, returned model field, deployment name, prompt hash,
request timestamp, token usage, retry count, and content-filter outcome.
