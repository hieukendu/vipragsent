import os
import json
import time

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. Load Azure configuration
# ============================================================

load_dotenv(".env", override=False)

api_key = os.getenv("AZURE_OPENAI_API_KEY")
base_url = os.getenv("AZURE_OPENAI_BASE_URL")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

if not api_key:
    raise RuntimeError("AZURE_OPENAI_API_KEY is missing")

if not base_url:
    raise RuntimeError("AZURE_OPENAI_BASE_URL is missing")


# ============================================================
# 2. Create Azure OpenAI v1 client
# ============================================================

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=60.0,       # 5 minutes for this diagnostic
    max_retries=0,       # no hidden SDK retries
)


# ============================================================
# 3. Exact sample to explain
# ============================================================

sample = {
    "comment": "mua Tab S8 dùng đi thui",
    "gold_labels": {
        "code_switching": 0,
        "emotion": "other",
        "idiom_figurative": 0,
        "implicit_sentiment": 0,
        "irony": 0,
        "mocking": 0,
        "polarity": "neutral",
        "sarcasm": 0
    },
    "sample_id": "fresh_visobert_10072"
}


# ============================================================
# 4. Prompt
#
# IMPORTANT:
# Gold labels are GIVEN.
# The model must explain them, not relabel the sample.
# ============================================================

prompt = f"""
You are generating a concise annotation rationale for a Vietnamese
pragmatic-language dataset.

The gold labels below are already fixed and authoritative.
DO NOT change, correct, reinterpret, or predict different labels.

Your only task is to explain why the provided Vietnamese comment is
consistent with ALL of the supplied gold labels.

Write the rationale in Vietnamese.

Requirements:
- Be concise but sufficiently specific.
- Ground the explanation only in the comment.
- Explain the pragmatic labels when relevant.
- Explain polarity and emotion.
- Do not invent hidden context.
- Do not claim sarcasm, irony, mocking, code-switching, figurative language,
  or implicit sentiment when the corresponding gold label is 0.
- Return only the required structured JSON object.

Sample:

{json.dumps(sample, ensure_ascii=False, indent=2)}
""".strip()


# ============================================================
# 5. Same strict rationale schema used conceptually by repo
# ============================================================

rationale_schema = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string"
        }
    },
    "required": ["rationale"],
    "additionalProperties": False
}


# ============================================================
# 6. Make ONE real Responses API request
# ============================================================

print("Calling Azure...")
print("Deployment:", deployment)
print("Timeout: 300 seconds")
print()

start = time.perf_counter()

try:
    response = client.responses.create(
        model=deployment,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "vipragsent_rationale",
                "strict": True,
                "schema": rationale_schema,
            }
        },
        temperature=0,
        max_output_tokens=512,
    )

except Exception as exc:
    elapsed = time.perf_counter() - start

    print(f"FAILED after {elapsed:.2f} seconds")
    print("Error type:", type(exc).__name__)
    print("Error:", str(exc))
    raise


elapsed = time.perf_counter() - start


# ============================================================
# 7. Show result
# ============================================================

print(f"SUCCESS after {elapsed:.2f} seconds")
print()

raw_text = response.output_text

print("Raw structured output:")
print(raw_text)
print()


# Parse returned JSON
parsed = json.loads(raw_text)

if set(parsed.keys()) != {"rationale"}:
    raise RuntimeError(
        f"Unexpected response keys: {list(parsed.keys())}"
    )

if not isinstance(parsed["rationale"], str) or not parsed["rationale"].strip():
    raise RuntimeError("Rationale is empty")

print("Rationale:")
print(parsed["rationale"])
print()


# ============================================================
# 8. Print token usage without exposing credentials
# ============================================================

usage = getattr(response, "usage", None)

if usage is not None:
    print("Usage:")
    print("  input_tokens :", getattr(usage, "input_tokens", None))
    print("  output_tokens:", getattr(usage, "output_tokens", None))

    input_details = getattr(usage, "input_tokens_details", None)

    if input_details is not None:
        print(
            "  cached_tokens:",
            getattr(input_details, "cached_tokens", None)
        )

print()
print("Response ID:", getattr(response, "id", None))