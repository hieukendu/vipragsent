from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.hashing import sha256_file
from vipragsent.orchestration.inventory import write_expected_runs
from vipragsent.orchestration.sequential import build_azure_job_inventory, load_execution_policy

RESEARCH_QUESTIONS = ("Q1a", "Q1b", "Q2", "Q3", "Q4")


def _experiment_prompt(row: dict[str, Any]) -> str:
    experiment_id = row["experiment_id"]
    return f"""# ViPragSent sequential experiment run: {experiment_id}

This runbook names exactly one inventory entry. Do not start another experiment or Azure job.

## Locked entry

- Experiment ID: `{experiment_id}`
- Research question: `{row['research_question']}`
- System ID: `{row['system_id']}`
- Display name: {row['display_name']}
- Variant: `{row['variant']}`
- Backbone: `{row['backbone']}`
- Seed: `{row['seed']}`
- Budget: `{row['budget']}`
- Task: `{row['task']}`
- Split: `{row['split']}`
- Dependencies: `{row['dependencies']}`
- Required Phase 15 assets: `{row['required_phase15_assets']}`
- Execution kind: `{row['execution_kind']}`
- Expected artifacts: `{row['expected_outputs']}`
- Selection metric: `{row['selection_metric']}`
- Evaluation protocol: `{row['evaluation_protocol']}`
- Reusable checkpoint key: `{row['reusable_checkpoint_key']}`
- Protocol resolution: `{row['protocol_resolution_status']}`

## Required command sequence

1. Run `python scripts/run_single_experiment.py --experiment-id {experiment_id} --stage preflight`.
2. If preflight is BLOCKED or FAIL, paste its report into the Codex chat and stop.
3. After preflight passes, run `python scripts/run_single_experiment.py --experiment-id {experiment_id} --stage all`.
4. On interruption, resume only this entry with `python scripts/run_single_experiment.py --experiment-id {experiment_id} --resume`.
5. Do not call the global DAG, change the locked protocol, tune on the test split, or substitute a model, dataset, seed, or checkpoint.

## Required review handoff

The run must complete these stages in order: preflight, train_or_reuse, evaluate_dev, freeze_selection, evaluate_test, export_artifacts, validate_artifacts, generate_review_summary.

Print the complete review summary with `python scripts/print_run_review_summary.py --run-id {experiment_id}` and paste it into the Codex chat. It must include `RUN_STATUS`, `USER_REVIEW_STATUS`, `NEXT_RUN_ALLOWED`, artifact hashes, and blockers.

The initial approval file must remain `PENDING_USER_APPROVAL`; never fabricate approval. After the summary is pasted, stop and wait for explicit user approval. Do not begin the next run automatically.

RUN_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO
"""


def _azure_prompt(job: dict[str, Any]) -> str:
    job_id = job["job_id"]
    return f"""# ViPragSent sequential Azure job: {job_id}

This runbook names exactly one Azure job. Do not start another job, experiment, batch, or global matrix.

## Locked job

- Job ID: `{job_id}`
- Job type: `{job['job_type']}`
- Research question: `{job['research_question']}`
- Task: `{job['task']}`
- Budget: `{job['budget']}`
- Split: `{job['split']}`
- Required assets: `{job['required_phase15_assets']}`
- Model family: `{job['model']}`

## Required command sequence

1. Run `python scripts/run_single_azure_job.py --job-id {job_id} --stage preflight`.
2. If preflight is BLOCKED or FAIL, paste the report into the Codex chat and stop.
3. After preflight passes, execute exactly this stage order: `preflight` -> `execute_api_job` -> `validate_responses` -> `export_artifacts` -> `validate_artifacts` -> `generate_review_summary`.
4. Run that locked sequence with `python scripts/run_single_azure_job.py --job-id {job_id} --stage all`.
5. On interruption, resume only this job with `python scripts/run_single_azure_job.py --job-id {job_id} --resume`.
6. Use only the frozen prompt/schema manifest for this job. Do not log secrets, use the direct OpenAI endpoint, or silently change demonstrations, deployment, budget, or retry policy.

## Required review handoff

Complete the sequential stages applicable to this job, print the complete review summary with `python scripts/print_run_review_summary.py --run-id {job_id}`, and paste it into the Codex chat. Include request/token usage, invalid-output accounting when applicable, artifact hashes, `RUN_STATUS`, `USER_REVIEW_STATUS`, and `NEXT_RUN_ALLOWED`.

The initial approval file must remain `PENDING_USER_APPROVAL`; never fabricate approval. After the summary is pasted, stop and wait for explicit user approval. Do not begin the next job automatically.

RUN_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO
"""


def _phase15_prompt(model: dict[str, Any]) -> str:
    family = model["name"]
    return f"""# ViPragSent Phase 15 model-family preparation: {family}

This future runbook is for exactly one model family: `{family}`. Do not download or verify any other model family in this run.

The setup task that generated this file must not execute Phase 15. Execute this runbook only after the setup is frozen and the user explicitly approves Phase 15.

## Locked model

- Model family: `{family}`
- Repository: `{model['repo_id']}`
- Revision: `{model['revision']}`
- Tokenizer revision: `{model['tokenizer_revision']}`
- Quantization: `{model['quantization']}`

## Required sequence

1. Confirm the runtime preflight and server prerequisites from `32_RUNTIME_PREFLIGHT_CHECKLIST.md`.
2. Download only this family with `python scripts/download_all_models.py --manifest configs/models/download_manifest.yaml --model-family {family}`.
3. Run the offline revision/tokenizer/model verification for this family with `python scripts/verify_model_smoke.py --manifest data/model_cache_manifest.json --model-family {family}`.
4. Run the locked forward/backward smoke and physical-batch probe for this family when the runtime checklist permits it. Use exactly `python scripts/probe_model_batch.py --model-family {family}` for the physical-batch probe.
5. Record the exact local revision, tokenizer revision, quantization, physical batch, and verification hashes.
6. Print the complete Phase 15 report and paste it into the Codex chat.

Do not start Phase 16, an experiment, an Azure job, or another model family. Stop with `PENDING_USER_APPROVAL` and wait for explicit approval before any next Phase 15 prompt.

PHASE15_STATUS: PASS | BLOCKED | FAIL
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO
"""


def _aggregation_prompt(research_question: str, *, final: bool = False) -> str:
    scope = "all research questions" if final else research_question
    argument = "all" if final else research_question
    return f"""# ViPragSent approved-run aggregation: {scope}

This runbook aggregates only completed, explicitly approved sequential runs for `{scope}`. It does not train, download models, call Azure, approve runs, or start another run.

## Required command

Run:

`python scripts/aggregate_approved_runs.py --research-question {argument}`

The command must reject any missing run, non-PASS run, missing prediction/metric artifact, unresolved protocol, or approval status other than `APPROVED` with a named approver and timestamp. Paste the complete aggregation report into the Codex chat.

Do not fabricate approval, alter the locked protocol, or begin a subsequent research question automatically. Stop after reporting the result and wait for explicit user direction. The approval gate remains `PENDING_USER_APPROVAL` until the user changes it.

AGGREGATION_STATUS: PASS | BLOCKED
USER_REVIEW_STATUS: PENDING
NEXT_RUN_ALLOWED: NO
"""


def _write_prompt(root: Path, directory: str, filename: str, content: str, *, kind: str, identifier: str, command: str) -> dict[str, Any]:
    path = root / directory / filename
    atomic_write_text(path, content)
    return {"kind": kind, "id": identifier, "path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "command": command}


def main() -> int:
    policy = load_execution_policy(ROOT)
    inventory = write_expected_runs(ROOT)
    prompt_entries: list[dict[str, Any]] = []
    for row in inventory["rows"]:
        prompt_entries.append(_write_prompt(
            ROOT, "prompts/sequential/experiments", f"{row['experiment_id']}.md", _experiment_prompt(row),
            kind="experiment", identifier=row["experiment_id"],
            command=f"python scripts/run_single_experiment.py --experiment-id {row['experiment_id']} --stage all",
        ))

    jobs = build_azure_job_inventory()
    atomic_write_json(ROOT / "reports/azure_job_inventory.json", {"schema_version": 1, "jobs": jobs, "job_count": len(jobs)})
    for job in jobs:
        prompt_entries.append(_write_prompt(
            ROOT, "prompts/sequential/azure", f"{job['job_id']}.md", _azure_prompt(job),
            kind="azure_job", identifier=job["job_id"],
            command=f"python scripts/run_single_azure_job.py --job-id {job['job_id']} --stage all",
        ))

    model_registry = yaml.safe_load((ROOT / "configs/models/model_registry.yaml").read_text(encoding="utf-8"))
    for family, model in model_registry["models"].items():
        model = dict(model) | {"name": family}
        prompt_entries.append(_write_prompt(
            ROOT, "prompts/sequential/phase15", f"{family}.md", _phase15_prompt(model),
            kind="phase15", identifier=family,
            command=f"python scripts/download_all_models.py --manifest configs/models/download_manifest.yaml --model-family {family}",
        ))

    for research_question in RESEARCH_QUESTIONS:
        prompt_entries.append(_write_prompt(
            ROOT, "prompts/sequential/aggregation", f"{research_question.casefold()}.md", _aggregation_prompt(research_question),
            kind="aggregation", identifier=research_question,
            command=f"python scripts/aggregate_approved_runs.py --research-question {research_question}",
        ))
    prompt_entries.append(_write_prompt(
        ROOT, "prompts/sequential/aggregation", "final_aggregation.md", _aggregation_prompt("all research questions", final=True),
        kind="final_aggregation", identifier="all",
        command="python scripts/aggregate_approved_runs.py --research-question all",
    ))

    manifest = {
        "schema_version": 3,
        "execution_policy": policy,
        "experiment_count": len(inventory["rows"]),
        "azure_job_count": len(jobs),
        "phase15_model_count": len(model_registry["models"]),
        "aggregation_count": len(RESEARCH_QUESTIONS),
        "inventory_hash": inventory["inventory_hash"],
        "prompt_count": len(prompt_entries),
        "prompts": prompt_entries,
        "approval_contract": {"status": "PENDING_USER_APPROVAL", "next_run_allowed": "NO"},
    }
    atomic_write_json(ROOT / "reports/sequential_prompt_manifest.json", manifest)
    atomic_write_json(ROOT / "reports/generated_sequential_prompts_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("experiment_count", "azure_job_count", "phase15_model_count", "aggregation_count", "prompt_count", "inventory_hash")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
