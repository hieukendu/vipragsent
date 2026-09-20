"""Build a machine-readable Q1a/Q2/Q3/Q4 run-lineage evidence file.

Only supplied reports and read-only implementation/artifact snapshots are
read. Generated output is kept below the paper evidence directory.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PAPER = Path(r"D:\vipragsent\paper")
EVIDENCE = PAPER / "evidence"
PACKAGE = Path(
    r"D:\vipragsent-pr\reports\hf_vipragsent_naacl_comparison_2026-09-15"
)
REMOTE_AUDIT = Path(
    r"D:\vipragsent-pr\reports\hf_vipragsent_remote_audit_2026-09-15"
)

SEED_RE = re.compile(r"(20260521|20260522|20260523)$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_index() -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(PACKAGE / "source_manifest.jsonl"):
        run_id = record.get("run_id", "")
        if (
            run_id.startswith("q1a_vipragsent_full_XLM_R_large_optimization")
            or run_id.startswith("xlmr_followup_q2_")
            or run_id.startswith("xlmr_followup_q3_")
            or run_id.startswith("xlmr_followup_q4_")
        ):
            output[run_id].append(record)
    return dict(output)


def selected_metric_index() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with (PACKAGE / "selected_run_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            output[row["run_id"]] = row
    return output


def calibration_index() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(PACKAGE / "calibration_records.jsonl"):
        run_id = record.get("run_id")
        if run_id:
            output[run_id] = record
    return output


def remote_run_manifests() -> dict[str, tuple[Path, dict[str, Any]]]:
    output: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in REMOTE_AUDIT.rglob("run_manifest.json"):
        run_dir = path.parent.name
        if not (
            run_dir.startswith("xlmr_followup_q2_")
            or run_dir.startswith("xlmr_followup_q3_")
            or run_dir.startswith("xlmr_followup_q4_")
        ):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        output[run_dir] = (path, record)
    return output


def target_manifests() -> dict[str, tuple[Path, dict[str, Any]]]:
    output: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in (
        PACKAGE / "raw_short" / "model"
    ).rglob("*optimization_manifest.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        run_id = record.get("run_id", "")
        if run_id.startswith("q1a_vipragsent_full_XLM_R_large_optimization"):
            output[run_id] = (path, record)
    return output


def config_summary(record: dict[str, Any]) -> dict[str, Any]:
    config = record.get("resolved_training_config") or {}
    if not isinstance(config, dict):
        config = {}
    keys = (
        "system_id",
        "variant_id",
        "active_uncertainty_tasks",
        "rationale_training",
        "rationale_inference",
        "rationale_beta",
        "rationale_target_max_length",
        "uncertainty_weighting_enabled",
        "learning_rate",
        "effective_batch_size",
        "physical_batch_size",
        "gradient_accumulation_steps",
        "maximum_epochs",
        "warmup_ratio",
        "scheduler",
        "patience",
        "selection_metric",
        "optimizer",
        "precision",
        "gradient_strategy",
        "loss_multipliers",
        "trial",
        "config_hash",
    )
    return {key: config[key] for key in keys if key in config}


def logical_seed(run_id: str) -> int | None:
    match = SEED_RE.search(run_id)
    return int(match.group(1)[-2:]) if match else None


def run_family(run_id: str) -> str:
    return re.sub(r"_2026052[123]$", "", run_id)


def q4_lineage_auxiliary(
    manifest_path: Path,
) -> dict[str, Any]:
    run_dir = manifest_path.parent
    output: dict[str, Any] = {}
    provenance_path = run_dir / "source" / "source_provenance.json"
    if provenance_path.exists():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            provenance = {}
        for key in (
            "source_run_id",
            "source_variant_id",
            "source_model_family",
            "followup_lane",
            "checkpoint_sha256",
            "source_checkpoint_sha256",
            "status",
        ):
            if key in provenance:
                output[key] = provenance[key]
    config_path = run_dir / "config_snapshot.yaml"
    if config_path.exists():
        config_text = config_path.read_text(encoding="utf-8")
        for key in ("followup_lane", "source_run_id", "source_checkpoint_id"):
            match = re.search(
                rf"^\s*{re.escape(key)}:\s*(.+?)\s*$",
                config_text,
                flags=re.MULTILINE,
            )
            if match:
                output[f"config_{key}"] = match.group(1).strip().strip("\"'")
    return output


def make_run_record(
    run_id: str,
    question: str,
    sources: list[dict[str, Any]],
    metrics: dict[str, Any],
    calibrations: dict[str, Any],
    manifests: dict[str, tuple[Path, dict[str, Any]]],
    target: dict[str, tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    source = sources[0] if sources else {}
    manifest_path: str | None = None
    manifest: dict[str, Any] = {}
    if run_id in target:
        manifest_path = str(target[run_id][0])
        manifest = target[run_id][1]
    elif run_id in manifests:
        manifest_path = str(manifests[run_id][0])
        manifest = manifests[run_id][1]
    q4_auxiliary: dict[str, Any] = {}
    if question == "Q4" and manifest_path:
        q4_auxiliary = q4_lineage_auxiliary(Path(manifest_path))
    resolved = config_summary(manifest)
    source_paths = {
        record.get("kind", "unknown"): record.get("source_path")
        for record in sources
    }
    calibration = calibrations.get(run_id, {})
    return {
        "question": question,
        "run_id": run_id,
        "run_family": run_family(run_id),
        "logical_seed": logical_seed(run_id),
        "actual_seed": 20260500 + logical_seed(run_id)
        if logical_seed(run_id) is not None
        else None,
        "system": source.get("system"),
        "scope": source.get("scope"),
        "backbone": source.get("backbone"),
        "artifact_repo": source.get("repo_id"),
        "artifact_source_paths": source_paths,
        "metric_record": {
            key: metrics.get(key)
            for key in (
                "status",
                "macro_pragmatic_f1",
                "implicit_sentiment",
                "sarcasm",
                "irony",
                "idiom_figurative",
                "code_switching",
                "mocking",
            )
            if key in metrics
        },
        "calibration_record": {
            key: calibration.get(key)
            for key in (
                "status",
                "macro_pragmatic_ece",
                "ece_mean",
                "ece_sd",
            )
            if key in calibration
        },
        "manifest_path": manifest_path,
        "manifest_status": manifest.get("status"),
        "manifest_seed": manifest.get("seed"),
        "manifest_config": resolved,
        "q4_source_run_id": q4_auxiliary.get("source_run_id")
        or q4_auxiliary.get("config_source_run_id")
        or manifest.get("source_run_id")
        or manifest.get("source_system_id"),
        "q4_followup_lane": q4_auxiliary.get("followup_lane")
        or q4_auxiliary.get("config_followup_lane"),
        "q4_source_variant_id": q4_auxiliary.get("source_variant_id"),
        "q4_source_checkpoint_sha256": q4_auxiliary.get(
            "source_checkpoint_sha256"
        )
        or q4_auxiliary.get("checkpoint_sha256"),
        "q4_lineage_auxiliary": q4_auxiliary,
        "q4_additional_training": manifest.get("additional_training"),
        "q4_inference_output_source": manifest.get("inference_output_source"),
        "q4_rationale_decoder_enabled_at_inference": manifest.get(
            "rationale_decoder_enabled_at_inference"
        ),
    }


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    sources = source_index()
    metrics = selected_metric_index()
    calibrations = calibration_index()
    manifests = remote_run_manifests()
    target = target_manifests()

    run_ids: list[tuple[str, str]] = []
    run_ids.extend(
        (run_id, "Q1a")
        for run_id in sorted(sources)
        if run_id.startswith("q1a_vipragsent_full_XLM_R_large_optimization")
    )
    run_ids.extend(
        (run_id, "Q2")
        for run_id in sorted(sources)
        if run_id.startswith("xlmr_followup_q2_")
    )
    run_ids.extend(
        (run_id, "Q3")
        for run_id in sorted(sources)
        if run_id.startswith("xlmr_followup_q3_")
    )
    run_ids.extend(
        (run_id, "Q4")
        for run_id in sorted(sources)
        if run_id.startswith("xlmr_followup_q4_")
    )
    runs = [
        make_run_record(
            run_id,
            question,
            sources.get(run_id, []),
            metrics.get(run_id, {}),
            calibrations,
            manifests,
            target,
        )
        for run_id, question in run_ids
    ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["question"]].append(run)
    family_summary: dict[str, list[dict[str, Any]]] = {}
    for question, question_runs in grouped.items():
        families: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in question_runs:
            families[run["run_family"]].append(run)
        family_summary[question] = [
            {
                "run_family": family,
                "run_count": len(items),
                "logical_seeds": sorted(
                    item["logical_seed"] for item in items if item["logical_seed"]
                ),
                "actual_seeds": sorted(
                    item["actual_seed"] for item in items if item["actual_seed"]
                ),
                "run_ids": [item["run_id"] for item in items],
                "config_summaries_observed": sorted(
                    {
                        json.dumps(
                            item["manifest_config"], sort_keys=True
                        )
                        for item in items
                        if item["manifest_config"]
                    }
                ),
            }
            for family, items in sorted(families.items())
        ]

    q1a = [run for run in runs if run["question"] == "Q1a"]
    q2 = [run for run in runs if run["question"] == "Q2"]
    q3 = [run for run in runs if run["question"] == "Q3"]
    q4 = [run for run in runs if run["question"] == "Q4"]

    protocol_differences = [
        {
            "dimension": "run identity",
            "q1a_v37": "q1a_vipragsent_full_XLM_R_large_optimization_*_v37_2026052[123]",
            "q2": "xlmr_followup_q2_*_2026052[123]",
            "q3": "xlmr_followup_q3_full_{32,64,128,256,512,full}_2026052[123]",
            "q4": "xlmr_followup_q4_full_2026052[123]",
            "interpretation": (
                "Q2/Q3/Q4 are distinct follow-up run families; their metrics "
                "must not be relabeled as the Q1a V37 training runs."
            ),
        },
        {
            "dimension": "seed identity",
            "q1a_v37": [20260521, 20260522, 20260523],
            "q2": sorted({run["actual_seed"] for run in q2}),
            "q3": sorted({run["actual_seed"] for run in q3}),
            "q4": sorted({run["actual_seed"] for run in q4}),
            "interpretation": (
                "The artifact run IDs encode actual date-coded seeds; logical "
                "labels 21/22/23 are shorthand only."
            ),
        },
        {
            "dimension": "training and selection",
            "q1a_v37": (
                "Manifest records warmup_ratio=0.2, cosine scheduler, "
                "patience=10, trial=v37, dev_macro_pragmatic_f1 selection."
            ),
            "q2": (
                "Full follow-up manifest records warmup_ratio=0.1, linear "
                "scheduler, patience=2, system_id=xlmr_followup_full."
            ),
            "q3": (
                "Low-resource budgets change training masks/budget and the "
                "full Q3 family records dev_sarcasm_binary_macro_f1 selection."
            ),
            "q4": (
                "Outer run is artifact_extraction with resolved_training_config "
                "not applicable; source is the same-seed Q1a V37 checkpoint."
            ),
        },
        {
            "dimension": "rationale and uncertainty components",
            "q1a_v37": (
                "Full XLM-R variant records rationale_training=true, "
                "rationale_inference=false, beta=0.3, and uncertainty "
                "weighting enabled across eight classification tasks."
            ),
            "q2": (
                "Ablation families explicitly remove rationale, polarity "
                "auxiliary, emotion auxiliary, uncertainty weighting, or the "
                "multi-task bundle; treat each as its named variant."
            ),
            "q3": (
                "Q3 full/budget families are separate low-resource runs, not "
                "the Q1a optimization cohort."
            ),
            "q4": (
                "Calibration consumes direct classification probabilities from "
                "the source V37 checkpoint; it is not rationale generation or "
                "new training."
            ),
        },
    ]

    report = {
        "report_type": "Q1a/Q2/Q3/Q4 protocol lineage",
        "created_utc": "2026-09-17",
        "seed_protocol": {
            "logical_seed_labels": [21, 22, 23],
            "actual_recorded_seeds": [20260521, 20260522, 20260523],
            "interpretation": "Use actual date-coded seeds in provenance.",
        },
        "q1a_v37_target_runs": q1a,
        "followup_runs": {
            "Q2": q2,
            "Q3": q3,
            "Q4": q4,
        },
        "family_summary": family_summary,
        "protocol_differences": protocol_differences,
        "q4_lineage_rule": (
            "Q4 primary runs are calibration/extraction follow-ups from the "
            "same-seed Q1a V37 full checkpoint; they are not independent "
            "training replicates."
        ),
        "evidence_boundary": {
            "run_identity_and_seed_coverage": "RECORDED_ARTIFACT_METADATA",
            "training_protocol_details": (
                "RECORDED_WHERE_MANIFEST_PRESENT; otherwise do not infer"
            ),
            "independent_rerun": "NOT_PERFORMED",
        },
    }
    (EVIDENCE / "protocol_lineage.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "q1a_v37_runs": len(q1a),
                "q2_runs": len(q2),
                "q3_runs": len(q3),
                "q4_runs": len(q4),
                "output": str(EVIDENCE / "protocol_lineage.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
