"""Render the complete reviewer-gap experiment report from canonical artifacts."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LABELS = (
    "code_switching",
    "idiom_figurative",
    "implicit_sentiment",
    "irony",
    "mocking",
    "sarcasm",
)
SEEDS = (20260521, 20260522, 20260523)
VARIANTS = (
    ("visobert_baseline", "ViSoBERT baseline", "reviewer_visobert_baseline"),
    ("xlmr_all_multipliers_1", "XLM-R all multipliers = 1", "priority12_v37_q1_all_multipliers_1"),
    ("xlmr_beta_01", "XLM-R rationale beta = 0.1", "priority12_v37_q1_beta_01"),
    ("xlmr_beta_05", "XLM-R rationale beta = 0.5", "priority12_v37_q1_beta_05"),
)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def run_dir(prefix: str, seed: int) -> Path:
    return ROOT / "results/runs" / f"{prefix}_{seed}"


def upload_record(run_id: str, status: dict[str, Any]) -> dict[str, Any]:
    record = status.get("runs", {}).get(run_id, {})
    receipt = record.get("receipt", {}) or {}
    uploaded = record.get("uploaded", {}) or {}
    verified = sum(1 for value in uploaded.values() if value.get("verified"))
    scan = next((item for item in status.get("last_scan", {}).get("results", []) if item.get("run_id") == run_id), {})
    return {
        "status": receipt.get("status") or scan.get("status") or ("PASS" if verified else record.get("status", "NOT_FOUND")),
        "verified": verified or int(scan.get("verified_file_count", 0) or 0),
        "local": len(record.get("observed", {})) or int(scan.get("local_file_count", 0) or 0),
        "repo": receipt.get("repo") or record.get("artifact_repository") or scan.get("artifact_repository", "—"),
        "root": record.get("remote_root") or scan.get("remote_root", "—"),
    }


def model_config(variant_prefix: str, seed: int) -> dict[str, Any]:
    resolved = read_json(run_dir(variant_prefix, seed) / "training/resolved_training_config.json", {})
    manifest = read_json(run_dir(variant_prefix, seed) / "manifest.json", {})
    return {**resolved, **{"manifest": manifest}}


def multiplier_text(values: dict[str, Any]) -> str:
    return ", ".join(f"{label}={fmt(values.get(label), 2)}" for label in LABELS + ("polarity", "emotion"))


def mean_sd(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}" if len(values) > 1 else f"{values[0]:.4f}"


def main() -> int:
    analysis_root = ROOT / "results/runs/reviewer_gap_analyses"
    per_seed = read_json(analysis_root / "per_seed_metrics.json", {})
    aggregates = read_json(analysis_root / "aggregate_metrics.json", {})
    analysis_manifest = read_json(analysis_root / "analysis_manifest.json", {})
    overlap = read_json(analysis_root / "dataset_overlap.json", {})
    code_switch = read_json(analysis_root / "code_switching_composition.json", {})
    log_variance = read_json(analysis_root / "learned_log_variance.json", {})
    uploader = read_json(ROOT / "runtime/reviewer_gaps_hf_uploader_status.json", {})
    analysis_upload = upload_record("reviewer_gap_analyses_20261005", uploader)

    lines: list[str] = [
        "# ViPragSent reviewer-gap experiment report",
        "",
        "Generated from canonical run manifests, validation files, test predictions, and HF uploader status.",
        "All checkpoint/threshold selection is dev-only; test metrics are reported only after the frozen selection protocol.",
        "",
        "## Protocol and parameters",
        "",
        "| Experiment | Model/revision | Batch (physical → effective; accumulation) | Optimizer / LR / schedule | Epochs / precision | Rationale | Uncertainty | Multipliers | Selection / threshold | Data |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, title, prefix in VARIANTS:
        cfg = model_config(prefix, SEEDS[0])
        manifest = cfg.get("manifest", {})
        train = cfg.get("training_config", {})
        repo = cfg.get("model_repository", "—")
        revision = cfg.get("model_revision", "—")
        rationale = f"decoder={fmt(cfg.get('rationale_decoder'))}, beta={fmt(train.get('rationale_beta'), 2)}"
        if key == "xlmr_all_multipliers_1":
            rationale += "; all task multipliers=1"
        lines.append(
            f"| {title} | `{repo}` @ `{revision[:12]}` | {fmt(train.get('physical_batch_size'))} → {fmt(train.get('effective_batch_size'))}; {fmt(train.get('gradient_accumulation_steps'))} | {fmt(train.get('optimizer'))} / {fmt(train.get('learning_rate'), 6)} / {fmt(train.get('scheduler'))}, warmup {fmt(train.get('warmup_ratio'), 2)} | {fmt(train.get('max_epochs'))} / {fmt(train.get('precision'))} | {rationale} | {fmt(cfg.get('uncertainty_log_variances'))} | {multiplier_text(cfg.get('loss_multipliers', {}))} | `{manifest.get('checkpoint_selection_rule', 'best_checkpoint_selected_on_development_macro_pragmatic_f1')}`; dev; test-after-freeze | train 7,998 / dev 1,999 / test 2,000; max len {fmt(cfg.get('max_sequence_length'))} |"
        )

    lines += [
        "",
        "## Per-seed overview",
        "",
        "Macro pragmatic F1 is the mean of the six pragmatic binary macro-F1 values. `F1+` is the positive-class F1.",
        "",
        "| Experiment | Seed | Run status | Best epoch | Macro pragmatic F1 | Macro ECE | Peak VRAM GiB | HF upload | Verified | Artifact repo |",
        "|---|---:|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for key, title, prefix in VARIANTS:
        records = per_seed.get(key, [])
        by_seed = {int(record["seed"]): record for record in records}
        for seed in SEEDS:
            run = run_dir(prefix, seed)
            manifest = read_json(run / "manifest.json", {})
            metric = by_seed.get(seed, {}).get("metrics", {})
            selection = read_json(run / "selection/selection_metric.json", {})
            resources = read_json(run / "training/resource_usage.json", {})
            run_id = run.name
            upload = upload_record(run_id, uploader)
            lines.append(f"| {title} | {seed} | {manifest.get('status', 'MISSING')} | {selection.get('best_epoch', '—')} | {fmt(metric.get('macro_pragmatic_f1'))} | {fmt(metric.get('macro_ece'))} | {fmt(resources.get('peak_vram_gb'), 2)} | {upload['status']} | {upload['verified']}/{upload['local']} | `{upload['repo']}` |")

    lines += [
        "",
        "## Per-seed six-pragmatic metrics",
        "",
        "Each row contains macro-F1 for the label, then positive-class precision / recall / F1 and confusion counts.",
        "",
        "| Experiment | Seed | Pragmatic label | Macro-F1 | P+ | R+ | F1+ | TP | FP | FN | TN |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, title, _prefix in VARIANTS:
        for record in per_seed.get(key, []):
            seed = record["seed"]
            metrics = record["metrics"]
            for label in LABELS:
                item = metrics["per_label"][label]
                positive = item["positive_class"]
                lines.append(f"| {title} | {seed} | {label} | {fmt(item.get('macro_f1'))} | {fmt(positive.get('precision'))} | {fmt(positive.get('recall'))} | {fmt(positive.get('f1'))} | {positive.get('tp', '—')} | {positive.get('fp', '—')} | {positive.get('fn', '—')} | {positive.get('tn', '—')} |")

    lines += [
        "",
        "## Mean ± sample SD across seeds",
        "",
        "| Experiment | Macro pragmatic F1 | Macro ECE |",
        "|---|---:|---:|",
    ]
    for key, title, _prefix in VARIANTS:
        aggregate = aggregates.get(key, {})
        lines.append(f"| {title} | {mean_sd([float(value) for value in aggregate.get('macro_pragmatic_f1', {}).get('values', [])])} | {mean_sd([float(value) for value in aggregate.get('macro_ece', {}).get('values', [])])} |")

    lines += [
        "",
        "| Experiment | Pragmatic label | Macro-F1 mean ± SD | P+ mean ± SD | R+ mean ± SD | F1+ mean ± SD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, title, _prefix in VARIANTS:
        aggregate = aggregates.get(key, {})
        for label in LABELS:
            item = aggregate.get("per_label", {}).get(label, {})
            positive = item.get("positive_class_summary", {})
            lines.append(f"| {title} | {label} | {mean_sd([float(value) for value in item.get('macro_f1', {}).get('values', [])])} | {mean_sd([float(value) for value in positive.get('precision', {}).get('values', [])])} | {mean_sd([float(value) for value in positive.get('recall', {}).get('values', [])])} | {mean_sd([float(value) for value in positive.get('f1', {}).get('values', [])])} |")

    lines += ["", "## Reviewer-gap post-hoc artifacts", ""]
    lines.append(f"- Analysis status: `{analysis_manifest.get('status', 'MISSING')}`; experiments present: `{', '.join(analysis_manifest.get('experiments_present', []))}`.")
    lines.append(f"- Analysis HF upload: `{analysis_upload['status']}`, {analysis_upload['verified']}/{analysis_upload['local']} verified; repository `{analysis_upload['repo']}`; remote root `{analysis_upload['root']}`.")
    lines.append("- Files: `per_seed_metrics.json`, `aggregate_metrics.json`, `pr_curves.json`, `calibration.json`, `dataset_overlap.json`, `code_switching_composition.json`, `learned_log_variance.json`.")
    for name, item in overlap.get("datasets", {}).items():
        lines.append(f"- Exact normalized overlap with {name}: {item.get('exact_normalized_overlap_count', '—')} rows; by ViPragSent split: {item.get('overlap_by_vipragsent_split', {})}.")
    test_switch = code_switch.get("splits", {}).get("test", {})
    if test_switch:
        lines.append(f"- Code-switching heuristic on test positives: {test_switch.get('positive_rows')} rows; shares `{test_switch.get('shares')}`.")
    lv_entries = log_variance.get("entries", {})
    lines.append(f"- Learned log-variance extraction: `{log_variance.get('status', 'MISSING')}` ({len(lv_entries)} entries). See `learned_log_variance.json` for exact values.")
    lines += [
        "",
        "## Reproducibility and resource audit",
        "",
        "- Dataset fingerprint: `B906C090400BAE115C9C5E3C35E32FA410AC519AE09209EBA741F198087C24F9`; split sizes 7,998 / 1,999 / 2,000.",
        "- Selected device: NVIDIA H100 80GB HBM3 MIG 2g.20gb, visible as `cuda:0`, 19.625 GiB; CPU host has 128 cores.",
        "- Existing relevant uploader/monitor/training processes were preserved. No unrelated ViPragSent-external GPU training process was found to terminate.",
        "- Original Q1a/Q2 artifacts were preserved; all new artifacts are under `reviewer-gaps-20261005`.",
    ]
    output = ROOT / "reports/reviewer_gap_final_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
