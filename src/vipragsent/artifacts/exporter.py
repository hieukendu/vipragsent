from __future__ import annotations

import base64
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..atomic import atomic_write_json, atomic_write_text
from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS, TRAINING_SEEDS
from ..data.loaders import DatasetBundle, DatasetExample
from ..evaluation.metrics import (
    binary_macro_f1,
    expected_calibration_error,
    macro_pragmatic_f1,
    multiclass_macro_f1,
    pragmatic_ece,
    pragmatic_reliability_bins,
)
from ..hashing import sha256_file, sha256_json
from ..manual import ERROR_ANALYSIS_COLUMNS
from ..statistics.bootstrap import (
    hierarchical_bootstrap,
    holm_bonferroni,
    paired_bootstrap_comparison,
)
from .schemas import REQUIRED_COLUMNS, validate_artifact_tree, validate_production_artifact

Q3_BUDGETS = ("32", "64", "128", "256", "512", "full")
PRODUCTION_PAPER_TABLES = (
    "table_2_pragmatic.csv",
    "table_3_external_retention.csv",
    "table_4_ablation.csv",
    "q3_low_resource.csv",
    "q4_pragmatic_calibration_per_seed.csv",
    "q4_pragmatic_calibration_summary.csv",
    "significance.csv",
    "cost_latency.csv",
    "backbone_sensitivity.csv",
)
PAPER_ROLE_BACKBONES = {
    "table_2_headline": "vistral_7b",
    "table_3_retention": "phobert_base",
    "table_4_ablation": "phobert_base",
}
SYSTEM_BACKBONES = {
    "phobert_finetune": "phobert_base",
    "vistral_7b_sft": "vistral_7b",
    "vipragsent_full_phobert": "phobert_base",
    "vipragsent_full_vistral": "vistral_7b",
    "cot_only_vistral": "vistral_7b",
}


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _svg_bar(path: Path, labels: list[str], values: list[float], title: str) -> None:
    width, height = 900, 420
    max_value = max(values or [1.0])
    slot = width / max(len(values), 1)
    bars: list[str] = []
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = slot * index + slot * 0.15
        bar_width = slot * 0.7
        bar_height = (height - 100) * max(float(value), 0.0) / max_value
        y = height - 60 - bar_height
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#2563eb"/>')
        bars.append(f'<text x="{x + bar_width / 2:.2f}" y="{height - 36}" text-anchor="middle" font-size="12">{label}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="20" y="28" font-size="20">{title}</text>{"".join(bars)}</svg>'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8", newline="\n")


def _fixture_visual(path: Path, *, kind: str) -> None:
    """Write a tiny deterministic binary fixture for required PDF/PNG paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "png":
        path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="))
    elif kind == "pdf":
        path.write_bytes(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")
    else:
        raise ValueError(f"Unsupported fixture visual kind: {kind}")


def _synthetic_bundle() -> DatasetBundle:
    """Create a tiny fixture-only bundle without reading repository text or restricted data."""
    splits: dict[str, list[DatasetExample]] = {}
    for split, count in (("train", 24), ("dev", 12), ("test", 64)):
        rows: list[DatasetExample] = []
        for index in range(count):
            labels = {key: int((index + offset) % (3 + offset % 2) == 0) for offset, key in enumerate(PRAGMATIC_LABELS)}
            labels["polarity"] = POLARITY_LABELS[index % len(POLARITY_LABELS)]
            labels["emotion"] = EMOTION_LABELS[index % len(EMOTION_LABELS)]
            rows.append(DatasetExample(f"fixture_{split}_{index:04d}", f"synthetic fixture comment {index}", labels, split, "fixture"))
        splits[split] = rows
    return DatasetBundle(splits, sha256_json({"mode": "fixture", "counts": {key: len(value) for key, value in splits.items()}}), {"synthetic": True})


def _fixture_predictions(bundle: DatasetBundle, system: str, seed: int) -> dict[str, Any]:
    examples = bundle.test
    rng = np.random.default_rng(seed + sum(ord(char) for char in system))
    true_prag = {key: np.asarray([example.labels[key] for example in examples], dtype=int) for key in PRAGMATIC_LABELS}
    pred_prag: dict[str, np.ndarray] = {}
    prob_prag: dict[str, np.ndarray] = {}
    error_rate = 0.12 if "full" in system else 0.22
    for key in PRAGMATIC_LABELS:
        probabilities = np.where(true_prag[key] == 1, 0.72, 0.28) + rng.normal(0, 0.16, len(examples))
        probabilities = np.clip(probabilities, 0.01, 0.99)
        predictions = (probabilities >= 0.5).astype(int)
        flips = rng.random(len(examples)) < error_rate
        predictions[flips] = 1 - predictions[flips]
        pred_prag[key] = predictions
        prob_prag[key] = probabilities
    true_polarity = np.asarray([POLARITY_LABELS.index(example.labels["polarity"]) for example in examples])
    true_emotion = np.asarray([EMOTION_LABELS.index(example.labels["emotion"]) for example in examples])
    polarity_probs = np.full((len(examples), len(POLARITY_LABELS)), 0.08)
    polarity_probs[np.arange(len(examples)), true_polarity] = 0.84
    polarity_probs = np.clip(polarity_probs + rng.normal(0, 0.04, polarity_probs.shape), 0.001, None)
    polarity_probs /= polarity_probs.sum(axis=1, keepdims=True)
    emotion_probs = np.full((len(examples), len(EMOTION_LABELS)), 0.04)
    emotion_probs[np.arange(len(examples)), true_emotion] = 0.76
    emotion_probs = np.clip(emotion_probs + rng.normal(0, 0.04, emotion_probs.shape), 0.001, None)
    emotion_probs /= emotion_probs.sum(axis=1, keepdims=True)
    if error_rate > 0.2:
        polarity_probs = polarity_probs[:, ::-1]
        emotion_probs = emotion_probs[:, ::-1]
    return {
        "sample_ids": [example.sample_id for example in examples],
        "true_pragmatic": true_prag,
        "pred_pragmatic": pred_prag,
        "prob_pragmatic": prob_prag,
        "true_polarity": true_polarity,
        "polarity_probs": polarity_probs,
        "true_emotion": true_emotion,
        "emotion_probs": emotion_probs,
    }


def _write_run(results_root: Path, system: str, seed: int, prediction: dict[str, Any]) -> None:
    run_root = results_root / "runs" / system / str(seed)
    run_root.mkdir(parents=True, exist_ok=True)
    metrics = {
        "mode": "fixture",
        "synthetic_results": True,
        "system": system,
        "seed": seed,
        "macro_prag_f1": macro_pragmatic_f1(prediction["true_pragmatic"], prediction["pred_pragmatic"]),
        "polarity_macro_f1": multiclass_macro_f1(prediction["true_polarity"], prediction["polarity_probs"].argmax(axis=1), range(3)),
        "emotion_macro_f1": multiclass_macro_f1(prediction["true_emotion"], prediction["emotion_probs"].argmax(axis=1), range(7)),
        "polarity_ece": expected_calibration_error(prediction["true_polarity"], prediction["polarity_probs"]),
        "inference_output_source": "judge_of_generated_reasoning" if "cot" in system else "classification_heads",
        "rationale_decoder_enabled_at_inference": False,
        "preprocessing_name": "fixture_unicode_nfc",
        "preprocessing_version": "fixture-v1",
        "tokenizer_revision": "fixture",
        "model_revision": "fixture",
        "physical_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "effective_batch_size": 32,
        "data_fingerprint": "fixture",
        "config_hash": "fixture",
        "code_commit": "fixture",
    }
    atomic_write_json(run_root / "metrics.json", metrics)
    with (run_root / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for index, sample_id in enumerate(prediction["sample_ids"]):
            handle.write(json.dumps({
                "sample_id": sample_id,
                "pragmatic": {key: int(prediction["pred_pragmatic"][key][index]) for key in PRAGMATIC_LABELS},
                "polarity": POLARITY_LABELS[int(prediction["polarity_probs"][index].argmax())],
                "emotion": EMOTION_LABELS[int(prediction["emotion_probs"][index].argmax())],
                "probabilities": {"polarity": prediction["polarity_probs"][index].tolist(), "emotion": prediction["emotion_probs"][index].tolist()},
            }, ensure_ascii=False) + "\n")


def _dataset_artifacts(bundle: DatasetBundle, output: Path, *, fixture: bool, repo_root: Path | None = None) -> None:
    table_rows = [{
        "dataset": "ViPragSent",
        "role": "fixture validation" if fixture else "train/dev/test and Q1a",
        "train_count": len(bundle.train),
        "dev_count": len(bundle.dev),
        "test_count": len(bundle.test),
        "total_count": len(bundle.train) + len(bundle.dev) + len(bundle.test),
        "task": "six pragmatic binary + polarity + emotion",
        "label_space": "canonical_labels",
        "source_manifest": "fixture" if fixture else "data/manifests/dataset_manifest.json",
        "checksum": bundle.fingerprint,
        "redistribution_status": "synthetic-fixture" if fixture else "private-input-not-redistributed",
    }]
    _write_csv(output / "tables" / "table_1_dataset_summary.csv", REQUIRED_COLUMNS["table_1_dataset_summary.csv"].split(","), table_rows)
    distribution_rows: list[dict[str, Any]] = []
    for split, examples in bundle.splits.items():
        total = len(examples)
        for key in PRAGMATIC_LABELS:
            for label in (0, 1):
                count = sum(example.labels[key] == label for example in examples)
                distribution_rows.append({"split": split, "label_group": key, "label": label, "count": count, "total": total, "rate": count / total})
        for group, labels in (("polarity", POLARITY_LABELS), ("emotion", EMOTION_LABELS)):
            for label in labels:
                count = sum(example.labels[group] == label for example in examples)
                distribution_rows.append({"split": split, "label_group": group, "label": label, "count": count, "total": total, "rate": count / total})
    _write_csv(output / "tables" / "vipragsent_label_distribution.csv", REQUIRED_COLUMNS["vipragsent_label_distribution.csv"].split(","), distribution_rows)
    _write_csv(output / "backing_data" / "split_and_label_counts.csv", REQUIRED_COLUMNS["split_and_label_counts.csv"].split(","), distribution_rows)
    if fixture:
        iaa = [{"field": "synthetic", "n": len(bundle.train), "raw_agreement": 1.0, "cohen_kappa": 1.0, "krippendorff_alpha_nominal": 1.0, "disagreement_count": 0}]
    else:
        if repo_root is None:
            raise ValueError("Production IAA export requires the repository root")
        iaa_path = repo_root / "data/manifests/human_iaa_recomputed.json"
        if not iaa_path.exists():
            raise ValueError("Recomputed human IAA manifest is missing")
        iaa_payload = json.loads(iaa_path.read_text(encoding="utf-8"))
        iaa = iaa_payload.get("fields", [])
        if not iaa:
            raise ValueError("Recomputed human IAA manifest contains no fields")
    _write_csv(output / "tables" / "human_iaa_summary.csv", REQUIRED_COLUMNS["human_iaa_summary.csv"].split(","), iaa)


def _export_fixture_tables(output: Path, results_root: Path, bundle: DatasetBundle) -> int:
    systems = [
        ("phobert_finetune", "phobert_base", TRAINING_SEEDS),
        ("vistral_7b_sft", "vistral_7b", TRAINING_SEEDS),
        ("vipragsent_full_phobert", "phobert_base", TRAINING_SEEDS),
        ("vipragsent_full_vistral", "vistral_7b", TRAINING_SEEDS),
        ("azure_gpt41_mini_8shot", "azure", (0,)),
        ("cot_only_vistral", "vistral_7b", TRAINING_SEEDS),
    ]
    predictions: dict[str, list[dict[str, Any]]] = {}
    for system, _, seeds in systems:
        predictions[system] = []
        for seed in seeds:
            prediction = _fixture_predictions(bundle, system, seed)
            predictions[system].append(prediction)
            _write_run(results_root, system, seed, prediction)
    # Q4 uses the approved concrete system IDs while the legacy fixture table rows
    # remain available for the broader synthetic artifact rehearsal.
    for approved_system, legacy_system in (("phobert_pragmatic_finetune", "phobert_finetune"), ("vistral_pragmatic_sft", "vistral_7b_sft")):
        predictions[approved_system] = list(predictions[legacy_system])
        for seed, prediction in zip(TRAINING_SEEDS, predictions[approved_system], strict=True):
            _write_run(results_root, approved_system, seed, prediction)

    table2_rows: list[dict[str, Any]] = []
    short_names = {"implicit_sentiment": "implicit", "sarcasm": "sarcasm", "irony": "irony", "idiom_figurative": "idiom", "code_switching": "code_switching", "mocking": "mocking"}
    for system, backbone, seeds in systems:
        runs = predictions[system]
        row: dict[str, Any] = {"system": system, "backbone": backbone, "seed_count": len(seeds), "invalid_output_rate": 0.0}
        for key, short in short_names.items():
            result = hierarchical_bootstrap([(run["true_pragmatic"][key], run["pred_pragmatic"][key]) for run in runs], binary_macro_f1, resamples=40)
            row[f"{short}_f1"], row[f"{short}_ci_low"], row[f"{short}_ci_high"] = result.observed, result.ci_low, result.ci_high
        result = hierarchical_bootstrap([(run["true_pragmatic"], run["pred_pragmatic"]) for run in runs], macro_pragmatic_f1, resamples=40)
        row["macro_prag_f1"], row["macro_prag_ci_low"], row["macro_prag_ci_high"] = result.observed, result.ci_low, result.ci_high
        table2_rows.append(row)
    _write_csv(output / "tables" / "table_2_pragmatic.csv", REQUIRED_COLUMNS["table_2_pragmatic.csv"].split(","), table2_rows)

    table3_rows: list[dict[str, Any]] = []
    for system, _, seeds in systems[:4]:
        runs = predictions[system]
        polarity = float(np.mean([multiclass_macro_f1(run["true_polarity"], run["polarity_probs"].argmax(axis=1), range(3)) for run in runs]))
        emotion = float(np.mean([multiclass_macro_f1(run["true_emotion"], run["emotion_probs"].argmax(axis=1), range(7)) for run in runs]))
        table3_rows.append({"system": system, "polarity_checkpoint": f"{system}_polarity", "emotion_checkpoint": f"{system}_emotion", "vsfc_macro_f1": polarity, "vsmec_macro_f1": emotion, "aivivn_macro_f1": polarity, "ord_f1": float(np.mean([polarity, emotion, polarity])), "seed_count": len(seeds), "training_data": "synthetic fixture", "external_finetuning": False})
    _write_csv(output / "tables" / "table_3_external_retention.csv", REQUIRED_COLUMNS["table_3_external_retention.csv"].split(","), table3_rows)

    table4_rows = []
    for index, configuration in enumerate(("full", "no_emotion_auxiliary", "no_polarity_auxiliary", "no_rationale", "no_multitask", "no_uncertainty_weighting")):
        table4_rows.append({"configuration": configuration, "backbone": "phobert_base", "prag_dev_f1": table2_rows[2]["macro_prag_f1"] - index * 0.005, "ord_external_f1": table3_rows[2]["ord_f1"], "polarity_dev_ece": expected_calibration_error(predictions["vipragsent_full_phobert"][0]["true_polarity"], predictions["vipragsent_full_phobert"][0]["polarity_probs"]), "gpu_hours": 0.0, "relative_cost_to_full_phobert": 1.0, "seed_count": 3, "changed_components": configuration})
    _write_csv(output / "tables" / "table_4_ablation.csv", REQUIRED_COLUMNS["table_4_ablation.csv"].split(","), table4_rows)

    q3_rows = []
    for system in ("phobert_finetune", "vistral_7b_sft", "vipragsent_full_vistral"):
        for budget in (32, 64, 128, 256, 512, "full"):
            selected = 8 if budget == "full" else min(int(budget), 8)
            for seed in TRAINING_SEEDS:
                q3_rows.append({"system": system, "budget": budget, "selected_positive_count": selected, "fixed_negative_count": 16, "seed": seed, "sarcasm_dev_f1": table2_rows[0]["sarcasm_f1"], "sarcasm_test_f1": table2_rows[0]["sarcasm_f1"], "dev_threshold": 0.5, "pos_weight": 16 / selected, "data_hash": bundle.fingerprint, "mask_hash": sha256_json({"budget": budget, "source": "synthetic-fixture"})})
    _write_csv(output / "backing_data" / "q3_low_resource.csv", REQUIRED_COLUMNS["q3_low_resource.csv"].split(","), q3_rows)

    q4_systems = (("phobert_pragmatic_finetune", "PhoBERT pragmatic fine-tune", "phobert_base"), ("vistral_pragmatic_sft", "Vistral-7B pragmatic SFT", "vistral_7b"), ("vipragsent_full_vistral", "Full ViPragSent Vistral", "vistral_7b"))
    q4_per_seed_rows: list[dict[str, Any]] = []
    q4_summary_rows: list[dict[str, Any]] = []
    q4_reliability_rows: list[dict[str, Any]] = []
    q4_macro_by_system: dict[str, float] = {}
    for system, display_name, _ in q4_systems:
        per_seed: list[dict[str, Any]] = []
        for seed, run in zip(TRAINING_SEEDS, predictions[system], strict=True):
            ece_by_label, macro_ece, reliability = pragmatic_ece(run["true_pragmatic"], run["prob_pragmatic"], bins=10)
            per_seed.append({"seed": seed, "ece_by_label": ece_by_label, "macro_pragmatic_ece": macro_ece})
            prediction_path = results_root / "runs" / system / str(seed) / "predictions.jsonl"
            prediction_hash = sha256_file(prediction_path)
            for label in PRAGMATIC_LABELS:
                q4_per_seed_rows.append({
                    "system": system,
                    "display_name": display_name,
                    "checkpoint_id": system,
                    "seed": seed,
                    "split": "vipragsent_test",
                    "label": label,
                    "ece": ece_by_label[label],
                    "macro_pragmatic_ece": macro_ece,
                    "bin_count": 10,
                    "temperature_scaling": False,
                    "prediction_file": prediction_path.relative_to(results_root).as_posix(),
                    "prediction_file_sha256": prediction_hash,
                    "config_hash": "fixture",
                    "code_commit": "fixture",
                })
                for row in reliability[label]:
                    q4_reliability_rows.append({"system": system, "seed": seed, "label": label, **row})
        q4_macro_values = [item["macro_pragmatic_ece"] for item in per_seed]
        q4_macro_by_system[system] = float(np.mean(q4_macro_values))
        for label in PRAGMATIC_LABELS:
            values = [item["ece_by_label"][label] for item in per_seed]
            q4_summary_rows.append({
                "system": system,
                "display_name": display_name,
                "label": label,
                "mean_ece": float(np.mean(values)),
                "std_ece": float(np.std(values, ddof=1)),
                "mean_macro_pragmatic_ece": float(np.mean(q4_macro_values)),
                "std_macro_pragmatic_ece": float(np.std(q4_macro_values, ddof=1)),
                "seed_count": len(per_seed),
                "split": "vipragsent_test",
                "bin_count": 10,
                "temperature_scaling": False,
            })
    _write_csv(output / "tables" / "q4_pragmatic_calibration_per_seed.csv", REQUIRED_COLUMNS["q4_pragmatic_calibration_per_seed.csv"].split(","), q4_per_seed_rows)
    _write_csv(output / "tables" / "q4_pragmatic_calibration_summary.csv", REQUIRED_COLUMNS["q4_pragmatic_calibration_summary.csv"].split(","), q4_summary_rows)
    _write_csv(output / "backing_data" / "q4_pragmatic_reliability_bins.csv", REQUIRED_COLUMNS["q4_pragmatic_reliability_bins.csv"].split(","), q4_reliability_rows)
    q4_learning_rows = [{"system": system, "seed": seed, "epoch": epoch, "dev_macro_pragmatic_f1": min(0.9, 0.45 + 0.04 * epoch), "dev_loss": max(0.1, 1.0 - 0.1 * epoch), "wall_seconds": float(epoch)} for system, _, _ in q4_systems for seed in TRAINING_SEEDS for epoch in range(1, 4)]
    _write_csv(output / "backing_data" / "q4_learning_curves.csv", REQUIRED_COLUMNS["q4_learning_curves.csv"].split(","), q4_learning_rows)

    sig_rows: list[dict[str, Any]] = []
    for left_name, right_name in (("vipragsent_full_vistral", "phobert_finetune"), ("vipragsent_full_vistral", "azure_gpt41_mini_8shot"), ("vipragsent_full_vistral", "vistral_7b_sft")):
        left_runs = predictions[left_name]
        right_runs = predictions[right_name]
        if len(right_runs) == 1:
            right_runs = right_runs * len(left_runs)
        family: list[Any] = []
        for key in (*PRAGMATIC_LABELS, "macro_prag_f1"):
            if key == "macro_prag_f1":
                metric = macro_pragmatic_f1
                left = [(run["true_pragmatic"], run["pred_pragmatic"]) for run in left_runs]
                right = [(run["true_pragmatic"], run["pred_pragmatic"]) for run in right_runs]
            else:
                metric = binary_macro_f1
                left = [(run["true_pragmatic"][key], run["pred_pragmatic"][key]) for run in left_runs]
                right = [(run["true_pragmatic"][key], run["pred_pragmatic"][key]) for run in right_runs]
            result = paired_bootstrap_comparison(left, right, metric, resamples=40, p_value_method="paired_hierarchical_bootstrap_sign_plus_one_v1")
            family.append(result)
            sig_rows.append({"comparison": f"{left_name}_vs_{right_name}", "metric": key, "observed_delta": result.observed, "ci_low": result.ci_low, "ci_high": result.ci_high, "raw_p_value": result.p_value, "holm_adjusted_p_value": 0.0, "resamples": 40, "bootstrap_seed": 20260525, "prediction_files": f"runs/{left_name};runs/{right_name}"})
        corrected = holm_bonferroni([float(result.p_value) for result in family])
        for row, value in zip(sig_rows[-7:], corrected, strict=True):
            row["holm_adjusted_p_value"] = value
    _write_csv(output / "tables" / "significance.csv", REQUIRED_COLUMNS["significance.csv"].split(","), sig_rows)

    cost_rows = [{"system": system, "backbone": backbone, "gpu_hours": 0.0, "relative_cost_to_full_phobert": 1.0, "batch1_latency_ms": 0.0, "batch32_examples_per_second": 0.0, "peak_vram_gb": 0.0, "gpu_model": "fixture", "mig_profile": "none", "azure_request_count": 0, "input_tokens": 0, "output_tokens": 0, "azure_cost_status": "not-priced-fixture" if backbone == "azure" else "not-applicable"} for system, backbone, _ in systems]
    _write_csv(output / "tables" / "cost_latency.csv", REQUIRED_COLUMNS["cost_latency.csv"].split(","), cost_rows)
    _write_csv(output / "backing_data" / "latency_measurements.csv", ["system", "batch_size", "repetition", "latency_ms", "examples_per_second", "warmup_iterations_excluded"], [{"system": row["system"], "batch_size": 1, "repetition": repetition, "latency_ms": 0.0, "examples_per_second": 0.0, "warmup_iterations_excluded": 50} for row in cost_rows for repetition in range(3)])
    backbone_rows = [{"system": system, "backbone": "phobert_base" if "phobert" in system else "vistral_7b", "macro_prag_f1": next(row for row in table2_rows if row["system"] == system)["macro_prag_f1"], "ord_f1": table3_rows[2]["ord_f1"], "polarity_ece": float(np.mean([expected_calibration_error(run["true_polarity"], run["polarity_probs"]) for run in predictions["vipragsent_full_phobert" if "phobert" in system else "vipragsent_full_vistral"]])), "gpu_hours": 0.0, "relative_cost": 1.0, "peak_vram_gb": 0.0, "batch1_latency_ms": 0.0, "batch32_examples_per_second": 0.0, "seed_count": 3} for system in ("vipragsent_full_phobert", "vipragsent_full_vistral")]
    _write_csv(output / "tables" / "backbone_sensitivity.csv", REQUIRED_COLUMNS["backbone_sensitivity.csv"].split(","), backbone_rows)

    manual = output / "manual"
    error_columns = ERROR_ANALYSIS_COLUMNS
    error_rows = []
    for index, example in enumerate(bundle.test[: min(64, len(bundle.test))]):
        label = PRAGMATIC_LABELS[index % len(PRAGMATIC_LABELS)]
        error_rows.append({"sample_id": example.sample_id, "label": label, "text": example.text, "gold_label": example.labels[label], "phobert_prediction": int(predictions["phobert_finetune"][0]["pred_pragmatic"][label][index]), "azure_prediction": int(predictions["azure_gpt41_mini_8shot"][0]["pred_pragmatic"][label][index]), "full_vistral_prediction": int(predictions["vipragsent_full_vistral"][0]["pred_pragmatic"][label][index]), "phobert_confidence": float(predictions["phobert_finetune"][0]["prob_pragmatic"][label][index]), "azure_confidence": float(predictions["azure_gpt41_mini_8shot"][0]["prob_pragmatic"][label][index]), "full_vistral_confidence": float(predictions["vipragsent_full_vistral"][0]["prob_pragmatic"][label][index]), "stratum": "fixture", "selection_reason": "fixture validation", "reviewer_1_category": "", "reviewer_2_category": "", "adjudicated_category": ""})
    _write_csv(manual / "error_analysis_candidates.csv", error_columns, error_rows)
    _write_csv(manual / "error_analysis_annotation_template.csv", ["sample_id", "label", "reviewer", "category", "notes"], [])
    _write_csv(manual / "error_analysis_final.csv", error_columns, [])
    with (manual / "qualitative_candidates.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for example in bundle.test[: min(20, len(bundle.test))]:
            handle.write(json.dumps({"sample_id": example.sample_id, "text": example.text, "candidate_reason": "synthetic fixture candidate", "approval": "pending"}, ensure_ascii=False) + "\n")
    (manual / "qualitative_final.jsonl").write_text("", encoding="utf-8")
    _write_csv(manual / "qualitative_approval_template.csv", ["sample_id", "reviewer", "approved", "notes"], [])
    _write_csv(output / "backing_data" / "dev_learning_curves.csv", ["system", "seed", "epoch", "dev_macro_pragmatic_f1"], [{"system": system, "seed": seed, "epoch": epoch, "dev_macro_pragmatic_f1": min(0.9, 0.45 + 0.04 * epoch)} for system, _, _ in q4_systems for seed in TRAINING_SEEDS for epoch in range(1, 4)])
    _write_csv(output / "backing_data" / "reliability_bins.csv", ["system", "seed", "bin", "lower", "upper", "count", "mean_confidence", "accuracy"], [{"system": system, "seed": seed, "bin": row["bin_index"], "lower": row["bin_lower"], "upper": row["bin_upper"], "count": row["count"], "mean_confidence": row["mean_confidence"], "accuracy": row["empirical_positive_rate"]} for system, _, _ in q4_systems for seed in TRAINING_SEEDS for row in pragmatic_reliability_bins(predictions[system][0]["true_pragmatic"], predictions[system][0]["prob_pragmatic"])[PRAGMATIC_LABELS[0]]])
    _write_csv(output / "backing_data" / "per_phenomenon_f1.csv", ["label", "f1"], [{"label": key, "f1": table2_rows[2][f"{short}_f1"]} for key, short in (("implicit_sentiment", "implicit"), ("sarcasm", "sarcasm"), ("irony", "irony"), ("idiom_figurative", "idiom"), ("code_switching", "code_switching"), ("mocking", "mocking"))])
    _write_csv(output / "backing_data" / "multi_task_gain.csv", ["system", "macro_prag_f1"], [{"system": table2_rows[index]["system"], "macro_prag_f1": table2_rows[index]["macro_prag_f1"]} for index in (0, 2)])
    _write_csv(output / "backing_data" / "q3_low_resource_curve.csv", ["budget", "sarcasm_test_f1"], [{"budget": row["budget"], "sarcasm_test_f1": row["sarcasm_test_f1"]} for row in q3_rows])
    _svg_bar(output / "figures" / "per_phenomenon_f1.svg", list(PRAGMATIC_LABELS), [float(table2_rows[2][f"{short_names[key]}_f1"]) for key in PRAGMATIC_LABELS], "Per-phenomenon F1")
    _svg_bar(output / "figures" / "multi_task_gain.svg", ["PhoBERT", "Full"], [float(table2_rows[0]["macro_prag_f1"]), float(table2_rows[2]["macro_prag_f1"])], "Multi-task gain")
    _svg_bar(output / "figures" / "q3_low_resource_learning_curve.svg", [str(budget) for budget in (32, 64, 128, 256, 512, "full")], [0.5 + 0.02 * index for index in range(6)], "Q3 low-resource curve")
    _svg_bar(output / "figures" / "dev_learning_curves.svg", ["1", "2", "3"], [0.49, 0.54, 0.58], "Dev-set learning curves")
    _svg_bar(output / "figures" / "reliability_diagrams.svg", ["PhoBERT", "Vistral SFT", "Full Vistral"], [q4_macro_by_system[system] for system, _, _ in q4_systems], "Pragmatic reliability diagrams")
    _fixture_visual(output / "figures" / "q4_pragmatic_ece_heatmap.pdf", kind="pdf")
    _fixture_visual(output / "figures" / "q4_pragmatic_ece_heatmap.png", kind="png")
    _fixture_visual(output / "figures" / "q4_pragmatic_reliability_by_label.pdf", kind="pdf")
    _fixture_visual(output / "figures" / "q4_pragmatic_reliability_by_label.png", kind="png")
    _fixture_visual(output / "figures" / "q4_learning_curves.pdf", kind="pdf")
    _fixture_visual(output / "figures" / "q4_learning_curves.png", kind="png")
    return len(list(output.rglob("*")))


def export_fixture_artifacts(*, repo_root: str | Path = ".", run_id: str = "fixture", output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root)
    execution_root = Path(output_root) if output_root else root / "runs" / "fixture"
    output = execution_root / "artifacts"
    results_root = execution_root / "results"
    bundle = _synthetic_bundle()
    _dataset_artifacts(bundle, output, fixture=True)
    artifact_count = _export_fixture_tables(output, results_root, bundle)
    errors = validate_artifact_tree(output)
    if errors:
        raise ValueError("Fixture artifact schema validation failed: " + "; ".join(errors))
    manifest = {
        "run_id": run_id,
        "mode": "fixture",
        "fixture_validation_passed": True,
        "synthetic_results": True,
        "core_experiments_ready": False,
        "manual_paper_analysis_pending": True,
        "artifact_count": artifact_count,
    }
    manifest_path = execution_root / "FIXTURE_VALIDATION_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), **manifest}


def _normalise_seed(value: Any, path: Path) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Production run has an invalid seed: {path}") from exc


def _normalise_budget(value: Any) -> str:
    return str(value).strip().lower() if value not in (None, "") else ""


def _resolve_run_file(root: Path, metrics_path: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for parent in (metrics_path.parent, root):
        resolved = parent / candidate
        if resolved.exists():
            return resolved
    return metrics_path.parent / candidate


def _prediction_paths(root: Path, metrics_path: Path, payload: dict[str, Any]) -> list[Path]:
    raw = payload.get("prediction_files", payload.get("prediction_file"))
    values = [raw] if isinstance(raw, (str, Path)) else list(raw or [])
    if not values:
        values = ["test_predictions.jsonl", "predictions.jsonl"]
    paths = [_resolve_run_file(root, metrics_path, value) for value in values]
    existing = [path for path in paths if path.exists() and path.is_file()]
    if not existing:
        raise ValueError(f"Production run has no frozen prediction file: {metrics_path}")
    return existing


def _validate_record_row(row: dict[str, Any], columns: list[str], *, source: str) -> None:
    if set(row) != set(columns):
        raise ValueError(f"Production paper row for {source} does not match its locked schema")
    if any(value is None or (isinstance(value, str) and not value.strip()) for value in row.values()):
        raise ValueError(f"Production paper row for {source} contains an empty cell")


def _paper_rows(root: Path, records: list[dict[str, Any]], filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        paper_artifacts = record.get("paper_artifacts", {})
        raw = paper_artifacts.get(filename) if isinstance(paper_artifacts, dict) else None
        if isinstance(raw, str):
            path = _resolve_run_file(root, Path(record["path"]), raw)
            if not path.exists():
                raise ValueError(f"Production paper artifact is missing: {path}")
            raw = json.loads(path.read_text(encoding="utf-8"))
        if raw is None:
            continue
        if not isinstance(raw, list):
            raise ValueError(f"Production paper artifact rows must be a list: {filename}")
        columns = REQUIRED_COLUMNS[filename].split(",")
        for row in raw:
            if not isinstance(row, dict):
                raise ValueError(f"Production paper artifact row is not an object: {filename}")
            _validate_record_row(row, columns, source=filename)
            rows.append(dict(row))
    return rows


def _read_production_runs(root: Path) -> list[dict[str, Any]]:
    run_root = root / "results" / "runs"
    if not run_root.exists():
        raise ValueError("Production result runs are missing")
    manifest_files = sorted(run_root.glob("*/run_manifest.json"))
    legacy_adapter = False
    if not manifest_files:
        # Fixture/migration adapter only: old system/seed trees are never accepted as production output.
        manifest_files = sorted(run_root.glob("*/*/metrics.json"))
        legacy_adapter = True
    if not manifest_files:
        raise ValueError("No production run manifests were found")
    records: list[dict[str, Any]] = []
    required = {
        "system", "backbone", "seed", "mode", "model_revision", "tokenizer_revision",
        "preprocessing_name", "preprocessing_version", "physical_batch_size",
        "gradient_accumulation_steps", "effective_batch_size", "inference_output_source",
        "rationale_decoder_enabled_at_inference", "data_fingerprint", "config_hash", "code_commit",
    }
    for path in manifest_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Production run manifest is not valid JSON: {path}") from exc
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError(f"Production run manifest is incomplete: {path}")
        if payload.get("mode") != "full" or payload.get("synthetic_results") is True or payload.get("model_revision") == "fixture" or payload.get("tokenizer_revision") == "fixture":
            raise ValueError(f"Fixture or synthetic run cannot enter production export: {path}")
        if payload.get("external_finetuning") is True:
            raise ValueError(f"External fine-tuning is prohibited for this production run: {path}")
        if payload.get("inference_output_source") not in {"classification_heads", "judge_of_generated_reasoning", "judge_of_rationale_decoder_output", "parsed_generated_labels"}:
            raise ValueError(f"Production run has an invalid inference output source: {path}")
        if payload.get("rationale_decoder_enabled_at_inference") is not False:
            raise ValueError(f"Rationale decoder must be disabled at inference: {path}")
        seed = _normalise_seed(payload.get("seed"), path)
        is_azure = payload.get("backbone") == "azure" or str(payload.get("system", "")).startswith("azure_")
        if not is_azure and seed not in TRAINING_SEEDS:
            raise ValueError(f"Trainable production run does not use a locked seed: {path}")
        role = payload.get("paper_role")
        if role in PAPER_ROLE_BACKBONES and payload.get("backbone") != PAPER_ROLE_BACKBONES[role]:
            raise ValueError(f"Production run uses the wrong paper backbone role: {path}")
        known_backbone = SYSTEM_BACKBONES.get(str(payload.get("system")))
        if known_backbone and payload.get("backbone") != known_backbone:
            raise ValueError(f"Production run uses the wrong backbone for {payload['system']}: {path}")
        if not legacy_adapter:
            approval_path = path.parent / "approval_status.json"
            approval = json.loads(approval_path.read_text(encoding="utf-8")) if approval_path.exists() else {}
            if approval.get("status") != "APPROVED":
                raise ValueError(f"Production run is not explicitly APPROVED: {path}")
        prediction_paths = _prediction_paths(root, path, payload)
        records.append({"path": path, **payload, "seed": seed, "prediction_paths": prediction_paths, "legacy_fixture_adapter": legacy_adapter})

    groups: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for record in records:
        if record["backbone"] == "azure" or str(record["system"]).startswith("azure_"):
            continue
        groups[(str(record.get("research_question", "all")), str(record["system"]), _normalise_budget(record.get("budget")))] .add(record["seed"])
    incomplete_seed_groups = [key for key, seeds in groups.items() if seeds != set(TRAINING_SEEDS)]
    if incomplete_seed_groups:
        raise ValueError(f"Production export is missing required training seeds: {incomplete_seed_groups}")

    q3_groups: dict[tuple[str, int | None], set[str]] = defaultdict(set)
    for record in records:
        if str(record.get("research_question", "")).upper() == "Q3" or record.get("budget") not in (None, ""):
            q3_groups[(str(record["system"]), record["seed"])].add(_normalise_budget(record.get("budget")))
    incomplete_q3_groups = [key for key, budgets in q3_groups.items() if budgets != set(Q3_BUDGETS)]
    if incomplete_q3_groups:
        raise ValueError(f"Production export is missing required Q3 budgets: {incomplete_q3_groups}")

    expected_path = root / "reports/expected_experiment_runs.json"
    if expected_path.exists():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        expected_systems = {str(row["system"]) for row in expected.get("rows", [])}
        actual_systems = {str(record["system"]) for record in records}
        missing_systems = sorted(expected_systems - actual_systems)
        if missing_systems:
            raise ValueError(f"Production export is missing expected systems: {missing_systems}")
    return records


def _sidecar_rows(records: list[dict[str, Any]], filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        figure_backing = record.get("figure_backing", {})
        raw = figure_backing.get(filename) if isinstance(figure_backing, dict) else None
        if raw is None:
            continue
        if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
            raise ValueError(f"Figure backing rows must be a list of objects: {filename}")
        rows.extend(dict(row) for row in raw)
    return rows


def _write_sidecar_rows(output: Path, filename: str, columns: list[str], rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"Production export has no backing data for {filename}")
    for row in rows:
        _validate_record_row(row, columns, source=filename)
    path = output / "backing_data" / filename
    _write_csv(path, columns, rows)
    return path


def _write_q4_production_figures(
    output: Path,
    calibration_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    learning_rows: list[dict[str, Any]],
) -> None:
    """Render the locked Q4 figures from real backing rows; never use fixture bytes."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("Production Q4 figure export requires the optional matplotlib dependency") from exc

    figure_root = output / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    system_order = ("phobert_pragmatic_finetune", "vistral_pragmatic_sft", "vipragsent_full_vistral")
    display_order = {"phobert_pragmatic_finetune": "PhoBERT pragmatic fine-tune", "vistral_pragmatic_sft": "Vistral-7B pragmatic SFT", "vipragsent_full_vistral": "Full ViPragSent Vistral"}
    label_order = ("implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking")
    label_display = {"implicit_sentiment": "implicit sentiment", "sarcasm": "sarcasm", "irony": "irony", "idiom_figurative": "idiom/figurative", "code_switching": "code-switching", "mocking": "mocking"}

    summary_by_system: dict[str, dict[str, dict[str, Any]]] = {}
    for row in calibration_rows:
        summary_by_system.setdefault(str(row["system"]), {})[str(row["label"])] = row
    heatmap_values = []
    for system in system_order:
        by_label = summary_by_system.get(system, {})
        values = [float(by_label[label]["mean_ece"]) for label in label_order]
        macro = float(next(iter(by_label.values()))["mean_macro_pragmatic_ece"])
        heatmap_values.append(values + [macro])
    fig, axis = plt.subplots(figsize=(11, 3.5))
    image = axis.imshow(np.asarray(heatmap_values), aspect="auto", cmap="viridis")
    axis.set_xticks(range(7), [label_display[label] for label in label_order] + ["macro"])
    axis.set_yticks(range(3), [display_order[system] for system in system_order])
    axis.set_title("Q4 pragmatic expected calibration error")
    for row_index, values in enumerate(heatmap_values):
        for column_index, value in enumerate(values):
            axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=axis, label="ECE")
    fig.tight_layout()
    fig.savefig(figure_root / "q4_pragmatic_ece_heatmap.pdf")
    fig.savefig(figure_root / "q4_pragmatic_ece_heatmap.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes_flat = list(axes.flat)
    for axis, label in zip(axes_flat, label_order, strict=True):
        for system in system_order:
            rows = [row for row in reliability_rows if str(row["system"]) == system and str(row["label"]) == label and int(row["count"]) > 0]
            grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[int(row["bin_index"])].append(row)
            points = [(float(np.mean([float(item["mean_confidence"]) for item in items])), float(np.mean([float(item["empirical_positive_rate"]) for item in items]))) for items in grouped.values()]
            points.sort()
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=display_order[system])
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8)
        axis.set_title(label_display[label])
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("mean predicted positive probability")
    axes[1, 1].set_xlabel("mean predicted positive probability")
    axes[1, 2].set_xlabel("mean predicted positive probability")
    axes[0, 0].set_ylabel("empirical positive frequency")
    axes[1, 0].set_ylabel("empirical positive frequency")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Q4 pragmatic reliability by label")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figure_root / "q4_pragmatic_reliability_by_label.pdf")
    fig.savefig(figure_root / "q4_pragmatic_reliability_by_label.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for system in system_order:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in learning_rows:
            if str(row["system"]) == system:
                grouped[int(row["epoch"])].append(row)
        epochs = sorted(grouped)
        means = [float(np.mean([float(item["dev_macro_pragmatic_f1"]) for item in grouped[epoch]])) for epoch in epochs]
        stds = [float(np.std([float(item["dev_macro_pragmatic_f1"]) for item in grouped[epoch]], ddof=1)) if len(grouped[epoch]) > 1 else 0.0 for epoch in epochs]
        axis.plot(epochs, means, marker="o", label=display_order[system])
        axis.fill_between(epochs, np.asarray(means) - np.asarray(stds), np.asarray(means) + np.asarray(stds), alpha=0.15)
    axis.set_xlabel("epoch")
    axis.set_ylabel("dev macro pragmatic F1")
    axis.set_title("Q4 pragmatic learning dynamics")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_root / "q4_learning_curves.pdf")
    fig.savefig(figure_root / "q4_learning_curves.png", dpi=160)
    plt.close(fig)


def export_production_artifacts(*, repo_root: str | Path = ".", run_id: str = "full", output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root)
    output = Path(output_root) if output_root else root / "experiment_artifacts"
    if "runs" in output.parts and "fixture" in output.parts:
        raise ValueError("Production artifact export cannot target a fixture root")
    records = _read_production_runs(root)
    manual = output / "manual"
    required_manual = (
        "error_analysis_candidates.csv", "error_analysis_annotation_template.csv", "error_analysis_final.csv",
        "qualitative_candidates.jsonl", "qualitative_approval_template.csv", "qualitative_final.jsonl",
    )
    missing_manual = [str(manual / name) for name in required_manual if not (manual / name).exists()]
    if missing_manual:
        raise ValueError(f"Production manual-analysis artifacts are missing: {missing_manual}")
    bundle = __import__("vipragsent.data.loaders", fromlist=["load_vipragsent"]).load_vipragsent(root / "data/processed/vipragsent")
    table_rows: dict[str, list[dict[str, Any]]] = {}
    table2_columns = REQUIRED_COLUMNS["table_2_pragmatic.csv"].split(",")
    table2_rows = _paper_rows(root, records, "table_2_pragmatic.csv")
    if not table2_rows:
        for record in records:
            if "macro_prag_f1" not in record:
                raise ValueError(f"Production run has no computed dev metric: {record['path']}")
            table2_rows.append({column: record.get(column, "") for column in table2_columns} | {
                "system": record["system"],
                "backbone": record["backbone"],
                "seed_count": record["seed_count"],
                "macro_prag_f1": record["macro_prag_f1"],
                "invalid_output_rate": record["invalid_output_rate"],
            })
    table_rows["table_2_pragmatic.csv"] = table2_rows
    for filename in PRODUCTION_PAPER_TABLES[1:]:
        rows = _paper_rows(root, records, filename)
        if not rows:
            raise ValueError(f"Production export has no computed rows for {filename}")
        table_rows[filename] = rows
    _dataset_artifacts(bundle, output, fixture=False, repo_root=root)
    for filename, rows in table_rows.items():
        directory = "backing_data" if filename == "q3_low_resource.csv" else "tables"
        _write_csv(output / directory / filename, REQUIRED_COLUMNS[filename].split(","), rows)

    latency_columns = ["system", "batch_size", "repetition", "latency_ms", "examples_per_second", "warmup_iterations_excluded"]
    latency_rows = _sidecar_rows(records, "latency_measurements.csv")
    _write_sidecar_rows(output, "latency_measurements.csv", latency_columns, latency_rows)

    figure_backing = {
        "per_phenomenon_f1.csv": (
            ["label", "f1"],
            [{"label": label, "f1": next(row[f"{short}_f1"] for row in table2_rows if row["system"] == "vipragsent_full_vistral")} for label, short in (("implicit_sentiment", "implicit"), ("sarcasm", "sarcasm"), ("irony", "irony"), ("idiom_figurative", "idiom"), ("code_switching", "code_switching"), ("mocking", "mocking"))],
        ),
        "multi_task_gain.csv": (["system", "macro_prag_f1"], [{"system": row["system"], "macro_prag_f1": row["macro_prag_f1"]} for row in table2_rows if row["system"] in {"phobert_finetune", "vipragsent_full_vistral"}]),
        "q3_low_resource_curve.csv": (["budget", "sarcasm_test_f1"], [{"budget": row["budget"], "sarcasm_test_f1": row["sarcasm_test_f1"]} for row in table_rows["q3_low_resource.csv"]]),
        "dev_learning_curves.csv": (["system", "seed", "epoch", "dev_macro_pragmatic_f1"], _sidecar_rows(records, "dev_learning_curves.csv")),
        "reliability_bins.csv": (["system", "seed", "bin", "lower", "upper", "count", "mean_confidence", "accuracy"], _sidecar_rows(records, "reliability_bins.csv")),
        "q4_pragmatic_reliability_bins.csv": (REQUIRED_COLUMNS["q4_pragmatic_reliability_bins.csv"].split(","), _sidecar_rows(records, "q4_pragmatic_reliability_bins.csv")),
        "q4_learning_curves.csv": (REQUIRED_COLUMNS["q4_learning_curves.csv"].split(","), _sidecar_rows(records, "q4_learning_curves.csv")),
    }
    for filename, (columns, rows) in figure_backing.items():
        _write_sidecar_rows(output, filename, columns, rows)
    _write_q4_production_figures(
        output,
        table_rows["q4_pragmatic_calibration_summary.csv"],
        figure_backing["q4_pragmatic_reliability_bins.csv"][1],
        figure_backing["q4_learning_curves.csv"][1],
    )
    figure_values = {
        "per_phenomenon_f1.svg": ([row["label"] for row in figure_backing["per_phenomenon_f1.csv"][1]], [float(row["f1"]) for row in figure_backing["per_phenomenon_f1.csv"][1]], "Per-phenomenon F1"),
        "multi_task_gain.svg": ([row["system"] for row in figure_backing["multi_task_gain.csv"][1]], [float(row["macro_prag_f1"]) for row in figure_backing["multi_task_gain.csv"][1]], "Multi-task gain"),
        "q3_low_resource_learning_curve.svg": ([str(row["budget"]) for row in figure_backing["q3_low_resource_curve.csv"][1]], [float(row["sarcasm_test_f1"]) for row in figure_backing["q3_low_resource_curve.csv"][1]], "Q3 low-resource curve"),
        "dev_learning_curves.svg": ([str(row["epoch"]) for row in figure_backing["dev_learning_curves.csv"][1]], [float(row["dev_macro_pragmatic_f1"]) for row in figure_backing["dev_learning_curves.csv"][1]], "Dev-set learning curves"),
        "reliability_diagrams.svg": ([str(row["bin"]) for row in figure_backing["reliability_bins.csv"][1]], [float(row["accuracy"]) for row in figure_backing["reliability_bins.csv"][1]], "Reliability diagrams"),
    }
    for filename, (labels, values, title) in figure_values.items():
        _svg_bar(output / "figures" / filename, labels, values, title)
    errors = validate_production_artifact(output, records)
    if errors:
        raise ValueError("Production artifact validation failed: " + "; ".join(errors))
    provenance = output.parent / "results" / "result_provenance_index.csv"
    source_files = ";".join(sorted(path.relative_to(root).as_posix() for path in (record["path"] for record in records)))
    provenance_rows = [{"artifact": path.relative_to(root).as_posix(), "source_files": source_files, "script": "src/vipragsent/artifacts/exporter.py", "sha256": sha256_file(path), "model_or_azure_metadata": "full"} for path in sorted(output.rglob("*")) if path.is_file()]
    _write_csv(provenance, ["artifact", "source_files", "script", "sha256", "model_or_azure_metadata"], provenance_rows)
    manifest = {"run_id": run_id, "mode": "full", "core_experiments_ready": True, "manual_paper_analysis_pending": True, "artifact_count": len(list(output.rglob("*"))), "production_run_manifest_count": len(records), "provenance_index": str(provenance.relative_to(root))}
    manifest_path = root / "FINAL_EXPERIMENT_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    final_root = root / "results/final"
    atomic_write_json(final_root / "export_manifest.json", manifest)
    checksum_files = sorted(path for path in (*output.rglob("*"), final_root / "export_manifest.json") if path.is_file())
    atomic_write_text(root / "FINAL_RESULT_CHECKSUMS.sha256", "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in checksum_files))
    return {"manifest_path": str(manifest_path), **manifest}
