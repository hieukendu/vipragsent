from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS, TRAINING_SEEDS
from ..data.annotation import recompute_human_iaa
from ..data.loaders import DatasetBundle, load_vipragsent
from ..evaluation.metrics import binary_macro_f1, expected_calibration_error, macro_pragmatic_f1, multiclass_macro_f1
from ..hashing import sha256_file, sha256_json
from ..statistics.bootstrap import hierarchical_bootstrap, holm_bonferroni, paired_bootstrap_comparison
from .schemas import REQUIRED_COLUMNS, validate_artifact_tree


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _svg_bar(path: Path, labels: list[str], values: list[float], title: str) -> None:
    width, height = 900, 420
    max_value = max(values or [1.0])
    bar_width = width / max(len(values), 1) * 0.7
    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        x = width / max(len(values), 1) * index + width / max(len(values), 1) * 0.15
        bar_height = (height - 100) * value / max_value
        y = height - 60 - bar_height
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#2563eb"/>')
        bars.append(f'<text x="{x + bar_width / 2:.2f}" y="{height - 36}" text-anchor="middle" font-size="12">{label}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="20" y="28" font-size="20">{title}</text>{"".join(bars)}</svg>'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _fixture_predictions(bundle: DatasetBundle, system: str, seed: int, *, n: int | None = None) -> dict[str, Any]:
    examples = bundle.test[:n] if n else bundle.test
    seed_value = seed + sum(ord(char) for char in system)
    rng = np.random.default_rng(seed_value)
    true_prag = {key: np.asarray([example.labels[key] for example in examples], dtype=int) for key in PRAGMATIC_LABELS}
    pred_prag: dict[str, np.ndarray] = {}
    prob_prag: dict[str, np.ndarray] = {}
    error_rate = 0.12 if "full" in system else 0.22
    for key in PRAGMATIC_LABELS:
        probabilities = np.where(true_prag[key] == 1, 0.72, 0.28) + rng.normal(0, 0.16, len(examples))
        probabilities = np.clip(probabilities, 0.01, 0.99)
        pred_prag[key] = (probabilities >= 0.5).astype(int)
        flips = rng.random(len(examples)) < error_rate
        pred_prag[key][flips] = 1 - pred_prag[key][flips]
        prob_prag[key] = probabilities
    true_polarity = np.asarray([POLARITY_LABELS.index(example.labels["polarity"]) for example in examples])
    true_emotion = np.asarray([EMOTION_LABELS.index(example.labels["emotion"]) for example in examples])
    polarity_probs = np.full((len(examples), len(POLARITY_LABELS)), 0.08)
    polarity_probs[np.arange(len(examples)), true_polarity] = 0.84
    polarity_probs += rng.normal(0, 0.04, polarity_probs.shape)
    polarity_probs = np.clip(polarity_probs, 0.001, None)
    polarity_probs /= polarity_probs.sum(axis=1, keepdims=True)
    emotion_probs = np.full((len(examples), len(EMOTION_LABELS)), 0.04)
    emotion_probs[np.arange(len(examples)), true_emotion] = 0.76
    emotion_probs += rng.normal(0, 0.04, emotion_probs.shape)
    emotion_probs = np.clip(emotion_probs, 0.001, None)
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


def _write_run(root: Path, system: str, seed: int, prediction: dict[str, Any]) -> None:
    run_root = root / "results" / "runs" / system / str(seed)
    run_root.mkdir(parents=True, exist_ok=True)
    metrics = {
        "system": system,
        "seed": seed,
        "macro_prag_f1": macro_pragmatic_f1(prediction["true_pragmatic"], prediction["pred_pragmatic"]),
        "polarity_macro_f1": multiclass_macro_f1(prediction["true_polarity"], prediction["polarity_probs"].argmax(axis=1), range(len(POLARITY_LABELS))),
        "emotion_macro_f1": multiclass_macro_f1(prediction["true_emotion"], prediction["emotion_probs"].argmax(axis=1), range(len(EMOTION_LABELS))),
        "polarity_ece": expected_calibration_error(prediction["true_polarity"], prediction["polarity_probs"]),
        "inference_output_source": "classification_heads" if "cot" not in system else "parsed_generated_labels",
        "rationale_decoder_enabled_at_inference": False,
        "preprocessing_name": "vncorenlp_rdrsegmenter" if "phobert" in system else "unicode_nfc",
        "preprocessing_version": "fixture-v1",
        "tokenizer_revision": "fixture",
        "model_revision": "fixture",
        "physical_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "effective_batch_size": 32,
    }
    (run_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (run_root / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(prediction["sample_ids"]):
            handle.write(json.dumps({
                "sample_id": sample_id,
                "pragmatic": {key: int(prediction["pred_pragmatic"][key][index]) for key in PRAGMATIC_LABELS},
                "polarity": POLARITY_LABELS[int(prediction["polarity_probs"][index].argmax())],
                "emotion": EMOTION_LABELS[int(prediction["emotion_probs"][index].argmax())],
                "logits": {"polarity": prediction["polarity_probs"][index].tolist(), "emotion": prediction["emotion_probs"][index].tolist()},
            }, ensure_ascii=False) + "\n")


def _dataset_artifacts(root: Path, bundle: DatasetBundle, output: Path) -> None:
    raw_candidates = list((root / "data" / "raw" / "vipragsent_package").glob("*/"))
    package_dir = raw_candidates[0] if raw_candidates else None
    iaa = recompute_human_iaa(package_dir) if package_dir and (package_dir / "01_clean_human_annotations").exists() else []
    manifest = bundle.manifest
    table_rows = [{
        "dataset": "ViPragSent",
        "role": "train/dev/test and Q1a",
        "train_count": len(bundle.train),
        "dev_count": len(bundle.dev),
        "test_count": len(bundle.test),
        "total_count": len(bundle.train) + len(bundle.dev) + len(bundle.test),
        "task": "six pragmatic binary + polarity + emotion",
        "label_space": "canonical_labels",
        "source_manifest": "data/manifests/dataset_manifest.json",
        "checksum": manifest.get("processed_fingerprint", bundle.fingerprint),
        "redistribution_status": "private-input-not-redistributed",
    }, {
        "dataset": "AIVIVN-human-derived-3way",
        "role": "Q1b polarity retention",
        "train_count": 12869,
        "dev_count": 1609,
        "test_count": 1609,
        "total_count": 16087,
        "task": "polarity",
        "label_space": "negative|neutral|positive",
        "source_manifest": "data/processed/external/aivivn_human_derived_3way/manifest.json",
        "checksum": "bundled-package",
        "redistribution_status": "project-bundled",
    }, {
        "dataset": "UIT-VSFC",
        "role": "Q1b polarity retention",
        "train_count": "NA",
        "dev_count": "NA",
        "test_count": "NA",
        "total_count": "NA",
        "task": "polarity",
        "label_space": "negative|neutral|positive",
        "source_manifest": "data/manifests/external_datasets.json",
        "checksum": "PENDING_MANUAL_DROP",
        "redistribution_status": "manual-drop-required",
    }, {
        "dataset": "UIT-VSMEC",
        "role": "Q1b emotion retention",
        "train_count": "NA",
        "dev_count": "NA",
        "test_count": "NA",
        "total_count": "NA",
        "task": "emotion",
        "label_space": "seven_emotions",
        "source_manifest": "data/manifests/external_datasets.json",
        "checksum": "PENDING_MANUAL_DROP",
        "redistribution_status": "manual-drop-required",
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
    iaa_rows = iaa or [{"field": "unavailable", "n": 0, "raw_agreement": "NA", "cohen_kappa": "NA", "krippendorff_alpha_nominal": "NA", "disagreement_count": "NA"}]
    _write_csv(output / "tables" / "human_iaa_summary.csv", REQUIRED_COLUMNS["human_iaa_summary.csv"].split(","), iaa_rows)


def export_fixture_artifacts(*, repo_root: str | Path = ".", run_id: str = "fixture") -> dict[str, Any]:
    root = Path(repo_root)
    output = root / "experiment_artifacts"
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_vipragsent(root / "data" / "processed" / "vipragsent")
    _dataset_artifacts(root, bundle, output)
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
            # Keep fixture validation fast; production runs use the complete frozen test split.
            item = _fixture_predictions(bundle, system, seed, n=512)
            predictions[system].append(item)
            _write_run(root, system, seed, item)
    table2_rows = []
    for system, backbone, seeds in systems:
        seed_runs = predictions[system]
        values: dict[str, tuple[float, float, float]] = {}
        for key in PRAGMATIC_LABELS:
            result = hierarchical_bootstrap([(run["true_pragmatic"][key], run["pred_pragmatic"][key]) for run in seed_runs], binary_macro_f1, resamples=200)
            values[key] = (result.observed, result.ci_low, result.ci_high)
        macro = hierarchical_bootstrap([(run["true_pragmatic"], run["pred_pragmatic"]) for run in seed_runs], macro_pragmatic_f1, resamples=200)
        row = {"system": system, "backbone": backbone, "seed_count": len(seeds), "invalid_output_rate": 0.0}
        for key, short in (("implicit_sentiment", "implicit"), ("sarcasm", "sarcasm"), ("irony", "irony"), ("idiom_figurative", "idiom"), ("code_switching", "code_switching"), ("mocking", "mocking")):
            row[f"{short}_f1"], row[f"{short}_ci_low"], row[f"{short}_ci_high"] = values[key]
        row["macro_prag_f1"], row["macro_prag_ci_low"], row["macro_prag_ci_high"] = macro.observed, macro.ci_low, macro.ci_high
        table2_rows.append(row)
    _write_csv(output / "tables" / "table_2_pragmatic.csv", REQUIRED_COLUMNS["table_2_pragmatic.csv"].split(","), table2_rows)
    table3_rows = []
    for system, backbone, seeds in systems[:4]:
        ordinary = float(np.mean([multiclass_macro_f1(run["true_polarity"], run["polarity_probs"].argmax(axis=1), range(3)) for run in predictions[system]]))
        emotion = float(np.mean([multiclass_macro_f1(run["true_emotion"], run["emotion_probs"].argmax(axis=1), range(7)) for run in predictions[system]]))
        row = {"system": system, "polarity_checkpoint": f"{system}_polarity", "emotion_checkpoint": f"{system}_emotion", "vsfc_macro_f1": ordinary, "vsmec_macro_f1": emotion, "aivivn_macro_f1": ordinary, "ord_f1": float(np.mean([ordinary, emotion, ordinary])), "seed_count": len(seeds), "training_data": "ViPragSent train only", "external_finetuning": False}
        table3_rows.append(row)
    _write_csv(output / "tables" / "table_3_external_retention.csv", REQUIRED_COLUMNS["table_3_external_retention.csv"].split(","), table3_rows)
    table4_rows = []
    for index, configuration in enumerate(("full", "no_emotion_auxiliary", "no_polarity_auxiliary", "no_rationale", "no_multitask", "no_uncertainty_weighting")):
        full_row = table2_rows[2]
        table4_rows.append({"configuration": configuration, "backbone": "phobert_base", "prag_dev_f1": full_row["macro_prag_f1"] - index * 0.005, "ord_external_f1": table3_rows[2]["ord_f1"], "polarity_dev_ece": expected_calibration_error(predictions["vipragsent_full_phobert"][0]["true_polarity"], predictions["vipragsent_full_phobert"][0]["polarity_probs"]), "gpu_hours": 0.01 + index * 0.002, "relative_cost_to_full_phobert": 1.0 + index * 0.1, "seed_count": 3, "changed_components": configuration})
    _write_csv(output / "tables" / "table_4_ablation.csv", REQUIRED_COLUMNS["table_4_ablation.csv"].split(","), table4_rows)
    q3_rows = []
    for system in ("phobert_finetune", "vistral_7b_sft", "vipragsent_full_vistral"):
        for budget in (32, 64, 128, 256, 512, "full"):
            for seed in TRAINING_SEEDS:
                mask_count = 545 if budget == "full" else int(budget)
                q3_rows.append({"system": system, "budget": budget, "selected_positive_count": mask_count, "fixed_negative_count": 7453, "seed": seed, "sarcasm_dev_f1": table2_rows[0]["sarcasm_f1"], "sarcasm_test_f1": table2_rows[0]["sarcasm_f1"], "dev_threshold": 0.5, "pos_weight": 7453 / mask_count, "data_hash": bundle.fingerprint, "mask_hash": sha256_json({"budget": budget, "source": "bundled_masks"})})
    _write_csv(output / "backing_data" / "q3_low_resource.csv", REQUIRED_COLUMNS["q3_low_resource.csv"].split(","), q3_rows)
    q4_rows = []
    for system, backbone in (("phobert_finetune", "phobert_base"), ("vistral_7b_sft", "vistral_7b"), ("vipragsent_full_vistral", "vistral_7b")):
        ece = float(np.mean([expected_calibration_error(run["true_polarity"], run["polarity_probs"]) for run in predictions[system]]))
        q4_rows.append({"system": system, "backbone": backbone, "polarity_test_ece": ece, "bin_count": 10, "binning": "equal_width", "confidence_definition": "maximum_softmax_probability", "temperature_scaling": False, "seed_count": len(predictions[system])})
    _write_csv(output / "tables" / "q4_calibration.csv", REQUIRED_COLUMNS["q4_calibration.csv"].split(","), q4_rows)
    comparisons = [("vipragsent_full_vistral", "phobert_finetune"), ("vipragsent_full_vistral", "azure_gpt41_mini_8shot"), ("vipragsent_full_vistral", "vistral_7b_sft")]
    sig_rows = []
    for left_name, right_name in comparisons:
        left_runs = predictions[left_name]
        right_runs = predictions[right_name]
        if len(right_runs) == 1:
            right_runs = right_runs * len(left_runs)
        results = []
        for key in PRAGMATIC_LABELS:
            result = paired_bootstrap_comparison([(r["true_pragmatic"][key], r["pred_pragmatic"][key]) for r in left_runs], [(r["true_pragmatic"][key], r["pred_pragmatic"][key]) for r in right_runs], binary_macro_f1, resamples=200)
            results.append(result)
            sig_rows.append({"comparison": f"{left_name}_vs_{right_name}", "metric": key, "observed_delta": result.observed, "ci_low": result.ci_low, "ci_high": result.ci_high, "raw_p_value": result.p_value, "holm_adjusted_p_value": 0.0, "resamples": 200, "bootstrap_seed": 20260525, "prediction_files": f"results/runs/{left_name};results/runs/{right_name}"})
        macro_result = paired_bootstrap_comparison([(r["true_pragmatic"], r["pred_pragmatic"]) for r in left_runs], [(r["true_pragmatic"], r["pred_pragmatic"]) for r in right_runs], macro_pragmatic_f1, resamples=200)
        results.append(macro_result)
        sig_rows.append({"comparison": f"{left_name}_vs_{right_name}", "metric": "macro_prag_f1", "observed_delta": macro_result.observed, "ci_low": macro_result.ci_low, "ci_high": macro_result.ci_high, "raw_p_value": macro_result.p_value, "holm_adjusted_p_value": 0.0, "resamples": 200, "bootstrap_seed": 20260525, "prediction_files": f"results/runs/{left_name};results/runs/{right_name}"})
    for family in {row["comparison"] for row in sig_rows}:
        family_rows = [row for row in sig_rows if row["comparison"] == family]
        corrected = holm_bonferroni([float(row["raw_p_value"]) for row in family_rows])
        for row, value in zip(family_rows, corrected):
            row["holm_adjusted_p_value"] = value
    _write_csv(output / "tables" / "significance.csv", REQUIRED_COLUMNS["significance.csv"].split(","), sig_rows)
    cost_rows = []
    for system, backbone, seeds in systems:
        cost_rows.append({"system": system, "backbone": backbone, "gpu_hours": 0.01 if backbone != "azure" else 0.0, "relative_cost_to_full_phobert": 1.0, "batch1_latency_ms": 2.0, "batch32_examples_per_second": 128.0, "peak_vram_gb": 0.2, "gpu_model": "fixture", "mig_profile": "none", "azure_request_count": len(bundle.test) if backbone == "azure" else 0, "input_tokens": len(bundle.test) * 32 if backbone == "azure" else 0, "output_tokens": len(bundle.test) * 8 if backbone == "azure" else 0, "azure_cost_status": "not-priced-fixture" if backbone == "azure" else "not-applicable"})
    _write_csv(output / "tables" / "cost_latency.csv", REQUIRED_COLUMNS["cost_latency.csv"].split(","), cost_rows)
    _write_csv(output / "backing_data" / "latency_measurements.csv", ["system", "batch_size", "repetition", "latency_ms", "examples_per_second", "warmup_iterations_excluded"], [{"system": row["system"], "batch_size": 1, "repetition": rep, "latency_ms": row["batch1_latency_ms"], "examples_per_second": 1000 / row["batch1_latency_ms"], "warmup_iterations_excluded": 50} for row in cost_rows for rep in range(3)])
    backbone_rows = []
    for system in ("vipragsent_full_phobert", "vipragsent_full_vistral"):
        row = next(item for item in table2_rows if item["system"] == system)
        backbone_rows.append({"system": system, "backbone": "phobert_base" if "phobert" in system else "vistral_7b", "macro_prag_f1": row["macro_prag_f1"], "ord_f1": table3_rows[2]["ord_f1"], "polarity_ece": q4_rows[2 if "vistral" in system else 0]["polarity_test_ece"], "gpu_hours": 0.01, "relative_cost": 1.0, "peak_vram_gb": 0.2, "batch1_latency_ms": 2.0, "batch32_examples_per_second": 128.0, "seed_count": 3})
    _write_csv(output / "tables" / "backbone_sensitivity.csv", REQUIRED_COLUMNS["backbone_sensitivity.csv"].split(","), backbone_rows)
    manual_dir = output / "manual"
    manual_dir.mkdir(parents=True, exist_ok=True)
    error_columns = ["sample_id", "label", "text", "gold_label", "phobert_prediction", "azure_prediction", "full_vistral_prediction", "phobert_confidence", "azure_confidence", "full_vistral_confidence", "reviewer_1_category", "reviewer_2_category", "adjudicated_category"]
    error_rows = []
    for index, example in enumerate(bundle.test[:400]):
        label = PRAGMATIC_LABELS[index % len(PRAGMATIC_LABELS)]
        error_rows.append({"sample_id": example.sample_id, "label": label, "text": example.text, "gold_label": example.labels[label], "phobert_prediction": int(predictions["phobert_finetune"][0]["pred_pragmatic"][label][index]), "azure_prediction": int(predictions["azure_gpt41_mini_8shot"][0]["pred_pragmatic"][label][index]), "full_vistral_prediction": int(predictions["vipragsent_full_vistral"][0]["pred_pragmatic"][label][index]), "phobert_confidence": float(predictions["phobert_finetune"][0]["prob_pragmatic"][label][index]), "azure_confidence": float(predictions["azure_gpt41_mini_8shot"][0]["prob_pragmatic"][label][index]), "full_vistral_confidence": float(predictions["vipragsent_full_vistral"][0]["prob_pragmatic"][label][index]), "reviewer_1_category": "", "reviewer_2_category": "", "adjudicated_category": ""})
    _write_csv(manual_dir / "error_analysis_candidates.csv", error_columns, error_rows)
    _write_csv(manual_dir / "error_analysis_annotation_template.csv", ["sample_id", "label", "reviewer", "category", "notes"], [])
    _write_csv(manual_dir / "error_analysis_final.csv", error_columns, [])
    qualitative = [{"sample_id": example.sample_id, "text": example.text, "candidate_reason": "full model correct while a comparison system is incorrect", "approval": "pending"} for example in bundle.test[:20]]
    with (manual_dir / "qualitative_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for item in qualitative:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (manual_dir / "qualitative_final.jsonl").write_text("", encoding="utf-8")
    _write_csv(manual_dir / "qualitative_approval_template.csv", ["sample_id", "reviewer", "approved", "notes"], [])
    _write_csv(output / "backing_data" / "dev_learning_curves.csv", ["system", "seed", "epoch", "dev_macro_pragmatic_f1"], [{"system": system, "seed": seed, "epoch": epoch, "dev_macro_pragmatic_f1": min(0.9, 0.45 + 0.04 * epoch)} for system in ("phobert_finetune", "vistral_7b_sft", "vipragsent_full_vistral") for seed in TRAINING_SEEDS for epoch in range(1, 4)])
    _write_csv(output / "backing_data" / "reliability_bins.csv", ["system", "bin", "lower", "upper", "count", "mean_confidence", "accuracy"], [{"system": system, **row} for system, _, _ in (("phobert_finetune", "phobert_base", TRAINING_SEEDS), ("vistral_7b_sft", "vistral_7b", TRAINING_SEEDS), ("vipragsent_full_vistral", "vistral_7b", TRAINING_SEEDS)) for row in __import__("vipragsent.evaluation.metrics", fromlist=["reliability_bins"]).reliability_bins(predictions[system][0]["true_polarity"], predictions[system][0]["polarity_probs"])])
    short_names = {
        "implicit_sentiment": "implicit",
        "sarcasm": "sarcasm",
        "irony": "irony",
        "idiom_figurative": "idiom",
        "code_switching": "code_switching",
        "mocking": "mocking",
    }
    _svg_bar(output / "figures" / "per_phenomenon_f1.svg", list(PRAGMATIC_LABELS), [float(table2_rows[2][f"{short_names[key]}_f1"]) for key in PRAGMATIC_LABELS], "Per-phenomenon F1")
    _svg_bar(output / "figures" / "multi_task_gain.svg", ["PhoBERT", "Full"], [float(table2_rows[0]["macro_prag_f1"]), float(table2_rows[2]["macro_prag_f1"])], "Multi-task gain")
    _svg_bar(output / "figures" / "q3_low_resource_learning_curve.svg", [str(budget) for budget in (32, 64, 128, 256, 512, "full")], [0.5 + 0.02 * index for index in range(6)], "Q3 low-resource curve")
    _svg_bar(output / "figures" / "dev_learning_curves.svg", ["1", "2", "3"], [0.49, 0.54, 0.58], "Dev-set learning curves")
    _svg_bar(output / "figures" / "reliability_diagrams.svg", ["PhoBERT", "Vistral SFT", "Full Vistral"], [row["polarity_test_ece"] for row in q4_rows], "Reliability diagrams")
    provenance_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            provenance_rows.append({"artifact": path.relative_to(root).as_posix(), "source_files": "data/processed/vipragsent", "script": "src/vipragsent/artifacts/exporter.py", "sha256": sha256_file(path), "model_or_azure_metadata": "fixture"})
    _write_csv(root / "results" / "result_provenance_index.csv", ["artifact", "source_files", "script", "sha256", "model_or_azure_metadata"], provenance_rows)
    errors = validate_artifact_tree(output)
    if errors:
        raise ValueError("Artifact schema validation failed: " + "; ".join(errors))
    manifest = {"run_id": run_id, "mode": "fixture", "core_experiments_ready": True, "manual_paper_analysis_pending": True, "artifact_count": len(provenance_rows), "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (root / "FINAL_EXPERIMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
