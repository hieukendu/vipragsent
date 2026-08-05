from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..orchestration.provenance import validate_inference_provenance

REQUIRED_COLUMNS = {
    "table_2_pragmatic.csv": "system,backbone,seed_count,implicit_f1,implicit_ci_low,implicit_ci_high,sarcasm_f1,sarcasm_ci_low,sarcasm_ci_high,irony_f1,irony_ci_low,irony_ci_high,idiom_f1,idiom_ci_low,idiom_ci_high,code_switching_f1,code_switching_ci_low,code_switching_ci_high,mocking_f1,mocking_ci_low,mocking_ci_high,macro_prag_f1,macro_prag_ci_low,macro_prag_ci_high,invalid_output_rate",
    "table_3_external_retention.csv": "system,polarity_checkpoint,emotion_checkpoint,vsfc_macro_f1,vsmec_macro_f1,aivivn_macro_f1,ord_f1,seed_count,training_data,external_finetuning",
    "table_4_ablation.csv": "configuration,backbone,prag_dev_f1,ord_external_f1,polarity_dev_ece,gpu_hours,relative_cost_to_full_phobert,seed_count,changed_components",
    "q3_low_resource.csv": "system,budget,selected_positive_count,fixed_negative_count,seed,sarcasm_dev_f1,sarcasm_test_f1,dev_threshold,pos_weight,data_hash,mask_hash",
    "q4_pragmatic_calibration_per_seed.csv": "system,display_name,checkpoint_id,seed,split,label,ece,macro_pragmatic_ece,bin_count,temperature_scaling,prediction_file,prediction_file_sha256,config_hash,code_commit",
    "q4_pragmatic_calibration_summary.csv": "system,display_name,label,mean_ece,std_ece,mean_macro_pragmatic_ece,std_macro_pragmatic_ece,seed_count,split,bin_count,temperature_scaling",
    "significance.csv": "comparison,metric,observed_delta,ci_low,ci_high,raw_p_value,holm_adjusted_p_value,resamples,bootstrap_seed,prediction_files",
    "cost_latency.csv": "system,backbone,gpu_hours,relative_cost_to_full_phobert,batch1_latency_ms,batch32_examples_per_second,peak_vram_gb,gpu_model,mig_profile,azure_request_count,input_tokens,output_tokens,azure_cost_status",
    "backbone_sensitivity.csv": "system,backbone,macro_prag_f1,ord_f1,polarity_ece,gpu_hours,relative_cost,peak_vram_gb,batch1_latency_ms,batch32_examples_per_second,seed_count",
    "table_1_dataset_summary.csv": "dataset,role,train_count,dev_count,test_count,total_count,task,label_space,source_manifest,checksum,redistribution_status",
    "vipragsent_label_distribution.csv": "split,label_group,label,count,total,rate",
    "human_iaa_summary.csv": "field,n,raw_agreement,cohen_kappa,krippendorff_alpha_nominal,disagreement_count",
    "split_and_label_counts.csv": "split,label_group,label,count,total,rate",
    "q4_pragmatic_reliability_bins.csv": "system,seed,label,bin_index,bin_lower,bin_upper,count,mean_confidence,empirical_positive_rate,absolute_gap",
    "q4_learning_curves.csv": "system,seed,epoch,dev_macro_pragmatic_f1,dev_loss,wall_seconds",
}

REQUIRED_PRODUCTION_FIGURES = (
    "q4_pragmatic_ece_heatmap.pdf",
    "q4_pragmatic_ece_heatmap.png",
    "q4_pragmatic_reliability_by_label.pdf",
    "q4_pragmatic_reliability_by_label.png",
    "q4_learning_curves.pdf",
    "q4_learning_curves.png",
)


def _expected(filename: str) -> list[str]:
    return REQUIRED_COLUMNS[filename].split(",")


def validate_artifact_tree(root: str | Path) -> list[str]:
    root = Path(root)
    errors: list[str] = []
    for filename, columns in REQUIRED_COLUMNS.items():
        directory = "backing_data" if filename in {"q3_low_resource.csv", "split_and_label_counts.csv", "q4_pragmatic_reliability_bins.csv", "q4_learning_curves.csv"} else "tables"
        path = root / directory / filename
        if not path.exists():
            errors.append(str(path))
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            actual = next(csv.reader(handle), [])
        if actual != _expected(filename):
            errors.append(f"{path}: expected columns {_expected(filename)}, got {actual}")
    return errors


def validate_production_artifact(root: str | Path, run_records: list[dict[str, Any]]) -> list[str]:
    """Validate a release-shaped tree without accepting fixture provenance."""
    errors = validate_artifact_tree(root)
    errors.extend(str(Path(root) / "figures" / filename) for filename in REQUIRED_PRODUCTION_FIGURES if not (Path(root) / "figures" / filename).exists())
    if not run_records:
        errors.append("production export has no real run records")
    for record in run_records:
        if record.get("mode") != "full":
            errors.append(f"run {record.get('system')} is not mode=full")
        if record.get("synthetic_results") is True:
            errors.append(f"run {record.get('system')} contains synthetic results")
        if record.get("model_revision") == "fixture" or record.get("tokenizer_revision") == "fixture":
            errors.append(f"run {record.get('system')} uses fixture revisions")
        errors.extend(validate_inference_provenance(record, source=f"run {record.get('system')}", allow_fixture_parser=False))
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("mode") == "fixture" or payload.get("synthetic_results") is True:
                errors.append(f"fixture provenance found in production artifact: {path}")
        if path.suffix in {".csv", ".jsonl"} and "fixture" in path.read_text(encoding="utf-8", errors="ignore").casefold():
            errors.append(f"fixture marker found in production artifact: {path}")
    return errors
