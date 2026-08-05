from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vipragsent.constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from vipragsent.data.collation import BatchCollator
from vipragsent.data.loaders import DatasetExample
from vipragsent.data.preprocessing import PreprocessingSpec, TextPreprocessor
from vipragsent.evaluation.external_retention import (
    NormalizedExternalExample,
    evaluate_external_retention,
)
from vipragsent.models.variants import VariantConfig, build_dummy_model
from vipragsent.orchestration.aggregation import _q3_rows, _q4_summary, _table4
from vipragsent.orchestration.contracts import RunContext, RunEntry
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.review import validate_review_summary
from vipragsent.orchestration.run_store import RunStore
from vipragsent.orchestration.stage_registry import (
    _review_summary,
    build_single_experiment_stage_registry,
)
from vipragsent.orchestration.system_registry import (
    load_execution_registry,
    resolve_execution_spec,
    validate_execution_registry,
)
from vipragsent.runtime.batch_probe import probe_physical_batch
from vipragsent.training.class_weights import (
    compute_train_only_class_weights,
    synthetic_class_weights,
)
from vipragsent.training.config_resolver import resolve_training_config
from vipragsent.training.optimizers import build_optimizer
from vipragsent.training.schedulers import build_scheduler


def _entry(root: Path, run_id: str) -> RunEntry:
    row = next(item for item in build_expected_runs(root)["rows"] if item["run_id"] == run_id)
    return RunEntry.from_mapping(row, run_id=run_id)


def _labels(index: int = 0) -> dict[str, int | str]:
    return {
        **{label: (index + offset) % 2 for offset, label in enumerate(PRAGMATIC_LABELS)},
        "polarity": POLARITY_LABELS[index % len(POLARITY_LABELS)],
        "emotion": EMOTION_LABELS[index % len(EMOTION_LABELS)],
    }


def _example(sample_id: str, index: int = 0) -> DatasetExample:
    return DatasetExample(sample_id, f"fixture text {index}", _labels(index), "train")


def test_execution_registry_is_exact_and_unknown_systems_block_before_model_build() -> None:
    root = Path(".").resolve()
    report = validate_execution_registry(root)
    assert report["status"] == "PASS"
    specs = load_execution_registry(root)
    inventory_ids = {row["system_id"] for row in build_expected_runs(root)["rows"]}
    assert set(specs) == inventory_ids
    with pytest.raises(ValueError, match="BLOCKED before model construction"):
        resolve_execution_spec(root, "full_phobert_substring_variant")


def test_training_resolver_golden_values_cover_encoder_7b_and_no_uncertainty() -> None:
    root = Path(".").resolve()
    cases = (
        ("q1a_phobert_pragmatic_finetune_20260521", 32, "AdamW", 2e-5, "linear", False, False),
        ("q1a_vipragsent_full_vistral_20260521", 2, "paged_adamw_8bit", 1e-4, "cosine", True, True),
        ("q2_no_uncertainty_weighting_20260521", 32, "AdamW", 2e-5, "linear", False, True),
    )
    for run_id, physical, optimizer, learning_rate, scheduler, uncertainty, rationale in cases:
        entry = _entry(root, run_id)
        spec = resolve_execution_spec(root, entry.system_id)
        resolved = resolve_training_config(entry, spec, root=root, runtime_status={"successful_batch": physical})
        assert (resolved.optimizer, resolved.learning_rate, resolved.scheduler) == (optimizer, learning_rate, scheduler)
        assert resolved.effective_batch_size == (16 if "vistral" in entry.system_id else 32)
        assert resolved.gradient_accumulation_steps == resolved.effective_batch_size // physical
        assert resolved.uncertainty_weighting_enabled is uncertainty
        assert resolved.rationale_training is rationale
        assert resolved.rationale_inference is False
        if "vistral" in entry.system_id:
            assert resolved.qlora["quantization"] == {"load_in_4bit": True, "quant_type": "nf4", "double_quant": True, "compute_dtype": "bf16"}
            assert resolved.qlora["lora"]["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_class_weights_use_only_train_split_and_persist_content_hash() -> None:
    rows = [_example(f"row-{index}", index) for index in range(8)]
    weights = compute_train_only_class_weights(rows, dataset_hash="data", code_commit="commit")
    assert weights.source_split == "train"
    assert weights.counts["pragmatic"]["sarcasm"]["positive"] == 4
    assert weights.pragmatic_pos_weight["sarcasm"] == pytest.approx(1.0)
    assert set(weights.as_dict()) >= {"counts", "content_hash", "dataset_hash", "code_commit"}
    with pytest.raises(ValueError, match="only be computed"):
        compute_train_only_class_weights([DatasetExample("dev", "text", _labels(), "dev")], dataset_hash="data", code_commit="commit")


def test_optimizer_and_scheduler_keep_uncertainty_out_of_decay() -> None:
    model = build_dummy_model(VariantConfig(name="vipragsent_full", hidden_size=8, vocab_size=32))
    aggregator = torch.nn.Parameter(torch.tensor(0.0))
    optimizer, summary = build_optimizer(model, optimizer_name="AdamW", learning_rate=2e-5, weight_decay=0.01, uncertainty_parameters=[aggregator])
    assert summary["optimizer"] == "AdamW"
    assert any(group["name"] == "uncertainty_no_decay" for group in summary["groups"])
    scheduler, scheduler_summary = build_scheduler(optimizer, scheduler_name="linear", warmup_ratio=0.1, total_steps=10)
    assert scheduler_summary["total_optimizer_steps"] == 10
    assert scheduler is not None


def test_collator_emits_rationale_and_q3_masks_without_masking_other_tasks() -> None:
    rows = [_example("positive", 0), _example("negative", 1)]
    tokenizer = __import__("vipragsent.data.preprocessing", fromlist=["DummyTokenizer"]).DummyTokenizer()
    preprocessor = TextPreprocessor(PreprocessingSpec("fixture", "unicode_nfc", "fixture-v1", execution_mode="fixture"))
    masks = {
        "32": {
            "positive": {"sample_id": "positive", "is_sarcasm_positive": "1", "positive_selected_for_budget": "1", "sarcasm_target_mask": "1", "rationale_loss_mask": "1"},
            "negative": {"sample_id": "negative", "is_sarcasm_positive": "0", "positive_selected_for_budget": "0", "sarcasm_target_mask": "1", "rationale_loss_mask": "0"},
        }
    }
    collator = BatchCollator(tokenizer, preprocessor, q3_masks=masks, budget="32", mask_hash="mask", class_weights=synthetic_class_weights().as_dict(), rationale_records={"positive": {"rationale": "cue"}})
    batch = collator(rows)
    assert torch.equal(batch["target_masks"]["sarcasm"], torch.tensor([1.0, 1.0]))
    assert batch["rationale_loss_mask"].tolist() == [1.0, 0.0]
    assert batch["budget_pos_weight"] == pytest.approx(1.0)
    assert batch["polarity_weight"].shape == (len(POLARITY_LABELS),)
    assert batch["emotion_weight"].shape == (len(EMOTION_LABELS),)


def test_external_retention_is_test_only_and_writes_required_outputs(tmp_path: Path) -> None:
    datasets = {
        "vsfc": [NormalizedExternalExample("v0", "text", "positive"), NormalizedExternalExample("v1", "text", "negative"), NormalizedExternalExample("v2", "text", "neutral")],
        "vsmec": [NormalizedExternalExample("m0", "text", "anger"), NormalizedExternalExample("m1", "text", "sadness"), NormalizedExternalExample("m2", "text", "other")],
        "aivivn": [NormalizedExternalExample("a0", "text", "positive"), NormalizedExternalExample("a1", "text", "negative"), NormalizedExternalExample("a2", "text", "neutral")],
    }
    predictions = {key: {row.sample_id: row.label for row in rows} for key, rows in datasets.items()}
    result = evaluate_external_retention(datasets, predictions, source_checkpoint_id="ckpt", source_seed=20260521, external_manifest_hash="manifest", output_root=tmp_path)
    assert result["external_finetuning"] is False
    assert result["optimizer_steps"] == 0
    assert (tmp_path / "metrics/external_retention_metrics.json").exists()
    assert (tmp_path / "predictions/uit_vsfc_test_predictions.jsonl").exists()


def test_variant_isolation_and_generation_dispatch() -> None:
    full = build_dummy_model(VariantConfig(name="vipragsent_full", rationale_enabled_for_training=True, hidden_size=8, vocab_size=32))
    no_rationale = build_dummy_model(VariantConfig(name="no_rationale", hidden_size=8, vocab_size=32))
    no_uncertainty = build_dummy_model(VariantConfig(name="no_uncertainty_weighting", hidden_size=8, vocab_size=32))
    generation = build_dummy_model(VariantConfig(name="cot_only_vistral", backbone_family="causal", hidden_size=8, vocab_size=32))
    assert full.rationale_decoder is not None
    assert no_rationale.rationale_decoder is None
    assert no_uncertainty.rationale_decoder is not None
    assert no_uncertainty.config.has_uncertainty_weighting is False
    assert generation.inference_output_source == "parsed_generated_labels"
    assert generation.config.active_tasks == set()


def test_batch_probe_requires_exact_divisibility_and_records_oom_evidence(tmp_path: Path) -> None:
    def probe(batch: int) -> bool:
        if batch == 8:
            raise RuntimeError("CUDA out of memory")
        return batch in {4, 2}

    result = probe_physical_batch(tmp_path, "synthetic", probe=probe, candidate_order=[8, 4, 2], effective_batch_size=8, hardware_identity="synthetic")
    assert result["status"] == "PASS"
    assert result["successful_batch"] == 4
    assert result["gradient_accumulation_steps"] == 2
    assert result["oom_evidence"][0]["batch"] == 8


def test_aggregation_refuses_missing_values_and_uses_external_ord(tmp_path: Path) -> None:
    full_root = tmp_path / "full"
    variant_root = tmp_path / "variant"
    for run_root, ord_value in ((full_root, 0.5), (variant_root, 0.8)):
        (run_root / "metrics").mkdir(parents=True)
        (run_root / "metrics/external_retention_metrics.json").write_text(json.dumps({"ord_f1": ord_value}), encoding="utf-8")
    base = {"best_dev_metric": 0.7, "polarity_dev_ece": 0.2, "successful_gpu_hours": 2.0, "changed_components": {"optimizer": "same"}, "backbone": "phobert_base", "seed": 20260521}
    full = {"run_id": "full", "run_root": str(full_root), "summary": {"variant": "full", **base}}
    variant = {"run_id": "variant", "run_root": str(variant_root), "summary": {"variant": "variant", **(base | {"successful_gpu_hours": 1.0})}}
    rows = _table4([full, variant])
    assert rows[1]["ord_external_f1"] == pytest.approx(0.8)
    assert rows[1]["relative_cost_to_full_phobert"] == pytest.approx(0.5)
    (variant_root / "metrics/external_retention_metrics.json").unlink()
    with pytest.raises(ValueError, match="external retention"):
        _table4([full, variant])


def test_q3_and_q4_aggregations_reject_incomplete_cartesian_inputs() -> None:
    q3_record = {"run_id": "q3", "summary": {"system_id": "phobert_pragmatic_finetune", "budget": "32", "seed": 20260521, "selected_positive_count": 32, "fixed_negative_count": 7453, "sarcasm_dev_f1": 0.5, "frozen_thresholds": {"sarcasm": 0.5}, "budget_pos_weight": 7453 / 32, "q3_mask_hash": "mask", "dataset_fingerprint": "data", "per_label_test_metrics": {"sarcasm_f1": 0.4}}}
    assert _q3_rows([q3_record])[0]["selected_positive_count"] == 32
    with pytest.raises(ValueError, match="exactly three"):
        _q4_summary([{"system": "full", "label": "sarcasm", "display_name": "Full", "seed": 20260521, "ece": 0.1, "macro_pragmatic_ece": 0.2}])


@pytest.mark.integration_slow
def test_synthetic_full_sequential_run_is_review_gated_and_hash_valid(tmp_path: Path) -> None:
    root = Path(".").resolve()
    entry = _entry(root, "q1a_vipragsent_full_vistral_20260521")
    context = RunContext(root, entry, fixture=True, run_root=tmp_path / "run")
    store = RunStore(context)
    state = store.initialize()
    handlers = build_single_experiment_stage_registry(root, entry, context)
    for name in entry.stages[:-1]:
        store.start_stage(state, name)
        outcome = handlers[name]()
        payload = outcome.as_dict()
        assert payload["status"] == "PASS", payload
        assert all((context.run_root / path).exists() for path in payload["expected_files"])
        store.write_checksums()
        store.complete_stage(state, name, payload)
        state = store.load()
    store.start_stage(state, "generate_review_summary")
    outcome = _review_summary(context, entry, state)
    assert outcome.status == "PASS", outcome
    store.complete_stage(state, "generate_review_summary", outcome.as_dict())
    store.write_checksums()
    summary = json.loads((context.run_root / "review_summary.json").read_text(encoding="utf-8"))
    assert validate_review_summary(summary, completed=True) == []
    assert summary["USER_REVIEW_STATUS"] == "PENDING"
    assert store.validate_checksums() == []
