from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vipragsent.constants import PRAGMATIC_LABELS, TRAINING_SEEDS
from vipragsent.evaluation.production import evaluate_q4_seed, evaluate_q4_seeds
from vipragsent.models.variants import VariantConfig, build_dummy_model
from vipragsent.orchestration.sequential import build_azure_job_inventory, load_execution_policy
from vipragsent.protocol import validate_protocol_resolution
from vipragsent.statistics.bootstrap import APPROVED_P_VALUE_METHOD, paired_bootstrap_comparison
from vipragsent.training.engine import TrainingConfig, TrainingEngine


def test_approved_protocol_is_resolved_and_sequential_policy_is_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    resolution = validate_protocol_resolution(root)
    assert resolution["passed"] is True
    assert not resolution["scientific_protocol_conflicts"]
    policy = load_execution_policy(root)
    assert policy["execution_policy"] == "sequential_review_gated"
    assert policy["global_full_dag_enabled"] is False
    assert policy["automatic_next_run"] is False


def test_no_auxiliary_vistral_has_a_distinct_six_task_fingerprint(tmp_path: Path) -> None:
    model = build_dummy_model(VariantConfig(name="vipragsent_no_auxiliary_vistral", backbone_family="causal", hidden_size=12, vocab_size=32))
    assert model.config.loss_aggregation == "homoscedastic_uncertainty"
    assert model.config.uncertainty_task_keys == PRAGMATIC_LABELS
    assert model.heads.polarity is None
    assert model.heads.emotion is None
    assert not any(name.startswith("heads.polarity") or name.startswith("heads.emotion") for name, _ in model.named_parameters())
    engine = TrainingEngine(model, TrainingConfig(max_epochs=1, precision="fp32"), run_id="no-aux", checkpoint_root=tmp_path / "checkpoints")
    assert set(engine.loss_aggregator.log_variances) == set(PRAGMATIC_LABELS)
    group_names = {group["name"] for group in engine.optimizer.param_groups}
    assert "uncertainty_no_decay" in group_names
    assert not any("polarity" in name or "emotion" in name for name, _ in model.named_parameters())


def test_q4_uses_six_raw_positive_probability_ece_values() -> None:
    true = {label: [0, 1, 0, 1] for label in PRAGMATIC_LABELS}
    probabilities = {label: [0.05, 0.95, 0.25, 0.75] for label in PRAGMATIC_LABELS}
    per_seed = [evaluate_q4_seed(probabilities, true, seed=seed) for seed in TRAINING_SEEDS]
    summary = evaluate_q4_seeds(per_seed)
    assert len(per_seed[0]["reliability_bins"][PRAGMATIC_LABELS[0]]) == 10
    assert per_seed[0]["reliability_bins"][PRAGMATIC_LABELS[0]][1]["count"] == 0
    assert summary["temperature_scaling"] is False
    assert summary["probability_aggregation"] == "none"
    assert set(summary["per_label"]) == set(PRAGMATIC_LABELS)
    assert summary["std_macro_pragmatic_ece"] == 0.0


def test_significance_uses_plus_one_and_rejects_legacy_mid_p() -> None:
    true = [0, 1, 0, 1]
    left = [(true, [0, 1, 0, 1])] * 3
    right = [(true, [1, 1, 0, 1])] * 3
    result = paired_bootstrap_comparison(left, right, lambda y, p: float(sum(a == b for a, b in zip(y, p, strict=True))), resamples=20, seed=7, p_value_method=APPROVED_P_VALUE_METHOD)
    assert result.p_value is not None
    with pytest.raises(ValueError, match="prohibited"):
        paired_bootstrap_comparison(left, right, lambda y, p: 0.0, resamples=2, p_value_method="mid_p_two_sided")


def test_azure_job_inventory_has_only_single_job_types() -> None:
    jobs = build_azure_job_inventory()
    assert len(jobs) == 11
    assert {job["job_type"] for job in jobs} <= {
        "rationale_generation", "pragmatic_zero_shot", "pragmatic_8_shot", "polarity_dedicated_prompt", "emotion_dedicated_prompt", "q3_budget_specific_pragmatic_8_shot"
    }
