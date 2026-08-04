from __future__ import annotations

import torch

from vipragsent.models.backbones import DummyBackbone, pool_hidden_states
from vipragsent.models.generation import parse_cot_generation
from vipragsent.models.losses import UncertaintyWeightedMultiTaskLoss
from vipragsent.models.variants import IndependentCheckpointBundle, VariantConfig, build_dummy_model


def test_pooling_rules_use_first_token_for_encoder_and_masked_mean_for_causal() -> None:
    hidden = torch.tensor([[[1.0], [2.0], [9.0]], [[3.0], [5.0], [7.0]]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
    assert torch.equal(pool_hidden_states(hidden, mask, "encoder").squeeze(-1), torch.tensor([2.0, 3.0]))
    assert torch.equal(pool_hidden_states(hidden, mask, "causal").squeeze(-1), torch.tensor([5.5, 4.0]))


def test_full_model_forward_shapes_and_decoder_is_disabled_at_inference() -> None:
    model = build_dummy_model(VariantConfig(name="vipragsent_full_vistral", backbone_family="causal", rationale_enabled_for_training=True))
    input_ids = torch.randint(3, 100, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    rationale_input_ids = torch.tensor([[1, 11, 12, 13, 14, 2], [1, 21, 22, 23, 24, 2]])
    rationale_attention_mask = torch.ones_like(rationale_input_ids)
    result = model(
        input_ids,
        attention_mask,
        rationale_input_ids=rationale_input_ids,
        rationale_attention_mask=rationale_attention_mask,
    )
    assert set(result["logits"]) == {"implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking", "polarity", "emotion"}
    assert result["rationale_logits"].shape == (2, 5, 512)
    assert result["rationale_labels"].shape == (2, 5)
    assert torch.all(result["rationale_labels"][:, -1] == 2)
    assert model.inference_output_source == "classification_heads"
    assert model.rationale_decoder_enabled_at_inference is False


def test_rationale_decoder_is_causal_and_future_tokens_do_not_change_prior_logits() -> None:
    torch.manual_seed(7)
    model = build_dummy_model(VariantConfig(name="vipragsent_full_vistral", backbone_family="causal", rationale_enabled_for_training=True))
    model.eval()
    input_ids = torch.randint(3, 100, (1, 8))
    attention_mask = torch.ones_like(input_ids)
    first_target = torch.tensor([[1, 31, 32, 33, 34, 2]])
    changed_target = torch.tensor([[1, 31, 32, 93, 94, 2]])
    first = model(input_ids, attention_mask, rationale_input_ids=first_target)
    changed = model(input_ids, attention_mask, rationale_input_ids=changed_target)
    assert torch.allclose(first["rationale_logits"][:, :2], changed["rationale_logits"][:, :2])


def test_uncertainty_loss_has_eight_parameters_and_finite_gradient() -> None:
    aggregator = UncertaintyWeightedMultiTaskLoss()
    losses = {name: torch.tensor(0.5, requires_grad=True) for name in aggregator.log_variances}
    total, _ = aggregator(losses)
    total.backward()
    assert len(aggregator.log_variances) == 8
    assert torch.isfinite(total)


def test_uncertainty_formula_clamps_and_excludes_inactive_tasks() -> None:
    aggregator = UncertaintyWeightedMultiTaskLoss()
    aggregator.log_variances["sarcasm"].data.fill_(9.0)
    loss = torch.tensor(2.0, requires_grad=True)
    total, components = aggregator({"sarcasm": loss})
    expected = 0.5 * torch.exp(torch.tensor(-5.0)) * loss + 0.5 * torch.tensor(5.0)
    assert torch.allclose(components["sarcasm"], expected)
    assert set(components) == {"sarcasm"}
    total.backward()
    assert torch.isfinite(loss.grad)


def test_no_multitask_is_eight_independent_single_task_components() -> None:
    config = VariantConfig(name="no_multitask", backbone_family="encoder", hidden_size=16, vocab_size=64)
    bundle = IndependentCheckpointBundle(lambda: DummyBackbone(64, 16), config)
    assert len(bundle.components) == 8
    assert bundle.checkpoint_ids[-2:] == ("phobert_pol_single", "phobert_emo_single")
    assert all(len(component.active_head_keys) == 1 for component in bundle.components.values())


def test_cot_parser_accepts_only_strict_canonical_labels() -> None:
    parsed = parse_cot_generation('<RATIONALE>cues</RATIONALE><LABELS>{"implicit_sentiment":0,"sarcasm":0,"irony":0,"idiom_figurative":0,"code_switching":0,"mocking":0,"polarity":"neutral","emotion":"other",}</LABELS>')
    assert parsed.repaired_punctuation is True
    assert parsed.labels["polarity"] == "neutral"
