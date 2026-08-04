from __future__ import annotations

import torch

from vipragsent.models.backbones import DummyBackbone, pool_hidden_states
from vipragsent.models.generation import parse_cot_generation
from vipragsent.models.losses import UncertaintyWeightedMultiTaskLoss
from vipragsent.models.variants import VariantConfig, build_dummy_model


def test_pooling_rules_use_first_token_for_encoder_and_masked_mean_for_causal() -> None:
    hidden = torch.tensor([[[1.0], [2.0], [9.0]], [[3.0], [5.0], [7.0]]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
    assert torch.equal(pool_hidden_states(hidden, mask, "encoder").squeeze(-1), torch.tensor([2.0, 3.0]))
    assert torch.equal(pool_hidden_states(hidden, mask, "causal").squeeze(-1), torch.tensor([5.5, 4.0]))


def test_full_model_forward_shapes_and_decoder_is_disabled_at_inference() -> None:
    model = build_dummy_model(VariantConfig(name="vipragsent_full_vistral", backbone_family="causal", rationale_enabled_for_training=True))
    input_ids = torch.randint(3, 100, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    result = model(input_ids, attention_mask, rationale_input_ids=torch.randint(3, 100, (2, 6)))
    assert set(result["logits"]) == {"implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking", "polarity", "emotion"}
    assert result["rationale_logits"].shape == (2, 6, 512)
    assert model.inference_output_source == "classification_heads"
    assert model.rationale_decoder_enabled_at_inference is False


def test_uncertainty_loss_has_eight_parameters_and_finite_gradient() -> None:
    aggregator = UncertaintyWeightedMultiTaskLoss()
    losses = {name: torch.tensor(0.5, requires_grad=True) for name in aggregator.log_variances}
    total, _ = aggregator(losses)
    total.backward()
    assert len(aggregator.log_variances) == 8
    assert torch.isfinite(total)


def test_cot_parser_accepts_only_strict_canonical_labels() -> None:
    parsed = parse_cot_generation('<RATIONALE>cues</RATIONALE><LABELS>{"implicit_sentiment":0,"sarcasm":0,"irony":0,"idiom_figurative":0,"code_switching":0,"mocking":0,"polarity":"neutral","emotion":"other",}</LABELS>')
    assert parsed.repaired_punctuation is True
    assert parsed.labels["polarity"] == "neutral"
