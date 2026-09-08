from __future__ import annotations

import unicodedata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_full_xlmr_large_experiment import _build_production_preprocessor, _loss_multipliers, _preprocessing_kwargs


def test_xlmr_runner_uses_locked_unicode_preprocessing() -> None:
    preprocessor = _build_production_preprocessor(
        "xlmr_large",
        tokenizer_revision="tokenizer-revision",
        model_revision="model-revision",
        **_preprocessing_kwargs(),
    )

    assert preprocessor.spec.preprocessing_name == "unicode_nfc"
    assert preprocessor.spec.preprocessing_version == "locked-v1"
    text = "a\u0301"
    assert preprocessor.prepare_text(text) == unicodedata.normalize("NFC", text)


def test_xlmr_runner_preserves_global_profile_and_applies_head_override() -> None:
    args = type(
        "Args",
        (),
        {
            "pragmatic_loss_multiplier": 1.1,
            "auxiliary_loss_multiplier": 1.0,
            "sarcasm_loss_multiplier": 1.3,
            "mocking_loss_multiplier": 1.2,
        },
    )()

    multipliers = _loss_multipliers(args)

    assert multipliers["implicit_sentiment"] == 1.1
    assert multipliers["sarcasm"] == 1.3
    assert multipliers["mocking"] == 1.2
    assert multipliers["polarity"] == 1.0
    assert multipliers["emotion"] == 1.0
