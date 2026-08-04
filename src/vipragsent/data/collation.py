from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from .loaders import DatasetExample
from .preprocessing import DummyTokenizer, TextPreprocessor


@dataclass
class BatchCollator:
    tokenizer: DummyTokenizer
    preprocessor: TextPreprocessor

    def __call__(self, examples: Iterable[DatasetExample]) -> dict[str, object]:
        rows = list(examples)
        encoded = self.tokenizer.batch_encode([self.preprocessor.prepare_text(row.text) for row in rows])
        batch = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "sample_ids": [row.sample_id for row in rows],
            "targets": {
                **{key: torch.tensor([row.labels[key] for row in rows], dtype=torch.float32) for key in PRAGMATIC_LABELS},
                "polarity": torch.tensor([POLARITY_LABELS.index(row.labels["polarity"]) for row in rows], dtype=torch.long),
                "emotion": torch.tensor([EMOTION_LABELS.index(row.labels["emotion"]) for row in rows], dtype=torch.long),
            },
        }
        return batch
