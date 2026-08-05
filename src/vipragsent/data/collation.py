from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from ..orchestration.status import RuntimeBlocked
from .loaders import DatasetExample
from .preprocessing import DummyTokenizer, TextPreprocessor


@dataclass
class BatchCollator:
    tokenizer: Any
    preprocessor: TextPreprocessor
    q3_masks: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None
    budget: str | None = None
    mask_hash: str | None = None

    def __call__(self, examples: Iterable[DatasetExample]) -> dict[str, object]:
        rows = list(examples)
        if isinstance(self.tokenizer, DummyTokenizer) and self.preprocessor.spec.execution_mode != "fixture":
            raise RuntimeBlocked("DummyTokenizer is available only in fixture mode")
        texts = [self.preprocessor.prepare_text(row.text) for row in rows]
        if hasattr(self.tokenizer, "batch_encode"):
            encoded = self.tokenizer.batch_encode(texts, max_length=self.preprocessor.spec.max_length)
        else:
            encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=self.preprocessor.spec.max_length, return_tensors="pt")
            encoded = {key: value.tolist() for key, value in encoded.items()}
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
        if self.budget is not None:
            if self.q3_masks is None or self.budget not in self.q3_masks:
                raise ValueError(f"Q3 mask for budget {self.budget!r} is unavailable")
            rows_by_id = self.q3_masks[self.budget]
            try:
                selected = [rows_by_id[row.sample_id] for row in rows]
            except KeyError as exc:
                raise ValueError(f"Q3 mask is missing sample ID {exc.args[0]}") from exc
            sarcasm_mask = torch.tensor([int(item["sarcasm_target_mask"]) for item in selected], dtype=torch.float32)
            rationale_mask = torch.tensor([int(item["rationale_loss_mask"]) for item in selected], dtype=torch.float32)
            selected_positive_count = sum(int(item["positive_selected_for_budget"]) for item in rows_by_id.values())
            fixed_negative_count = sum(1 for item in rows_by_id.values() if int(item["is_sarcasm_positive"]) == 0)
            if selected_positive_count <= 0:
                raise ValueError(f"Q3 budget {self.budget} has no selected positives")
            batch["budget"] = self.budget
            batch["sarcasm_target_mask"] = sarcasm_mask
            batch["rationale_loss_mask"] = rationale_mask
            batch["selected_positive_count"] = selected_positive_count
            batch["fixed_negative_count"] = fixed_negative_count
            batch["budget_pos_weight"] = fixed_negative_count / selected_positive_count
            batch["mask_hash"] = self.mask_hash
        return batch
