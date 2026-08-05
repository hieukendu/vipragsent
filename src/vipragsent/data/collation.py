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
    class_weights: Mapping[str, Any] | None = None
    rationale_records: Mapping[str, Any] | None = None
    rationale_target_max_length: int = 160

    def _encode(self, texts: list[str], *, max_length: int) -> dict[str, list[list[int]]]:
        if hasattr(self.tokenizer, "batch_encode"):
            return self.tokenizer.batch_encode(texts, max_length=max_length)
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return {key: value.tolist() for key, value in encoded.items()}

    def __call__(self, examples: Iterable[DatasetExample]) -> dict[str, object]:
        rows = list(examples)
        if isinstance(self.tokenizer, DummyTokenizer) and self.preprocessor.spec.execution_mode != "fixture":
            raise RuntimeBlocked("DummyTokenizer is available only in fixture mode")
        texts = [self.preprocessor.prepare_text(row.text) for row in rows]
        encoded = self._encode(texts, max_length=self.preprocessor.spec.max_length)
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
        if self.class_weights is not None:
            weights = self.class_weights.as_dict() if hasattr(self.class_weights, "as_dict") else self.class_weights
            pragmatic = weights.get("pragmatic_pos_weight") or weights.get("pragmatic")
            polarity = weights.get("polarity_weight") or weights.get("class_weight", {}).get("polarity")
            emotion = weights.get("emotion_weight") or weights.get("class_weight", {}).get("emotion")
            if pragmatic is not None:
                batch["pragmatic_pos_weight"] = {key: float(value) for key, value in pragmatic.items()}
            if polarity is not None:
                batch["polarity_weight"] = torch.tensor([float(polarity[label]) for label in POLARITY_LABELS], dtype=torch.float32)
            if emotion is not None:
                batch["emotion_weight"] = torch.tensor([float(emotion[label]) for label in EMOTION_LABELS], dtype=torch.float32)
        if self.rationale_records is not None:
            rationale_texts: list[str] = []
            rationale_available: list[float] = []
            for row in rows:
                value = self.rationale_records.get(row.sample_id)
                if value is not None and not isinstance(value, Mapping):
                    raise ValueError(f"Rationale record for {row.sample_id} must be a canonical mapping")
                text = str(value.get("rationale", "") if isinstance(value, Mapping) else "").strip()
                rationale_available.append(float(bool(text)))
                rationale_texts.append(text or "<NO_RATIONALE>")
            target_texts = [f"<RATIONALE>\n{text}\n</RATIONALE>" for text in rationale_texts]
            rationale_encoded = self._encode(target_texts, max_length=self.rationale_target_max_length)
            rationale_ids = torch.tensor(rationale_encoded["input_ids"], dtype=torch.long)
            rationale_attention = torch.tensor(rationale_encoded["attention_mask"], dtype=torch.long)
            batch["rationale_input_ids"] = rationale_ids
            batch["rationale_attention_mask"] = rationale_attention
            batch["rationale_targets"] = rationale_ids.clone()
            batch["rationale_loss_mask"] = torch.tensor(rationale_available, dtype=torch.float32)
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
            if "rationale_loss_mask" in batch:
                batch["rationale_loss_mask"] = batch["rationale_loss_mask"] * rationale_mask
            else:
                batch["rationale_loss_mask"] = rationale_mask
            batch["target_masks"] = {"sarcasm": sarcasm_mask}
            batch["selected_positive_count"] = selected_positive_count
            batch["fixed_negative_count"] = fixed_negative_count
            batch["budget_pos_weight"] = fixed_negative_count / selected_positive_count
            if "pragmatic_pos_weight" in batch:
                batch["pragmatic_pos_weight"] = dict(batch["pragmatic_pos_weight"])
                batch["pragmatic_pos_weight"]["sarcasm"] = batch["budget_pos_weight"]
            batch["mask_hash"] = self.mask_hash
        return batch
