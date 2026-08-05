from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path


def raw_agreement(first: Sequence[object], second: Sequence[object]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Agreement inputs must have the same non-zero length")
    return sum(left == right for left, right in zip(first, second)) / len(first)


def cohen_kappa(first: Sequence[object], second: Sequence[object]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Kappa inputs must have the same non-zero length")
    n = len(first)
    observed = raw_agreement(first, second)
    categories = sorted(set(first) | set(second), key=str)
    p_first = {category: sum(value == category for value in first) / n for category in categories}
    p_second = {category: sum(value == category for value in second) / n for category in categories}
    expected = sum(p_first[category] * p_second[category] for category in categories)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def krippendorff_alpha_nominal(units: Iterable[Sequence[object]]) -> float:
    """Compute nominal alpha for complete or partially observed annotation units."""
    unit_values = [list(unit) for unit in units]
    unit_values = [values for values in unit_values if len(values) >= 2]
    if not unit_values:
        raise ValueError("At least one unit with two annotations is required")
    coincidence: Counter[tuple[object, object]] = Counter()
    category_counts: Counter[object] = Counter()
    total_pairs = 0
    for values in unit_values:
        counts = Counter(values)
        m = len(values)
        for category, count in counts.items():
            category_counts[category] += count
            coincidence[(category, category)] += count * (count - 1) / (m - 1)
        categories = list(counts)
        for left in categories:
            for right in categories:
                if left != right:
                    coincidence[(left, right)] += counts[left] * counts[right] / (m - 1)
        total_pairs += m
    total_coincidences = sum(coincidence.values())
    if total_coincidences == 0:
        return 1.0
    observed_disagreement = sum(value for (left, right), value in coincidence.items() if left != right) / total_coincidences
    total = sum(category_counts.values())
    expected_disagreement = 1.0 - sum(count * (count - 1) for count in category_counts.values()) / (total * (total - 1))
    return 1.0 - observed_disagreement / expected_disagreement if expected_disagreement else 1.0


def recompute_human_iaa(package_dir: str | Path) -> list[dict[str, object]]:
    root = Path(package_dir) / "01_clean_human_annotations"
    files = [root / "01_annotator_1_clean.csv", root / "02_annotator_2_clean.csv"]
    with files[0].open(encoding="utf-8-sig", newline="") as handle:
        first_rows = list(csv.DictReader(handle))
    with files[1].open(encoding="utf-8-sig", newline="") as handle:
        second_rows = list(csv.DictReader(handle))
    if [row["sample_id"] for row in first_rows] != [row["sample_id"] for row in second_rows]:
        raise ValueError("Annotator rows are not aligned")
    fields = (
        "implicit_sentiment",
        "sarcasm",
        "irony",
        "idiom_figurative",
        "code_switching",
        "mocking",
        "polarity",
        "emotion",
    )
    result: list[dict[str, object]] = []
    for field in fields:
        left = [row[field] for row in first_rows]
        right = [row[field] for row in second_rows]
        result.append({
            "field": field,
            "n": len(left),
            "raw_agreement": raw_agreement(left, right),
            "cohen_kappa": cohen_kappa(left, right),
            "krippendorff_alpha_nominal": krippendorff_alpha_nominal(zip(left, right)),
            "disagreement_count": sum(a != b for a, b in zip(left, right)),
        })
    return result
