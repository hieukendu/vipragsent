from __future__ import annotations

import pytest

from vipragsent.runtime.estimator import GENERATION_FACTORS, estimate_runtime
from vipragsent.runtime.scheduler import ResourcePolicy, StageSpec


def test_generation_speedup_is_monotonic_and_matches_49_hour_planning_values() -> None:
    report = estimate_runtime(
        specs=(
            StageSpec(
                stage_id="vistral_dev_generation",
                campaign_id="fixture_campaign",
                run_id="fixture_run",
                kind="generation",
                duration_minutes=49 * 60,
                resource_class="7b",
                generation=True,
            ),
        ),
        as_of="fixture",
        policy=ResourcePolicy.resource_aware(),
    )

    expected_hours = {
        1.0: 49.0,
        1.5: 32.666666666666664,
        2.0: 24.5,
        2.5: 19.6,
        3.0: 16.333333333333332,
        4.0: 12.25,
    }
    actual_hours = {
        factor: minutes / 60 for factor, minutes in report.generation_sensitivity.items()
    }

    assert tuple(report.generation_sensitivity) == GENERATION_FACTORS
    assert actual_hours == pytest.approx(expected_hours)
    makespans = [report.generation_sensitivity[factor] for factor in GENERATION_FACTORS]
    assert all(left >= right for left, right in zip(makespans, makespans[1:]))
