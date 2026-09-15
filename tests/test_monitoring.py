"""Drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rtml.config import DriftConfig
from rtml.errors import EvaluationError
from rtml.monitoring.drift import (
    classify,
    compare_distributions,
    drift_over_time,
    performance_over_time,
    population_stability_index,
)


@pytest.fixture
def config() -> DriftConfig:
    return DriftConfig()


def test_identical_distributions_have_negligible_psi() -> None:
    rng = np.random.default_rng(0)
    psi = population_stability_index(rng.normal(size=20_000), rng.normal(size=20_000))
    assert psi < 0.01


def test_psi_grows_with_the_size_of_the_shift() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(size=20_000)
    small = population_stability_index(reference, rng.normal(0.2, 1, 20_000))
    large = population_stability_index(reference, rng.normal(1.5, 1, 20_000))
    assert small < large
    assert large > 0.25


def test_psi_detects_a_variance_change_not_just_a_mean_shift() -> None:
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 20_000)
    psi = population_stability_index(reference, rng.normal(0, 3, 20_000))
    assert psi > 0.25


def test_psi_is_finite_when_a_bin_empties() -> None:
    """The formula takes a log of the ratio; an empty bin must not return infinity."""
    reference = np.concatenate([np.zeros(500), np.ones(500)])
    comparison = np.zeros(500)
    assert np.isfinite(population_stability_index(reference, comparison))


def test_psi_on_a_constant_feature() -> None:
    constant = np.full(100, 5.0)
    assert population_stability_index(constant, np.full(100, 5.0)) == 0.0
    assert population_stability_index(constant, np.full(100, 9.0)) == 1.0


def test_psi_rejects_empty_samples() -> None:
    with pytest.raises(EvaluationError):
        population_stability_index(np.array([]), np.array([1.0]))


def test_severity_needs_both_effect_size_and_significance(config: DriftConfig) -> None:
    """Either test alone produces alerts nobody trusts."""
    assert classify(psi=0.40, pvalue=1e-9, config=config) == "significant"
    # A large move that could be noise is a warning, not a page.
    assert classify(psi=0.40, pvalue=0.5, config=config) == "moderate"
    # A statistically certain but tiny move is not worth waking anyone.
    assert classify(psi=0.01, pvalue=1e-30, config=config) == "stable"
    assert classify(psi=0.15, pvalue=0.5, config=config) == "moderate"


def test_compare_flags_only_the_drifted_feature(config: DriftConfig) -> None:
    rng = np.random.default_rng(2)
    reference = pd.DataFrame({"moved": rng.normal(0, 1, 8000), "stable": rng.normal(0, 1, 8000)})
    comparison = pd.DataFrame({"moved": rng.normal(1.5, 1, 8000), "stable": rng.normal(0, 1, 8000)})
    report = compare_distributions(reference, comparison, ["moved", "stable"], config)

    by_name = {result.feature: result for result in report.results}
    assert by_name["moved"].severity == "significant"
    assert by_name["stable"].severity == "stable"
    assert report.should_alert()
    # Results are ordered worst-first so a report reads top-down.
    assert report.results[0].feature == "moved"


def test_no_alert_when_nothing_moved(config: DriftConfig) -> None:
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"f": rng.normal(size=8000)})
    comparison = pd.DataFrame({"f": rng.normal(size=8000)})
    assert not compare_distributions(reference, comparison, ["f"], config).should_alert()


def test_drift_over_time_uses_the_configured_windows(config: DriftConfig) -> None:
    rng = np.random.default_rng(4)
    rows = []
    for day in range(30):
        # A deliberate regime change part way through.
        centre = 0.0 if day < 23 else 2.0
        rows.append(pd.DataFrame({"tx_day": day, "score": rng.normal(centre, 0.5, 400)}))
    frame = pd.concat(rows, ignore_index=True)

    report = drift_over_time(frame, ["score"], config)
    assert report.comparison_window[1] == 29
    assert report.should_alert()


def test_drift_over_time_needs_enough_history(config: DriftConfig) -> None:
    frame = pd.DataFrame({"tx_day": [0, 0, 1], "score": [0.1, 0.2, 0.3]})
    with pytest.raises(EvaluationError, match="not enough history"):
        drift_over_time(frame, ["score"], config)


def test_performance_over_time_skips_windows_without_fraud() -> None:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "tx_day": np.repeat(np.arange(21), 100),
            "score": rng.random(2100),
            # Fraud only in the first week.
            "is_fraud": np.where(
                np.repeat(np.arange(21), 100) < 7, (rng.random(2100) < 0.1).astype(int), 0
            ),
        }
    )
    table = performance_over_time(frame, window_days=7)
    assert len(table) == 1
    assert {"precision", "recall", "alert_rate", "mean_score"} <= set(table.columns)
