"""Promotion gates, retraining triggers and the model registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rtml.config import PromotionConfig
from rtml.models.registry import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    ExperimentTracker,
    ModelRegistry,
)
from rtml.training.promotion import evaluate_promotion, should_retrain

CHAMPION = {"pr_auc": 0.70, "brier": 0.010, "positives": 500}


@pytest.fixture
def config() -> PromotionConfig:
    return PromotionConfig()


def test_a_clear_improvement_is_promoted(config: PromotionConfig) -> None:
    decision = evaluate_promotion(
        {"pr_auc": 0.75, "brier": 0.009, "positives": 500},
        CHAMPION,
        config,
        challenger_latency_p95=2.0,
        champion_latency_p95=1.5,
    )
    assert decision.promote
    assert decision.failed == []


def test_noise_sized_gains_are_rejected(config: PromotionConfig) -> None:
    """Without a margin the pipeline swaps models on every fluctuation."""
    decision = evaluate_promotion(
        {"pr_auc": 0.702, "brier": 0.010, "positives": 500}, CHAMPION, config
    )
    assert not decision.promote
    assert [gate.name for gate in decision.failed] == ["improvement_margin"]


def test_better_ranking_but_worse_calibration_is_rejected(config: PromotionConfig) -> None:
    """Scores feed a threshold, so calibration drift invalidates the operating point."""
    decision = evaluate_promotion(
        {"pr_auc": 0.76, "brier": 0.020, "positives": 500}, CHAMPION, config
    )
    assert not decision.promote
    assert "calibration" in [gate.name for gate in decision.failed]


def test_a_better_but_slower_model_is_rejected(config: PromotionConfig) -> None:
    decision = evaluate_promotion(
        {"pr_auc": 0.80, "brier": 0.009, "positives": 500},
        CHAMPION,
        config,
        challenger_latency_p95=60.0,
        champion_latency_p95=1.5,
    )
    assert not decision.promote
    assert "latency" in [gate.name for gate in decision.failed]


def test_too_little_fraud_to_judge_is_rejected(config: PromotionConfig) -> None:
    """PR-AUC on a handful of positives is not evidence of anything."""
    decision = evaluate_promotion(
        {"pr_auc": 0.95, "brier": 0.001, "positives": 9}, CHAMPION, config
    )
    assert not decision.promote
    assert "evaluation_sample_size" in [gate.name for gate in decision.failed]


def test_a_broken_first_model_is_still_rejected(config: PromotionConfig) -> None:
    """With no champion the relative gates are skipped, the absolute one is not."""
    decision = evaluate_promotion({"pr_auc": 0.05, "brier": 0.4, "positives": 500}, None, config)
    assert not decision.promote
    assert "absolute_pr_auc" in [gate.name for gate in decision.failed]


def test_a_sound_first_model_is_promoted(config: PromotionConfig) -> None:
    decision = evaluate_promotion({"pr_auc": 0.65, "brier": 0.01, "positives": 500}, None, config)
    assert decision.promote


def test_the_decision_explains_itself(config: PromotionConfig) -> None:
    """A refusal has to name the gate and the numbers, or nobody can act on it."""
    decision = evaluate_promotion(
        {"pr_auc": 0.701, "brier": 0.010, "positives": 500}, CHAMPION, config
    )
    assert "improvement_margin" in decision.reason()
    payload = decision.to_dict()
    assert payload["promote"] is False
    assert all({"gate", "passed", "detail"} <= set(gate) for gate in payload["gates"])


@pytest.mark.parametrize(
    ("drift", "age", "recent", "deployed", "expected"),
    [
        (True, 1, None, None, True),  # drift alert
        (False, 45, None, None, True),  # staleness
        (False, 1, 0.60, 0.70, True),  # measured performance drop
        (False, 1, 0.69, 0.70, False),  # within tolerance
        (False, 1, None, None, False),  # nothing fired
    ],
)
def test_retraining_triggers(
    drift: bool, age: int, recent: float | None, deployed: float | None, expected: bool
) -> None:
    fired, reason = should_retrain(drift, age, recent, deployed)
    assert fired is expected
    assert reason


def test_retraining_is_separate_from_deploying() -> None:
    """Deciding to retrain is cheap; deciding to deploy is not.

    A drift alert should start a training run, and that run's output must still
    face the promotion gates rather than shipping automatically.
    """
    fired, _ = should_retrain(
        drift_alert=True, days_since_training=1, recent_pr_auc=None, deployed_pr_auc=None
    )
    assert fired
    decision = evaluate_promotion(
        {"pr_auc": 0.10, "brier": 0.5, "positives": 500}, CHAMPION, PromotionConfig()
    )
    assert not decision.promote


# --- Model registry -------------------------------------------------------
#
# These exercise the path that once failed silently: training completed, every
# metric was tracked, and the registry stayed empty because the champion - always
# a gradient-boosted model - could not be serialised by the default flavour.


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


@pytest.mark.slow
def test_a_gradient_boosted_champion_reaches_the_registry(tracking_uri: str) -> None:
    """The flavour must match the model's library, or logging fails outright.

    MLflow's sklearn flavour defaults to a serialiser that refuses LightGBM and
    XGBoost boosters. Since every champion this pipeline produces is one of
    those two, a wrong flavour empties the registry while training still
    reports success.
    """
    lightgbm = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + rng.normal(scale=0.1, size=200) > 0).astype(int)
    model = lightgbm.LGBMClassifier(n_estimators=5, num_leaves=3, verbose=-1).fit(X, y)

    tracker = ExperimentTracker(tracking_uri, "registry-test", None)
    with tracker.run("challenger") as active:
        run_id = active.info.run_id
        uri = tracker.log_model(model, "lightgbm")

    assert uri is not None, "logging a LightGBM model returned no URI"

    registry = ModelRegistry(tracking_uri)
    version = registry.register(uri, "fraud-scorer", run_id=run_id)
    registry.promote("fraud-scorer", version)

    assert registry.get_alias_version("fraud-scorer", CHAMPION_ALIAS) == version
    recorded = registry.versions("fraud-scorer")
    assert [v["version"] for v in recorded] == [version]
    # search_model_versions leaves aliases empty on the SQL store; reporting
    # them from the version rows alone would say nothing is deployed.
    assert recorded[0]["aliases"] == [CHAMPION_ALIAS]


@pytest.mark.slow
def test_promotion_demotes_the_incumbent_rather_than_dropping_it(tracking_uri: str) -> None:
    """Rollback must be an alias move, not a retrain."""
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)

    registry = ModelRegistry(tracking_uri)
    tracker = ExperimentTracker(tracking_uri, "registry-test", None)
    versions = []
    for name in ("first", "second"):
        model = sklearn_linear.LogisticRegression().fit(X, y)
        with tracker.run(name) as active:
            uri = tracker.log_model(model, name)
            assert uri is not None
            versions.append(registry.register(uri, "fraud-scorer", run_id=active.info.run_id))
        registry.promote("fraud-scorer", versions[-1])

    assert registry.get_alias_version("fraud-scorer", CHAMPION_ALIAS) == versions[1]
    assert registry.get_alias_version("fraud-scorer", CHALLENGER_ALIAS) == versions[0]

    rolled_back = registry.rollback("fraud-scorer")
    assert rolled_back["champion"] == versions[0]
