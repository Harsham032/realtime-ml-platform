"""Metrics, thresholds and the temporal split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rtml.config import PipelineConfig, SplitConfig
from rtml.errors import EvaluationError
from rtml.evaluation.metrics import (
    bootstrap_metric,
    card_precision_at_k,
    precision_recall_table,
    ranking_metrics,
    recall_by_group,
    threshold_metrics,
)
from rtml.evaluation.thresholds import select_threshold
from rtml.training.splitting import split_by_day


@pytest.fixture
def imbalanced_scores() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 20_000
    labels = (rng.random(n) < 0.008).astype(int)
    scores = np.clip(rng.normal(0.1 + 0.5 * labels, 0.15), 0.0, 1.0)
    return labels, scores


def test_ranking_metrics_report_the_base_rate(imbalanced_scores) -> None:
    labels, scores = imbalanced_scores
    metrics = ranking_metrics(labels, scores)
    assert metrics["base_rate"] == pytest.approx(labels.mean())
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["positives"] + metrics["negatives"] == len(labels)


def test_roc_auc_flatters_an_imbalanced_problem(imbalanced_scores) -> None:
    """The reason PR-AUC leads everywhere in this repository.

    On a 0.8 percent base rate the same scores give a ROC-AUC that looks
    excellent and a PR-AUC that tells the truth.
    """
    labels, scores = imbalanced_scores
    metrics = ranking_metrics(labels, scores)
    assert metrics["roc_auc"] > metrics["pr_auc"] + 0.2


def test_a_perfect_ranking_scores_one() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    metrics = ranking_metrics(labels, scores)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_evaluating_without_positives_is_an_error() -> None:
    """Silently returning zero would hide a broken split."""
    with pytest.raises(EvaluationError, match="no fraud"):
        ranking_metrics(np.zeros(100, dtype=int), np.random.random(100))


def test_threshold_metrics_include_raw_counts(imbalanced_scores) -> None:
    labels, scores = imbalanced_scores
    metrics = threshold_metrics(labels, scores, 0.5)
    total = (
        metrics["true_positives"]
        + metrics["false_positives"]
        + metrics["false_negatives"]
        + metrics["true_negatives"]
    )
    assert total == len(labels)
    assert metrics["alerts"] == metrics["true_positives"] + metrics["false_positives"]


def test_threshold_selection_uses_the_requested_objective(imbalanced_scores) -> None:
    labels, scores = imbalanced_scores
    f1_choice = select_threshold(labels, scores, objective="f1")
    precision_choice = select_threshold(
        labels, scores, objective="recall_at_precision", target_precision=0.8
    )
    assert precision_choice["validation_precision"] >= 0.8
    # A stricter precision target demands a higher bar.
    assert precision_choice["threshold"] >= f1_choice["threshold"]


def test_unreachable_precision_target_fails_clearly() -> None:
    """A target above the curve's ceiling must fail loudly, not silently settle.

    The highest-scored transaction here is a false positive, so precision never
    reaches 1.0 no matter where the threshold goes - which is what makes the
    target genuinely unreachable rather than merely expensive.
    """
    labels = np.array([0, 1, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    with pytest.raises(EvaluationError, match="no threshold reaches precision"):
        select_threshold(labels, scores, objective="recall_at_precision", target_precision=0.9)


def test_a_reachable_precision_target_is_honoured() -> None:
    labels = np.array([0, 1, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    chosen = select_threshold(labels, scores, objective="recall_at_precision", target_precision=0.6)
    assert chosen["validation_precision"] >= 0.6


def test_bootstrap_interval_brackets_the_estimate(imbalanced_scores) -> None:
    labels, scores = imbalanced_scores
    point = ranking_metrics(labels, scores)["pr_auc"]
    interval = bootstrap_metric(labels, scores, "pr_auc", resamples=200)
    assert interval["pr_auc_ci_lower"] <= point <= interval["pr_auc_ci_upper"]


def test_bootstrap_is_reproducible(imbalanced_scores) -> None:
    labels, scores = imbalanced_scores
    a = bootstrap_metric(labels, scores, resamples=100, seed=5)
    b = bootstrap_metric(labels, scores, resamples=100, seed=5)
    assert a == b


def test_card_precision_scores_cards_not_transactions() -> None:
    """A card with three flagged transactions is one investigation, not three."""
    frame = pd.DataFrame(
        {
            "customer_id": [1, 1, 1, 2, 3, 4],
            "tx_day": [0] * 6,
            "score": [0.9, 0.85, 0.8, 0.7, 0.2, 0.1],
            "is_fraud": [1, 1, 1, 0, 0, 0],
        }
    )
    result = card_precision_at_k(frame, k=2)
    # Top 2 cards are 1 (fraud) and 2 (clean).
    assert result["card_precision_at_2"] == pytest.approx(0.5)


def test_card_precision_skips_days_without_fraud() -> None:
    """Precision is undefined on a day with nothing to find, not zero."""
    frame = pd.DataFrame(
        {
            "customer_id": [1, 2, 1, 2],
            "tx_day": [0, 0, 1, 1],
            "score": [0.9, 0.1, 0.9, 0.1],
            "is_fraud": [1, 0, 0, 0],
        }
    )
    result = card_precision_at_k(frame, k=1)
    assert result["evaluated_days"] == 1.0
    assert result["card_precision_at_1"] == pytest.approx(1.0)


def test_precision_recall_table_is_readable(imbalanced_scores) -> None:
    labels, scores = imbalanced_scores
    table = precision_recall_table(labels, scores, points=10)
    assert set(table.columns) == {"threshold", "precision", "recall"}
    assert len(table) <= 25


# ------------------------------------------------------------------ splits


def test_split_is_chronological_and_gapped(features: pd.DataFrame, config: PipelineConfig) -> None:
    split = split_by_day(features, config.split)
    assert split.train["tx_day"].max() < split.validation["tx_day"].min()
    gap = split.test["tx_day"].min() - split.validation["tx_day"].max()
    assert gap > config.split.delay_days - 1, "the delay gap is missing"


def test_split_parts_are_disjoint(features: pd.DataFrame, config: PipelineConfig) -> None:
    split = split_by_day(features, config.split)
    ids = [set(part["transaction_id"]) for part in (split.train, split.validation, split.test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])


def test_every_split_contains_fraud(features: pd.DataFrame, config: PipelineConfig) -> None:
    split = split_by_day(features, config.split)
    for part in (split.train, split.validation, split.test):
        assert part["is_fraud"].sum() > 0


def test_a_split_longer_than_the_data_is_rejected(features: pd.DataFrame) -> None:
    with pytest.raises(EvaluationError, match="data ends"):
        split_by_day(
            features, SplitConfig(train_days=500, delay_days=7, test_days=100, validation_days=10)
        )


def test_validation_comes_from_the_end_of_training(
    features: pd.DataFrame, config: PipelineConfig
) -> None:
    """Operating points are chosen on the most recent data the model may see."""
    split = split_by_day(features, config.split)
    assert split.validation["tx_day"].min() == split.boundaries["validation_start_day"]
    assert split.validation["tx_day"].max() == split.boundaries["train_end_day"] - 1


def test_recall_by_group_splits_a_misleading_aggregate() -> None:
    """A model that catches one scenario and misses another looks fine in aggregate."""
    labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    groups = np.array([1, 1, 1, 2, 2, 2, 0, 0, 0, 0])
    # Every scenario-1 fraud scores high, every scenario-2 fraud scores low.
    scores = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2])

    per_group = recall_by_group(labels, scores, groups, threshold=0.5)
    assert set(per_group) == {1, 2}
    assert per_group[1] == {"frauds": 3.0, "detected": 3.0, "recall": 1.0}
    assert per_group[2] == {"frauds": 3.0, "detected": 0.0, "recall": 0.0}

    # The aggregate averages the failure away.
    assert threshold_metrics(labels, scores, 0.5)["recall"] == pytest.approx(0.5)


def test_recall_by_group_ignores_negatives() -> None:
    """A group appears only if it labels a positive; a false positive belongs to none."""
    labels = np.array([1, 0, 0, 1])
    groups = np.array([2, 0, 0, 2])
    scores = np.array([0.8, 0.9, 0.1, 0.4])
    per_group = recall_by_group(labels, scores, groups, threshold=0.5)
    assert list(per_group) == [2]
    assert per_group[2]["recall"] == pytest.approx(0.5)


def test_recall_by_group_rejects_misaligned_groups() -> None:
    with pytest.raises(EvaluationError, match="align"):
        recall_by_group(np.array([1, 0]), np.array([0.9, 0.1]), np.array([1]), threshold=0.5)
