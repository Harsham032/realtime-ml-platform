"""Classification metrics for a heavily imbalanced problem.

At a fraud rate near 0.8 percent, accuracy is meaningless (a model predicting
"never fraud" scores 99.2 percent) and ROC-AUC is optimistic: it is dominated by
the vast negative class, so a large absolute number of false positives barely
moves it. Precision-recall AUC and precision at a fixed review capacity are the
metrics that track what an operations team experiences, and they are what this
module leads with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..errors import EvaluationError


def _validate(y_true: np.ndarray, y_score: np.ndarray) -> None:
    if len(y_true) != len(y_score):
        raise EvaluationError("y_true and y_score must have the same length")
    if len(y_true) == 0:
        raise EvaluationError("cannot evaluate an empty set")
    positives = int(np.sum(y_true))
    if positives == 0:
        raise EvaluationError(
            "the evaluation set contains no fraud; PR-AUC and recall are undefined"
        )


def ranking_metrics(y_true: ArrayLike, y_score: ArrayLike) -> dict[str, float]:
    """Threshold-free metrics.

    ``pr_auc`` is ``average_precision_score``, the step-wise summary of the
    precision-recall curve. It is preferred over the trapezoidal
    ``auc(recall, precision)``, which interpolates between operating points that
    no threshold actually produces.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    _validate(y_true_arr, y_score_arr)

    return {
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "brier": float(brier_score_loss(y_true_arr, np.clip(y_score_arr, 0.0, 1.0))),
        "positives": float(y_true_arr.sum()),
        "negatives": float(len(y_true_arr) - y_true_arr.sum()),
        "base_rate": float(y_true_arr.mean()),
    }


def threshold_metrics(y_true: ArrayLike, y_score: ArrayLike, threshold: float) -> dict[str, float]:
    """Metrics at one operating point, including the raw confusion counts.

    The counts are reported alongside the rates because "precision 0.62" and
    "4,100 false positives a week" are the same fact, and only the second tells
    an investigation team whether the model is usable.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred = (np.asarray(y_score, dtype=np.float64) >= threshold).astype(np.int64)
    _validate(y_true_arr, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    alerts = int(tp + fp)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_negatives": float(tn),
        "alerts": float(alerts),
        "alert_rate": float(alerts / len(y_true_arr)),
    }


def recall_by_group(
    y_true: ArrayLike,
    y_score: ArrayLike,
    groups: ArrayLike,
    threshold: float,
) -> dict[int, dict[str, float]]:
    """Recall at one operating point, split by the group each positive belongs to.

    Aggregate recall hides which kinds of fraud a model actually catches. A
    pipeline computing only transaction-level features detects the
    amount-threshold scenario and nothing else, yet still reports a respectable
    overall number because that scenario is not rare. Splitting recall by
    scenario makes that failure visible instead of letting it average away.

    Only positive rows carry a group, so negatives are ignored here: precision
    is not defined per scenario, since a false positive belongs to no scenario.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    group_arr = np.asarray(groups)
    if len(group_arr) != len(y_true_arr):
        raise EvaluationError("groups must align with labels")
    flagged = np.asarray(y_score, dtype=np.float64) >= threshold

    out: dict[int, dict[str, float]] = {}
    for group in sorted({int(g) for g in group_arr[y_true_arr == 1]}):
        member = (y_true_arr == 1) & (group_arr == group)
        total = int(member.sum())
        caught = int((member & flagged).sum())
        out[group] = {
            "frauds": float(total),
            "detected": float(caught),
            "recall": float(caught / total) if total else 0.0,
        }
    return out


def card_precision_at_k(
    frame: pd.DataFrame,
    *,
    k: int = 100,
    score_column: str = "score",
    label_column: str = "is_fraud",
    entity_column: str = "customer_id",
    day_column: str = "tx_day",
) -> dict[str, float]:
    """Mean daily precision over the ``k`` highest-risk cards.

    This is the operational metric from the Fraud Detection Handbook. An
    investigation team can review a fixed number of cards a day, not a fixed
    score threshold, so the question that matters is: of the ``k`` cards we
    flagged today, how many were actually compromised?

    Scoring is per card per day - a card is counted once however many
    transactions it made - because a team investigates a card, not a swipe.
    """
    required = {score_column, label_column, entity_column, day_column}
    missing = required - set(frame.columns)
    if missing:
        raise EvaluationError(f"card precision needs columns: {', '.join(sorted(missing))}")
    if k <= 0:
        raise EvaluationError("k must be positive")

    daily: list[float] = []
    for _, day_frame in frame.groupby(day_column, sort=True):
        # A card's risk for the day is its highest-scoring transaction; its
        # label is whether any of its transactions that day was fraudulent.
        by_card = day_frame.groupby(entity_column).agg(
            score=(score_column, "max"), label=(label_column, "max")
        )
        if by_card["label"].sum() == 0:
            # No fraud that day: precision is undefined rather than zero, so
            # the day is excluded instead of dragging the mean down.
            continue
        top = by_card.nlargest(min(k, len(by_card)), "score")
        daily.append(float(top["label"].mean()))

    if not daily:
        raise EvaluationError("no day in the evaluation window contained fraud")
    return {
        f"card_precision_at_{k}": float(np.mean(daily)),
        f"card_precision_at_{k}_std": float(np.std(daily)),
        "evaluated_days": float(len(daily)),
    }


def bootstrap_metric(
    y_true: ArrayLike,
    y_score: ArrayLike,
    metric: str = "pr_auc",
    *,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260101,
) -> dict[str, float]:
    """Percentile bootstrap interval for a ranking metric.

    Resamples are stratified by class. An unstratified bootstrap of a 0.8
    percent positive rate occasionally draws a sample with no positives at all,
    which makes the metric undefined and biases whatever is left.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    _validate(y_true_arr, y_score_arr)
    if resamples <= 0:
        return {}

    scorer = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}.get(metric)
    if scorer is None:
        raise EvaluationError(f"unsupported bootstrap metric: {metric}")

    positive_index = np.flatnonzero(y_true_arr == 1)
    negative_index = np.flatnonzero(y_true_arr == 0)
    rng = np.random.default_rng(seed)

    values = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        sample = np.concatenate(
            [
                rng.choice(positive_index, size=len(positive_index), replace=True),
                rng.choice(negative_index, size=len(negative_index), replace=True),
            ]
        )
        values[draw] = scorer(y_true_arr[sample], y_score_arr[sample])

    values.sort()
    tail = (1.0 - confidence) / 2.0
    return {
        f"{metric}_ci_lower": float(values[int(tail * resamples)]),
        f"{metric}_ci_upper": float(values[min(int((1.0 - tail) * resamples), resamples - 1)]),
    }


def precision_recall_table(y_true: ArrayLike, y_score: ArrayLike, points: int = 20) -> pd.DataFrame:
    """A readable slice of the precision-recall curve for reporting."""
    precision, recall, thresholds = precision_recall_curve(np.asarray(y_true), np.asarray(y_score))
    # precision_recall_curve returns one more point than thresholds.
    frame = pd.DataFrame(
        {"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]}
    )
    frame = frame[frame["recall"] > 0]
    if len(frame) <= points:
        return frame.reset_index(drop=True)
    step = max(len(frame) // points, 1)
    return frame.iloc[::step].reset_index(drop=True)
