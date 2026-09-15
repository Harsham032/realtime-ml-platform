"""Choosing an operating point.

A probability model is not a decision until a threshold turns it into one, and
the threshold is a business choice rather than a modelling one: it trades
investigator time against undetected fraud. What this module does is make the
choice explicit, reproducible, and - critically - made on validation data.

Selecting a threshold on the test set is the most common way a fraud model's
reported precision turns out to be unreachable in production.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import precision_recall_curve

from ..errors import EvaluationError

Objective = Literal["f1", "recall_at_precision", "precision_at_recall"]


def select_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    objective: Objective = "f1",
    target_precision: float = 0.6,
    target_recall: float = 0.6,
) -> dict[str, float | str]:
    """Pick a threshold on validation data and report what it achieves there.

    ``f1``
        The point maximising F1. A reasonable default when the cost of a missed
        fraud and of a wasted investigation are treated as comparable.
    ``recall_at_precision``
        The highest recall available while holding precision at or above
        ``target_precision``. Use when investigator capacity is the binding
        constraint.
    ``precision_at_recall``
        The highest precision available while holding recall at or above
        ``target_recall``. Use when a fraud loss target is the binding
        constraint.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    if len(y_true_arr) == 0 or y_true_arr.sum() == 0:
        raise EvaluationError("threshold selection needs a validation set containing fraud")

    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_score_arr)
    precision, recall = precision[:-1], recall[:-1]
    if len(thresholds) == 0:
        raise EvaluationError("the score distribution admits no usable threshold")

    if objective == "f1":
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(
                (precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0
            )
        index = int(np.argmax(f1))
        attained = "f1"
    elif objective == "recall_at_precision":
        feasible = np.flatnonzero(precision >= target_precision)
        if len(feasible) == 0:
            raise EvaluationError(
                f"no threshold reaches precision {target_precision:.2f}; "
                f"the best available is {precision.max():.3f}"
            )
        index = int(feasible[np.argmax(recall[feasible])])
        attained = "recall_at_precision"
    elif objective == "precision_at_recall":
        feasible = np.flatnonzero(recall >= target_recall)
        if len(feasible) == 0:
            raise EvaluationError(
                f"no threshold reaches recall {target_recall:.2f}; "
                f"the best available is {recall.max():.3f}"
            )
        index = int(feasible[np.argmax(precision[feasible])])
        attained = "precision_at_recall"
    else:
        raise EvaluationError(f"unknown threshold objective: {objective}")

    return {
        "threshold": float(thresholds[index]),
        "validation_precision": float(precision[index]),
        "validation_recall": float(recall[index]),
        "objective": attained,
    }
