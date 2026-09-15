"""Model factory.

Four families are trained because they fail differently on this problem, not to
pad a list:

``logistic_regression``
    A linear baseline. If a gradient-boosted tree cannot beat it, the features
    carry no interaction worth modelling and the extra complexity is not
    earning anything.
``random_forest``
    Bagged trees. Robust to the feature scaling that logistic regression needs
    and hard to overfit, but it does not chase the residual the way boosting
    does.
``xgboost`` and ``lightgbm``
    Gradient-boosted trees, the usual strongest performers on tabular fraud.
    Both are included because their handling of the imbalance differs -
    ``scale_pos_weight`` against ``is_unbalance`` - and the gap between them is
    worth measuring rather than assuming.

``isolation_forest`` is available separately as an unsupervised contrast: it
never sees a label, so it shows how much of the signal is reachable from outlier
structure alone.

Class imbalance is handled by reweighting rather than resampling. Undersampling
the negative class throws away most of the data, and oversampling the positive
class with only a few thousand frauds mostly duplicates rows, which boosted
trees overfit quickly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import ModelConfig
from ..errors import ModelError

SUPPORTED = ("logistic_regression", "random_forest", "xgboost", "lightgbm")


def scale_pos_weight(y: np.ndarray) -> float:
    """Ratio of negatives to positives, the standard boosting imbalance knob."""
    positives = float(np.sum(y))
    if positives == 0:
        raise ModelError("cannot compute a class weight without any positive example")
    return float((len(y) - positives) / positives)


def build_estimator(
    name: str, config: ModelConfig, y_train: np.ndarray, *, seed: int = 20260101
) -> BaseEstimator:
    """Instantiate one model family, configured for imbalance."""
    balanced = config.class_weight == "balanced"

    if name == "logistic_regression":
        params = dict(config.logistic_regression)
        # Scaling matters here and nowhere else: the tree models are invariant
        # to monotone feature transforms, logistic regression is not, and the
        # count features run to hundreds while the risk features sit in [0, 1].
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced" if balanced else None,
                        random_state=seed,
                        **params,
                    ),
                ),
            ]
        )

    if name == "random_forest":
        params = dict(config.random_forest)
        return RandomForestClassifier(
            class_weight="balanced_subsample" if balanced else None,
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - xgboost is a hard dependency
            raise ModelError("xgboost is not installed") from exc
        params = dict(config.xgboost)
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight(y_train) if balanced else 1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )

    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - lightgbm is a hard dependency
            raise ModelError("lightgbm is not installed") from exc
        params = dict(config.lightgbm)
        return LGBMClassifier(
            objective="binary",
            class_weight="balanced" if balanced else None,
            random_state=seed,
            verbose=-1,
            **params,
        )

    raise ModelError(f"unknown model: {name}. Supported: {', '.join(SUPPORTED)}")


def build_isolation_forest(config: ModelConfig, *, seed: int = 20260101) -> IsolationForest:
    """Unsupervised contrast model."""
    params: dict[str, Any] = dict(config.isolation_forest)
    return IsolationForest(random_state=seed, n_jobs=-1, **params)


def isolation_forest_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Map isolation-forest scores into [0, 1] with higher meaning more anomalous.

    ``score_samples`` returns a negative log-anomaly score where *lower* is more
    anomalous, which is the opposite orientation to every other model here.
    Min-max mapping makes it comparable; it is a ranking, not a probability, so
    it is excluded from calibration comparisons.
    """
    raw = -model.score_samples(X)
    span = raw.max() - raw.min()
    if span <= 0:
        return np.zeros_like(raw)
    return (raw - raw.min()) / span
