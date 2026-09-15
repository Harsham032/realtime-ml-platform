"""Training pipeline.

Trains each enabled model family on the same temporal split, selects an
operating point on validation, evaluates on the held-out test window, and
records everything to MLflow so a result can be traced back to the exact
configuration and data that produced it.

The order matters and is enforced rather than left to discipline: thresholds are
chosen on validation, never on test. A threshold tuned on the evaluation window
reports a precision the deployed model will not reach.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..errors import ModelError
from ..evaluation.metrics import (
    bootstrap_metric,
    card_precision_at_k,
    ranking_metrics,
    recall_by_group,
    threshold_metrics,
)
from ..evaluation.thresholds import select_threshold
from ..features.engineering import feature_names
from ..logging_utils import get_logger
from ..models.estimators import build_estimator, build_isolation_forest, isolation_forest_scores
from .splitting import TemporalSplit, split_by_day

logger = get_logger(__name__)

# Single rows scored when measuring serving latency. One row at a time is the
# honest figure for a streaming path that scores each event as it arrives, but
# it is slow for a bagged forest, so the sample is bounded: 200 draws pins p95
# tightly enough and keeps a full benchmark run to minutes rather than hours.
LATENCY_SAMPLE_SIZE = 200


@dataclass
class TrainedModel:
    """A fitted model with everything needed to judge and serve it."""

    name: str
    estimator: Any
    features: list[str]
    threshold: float
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    scenario_recall: dict[int, dict[str, float]] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0
    run_id: str | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(X))[:, 1]
        return isolation_forest_scores(self.estimator, X)


def environment_info() -> dict[str, str]:
    """Facts a reader needs to interpret timings and reproduce a run."""
    import sklearn

    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    for package in ("xgboost", "lightgbm", "mlflow"):
        try:
            info[package] = __import__(package).__version__
        except ImportError:  # pragma: no cover - all three are hard dependencies
            info[package] = "not installed"
    return info


def measure_latency(
    model: TrainedModel, X: np.ndarray, *, sample: int = LATENCY_SAMPLE_SIZE
) -> dict[str, float]:
    """Per-transaction scoring latency, measured one row at a time.

    Batch throughput flatters a streaming system: the consumer scores an event
    when it arrives, so single-row latency is the number that predicts the
    service level. Batch figures are reported alongside for capacity planning.
    """
    if len(X) == 0:
        return {}
    rows = X[: min(sample, len(X))]

    single: list[float] = []
    for index in range(len(rows)):
        row = rows[index : index + 1]
        start = time.perf_counter()
        model.predict_proba(row)
        single.append((time.perf_counter() - start) * 1000.0)

    start = time.perf_counter()
    model.predict_proba(rows)
    batch_ms = (time.perf_counter() - start) * 1000.0

    ordered = np.sort(np.asarray(single))
    return {
        "latency_mean_ms": float(ordered.mean()),
        "latency_p50_ms": float(np.percentile(ordered, 50)),
        "latency_p95_ms": float(np.percentile(ordered, 95)),
        "latency_p99_ms": float(np.percentile(ordered, 99)),
        "batch_throughput_rows_per_second": float(len(rows) / (batch_ms / 1000.0)),
        "latency_samples": float(len(ordered)),
    }


def _matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return frame[features].to_numpy(dtype=np.float32)


def train_one(
    name: str,
    split: TemporalSplit,
    config: PipelineConfig,
    features: list[str],
) -> TrainedModel:
    """Fit, tune the operating point on validation, and evaluate on test."""
    X_train, y_train = _matrix(split.train, features), split.train["is_fraud"].to_numpy()
    X_val, y_val = _matrix(split.validation, features), split.validation["is_fraud"].to_numpy()
    X_test, y_test = _matrix(split.test, features), split.test["is_fraud"].to_numpy()

    if name == "isolation_forest":
        estimator = build_isolation_forest(config.models, seed=config.run.seed)
        start = time.perf_counter()
        # Unsupervised: fitted on the training features with no labels at all.
        estimator.fit(X_train)
        fit_seconds = time.perf_counter() - start
    else:
        estimator = build_estimator(name, config.models, y_train, seed=config.run.seed)
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

    model = TrainedModel(
        name=name, estimator=estimator, features=features, threshold=0.5, fit_seconds=fit_seconds
    )

    val_scores = model.predict_proba(X_val)
    selection = select_threshold(
        y_val,
        val_scores,
        objective=config.evaluation.threshold_objective,
        target_precision=config.evaluation.target_precision,
        target_recall=config.evaluation.target_recall,
    )
    model.threshold = float(selection["threshold"])
    model.validation_metrics = {
        **ranking_metrics(y_val, val_scores),
        **{k: v for k, v in selection.items() if isinstance(v, float)},
    }

    test_scores = model.predict_proba(X_test)
    model.test_metrics = {
        **ranking_metrics(y_test, test_scores),
        **threshold_metrics(y_test, test_scores, model.threshold),
        **bootstrap_metric(
            y_test,
            test_scores,
            "pr_auc",
            resamples=config.evaluation.bootstrap_resamples,
            seed=config.run.seed,
        ),
    }

    scored = split.test[["customer_id", "tx_day", "is_fraud"]].copy()
    scored["score"] = test_scores
    model.test_metrics.update(card_precision_at_k(scored, k=config.evaluation.top_k_per_day))
    if "fraud_scenario" in split.test.columns:
        # Aggregate recall averages the scenarios together; a model that only
        # learned the amount rule still scores respectably. Keep them apart.
        model.scenario_recall = recall_by_group(
            y_test, test_scores, split.test["fraud_scenario"].to_numpy(), model.threshold
        )
    model.latency = measure_latency(model, X_test)

    logger.info(
        "model_trained",
        model=name,
        fit_seconds=round(fit_seconds, 2),
        test_pr_auc=round(model.test_metrics["pr_auc"], 4),
        test_roc_auc=round(model.test_metrics["roc_auc"], 4),
        latency_p95_ms=round(model.latency.get("latency_p95_ms", 0.0), 3),
    )
    return model


def train_all(
    frame: pd.DataFrame, config: PipelineConfig, *, include_isolation_forest: bool = True
) -> tuple[list[TrainedModel], TemporalSplit]:
    """Train every enabled family on one split and return them ranked."""
    features = feature_names(config.features)
    split = split_by_day(frame, config.split)

    names = list(config.models.enabled)
    if include_isolation_forest:
        names.append("isolation_forest")
    if not names:
        raise ModelError("no models are enabled")

    models = [train_one(name, split, config, features) for name in names]
    primary = config.evaluation.primary_metric
    models.sort(key=lambda m: m.test_metrics.get(primary, 0.0), reverse=True)
    return models, split
