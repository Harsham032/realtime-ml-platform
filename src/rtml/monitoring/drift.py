"""Drift detection.

Three things drift, and they mean different things:

*Feature drift* - the inputs move. Often benign (a marketing campaign shifts the
amount distribution) but it is the earliest available warning.

*Prediction drift* - the score distribution moves. More actionable than feature
drift because it is what the downstream alert queue actually sees. A model whose
mean score doubles will double the alert volume whether or not fraud changed.

*Performance drift* - precision and recall move. The only one that directly
measures harm, and the one you learn about last, because it needs labels and
labels arrive days late. That delay is precisely why the other two are
monitored: they are leading indicators for something you cannot yet measure.

Two complementary tests are used. The Population Stability Index summarises how
far a distribution has moved on an interpretable scale with conventional action
bands. The two-sample Kolmogorov-Smirnov test asks whether the move is larger
than sampling noise. PSI without a significance test flags noise on small
windows; KS without an effect size flags trivial moves as significant on large
ones, because with a million rows almost everything is significant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from ..config import DriftConfig
from ..errors import EvaluationError
from ..logging_utils import get_logger

logger = get_logger(__name__)

Severity = Literal["stable", "moderate", "significant"]


@dataclass
class DriftResult:
    """Drift for one feature between a reference and a comparison window."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    severity: Severity
    reference_mean: float
    comparison_mean: float
    reference_count: int
    comparison_count: int

    @property
    def mean_shift(self) -> float:
        return self.comparison_mean - self.reference_mean

    def to_dict(self) -> dict[str, float | str]:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 6),
            "ks_statistic": round(self.ks_statistic, 6),
            "ks_pvalue": round(self.ks_pvalue, 6),
            "severity": self.severity,
            "reference_mean": round(self.reference_mean, 6),
            "comparison_mean": round(self.comparison_mean, 6),
            "mean_shift": round(self.mean_shift, 6),
            "reference_count": self.reference_count,
            "comparison_count": self.comparison_count,
        }


@dataclass
class DriftReport:
    """Drift across every monitored feature, plus the summary an alert uses."""

    results: list[DriftResult] = field(default_factory=list)
    reference_window: tuple[int, int] = (0, 0)
    comparison_window: tuple[int, int] = (0, 0)

    @property
    def drifted(self) -> list[DriftResult]:
        return [r for r in self.results if r.severity != "stable"]

    @property
    def significant(self) -> list[DriftResult]:
        return [r for r in self.results if r.severity == "significant"]

    @property
    def max_psi(self) -> float:
        return max((r.psi for r in self.results), default=0.0)

    def should_alert(self) -> bool:
        return bool(self.significant)

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_window": list(self.reference_window),
            "comparison_window": list(self.comparison_window),
            "features_monitored": len(self.results),
            "features_drifted": len(self.drifted),
            "features_significant": len(self.significant),
            "max_psi": round(self.max_psi, 6),
            "alert": self.should_alert(),
            "features": [r.to_dict() for r in self.results],
        }


def population_stability_index(
    reference: np.ndarray, comparison: np.ndarray, *, bins: int = 10
) -> float:
    """PSI between two samples.

    Bin edges come from the reference distribution's quantiles, so bins carry
    roughly equal reference mass and a shift shows up wherever it happens rather
    than only in a crowded region. Empty bins are floored to a small epsilon
    because the formula takes a logarithm of the ratio and a genuinely empty
    comparison bin would otherwise return infinity.
    """
    if len(reference) == 0 or len(comparison) == 0:
        raise EvaluationError("PSI needs a non-empty reference and comparison sample")

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if len(edges) < 2:
        # A constant reference feature cannot drift in distribution; report the
        # change as binary rather than dividing by zero.
        return 0.0 if np.allclose(comparison, reference[0]) else 1.0

    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts, _ = np.histogram(reference, bins=edges)
    comparison_counts, _ = np.histogram(comparison, bins=edges)

    epsilon = 1e-6
    reference_share = np.maximum(reference_counts / len(reference), epsilon)
    comparison_share = np.maximum(comparison_counts / len(comparison), epsilon)
    return float(
        np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share))
    )


def classify(psi: float, pvalue: float, config: DriftConfig) -> Severity:
    """Combine effect size and significance into one action band.

    A move is only called significant when it is both large (PSI above the alert
    band) and unlikely to be noise (KS below the p-value threshold). Either
    alone produces alerts nobody trusts.
    """
    if psi >= config.psi_alert and pvalue < config.ks_alert_pvalue:
        return "significant"
    if psi >= config.psi_warn:
        return "moderate"
    return "stable"


def compare_distributions(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    features: list[str],
    config: DriftConfig,
) -> DriftReport:
    """Drift for every feature between two frames."""
    report = DriftReport()
    for feature in features:
        if feature not in reference.columns or feature not in comparison.columns:
            continue
        reference_values = reference[feature].to_numpy(dtype=np.float64)
        comparison_values = comparison[feature].to_numpy(dtype=np.float64)
        if len(reference_values) == 0 or len(comparison_values) == 0:
            continue

        psi = population_stability_index(reference_values, comparison_values, bins=config.bins)
        ks_statistic, ks_pvalue = stats.ks_2samp(reference_values, comparison_values)
        report.results.append(
            DriftResult(
                feature=feature,
                psi=psi,
                ks_statistic=float(ks_statistic),
                ks_pvalue=float(ks_pvalue),
                severity=classify(psi, float(ks_pvalue), config),
                reference_mean=float(reference_values.mean()),
                comparison_mean=float(comparison_values.mean()),
                reference_count=len(reference_values),
                comparison_count=len(comparison_values),
            )
        )

    report.results.sort(key=lambda r: r.psi, reverse=True)
    logger.info(
        "drift_computed",
        monitored=len(report.results),
        drifted=len(report.drifted),
        significant=len(report.significant),
        max_psi=round(report.max_psi, 4),
    )
    return report


def drift_over_time(
    frame: pd.DataFrame,
    features: list[str],
    config: DriftConfig,
    *,
    day_column: str = "tx_day",
) -> DriftReport:
    """Compare the most recent window against the reference window before it."""
    if day_column not in frame.columns:
        raise EvaluationError(f"the frame has no {day_column} column")

    last_day = int(frame[day_column].max())
    comparison_start = last_day - config.comparison_days + 1
    reference_start = comparison_start - config.reference_days

    reference = frame[
        (frame[day_column] >= reference_start) & (frame[day_column] < comparison_start)
    ]
    comparison = frame[frame[day_column] >= comparison_start]
    if len(reference) == 0 or len(comparison) == 0:
        raise EvaluationError(
            f"not enough history: need {config.reference_days + config.comparison_days} days"
        )

    report = compare_distributions(reference, comparison, features, config)
    report.reference_window = (reference_start, comparison_start - 1)
    report.comparison_window = (comparison_start, last_day)
    return report


def performance_over_time(
    frame: pd.DataFrame,
    *,
    score_column: str = "score",
    label_column: str = "is_fraud",
    day_column: str = "tx_day",
    threshold: float = 0.5,
    window_days: int = 7,
) -> pd.DataFrame:
    """Rolling precision, recall and alert volume by period.

    The lagging indicator: it needs labels, so in production every row here is
    only computable once the investigation queue has caught up.
    """
    from ..evaluation.metrics import threshold_metrics

    rows: list[dict[str, float]] = []
    days = frame[day_column]
    for start in range(int(days.min()), int(days.max()) + 1, window_days):
        window = frame[(days >= start) & (days < start + window_days)]
        if len(window) == 0 or window[label_column].sum() == 0:
            continue
        metrics = threshold_metrics(
            window[label_column].to_numpy(), window[score_column].to_numpy(), threshold
        )
        rows.append(
            {
                "window_start_day": float(start),
                "window_end_day": float(start + window_days - 1),
                "transactions": float(len(window)),
                "frauds": float(window[label_column].sum()),
                "mean_score": float(window[score_column].mean()),
                **{k: v for k, v in metrics.items() if k != "threshold"},
            }
        )
    return pd.DataFrame(rows)
