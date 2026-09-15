"""Promotion gates.

Retraining on a schedule and shipping whatever comes out is how a pipeline
quietly replaces a working model with a worse one. Every gate here is a measured
comparison against the incumbent on the same evaluation window, and a candidate
must clear all of them.

The gates, and why each exists:

``absolute floor``
    The challenger must be better than a fixed minimum. Catches a training run
    that silently produced garbage - an empty feature, a corrupted split - which
    a purely relative comparison would pass if the champion were also broken.
``improvement margin``
    It must beat the champion by more than noise. Without a margin, a pipeline
    promotes on every random fluctuation and the model changes weekly for no
    reason.
``sample size``
    The evaluation window must contain enough fraud for the comparison to mean
    anything. At a 0.8 percent base rate a short window can hold a handful of
    positives, and PR-AUC on those is meaningless.
``latency``
    A better model that is too slow to serve is not better. Bounded regression
    rather than an absolute limit, since the budget belongs to the deployment.
``calibration``
    Scores feed a threshold, so a model whose probabilities drift lose their
    threshold's meaning even if its ranking improved. Brier score guards it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import PromotionConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Gate:
    """One pass/fail check with the numbers that decided it."""

    name: str
    passed: bool
    detail: str
    value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass
class PromotionDecision:
    """Whether a challenger may replace the champion, and why."""

    promote: bool
    gates: list[Gate] = field(default_factory=list)
    challenger_metrics: dict[str, float] = field(default_factory=dict)
    champion_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def failed(self) -> list[Gate]:
        return [gate for gate in self.gates if not gate.passed]

    def reason(self) -> str:
        if self.promote:
            return "all gates passed"
        return "; ".join(f"{gate.name}: {gate.detail}" for gate in self.failed)

    def to_dict(self) -> dict[str, object]:
        return {
            "promote": self.promote,
            "reason": self.reason(),
            "gates": [gate.to_dict() for gate in self.gates],
            "challenger": self.challenger_metrics,
            "champion": self.champion_metrics,
        }


def evaluate_promotion(
    challenger: dict[str, float],
    champion: dict[str, float] | None,
    config: PromotionConfig,
    *,
    challenger_latency_p95: float | None = None,
    champion_latency_p95: float | None = None,
) -> PromotionDecision:
    """Run every gate and return the decision with its evidence.

    ``champion`` of ``None`` means nothing is deployed yet. The relative gates
    are skipped in that case - there is nothing to compare against - but the
    absolute floor and sample-size gates still apply, so a broken first model
    is not promoted just because it is first.
    """
    gates: list[Gate] = []
    challenger_pr_auc = float(challenger.get("pr_auc", 0.0))

    gates.append(
        Gate(
            name="absolute_pr_auc",
            passed=challenger_pr_auc >= config.min_pr_auc,
            detail=f"PR-AUC {challenger_pr_auc:.4f} against floor {config.min_pr_auc:.4f}",
            value=challenger_pr_auc,
            threshold=config.min_pr_auc,
        )
    )

    frauds = float(challenger.get("positives", 0.0))
    gates.append(
        Gate(
            name="evaluation_sample_size",
            passed=frauds >= config.min_evaluation_frauds,
            detail=f"{int(frauds)} frauds in the evaluation window, minimum {config.min_evaluation_frauds}",
            value=frauds,
            threshold=float(config.min_evaluation_frauds),
        )
    )

    if champion is not None:
        champion_pr_auc = float(champion.get("pr_auc", 0.0))
        improvement = challenger_pr_auc - champion_pr_auc
        gates.append(
            Gate(
                name="improvement_margin",
                passed=improvement >= config.min_pr_auc_improvement,
                detail=(
                    f"PR-AUC improvement {improvement:+.4f}, "
                    f"minimum {config.min_pr_auc_improvement:+.4f}"
                ),
                value=improvement,
                threshold=config.min_pr_auc_improvement,
            )
        )

        if config.require_calibration:
            challenger_brier = float(challenger.get("brier", 1.0))
            champion_brier = float(champion.get("brier", 1.0))
            regression = challenger_brier - champion_brier
            gates.append(
                Gate(
                    name="calibration",
                    passed=regression <= config.max_brier_regression,
                    detail=(
                        f"Brier regression {regression:+.5f}, "
                        f"allowed {config.max_brier_regression:+.5f}"
                    ),
                    value=regression,
                    threshold=config.max_brier_regression,
                )
            )

    if challenger_latency_p95 is not None and champion_latency_p95 is not None:
        regression = challenger_latency_p95 - champion_latency_p95
        gates.append(
            Gate(
                name="latency",
                passed=regression <= config.max_latency_regression_ms,
                detail=(
                    f"p95 latency regression {regression:+.2f}ms, "
                    f"allowed {config.max_latency_regression_ms:+.2f}ms"
                ),
                value=regression,
                threshold=config.max_latency_regression_ms,
            )
        )

    decision = PromotionDecision(
        promote=all(gate.passed for gate in gates),
        gates=gates,
        challenger_metrics=challenger,
        champion_metrics=champion or {},
    )
    logger.info(
        "promotion_evaluated",
        promote=decision.promote,
        failed_gates=[gate.name for gate in decision.failed],
    )
    return decision


def should_retrain(
    drift_alert: bool,
    days_since_training: int,
    recent_pr_auc: float | None,
    deployed_pr_auc: float | None,
    *,
    max_age_days: int = 30,
    performance_drop: float = 0.05,
) -> tuple[bool, str]:
    """Decide whether to *start* a retraining run.

    Kept separate from promotion on purpose: deciding to retrain is cheap and
    can be triggered liberally, while deciding to deploy the result is
    expensive to get wrong and is gated hard. Conflating them is what produces
    pipelines that auto-deploy on every drift blip.
    """
    if recent_pr_auc is not None and deployed_pr_auc is not None:
        drop = deployed_pr_auc - recent_pr_auc
        if drop >= performance_drop:
            return True, f"measured PR-AUC dropped {drop:.4f} below the deployed model"
    if drift_alert:
        return True, "a monitored feature drifted significantly"
    if days_since_training >= max_age_days:
        return True, f"the deployed model is {days_since_training} days old"
    return False, "no trigger fired"
