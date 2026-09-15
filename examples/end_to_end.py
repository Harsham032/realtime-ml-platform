#!/usr/bin/env python
"""End-to-end walkthrough: generate, train, stream, monitor.

Runs the whole platform on a small configuration in about a minute. No network,
no credentials, no external service.

    python examples/end_to_end.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtml.config import PipelineConfig
from rtml.data.simulator import SimulationConfig, simulate
from rtml.evaluation.metrics import ranking_metrics, threshold_metrics
from rtml.features.engineering import build_features
from rtml.logging_utils import configure_logging
from rtml.monitoring.drift import drift_over_time
from rtml.streaming import InProcessBroker, ScoringConsumer, publish_transactions
from rtml.training.pipeline import train_all
from rtml.training.promotion import evaluate_promotion, should_retrain


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    configure_logging("WARNING")
    root = Path(__file__).resolve().parents[1]
    config = PipelineConfig.from_yaml(root / "configs" / "fast.yaml").with_overrides(
        {
            "simulation.n_customers": 800,
            "simulation.n_terminals": 1600,
            "simulation.n_days": 80,
            "split.train_days": 49,
            "split.delay_days": 7,
            "split.test_days": 17,
            "split.validation_days": 12,
            "evaluation.bootstrap_resamples": 200,
        }
    )

    rule("1. Generate")
    started = time.perf_counter()
    result = simulate(SimulationConfig(**config.simulation.model_dump(), seed=config.run.seed))
    frame = result.transactions
    print(
        f"{len(frame):,} transactions in {time.perf_counter() - started:.1f}s  "
        f"fraud {frame['is_fraud'].mean():.3%} ({int(frame['is_fraud'].sum()):,} cases)"
    )
    by_scenario = frame[frame.is_fraud == 1]["fraud_scenario"].value_counts().sort_index()
    for scenario, count in by_scenario.items():
        print(f"    scenario {scenario}: {count:,}")

    rule("2. Features")
    started = time.perf_counter()
    featured = build_features(frame, config.features)
    print(
        f"15 features in {time.perf_counter() - started:.1f}s, no nulls: {featured.notna().all().all()}"
    )
    print(f"    terminal risk uses a {config.features.risk_delay_days}-day label delay")

    rule("3. Train")
    started = time.perf_counter()
    models, split = train_all(featured, config)
    print(f"{len(models)} models in {time.perf_counter() - started:.1f}s")
    print(
        f"    split: train {len(split.train):,} | val {len(split.validation):,} | "
        f"test {len(split.test):,} ({int(split.test['is_fraud'].sum()):,} frauds)"
    )
    print()
    header = (
        f"    {'model':22s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'prec':>6s} {'rec':>6s} {'p95 ms':>8s}"
    )
    print(header)
    for model in models:
        metrics = model.test_metrics
        print(
            f"    {model.name:22s} {metrics['pr_auc']:8.4f} {metrics['roc_auc']:8.4f} "
            f"{metrics['precision']:6.3f} {metrics['recall']:6.3f} "
            f"{model.latency.get('latency_p95_ms', 0):8.3f}"
        )
    best = models[0]
    print(f"\n    best: {best.name} (ranked by {config.evaluation.primary_metric})")
    print("    note: ROC-AUC barely separates these models; PR-AUC does. That gap is the point.")

    rule("4. Promotion gates")
    decision = evaluate_promotion(best.test_metrics, None, config.promotion)
    for gate in decision.gates:
        print(f"    [{'pass' if gate.passed else 'FAIL'}] {gate.name}: {gate.detail}")
    print(f"    -> {'promote' if decision.promote else 'hold'}")

    weaker = {**best.test_metrics, "pr_auc": best.test_metrics["pr_auc"] - 0.002}
    held = evaluate_promotion(weaker, best.test_metrics, config.promotion)
    print(f"    a 0.002 regression against this champion -> {held.reason()}")

    rule("5. Stream")
    broker = InProcessBroker(partitions=config.streaming.partitions)
    produced = publish_transactions(broker, split.test, "transactions", progress_every=0)
    consumer = ScoringConsumer(broker, best, config.features, threshold=best.threshold)
    consumer.warm_up(featured[featured["tx_day"] < split.boundaries["test_start_day"]])
    predictions = consumer.run(batch_size=config.streaming.batch_size)

    print(f"    produced {produced.published:,} at {produced.events_per_second:,.0f}/s")
    print(f"    scored   {consumer.stats.scored:,} at {consumer.stats.events_per_second:,.0f}/s")
    latency = consumer.stats.latency_summary()
    print(
        f"    latency  p50 {latency['scoring_latency_p50_ms']:.3f}ms  p95 {latency['scoring_latency_p95_ms']:.3f}ms"
    )
    print(f"    lag after draining: {broker.lag('transactions', consumer.group)}")

    labels = np.array([p["is_fraud"] for p in predictions])
    scores = np.array([p["score"] for p in predictions])
    streamed = {
        **ranking_metrics(labels, scores),
        **threshold_metrics(labels, scores, best.threshold),
    }
    print(
        f"    streamed PR-AUC {streamed['pr_auc']:.4f} vs batch {best.test_metrics['pr_auc']:.4f}"
    )

    rule("6. Drift")
    report = drift_over_time(
        featured, ["tx_amount", "customer_nb_tx_7d", "terminal_risk_7d"], config.drift
    )
    print(f"    days {report.reference_window} -> {report.comparison_window}")
    for item in report.results:
        print(
            f"    {item.feature:24s} PSI {item.psi:7.4f}  KS p {item.ks_pvalue:9.2e}  {item.severity}"
        )
    fired, reason = should_retrain(
        report.should_alert(), days_since_training=3, recent_pr_auc=None, deployed_pr_auc=None
    )
    print(f"    retrain recommended: {fired} ({reason})")

    rule("Done")
    print("    Full-scale measured results are in docs/results.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
