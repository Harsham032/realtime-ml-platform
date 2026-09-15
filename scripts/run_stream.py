#!/usr/bin/env python
"""Run the streaming path: produce transactions, score them, store predictions.

Exercises the same code on either broker. With ``--backend inprocess`` it needs
no external service, which is what makes the throughput and latency figures in
``docs/results.md`` reproducible from a clean checkout.

The consumer is warmed from history before serving by default. A cold consumer
has empty feature windows and scores measurably worse until they fill - see
``docs/results.md`` for how much - so warming is the realistic configuration and
``--cold`` exists to demonstrate the difference.

Usage::

    python scripts/run_stream.py --config configs/fast.yaml
    python scripts/run_stream.py --backend kafka --speedup 3600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtml.config import PipelineConfig, load_settings
from rtml.data.simulator import SimulationConfig, simulate
from rtml.data.store import PredictionStore
from rtml.evaluation.metrics import ranking_metrics, threshold_metrics
from rtml.features.engineering import build_features
from rtml.logging_utils import configure_logging, get_logger
from rtml.streaming import ScoringConsumer, build_broker, publish_transactions
from rtml.training.pipeline import train_all
from rtml.training.splitting import split_by_day

logger = get_logger("run_stream")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/fast.yaml")
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--backend", choices=["inprocess", "kafka"], default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--speedup", type=float, default=None, help="0 replays as fast as possible")
    parser.add_argument("--cold", action="store_true", help="do not warm the feature store first")
    parser.add_argument("--write-database", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = PipelineConfig.from_yaml(args.config)
    settings = load_settings()
    backend = args.backend or config.streaming.backend
    speedup = config.streaming.speedup if args.speedup is None else args.speedup

    features_path = Path(args.features)
    if features_path.is_file():
        frame = pd.read_parquet(features_path)
    else:
        result = simulate(SimulationConfig(**config.simulation.model_dump(), seed=config.run.seed))
        frame = build_features(result.transactions, config.features)

    split = split_by_day(frame, config.split)

    bundle_path = Path("artifacts/champion.joblib")
    if bundle_path.is_file():
        import joblib

        bundle = joblib.load(bundle_path)
        model, threshold, model_name = bundle["model"], bundle["threshold"], bundle["name"]
        logger.info("model_loaded", path=str(bundle_path), model=model_name)
    else:
        logger.info("no_champion_bundle_training_one")
        models, _ = train_all(frame, config, include_isolation_forest=False)
        model, threshold, model_name = models[0], models[0].threshold, models[0].name

    broker = build_broker(backend, settings, partitions=config.streaming.partitions)
    topic_in = settings.kafka_transactions_topic
    topic_out = settings.kafka_predictions_topic

    produced = publish_transactions(
        broker, split.test, topic_in, speedup=speedup, max_events=args.max_events
    )

    consumer = ScoringConsumer(
        broker,
        model,
        config.features,
        threshold=threshold,
        input_topic=topic_in,
        output_topic=topic_out,
        group=settings.kafka_consumer_group,
    )
    warmed = 0
    if not args.cold:
        history = frame[frame["tx_day"] < split.boundaries["test_start_day"]]
        started = time.perf_counter()
        warmed = consumer.warm_up(history)
        logger.info(
            "feature_store_warmed", rows=warmed, seconds=round(time.perf_counter() - started, 2)
        )

    predictions = consumer.run(
        max_events=args.max_events,
        batch_size=config.streaming.batch_size,
        poll_timeout=config.streaming.poll_timeout_seconds,
    )

    if args.write_database and predictions:
        store = PredictionStore(settings.database_url)
        store.create_all()
        store.write_predictions(predictions, model_name=model_name)
        store.close()

    labels = np.array([p["is_fraud"] for p in predictions])
    scores = np.array([p["score"] for p in predictions])
    quality = {**ranking_metrics(labels, scores), **threshold_metrics(labels, scores, threshold)}

    summary = {
        "backend": backend,
        "warmed_rows": warmed,
        "produced": produced.published,
        "producer_events_per_second": produced.events_per_second,
        "scored": consumer.stats.scored,
        "alerts": consumer.stats.alerts,
        "errors": consumer.stats.errors,
        "late_labels": consumer.stats.late_labels,
        "consumer_events_per_second": consumer.stats.events_per_second,
        **consumer.stats.latency_summary(),
        "quality": quality,
        "feature_state": consumer.store.state_size(),
    }
    output_dir = Path(args.output_dir or config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stream_report.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nbackend={backend}  warmed={warmed:,} rows")
    print(f"produced {produced.published:,} events at {produced.events_per_second:,.0f}/s")
    print(
        f"scored   {consumer.stats.scored:,} events at {consumer.stats.events_per_second:,.0f}/s "
        f"({consumer.stats.alerts:,} alerts, {consumer.stats.errors} errors, "
        f"{consumer.stats.late_labels} labels past the reordering allowance)"
    )
    for key, value in consumer.stats.latency_summary().items():
        print(f"  {key:32s} {value:8.3f}")
    print(
        f"\nstreamed quality: PR-AUC {quality['pr_auc']:.4f}  precision {quality['precision']:.3f}  "
        f"recall {quality['recall']:.3f}"
    )
    print(f"report: {output_dir / 'stream_report.json'}")
    broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
