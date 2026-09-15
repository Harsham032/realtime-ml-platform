#!/usr/bin/env python
"""Train every model family, track the runs, and gate a promotion.

The full MLOps loop in one command:

1. train each enabled family on the same temporal split;
2. record parameters, metrics and models to MLflow;
3. select the best by the configured primary metric;
4. run it against the deployed champion through the promotion gates;
5. promote only if every gate passes, and write the serving bundle.

Nothing is promoted on a schedule. A challenger that fails a gate is recorded
with the reason and the champion is left alone.

Usage::

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/fast.yaml --no-promote
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtml.config import PipelineConfig, load_settings
from rtml.data.simulator import SimulationConfig, simulate
from rtml.data.store import PredictionStore
from rtml.features.engineering import build_features, feature_names
from rtml.logging_utils import configure_logging, get_logger
from rtml.models.registry import CHAMPION_ALIAS, ExperimentTracker, ModelRegistry
from rtml.training.pipeline import environment_info, train_all
from rtml.training.promotion import evaluate_promotion

logger = get_logger("train")

REGISTERED_MODEL = "fraud-scorer"
SERVING_BUNDLE = Path("artifacts/champion.joblib")


def load_or_generate(config: PipelineConfig, features_path: Path | None) -> pd.DataFrame:
    """Read a prepared feature table, or generate one."""
    if features_path and features_path.is_file():
        logger.info("features_loaded", path=str(features_path))
        return pd.read_parquet(features_path)
    result = simulate(SimulationConfig(**config.simulation.model_dump(), seed=config.run.seed))
    return build_features(result.transactions, config.features)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--no-promote", action="store_true", help="evaluate the gates but never promote"
    )
    parser.add_argument("--no-tracking", action="store_true", help="skip MLflow entirely")
    parser.add_argument("--skip-isolation-forest", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    overrides = dict(item.split("=", 1) for item in args.set)
    config = PipelineConfig.from_yaml(args.config).with_overrides(overrides)
    settings = load_settings()

    frame = load_or_generate(config, Path(args.features) if args.features else None)
    started = time.perf_counter()
    models, split = train_all(
        frame, config, include_isolation_forest=not args.skip_isolation_forest
    )
    train_seconds = time.perf_counter() - started

    tracker = None
    if not args.no_tracking:
        try:
            tracker = ExperimentTracker(
                settings.mlflow_tracking_uri,
                settings.mlflow_experiment,
                settings.mlflow_artifact_root,
            )
        except Exception as exc:
            # Tracking is valuable but not worth losing a training run over.
            logger.warning("tracking_unavailable", error=str(exc))

    model_uris: dict[str, str | None] = {}
    model_runs: dict[str, str] = {}
    if tracker is not None:
        for model in models:
            with tracker.run(
                f"{config.run.name}-{model.name}", tags={"model": model.name}
            ) as active:
                model_runs[model.name] = active.info.run_id
                tracker.log_params(
                    {
                        "model": model.name,
                        "seed": config.run.seed,
                        "features": len(model.features),
                        "train_rows": len(split.train),
                        "hyperparameters": getattr(config.models, model.name, {}),
                        **{f"split_{k}": v for k, v in split.boundaries.items()},
                    }
                )
                tracker.log_metrics(
                    {f"validation_{k}": v for k, v in model.validation_metrics.items()}
                )
                tracker.log_metrics({f"test_{k}": v for k, v in model.test_metrics.items()})
                tracker.log_metrics(model.latency)
                tracker.log_metrics({"fit_seconds": model.fit_seconds})
                tracker.log_dict(
                    {"features": model.features, "threshold": model.threshold}, "model_card.json"
                )
                model_uris[model.name] = tracker.log_model(model.estimator, model.name)

    best = models[0]
    champion_metrics = None
    registry = None
    if not args.no_tracking:
        try:
            registry = ModelRegistry(settings.mlflow_tracking_uri)
        except Exception as exc:
            logger.warning("registry_unavailable", error=str(exc))

    if SERVING_BUNDLE.is_file():
        import joblib

        champion_metrics = joblib.load(SERVING_BUNDLE).get("metrics")

    decision = evaluate_promotion(
        best.test_metrics,
        champion_metrics,
        config.promotion,
        challenger_latency_p95=best.latency.get("latency_p95_ms"),
        champion_latency_p95=(
            (champion_metrics or {}).get("latency_p95_ms") if champion_metrics else None
        ),
    )

    promoted_version = None
    if decision.promote and not args.no_promote:
        SERVING_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(
            {
                "model": best,
                "name": best.name,
                "version": time.strftime("%Y%m%d%H%M%S"),
                "threshold": best.threshold,
                "features": best.features,
                "metrics": {**best.test_metrics, **best.latency},
            },
            SERVING_BUNDLE,
        )
        if registry is not None and model_uris.get(best.name):
            try:
                version = registry.register(
                    model_uris[best.name], REGISTERED_MODEL, run_id=model_runs.get(best.name)
                )
                registry.promote(REGISTERED_MODEL, version)
                promoted_version = version
            except Exception as exc:
                # Surfaced rather than swallowed: a silent registry failure
                # leaves the serving bundle and the registry disagreeing about
                # what is deployed, which is worse than a noisy warning.
                logger.error("registration_failed", error=str(exc))
                print(f"WARNING: model registration failed: {exc}")

        store = PredictionStore(settings.database_url)
        store.create_all()
        store.record_deployment(
            REGISTERED_MODEL,
            promoted_version or "local",
            CHAMPION_ALIAS,
            best.threshold,
            pr_auc=best.test_metrics.get("pr_auc"),
            reason=decision.reason(),
        )
        store.close()

    output_dir = Path(args.output_dir or config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": environment_info(),
        "config": args.config,
        "split": split.describe(),
        "train_seconds": train_seconds,
        "promotion": decision.to_dict(),
        "promoted_version": promoted_version,
        "feature_names": feature_names(config.features),
        "models": [
            {
                "name": m.name,
                "threshold": m.threshold,
                "fit_seconds": m.fit_seconds,
                "validation": m.validation_metrics,
                "test": m.test_metrics,
                "latency": m.latency,
                "scenario_recall": m.scenario_recall,
            }
            for m in models
        ],
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    primary = config.evaluation.primary_metric
    header = f"{'model':22s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'p95 ms':>8s} {'fit s':>8s}"
    print(
        f"\nTrained on {len(split.train):,} rows, evaluated on {len(split.test):,} "
        f"({int(split.test['is_fraud'].sum()):,} frauds). Ranked by {primary}.\n"
    )
    print(header)
    print("-" * len(header))
    for model in models:
        t = model.test_metrics
        print(
            f"{model.name:22s} {t['pr_auc']:8.4f} {t['roc_auc']:8.4f} {t['precision']:7.3f} "
            f"{t['recall']:7.3f} {t['f1']:7.3f} {model.latency.get('latency_p95_ms', 0):8.3f} {model.fit_seconds:8.1f}"
        )
    if best.scenario_recall:
        print(f"\n{best.name} recall by fraud scenario (threshold {best.threshold:.4f}):")
        for scenario, stats in sorted(best.scenario_recall.items()):
            print(
                f"  scenario {scenario}: {stats['recall']:6.3f} "
                f"({int(stats['detected']):,}/{int(stats['frauds']):,} frauds)"
            )

    print(f"\nbest: {best.name}")
    print(
        f"promotion: {'PROMOTED' if promoted_version or (decision.promote and not args.no_promote) else 'HELD'} - {decision.reason()}"
    )
    for gate in decision.gates:
        print(f"  [{'pass' if gate.passed else 'FAIL'}] {gate.name}: {gate.detail}")
    print(f"\nreport: {output_dir / 'training_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
