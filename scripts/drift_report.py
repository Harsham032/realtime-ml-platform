#!/usr/bin/env python
"""Produce a drift report over stored predictions or a feature table.

Compares the most recent window against the reference window before it, on
every model input plus the score itself, and writes a machine-readable report
alongside a readable summary.

Usage::

    python scripts/drift_report.py --config configs/default.yaml
    python scripts/drift_report.py --input reports/stream_predictions.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtml.config import PipelineConfig, load_settings
from rtml.data.store import PredictionStore
from rtml.features.engineering import feature_names
from rtml.logging_utils import configure_logging, get_logger
from rtml.monitoring.drift import drift_over_time, performance_over_time
from rtml.training.promotion import should_retrain

logger = get_logger("drift_report")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default="data/processed/features.parquet")
    parser.add_argument(
        "--from-database", action="store_true", help="read stored predictions instead"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = PipelineConfig.from_yaml(args.config)

    if args.from_database:
        store = PredictionStore(load_settings().database_url)
        frame = store.read_predictions()
        store.close()
        if frame.empty:
            print("no predictions stored; run scripts/run_stream.py --write-database first")
            return 1
        monitored = [c for c in ("score", "tx_amount") if c in frame.columns]
    else:
        path = Path(args.input)
        if not path.is_file():
            print(f"no feature table at {path}; run scripts/generate_data.py first")
            return 1
        frame = pd.read_parquet(path)
        monitored = [name for name in feature_names(config.features) if name in frame.columns]

    report = drift_over_time(frame, monitored, config.drift)

    performance = pd.DataFrame()
    if {"score", "is_fraud"}.issubset(frame.columns):
        performance = performance_over_time(frame, threshold=0.5)

    retrain, reason = should_retrain(
        drift_alert=report.should_alert(),
        days_since_training=0,
        recent_pr_auc=None,
        deployed_pr_auc=None,
    )

    output_dir = Path(args.output_dir or config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {**report.to_dict(), "retrain_recommended": retrain, "retrain_reason": reason}
    (output_dir / "drift_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not performance.empty:
        performance.to_csv(output_dir / "performance_over_time.csv", index=False)

    print(
        f"\nreference days {report.reference_window}  ->  comparison days {report.comparison_window}"
    )
    print(
        f"{len(report.results)} features monitored, {len(report.drifted)} drifted, "
        f"{len(report.significant)} significant (max PSI {report.max_psi:.4f})\n"
    )
    header = (
        f"{'feature':28s} {'PSI':>9s} {'KS p':>11s} {'ref mean':>11s} {'new mean':>11s}  severity"
    )
    print(header)
    print("-" * len(header))
    for result in report.results[:15]:
        print(
            f"{result.feature:28s} {result.psi:9.4f} {result.ks_pvalue:11.3e} "
            f"{result.reference_mean:11.4f} {result.comparison_mean:11.4f}  {result.severity}"
        )
    print(f"\nalert: {report.should_alert()}   retrain recommended: {retrain} ({reason})")
    print(f"report: {output_dir / 'drift_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
