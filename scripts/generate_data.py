#!/usr/bin/env python
"""Generate a labelled transaction dataset and its features.

Nothing is downloaded: the dataset is produced from a seed, so the same command
yields the same data on any machine. See ``docs/data-sources.md`` for the
generation procedure and what a simulator can and cannot tell you.

Usage::

    python scripts/generate_data.py --config configs/default.yaml
    python scripts/generate_data.py --config configs/fast.yaml --output data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtml.config import PipelineConfig, load_settings
from rtml.data.simulator import SimulationConfig, simulate
from rtml.data.store import PredictionStore
from rtml.features.engineering import build_features
from rtml.logging_utils import configure_logging, get_logger

logger = get_logger("generate_data")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--output", default="data/processed", help="where to write the parquet files"
    )
    parser.add_argument(
        "--write-database", action="store_true", help="also insert transactions into the database"
    )
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="configuration overrides"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    overrides = dict(item.split("=", 1) for item in args.set)
    config = PipelineConfig.from_yaml(args.config).with_overrides(overrides)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    result = simulate(SimulationConfig(**config.simulation.model_dump(), seed=config.run.seed))
    simulate_seconds = time.perf_counter() - started

    started = time.perf_counter()
    features = build_features(result.transactions, config.features)
    feature_seconds = time.perf_counter() - started

    result.transactions.to_parquet(output / "transactions.parquet", index=False)
    features.to_parquet(output / "features.parquet", index=False)
    result.customers.to_parquet(output / "customers.parquet", index=False)
    result.terminals.to_parquet(output / "terminals.parquet", index=False)

    if args.write_database:
        store = PredictionStore(load_settings().database_url)
        store.create_all()
        store.write_transactions(result.transactions)
        store.close()

    summary = {
        "config": args.config,
        "seed": config.run.seed,
        "stats": result.stats,
        "timings": {"simulate_seconds": simulate_seconds, "features_seconds": feature_seconds},
        "outputs": [str(output / name) for name in ("transactions.parquet", "features.parquet")],
    }
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
