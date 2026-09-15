"""Transaction event producer.

Replays a transaction frame onto the broker in timestamp order, keyed by
customer id so every event for a customer lands on one partition.

``speedup`` controls pacing. Zero replays as fast as the broker accepts, which
is what throughput measurement wants. A positive value replays in simulated real
time compressed by that factor - ``speedup=3600`` plays an hour of traffic per
second - which is what you want when watching the monitoring dashboards behave.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..logging_utils import get_logger
from .broker import Broker

logger = get_logger(__name__)

EVENT_COLUMNS = (
    "transaction_id",
    "customer_id",
    "terminal_id",
    "tx_datetime",
    "tx_amount",
    "tx_day",
    "is_fraud",
)


@dataclass
class ProducerStats:
    published: int = 0
    elapsed_seconds: float = 0.0

    @property
    def events_per_second(self) -> float:
        return self.published / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


def to_event(row: Any) -> dict[str, Any]:
    """Serialise one transaction into a broker event.

    The label travels with the event only because this is a replay of labelled
    historical data used to drive and score the pipeline offline. A production
    producer publishes no label - it does not have one yet - and the consumer
    never reads it for scoring; see ``consumer.py``.
    """
    timestamp = pd.Timestamp(row.tx_datetime)
    return {
        "transaction_id": int(row.transaction_id),
        "customer_id": int(row.customer_id),
        "terminal_id": int(row.terminal_id),
        "tx_datetime": timestamp.isoformat(),
        "tx_amount": float(row.tx_amount),
        "tx_day": int(row.tx_day),
        "is_fraud": int(getattr(row, "is_fraud", 0)),
    }


def publish_transactions(
    broker: Broker,
    frame: pd.DataFrame,
    topic: str,
    *,
    speedup: float = 0.0,
    max_events: int | None = None,
    progress_every: int = 100_000,
) -> ProducerStats:
    """Publish ``frame`` to ``topic`` in timestamp order."""
    ordered = frame.sort_values("tx_datetime", kind="stable")
    if max_events is not None:
        ordered = ordered.head(max_events)

    stats = ProducerStats()
    start = time.perf_counter()
    previous: pd.Timestamp | None = None

    for row in ordered.itertuples():
        if speedup > 0 and previous is not None:
            gap = (pd.Timestamp(row.tx_datetime) - previous).total_seconds() / speedup
            if gap > 0:
                time.sleep(min(gap, 1.0))
        previous = pd.Timestamp(row.tx_datetime)

        broker.produce(topic, key=str(int(row.customer_id)), value=to_event(row))
        stats.published += 1
        if progress_every and stats.published % progress_every == 0:
            logger.info("producer_progress", published=stats.published)

    stats.elapsed_seconds = time.perf_counter() - start
    logger.info(
        "producer_finished",
        published=stats.published,
        seconds=round(stats.elapsed_seconds, 2),
        events_per_second=round(stats.events_per_second, 1),
    )
    return stats
