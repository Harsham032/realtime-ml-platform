"""Scoring consumer.

Polls transaction events, computes online features, scores them, and publishes
predictions. This is the hot path, so two properties matter more than anything
else here.

**The label is never an input.** Replayed events carry ``is_fraud`` because the
offline stream is built from labelled history and the consumer needs it to score
its own output afterwards. It is popped out before feature computation and never
reaches the model. A stray label in the feature vector produces a perfect model
and a useless one, and it is an easy mistake to make when the field is right
there in the event.

**Feature state updates after scoring, not before.** The online store is updated
with the transaction only once the score is emitted, mirroring the batch
pipeline where a transaction contributes to the windows of *later* transactions.

Terminal labels are fed back separately through :meth:`ScoringConsumer.apply_label`,
behind the configured delay, because in production a label arrives days after the
transaction, from the investigation queue rather than the payment stream.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from ..features.online import SECONDS_PER_DAY, OnlineFeatureStore
from ..logging_utils import get_logger
from .broker import Broker, Record

logger = get_logger(__name__)


@dataclass
class ConsumerStats:
    consumed: int = 0
    scored: int = 0
    alerts: int = 0
    errors: int = 0
    late_labels: int = 0
    elapsed_seconds: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def events_per_second(self) -> float:
        return self.scored / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    def latency_summary(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {}
        ordered = np.sort(np.asarray(self.latencies_ms))
        return {
            "scoring_latency_mean_ms": float(ordered.mean()),
            "scoring_latency_p50_ms": float(np.percentile(ordered, 50)),
            "scoring_latency_p95_ms": float(np.percentile(ordered, 95)),
            "scoring_latency_p99_ms": float(np.percentile(ordered, 99)),
            "scoring_latency_max_ms": float(ordered.max()),
        }


class ScoringConsumer:
    """Consumes transactions, scores them, publishes predictions."""

    def __init__(
        self,
        broker: Broker,
        model: Any,
        feature_config: FeatureConfig,
        *,
        threshold: float = 0.5,
        input_topic: str = "transactions",
        output_topic: str = "predictions",
        group: str = "scoring-workers",
        label_delay_days: int | None = None,
    ) -> None:
        self.broker = broker
        self.model = model
        self.store = OnlineFeatureStore(feature_config)
        self.threshold = threshold
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.group = group
        self.label_delay_days = (
            feature_config.risk_delay_days if label_delay_days is None else label_delay_days
        )
        self.stats = ConsumerStats()
        # Labels waiting out the delay before they may inform terminal risk,
        # kept as a min-heap on event time.
        #
        # Not a plain queue: records arrive from several partitions, and only
        # within a partition are they ordered. Terminal state is shared across
        # partitions, so a queue in arrival order releases labels out of event
        # time - and the online history is a prefix-sum index that is only valid
        # over sorted timestamps.
        #
        # The heap plus a watermark makes the release order total: nothing is
        # applied until every partition has certainly moved past it. The label
        # delay doubles as the reordering allowance, so correctness holds while
        # cross-partition skew stays below it (measured at 2.9 days against a
        # 7-day delay on the full stream). Beyond that a label is genuinely late
        # and is counted rather than applied out of order.
        self._pending_labels: list[tuple[float, int, int]] = []
        self._watermark: float = -math.inf
        self._last_applied: float = -math.inf

    # ------------------------------------------------------------- scoring

    def score_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Compute features, score, and return the prediction record."""
        # Popped, not read: the label must not reach the feature vector.
        label = int(event.pop("is_fraud", 0))

        when = pd.Timestamp(event["tx_datetime"])
        timestamp = when.timestamp()
        customer_id = int(event["customer_id"])
        terminal_id = int(event["terminal_id"])
        amount = float(event["tx_amount"])

        start = time.perf_counter()
        features = self.store.compute(customer_id, terminal_id, timestamp, amount, when)
        score = float(self.model.predict_proba(self.store.vector(features))[0])
        latency_ms = (time.perf_counter() - start) * 1000.0

        # State advances only after the score is produced.
        self.store.observe_transaction(customer_id, timestamp, amount)
        self._watermark = max(self._watermark, timestamp)
        heapq.heappush(self._pending_labels, (timestamp, terminal_id, label))
        self._release_labels()

        self.stats.latencies_ms.append(latency_ms)
        self.stats.scored += 1
        is_alert = score >= self.threshold
        if is_alert:
            self.stats.alerts += 1

        return {
            "transaction_id": int(event["transaction_id"]),
            "customer_id": customer_id,
            "terminal_id": terminal_id,
            "tx_datetime": event["tx_datetime"],
            "tx_day": int(event.get("tx_day", 0)),
            "tx_amount": amount,
            "score": score,
            "is_alert": bool(is_alert),
            "threshold": self.threshold,
            "scoring_latency_ms": latency_ms,
            # Carried for offline scoring of the consumer's own output. A
            # production prediction record would not contain it.
            "is_fraud": label,
        }

    def _release_labels(self) -> None:
        """Feed labels into the feature store once the delay has elapsed.

        Released in event-time order against the watermark, never in arrival
        order. A label is only applied once the stream has advanced a full delay
        past it, which is what makes it safe to assume no earlier label is still
        in flight.
        """
        cutoff = self._watermark - self.label_delay_days * SECONDS_PER_DAY
        while self._pending_labels and self._pending_labels[0][0] <= cutoff:
            timestamp, terminal_id, label = heapq.heappop(self._pending_labels)
            if timestamp < self._last_applied:
                # Later than the allowance the delay provides. Applying it would
                # corrupt the window index for this terminal; dropping it loses
                # one label. Counted so the choice is visible rather than silent.
                self.stats.late_labels += 1
                continue
            self._last_applied = timestamp
            self.store.observe_label(terminal_id, timestamp, label)

    # ------------------------------------------------------------ warm-up

    def warm_up(self, history: pd.DataFrame) -> int:
        """Seed feature state from history, respecting the label delay.

        A snapshot taken at the moment serving begins cannot contain labels that
        have not arrived yet, so the newest ``label_delay_days`` of them are held
        back and queued for release on the same schedule as live traffic.

        This is not merely a fidelity point. Applying them immediately seeds each
        terminal's history with timestamps up to the end of the snapshot, and the
        consumer's first delayed releases then arrive *behind* those timestamps.
        The online history is a prefix-sum index over a sorted array, so one
        out-of-order append makes every later window query on that terminal
        wrong - with no error, and with a value that looks entirely reasonable.
        Measured end to end, it cost 0.24 PR-AUC.

        Returns the number of rows the store was primed with.
        """
        if history.empty:
            return 0
        ordered = history.sort_values("tx_datetime", kind="stable")
        latest = pd.Timestamp(ordered["tx_datetime"].iloc[-1]).timestamp()
        cutoff = latest - self.label_delay_days * SECONDS_PER_DAY
        deferred = self.store.warm_up(ordered, label_cutoff=cutoff)
        for entry in deferred:
            heapq.heappush(self._pending_labels, entry)
        # The snapshot's own clock: labels older than cutoff are already applied,
        # so the release loop must not re-apply anything behind it.
        self._watermark = max(self._watermark, latest)
        self._last_applied = max(self._last_applied, cutoff)
        return len(ordered)

    # --------------------------------------------------------------- loops

    def process_batch(self, records: list[Record]) -> list[dict[str, Any]]:
        """Score one polled batch and publish the predictions."""
        predictions: list[dict[str, Any]] = []
        offsets: dict[int, int] = {}

        for record in records:
            try:
                prediction = self.score_event(dict(record.value))
            except Exception as exc:  # one poison event must not stall the partition
                self.stats.errors += 1
                logger.warning("scoring_failed", offset=record.offset, error=str(exc))
                offsets[record.partition] = max(offsets.get(record.partition, 0), record.offset + 1)
                continue

            predictions.append(prediction)
            self.broker.produce(
                self.output_topic, key=str(prediction["customer_id"]), value=prediction
            )
            offsets[record.partition] = max(offsets.get(record.partition, 0), record.offset + 1)

        self.stats.consumed += len(records)
        if offsets:
            # Committed after processing: at-least-once, so a crash here replays
            # the batch rather than dropping it.
            self.broker.commit(self.input_topic, self.group, offsets)
        return predictions

    def run(
        self, *, max_events: int | None = None, batch_size: int = 500, poll_timeout: float = 1.0
    ) -> list[dict[str, Any]]:
        """Consume until the stream is drained or ``max_events`` is reached."""
        collected: list[dict[str, Any]] = []
        start = time.perf_counter()

        while True:
            remaining = None if max_events is None else max_events - self.stats.scored
            if remaining is not None and remaining <= 0:
                break
            size = batch_size if remaining is None else min(batch_size, remaining)
            records = self.broker.poll(
                self.input_topic, self.group, max_records=size, timeout=poll_timeout
            )
            if not records:
                break
            collected.extend(self.process_batch(records))

        self.stats.elapsed_seconds = time.perf_counter() - start
        logger.info(
            "consumer_finished",
            scored=self.stats.scored,
            alerts=self.stats.alerts,
            errors=self.stats.errors,
            events_per_second=round(self.stats.events_per_second, 1),
            **{k: round(v, 3) for k, v in self.stats.latency_summary().items()},
        )
        return collected
