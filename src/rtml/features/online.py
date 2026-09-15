"""Online feature computation.

The batch pipeline computes trailing windows with a groupby over the whole
history. A streaming consumer cannot: it sees one event at a time and must
produce the same feature vector in microseconds, without the future and without
rescanning the past.

This module keeps, per entity, a bounded deque of ``(timestamp, value)`` pairs
covering the longest window it needs, and evicts from the left as time advances.
Each update is amortised O(1) in the number of events that have fallen out of
the window.

**Training/serving skew is the failure mode this design exists to prevent.** A
model trained on batch features and served online features that differ even
slightly degrades in ways that look like data drift and get misdiagnosed for
weeks. ``tests/test_features.py`` replays a transaction stream through
both paths and asserts the vectors match exactly, which is the only way to know
they agree.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from ..errors import FeatureError
from ..features.engineering import feature_names

SECONDS_PER_DAY = 86_400


class _EntityHistory:
    """Timestamped values for one entity, with O(log n) window queries.

    The obvious implementation - a deque scanned backwards until the window is
    exhausted - is O(events in window) per query. With a 30-day window, three
    window sizes and two entity types that is several hundred Python-level
    iterations per transaction, which measured at roughly 11ms and capped the
    consumer near 90 events a second.

    This keeps parallel arrays plus a prefix sum, so a window query is two
    binary searches and a subtraction. Eviction advances a head index rather
    than shifting the arrays, and the arrays are compacted only when the dead
    prefix grows past half, which keeps append amortised O(1).
    """

    __slots__ = ("_timestamps", "_values", "_prefix", "_head")

    def __init__(self) -> None:
        self._timestamps: list[float] = []
        self._values: list[float] = []
        # _prefix[i] is the sum of the first i values, so a range sum is a
        # single subtraction.
        self._prefix: list[float] = [0.0]
        self._head: int = 0

    def __len__(self) -> int:
        return len(self._timestamps) - self._head

    def append(self, timestamp: float, value: float) -> None:
        """Add an observation. Timestamps must not go backwards.

        The whole structure rests on ``_timestamps`` being sorted: both window
        bounds are found with ``bisect_right``. One out-of-order append breaks
        that invariant and every later window query on this entity returns a
        wrong answer with no error and a plausible value. The check is a single
        comparison, and it converts a silent corruption into a stack trace.
        """
        if self._timestamps and timestamp < self._timestamps[-1]:
            raise FeatureError(
                "online history must be appended in non-decreasing timestamp order; "
                f"got {timestamp} after {self._timestamps[-1]}"
            )
        self._timestamps.append(timestamp)
        self._values.append(value)
        self._prefix.append(self._prefix[-1] + value)

    def evict_before(self, cutoff: float) -> None:
        """Drop entries at or before ``cutoff``.

        Strictly before is wrong: the batch windows are half-open,
        ``(t - window, t]``, so an entry exactly ``window`` old is outside.
        """
        index = bisect_right(self._timestamps, cutoff, self._head)
        self._head = index
        # Compact once the dead prefix dominates, so memory tracks live events.
        if self._head > 1024 and self._head * 2 > len(self._timestamps):
            self._timestamps = self._timestamps[self._head :]
            self._values = self._values[self._head :]
            self._prefix = [0.0]
            running = 0.0
            for value in self._values:
                running += value
                self._prefix.append(running)
            self._head = 0

    def window_sum_count(self, now: float, window_seconds: float) -> tuple[float, int]:
        """Sum and count over ``(now - window, now]``."""
        cutoff = now - window_seconds
        start = bisect_right(self._timestamps, cutoff, self._head)
        end = bisect_right(self._timestamps, now, start)
        if end <= start:
            return 0.0, 0
        return self._prefix[end] - self._prefix[start], end - start


class OnlineFeatureStore:
    """Incremental feature computation for a single transaction stream.

    Not thread-safe by design: one instance belongs to one consumer partition,
    which is what keeps the update path lock-free. Partitioning by customer id
    means a customer's history always lands on the same partition.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.names = feature_names(config)

        self._customer_amounts: dict[int, _EntityHistory] = defaultdict(_EntityHistory)
        self._terminal_labels: dict[int, _EntityHistory] = defaultdict(_EntityHistory)

        self._max_customer_window = max(config.customer_windows) * SECONDS_PER_DAY
        self._max_terminal_window = (
            max(config.terminal_windows) + config.risk_delay_days
        ) * SECONDS_PER_DAY

    # ----------------------------------------------------------------- state

    def observe_transaction(self, customer_id: int, timestamp: float, amount: float) -> None:
        """Record a transaction so later events can see it.

        Called after scoring, never before: including the current transaction in
        its own customer history before computing features would double-count
        it, since :meth:`compute` already adds it explicitly.
        """
        history = self._customer_amounts[customer_id]
        history.append(timestamp, amount)
        history.evict_before(timestamp - self._max_customer_window)

    def observe_label(self, terminal_id: int, timestamp: float, is_fraud: int) -> None:
        """Record a confirmed fraud label for a terminal.

        In production this is driven by the investigation feedback loop rather
        than by the transaction stream, which is exactly why the label carries
        its own timestamp: it is the time of the *transaction*, not of the
        confirmation.
        """
        history = self._terminal_labels[terminal_id]
        history.append(timestamp, float(is_fraud))
        history.evict_before(timestamp - self._max_terminal_window)

    # -------------------------------------------------------------- features

    def compute(
        self,
        customer_id: int,
        terminal_id: int,
        timestamp: float,
        amount: float,
        when: pd.Timestamp,
    ) -> dict[str, float]:
        """Feature vector for one transaction, in the batch pipeline's order."""
        features: dict[str, float] = {
            "tx_amount": float(amount),
            "tx_during_weekend": float(when.dayofweek >= 5),
            "tx_during_night": float(self._is_night(when.hour)),
        }

        customer = self._customer_amounts.get(customer_id)
        for window in self.config.customer_windows:
            seconds = window * SECONDS_PER_DAY
            total, count = customer.window_sum_count(timestamp, seconds) if customer else (0.0, 0)
            # The transaction being scored is part of its own window: its amount
            # is known at scoring time, and the batch pipeline includes it too.
            total += float(amount)
            count += 1
            features[f"customer_nb_tx_{window}d"] = float(count)
            features[f"customer_avg_amount_{window}d"] = float(total / count)

        terminal = self._terminal_labels.get(terminal_id)
        delay_seconds = self.config.risk_delay_days * SECONDS_PER_DAY
        if terminal is not None and delay_seconds > 0:
            fraud_delay, tx_delay = terminal.window_sum_count(timestamp, delay_seconds)
        else:
            fraud_delay, tx_delay = 0.0, 0

        for window in self.config.terminal_windows:
            if terminal is None:
                features[f"terminal_nb_tx_{window}d"] = 0.0
                features[f"terminal_risk_{window}d"] = 0.0
                continue
            total_seconds = (window + self.config.risk_delay_days) * SECONDS_PER_DAY
            fraud_total, tx_total = terminal.window_sum_count(timestamp, total_seconds)
            tx_window = tx_total - tx_delay
            fraud_window = fraud_total - fraud_delay
            features[f"terminal_nb_tx_{window}d"] = float(tx_window)
            features[f"terminal_risk_{window}d"] = (
                float(fraud_window / tx_window) if tx_window > 0 else 0.0
            )

        return features

    def vector(self, features: dict[str, float]) -> np.ndarray:
        """Order a feature mapping into the model's input vector.

        Positional, using the same ordering the batch pipeline defines, so a
        feature added in one place and not the other fails loudly here rather
        than silently mis-assigning columns at serving time.
        """
        return np.array([features[name] for name in self.names], dtype=np.float32).reshape(1, -1)

    def _is_night(self, hour: int) -> bool:
        start, end = self.config.night_start_hour, self.config.night_end_hour
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end

    def warm_up(
        self,
        frame: pd.DataFrame,
        *,
        apply_labels: bool = True,
        label_cutoff: float | None = None,
    ) -> list[tuple[float, int, int]]:
        """Prime the store from historical transactions before serving begins.

        A consumer started cold has empty windows, so every terminal looks
        unseen and every customer looks new. Its first weeks of predictions are
        measurably worse than the batch evaluation implies - see
        ``docs/results.md``, which quantifies the gap. This is not a quirk of
        the simulator: it is what happens to any stateful streaming model after
        a deployment, a restart or a partition reassignment.

        ``label_cutoff`` is the moment the snapshot was taken, minus the label
        delay. Labels newer than it had not arrived yet, so they are *not*
        applied; they are returned instead, for the caller to release on the
        same schedule as live traffic. Applying them here would seed the
        terminal history with timestamps that the consumer's first delayed
        releases then undercut, and an out-of-order append invalidates the
        prefix-sum index the window queries depend on.

        Returns the deferred labels as ``(timestamp, terminal_id, is_fraud)``,
        in timestamp order.
        """
        ordered = frame.sort_values("tx_datetime", kind="stable")
        deferred: list[tuple[float, int, int]] = []
        for row in ordered.itertuples():
            timestamp = pd.Timestamp(row.tx_datetime).timestamp()
            self.observe_transaction(int(row.customer_id), timestamp, float(row.tx_amount))
            if not (apply_labels and hasattr(row, "is_fraud")):
                continue
            if label_cutoff is not None and timestamp > label_cutoff:
                deferred.append((timestamp, int(row.terminal_id), int(row.is_fraud)))
            else:
                self.observe_label(int(row.terminal_id), timestamp, int(row.is_fraud))
        return deferred

    def state_size(self) -> dict[str, int]:
        """How much history is being retained, for the metrics endpoint."""
        return {
            "customers_tracked": len(self._customer_amounts),
            "terminals_tracked": len(self._terminal_labels),
            "customer_events": sum(len(h) for h in self._customer_amounts.values()),
            "terminal_events": sum(len(h) for h in self._terminal_labels.values()),
        }
