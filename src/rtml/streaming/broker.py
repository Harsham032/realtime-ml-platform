"""Event broker abstraction.

Two implementations behind one protocol:

``InProcessBroker``
    A real, working partitioned log with consumer groups and committed offsets.
    Not a stub - it implements the semantics the consumer depends on, so the
    whole streaming path runs, and is measured, with no external service. That
    is what makes the throughput and latency figures in ``docs/results.md``
    reproducible by anyone who clones the repository.

``KafkaBroker``
    The production path, over ``kafka-python``.

The abstraction exists because of one specific invariant. Online features are
stateful per customer, so every transaction for a customer must reach the *same*
consumer: partition by customer id, never round-robin. Both implementations
guarantee it, and ``tests/test_streaming.py`` asserts it for each.

Delivery is at-least-once: offsets commit after processing, so a crash between
processing and commit replays the batch. Scoring is idempotent - the same
transaction yields the same prediction - so a replay costs work, not
correctness. Exactly-once would need transactional writes to the prediction
store, which is a heavier guarantee than this system needs.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..errors import StreamError
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Record:
    """One event as delivered to a consumer."""

    topic: str
    partition: int
    offset: int
    key: str
    value: dict[str, Any]
    timestamp_ms: int = 0


def partition_for(key: str, partitions: int) -> int:
    """Stable partition assignment.

    Deliberately not the built-in ``hash``: Python randomises string hashing per
    process unless ``PYTHONHASHSEED`` is fixed, so a restart would reshuffle
    customers across partitions and scatter each one's feature history.
    """
    if partitions <= 0:
        raise StreamError("partitions must be positive")
    digest = 0
    for char in key:
        digest = (digest * 31 + ord(char)) & 0xFFFFFFFF
    return digest % partitions


@runtime_checkable
class Broker(Protocol):
    """Minimal produce/consume interface the pipeline depends on."""

    def produce(self, topic: str, key: str, value: dict[str, Any]) -> None: ...

    def poll(
        self, topic: str, group: str, *, max_records: int = 500, timeout: float = 1.0
    ) -> list[Record]: ...

    def commit(self, topic: str, group: str, offsets: dict[int, int]) -> None: ...

    def close(self) -> None: ...


class InProcessBroker:
    """A partitioned append-only log held in memory.

    Thread-safe so a producer and consumer can run concurrently in one process,
    which is how the local demo and the tests exercise the streaming path.
    """

    def __init__(self, partitions: int = 4) -> None:
        if partitions <= 0:
            raise StreamError("partitions must be positive")
        self.partitions = partitions
        self._log: dict[str, list[list[Record]]] = {}
        self._offsets: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
        self._lock = threading.RLock()

    def _topic(self, topic: str) -> list[list[Record]]:
        if topic not in self._log:
            self._log[topic] = [[] for _ in range(self.partitions)]
        return self._log[topic]

    def produce(self, topic: str, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            partitions = self._topic(topic)
            index = partition_for(key, self.partitions)
            offset = len(partitions[index])
            partitions[index].append(
                Record(topic=topic, partition=index, offset=offset, key=key, value=value)
            )

    def poll(
        self,
        topic: str,
        group: str,
        *,
        max_records: int = 500,
        timeout: float = 1.0,  # noqa: ARG002 - accepted for interface parity; records are already in memory
    ) -> list[Record]:
        """Return up to ``max_records`` unconsumed records for ``group``.

        Every partition contributes an equal share of the batch, so the consumer
        advances through all of them at the same rate.

        The share is not a nicety. Filling the batch from the first partition
        before touching the next hands a stateful consumer one partition's
        entire history, then restarts it at the beginning of time for the next
        one. Partitioning by customer keeps each *customer's* events together
        and in order, so customer features survive that; terminal state is
        shared across partitions and does not. Measured on the full stream, it
        left 75% of events being scored behind the consumer's own clock by a
        mean of 28 days, and cost 0.24 PR-AUC against the batch evaluation -
        with no error raised and no obviously wrong value anywhere.
        """
        with self._lock:
            partitions = self._topic(topic)
            if not partitions:
                return []
            committed = self._offsets[(topic, group)]
            cursors = [committed.get(index, 0) for index in range(len(partitions))]
            batch: list[Record] = []
            # Chunks small enough that every partition contributes before any
            # contributes twice, and passes repeated until the batch is full or
            # nothing is left - so max_records is still honoured when some
            # partitions run dry.
            share = max(1, max_records // len(partitions))
            progressed = True
            while len(batch) < max_records and progressed:
                progressed = False
                for index, records in enumerate(partitions):
                    if len(batch) >= max_records:
                        break
                    start = cursors[index]
                    take = min(share, max_records - len(batch), len(records) - start)
                    if take <= 0:
                        continue
                    batch.extend(records[start : start + take])
                    cursors[index] = start + take
                    progressed = True
            return batch

    def commit(self, topic: str, group: str, offsets: dict[int, int]) -> None:
        with self._lock:
            committed = self._offsets[(topic, group)]
            for partition, offset in offsets.items():
                # Offsets only move forward: a late commit from a replayed batch
                # must not rewind a partition and cause an infinite loop.
                committed[partition] = max(committed.get(partition, 0), offset)

    def committed_offsets(self, topic: str, group: str) -> dict[int, int]:
        with self._lock:
            return dict(self._offsets[(topic, group)])

    def size(self, topic: str) -> int:
        with self._lock:
            return sum(len(partition) for partition in self._topic(topic))

    def lag(self, topic: str, group: str) -> int:
        """Unconsumed records, the signal that says whether scoring keeps up."""
        with self._lock:
            partitions = self._topic(topic)
            committed = self._offsets[(topic, group)]
            return sum(
                len(records) - committed.get(index, 0) for index, records in enumerate(partitions)
            )

    def close(self) -> None:
        """Nothing to release; present so callers need not special-case backend."""


class KafkaBroker:
    """Kafka-backed broker.

    Producer and consumer are created lazily so importing this module, and
    running the offline test suite, never requires a reachable broker.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        *,
        partitions: int = 4,
        client_id: str = "rtml",
        acks: str | int = "all",
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.partitions = partitions
        self.client_id = client_id
        self.acks = acks
        self._producer: Any = None
        self._consumers: dict[tuple[str, str], Any] = {}

    def _get_producer(self) -> Any:
        if self._producer is None:
            try:
                from kafka import KafkaProducer
            except ImportError as exc:  # pragma: no cover - kafka-python is a dependency
                raise StreamError("kafka-python is not installed") from exc
            import json

            try:
                self._producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers.split(","),
                    client_id=self.client_id,
                    # 'all' waits for every in-sync replica. A risk decision is
                    # worth more than the few milliseconds a weaker ack saves.
                    acks=self.acks,
                    retries=5,
                    linger_ms=5,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8"),
                )
            except Exception as exc:
                raise StreamError(
                    f"could not reach Kafka at {self.bootstrap_servers}: {exc}"
                ) from exc
        return self._producer

    def _get_consumer(self, topic: str, group: str) -> Any:
        key = (topic, group)
        if key not in self._consumers:
            try:
                from kafka import KafkaConsumer
            except ImportError as exc:  # pragma: no cover - kafka-python is a dependency
                raise StreamError("kafka-python is not installed") from exc
            import json

            try:
                self._consumers[key] = KafkaConsumer(
                    topic,
                    bootstrap_servers=self.bootstrap_servers.split(","),
                    group_id=group,
                    # Offsets commit after processing, which is what makes
                    # delivery at-least-once rather than at-most-once.
                    enable_auto_commit=False,
                    auto_offset_reset="earliest",
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    key_deserializer=lambda k: k.decode("utf-8") if k else "",
                )
            except Exception as exc:
                raise StreamError(
                    f"could not reach Kafka at {self.bootstrap_servers}: {exc}"
                ) from exc
        return self._consumers[key]

    def produce(self, topic: str, key: str, value: dict[str, Any]) -> None:
        self._get_producer().send(topic, key=key, value=value)

    def poll(
        self, topic: str, group: str, *, max_records: int = 500, timeout: float = 1.0
    ) -> list[Record]:
        consumer = self._get_consumer(topic, group)
        polled = consumer.poll(timeout_ms=int(timeout * 1000), max_records=max_records)
        records: list[Record] = []
        for partition, messages in polled.items():
            for message in messages:
                records.append(
                    Record(
                        topic=topic,
                        partition=partition.partition,
                        offset=message.offset,
                        key=message.key or "",
                        value=message.value,
                        timestamp_ms=message.timestamp or 0,
                    )
                )
        return records

    def commit(
        self,
        topic: str,
        group: str,
        offsets: dict[int, int],  # noqa: ARG002 - the client tracks its own offsets
    ) -> None:
        self._get_consumer(topic, group).commit()

    def flush(self) -> None:
        if self._producer is not None:
            self._producer.flush()

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush()
            self._producer.close()
            self._producer = None
        for consumer in self._consumers.values():
            consumer.close()
        self._consumers.clear()


def build_broker(backend: str, settings: Any, *, partitions: int = 4) -> Broker:
    """Instantiate the broker named by ``backend``."""
    if backend == "inprocess":
        return InProcessBroker(partitions=partitions)
    if backend == "kafka":
        return KafkaBroker(settings.kafka_bootstrap_servers, partitions=partitions)
    raise StreamError(f"unknown stream backend: {backend}")
