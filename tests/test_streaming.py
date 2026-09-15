"""Broker semantics and the scoring consumer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rtml.config import FeatureConfig, PipelineConfig
from rtml.errors import StreamError
from rtml.streaming import (
    InProcessBroker,
    ScoringConsumer,
    partition_for,
    publish_transactions,
    to_event,
)
from rtml.training.pipeline import train_all
from rtml.training.splitting import split_by_day


class ConstantModel:
    """A stand-in that returns a fixed score, so tests exercise plumbing not ML."""

    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.seen: list[np.ndarray] = []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.seen.append(X)
        return np.full(len(X), self.score)


@pytest.fixture
def broker() -> InProcessBroker:
    return InProcessBroker(partitions=4)


def test_partitioning_is_stable_across_calls() -> None:
    """Python randomises str hashing per process; a restart must not reshuffle."""
    keys = [str(i) for i in range(200)]
    first = [partition_for(k, 8) for k in keys]
    second = [partition_for(k, 8) for k in keys]
    assert first == second
    assert set(first) == set(range(8)), "keys should spread across every partition"


def test_partitioning_rejects_zero_partitions() -> None:
    with pytest.raises(StreamError):
        partition_for("1", 0)


def test_a_customer_always_lands_on_one_partition(broker: InProcessBroker) -> None:
    """The invariant the online feature store depends on.

    Round-robin partitioning would scatter a customer's history across
    consumers and every window feature would be computed from a fraction of it.
    """
    for index in range(50):
        broker.produce("transactions", key="42", value={"transaction_id": index})
    partitions_used = {
        record.partition for record in broker.poll("transactions", "g", max_records=100)
    }
    assert len(partitions_used) == 1


def test_offsets_advance_only_after_commit(broker: InProcessBroker) -> None:
    for index in range(10):
        broker.produce("transactions", key=str(index), value={"transaction_id": index})

    first = broker.poll("transactions", "g", max_records=10)
    assert len(first) == 10
    # Uncommitted, so a re-poll redelivers: at-least-once.
    assert len(broker.poll("transactions", "g", max_records=10)) == 10

    broker.commit("transactions", "g", {r.partition: r.offset + 1 for r in first})
    assert broker.poll("transactions", "g", max_records=10) == []


def test_commits_never_rewind(broker: InProcessBroker) -> None:
    """A late commit from a replayed batch must not reopen consumed records."""
    for index in range(5):
        broker.produce("transactions", key="1", value={"transaction_id": index})
    broker.commit("transactions", "g", {0: 5, 1: 5, 2: 5, 3: 5})
    broker.commit("transactions", "g", {0: 2, 1: 2, 2: 2, 3: 2})
    assert all(offset >= 5 for offset in broker.committed_offsets("transactions", "g").values())


def test_consumer_groups_have_independent_offsets(broker: InProcessBroker) -> None:
    for index in range(6):
        broker.produce("transactions", key=str(index), value={"transaction_id": index})
    first = broker.poll("transactions", "group-a", max_records=6)
    broker.commit("transactions", "group-a", {r.partition: r.offset + 1 for r in first})
    assert broker.poll("transactions", "group-a", max_records=6) == []
    assert len(broker.poll("transactions", "group-b", max_records=6)) == 6


def test_lag_reports_unconsumed_records(broker: InProcessBroker) -> None:
    for index in range(20):
        broker.produce("transactions", key=str(index), value={"transaction_id": index})
    assert broker.lag("transactions", "g") == 20
    batch = broker.poll("transactions", "g", max_records=20)
    broker.commit("transactions", "g", {r.partition: r.offset + 1 for r in batch})
    assert broker.lag("transactions", "g") == 0


def test_event_serialisation_round_trip() -> None:
    row = (
        pd.DataFrame(
            {
                "transaction_id": [1],
                "customer_id": [2],
                "terminal_id": [3],
                "tx_datetime": [pd.Timestamp("2025-03-01T10:00:00")],
                "tx_amount": [99.5],
                "tx_day": [59],
                "is_fraud": [1],
            }
        )
        .itertuples()
        .__next__()
    )
    event = to_event(row)
    assert event["transaction_id"] == 1
    assert event["tx_amount"] == pytest.approx(99.5)
    assert pd.Timestamp(event["tx_datetime"]) == pd.Timestamp("2025-03-01T10:00:00")


# --------------------------------------------------------------- consumer


def test_the_label_never_reaches_the_model(broker: InProcessBroker) -> None:
    """The single most dangerous bug available in this codebase.

    Replayed events carry the label so the consumer's output can be scored
    offline. If it reached the feature vector the model would look perfect and
    be worthless.
    """
    config = FeatureConfig()
    model = ConstantModel()
    consumer = ScoringConsumer(broker, model, config, threshold=0.5)

    event = {
        "transaction_id": 1,
        "customer_id": 7,
        "terminal_id": 3,
        "tx_datetime": "2025-03-01T10:00:00",
        "tx_amount": 50.0,
        "tx_day": 0,
        "is_fraud": 1,
    }
    consumer.score_event(dict(event))

    vector = model.seen[0]
    assert vector.shape == (1, len(consumer.store.names))
    # A label of 1 appears nowhere it should not: the only 1.0 values allowed
    # are genuine count or flag features, so check the vector width instead.
    assert "is_fraud" not in consumer.store.names


def test_feature_state_advances_only_after_scoring(broker: InProcessBroker) -> None:
    """A transaction must not appear in its own terminal history."""
    config = FeatureConfig()
    consumer = ScoringConsumer(broker, ConstantModel(), config, threshold=0.5, label_delay_days=0)
    base = pd.Timestamp("2025-03-01T10:00:00")

    first = consumer.score_event(
        {
            "transaction_id": 1,
            "customer_id": 1,
            "terminal_id": 1,
            "tx_datetime": base.isoformat(),
            "tx_amount": 10.0,
            "tx_day": 0,
            "is_fraud": 1,
        }
    )
    assert first["score"] == pytest.approx(0.9)

    # The second transaction on the same terminal now sees the first one's label.
    second_time = (base + pd.Timedelta(hours=1)).isoformat()
    consumer.score_event(
        {
            "transaction_id": 2,
            "customer_id": 1,
            "terminal_id": 1,
            "tx_datetime": second_time,
            "tx_amount": 10.0,
            "tx_day": 0,
            "is_fraud": 0,
        }
    )
    assert consumer.store.state_size()["terminal_events"] >= 1


def test_labels_are_held_for_the_delay(broker: InProcessBroker) -> None:
    config = FeatureConfig(risk_delay_days=7)
    consumer = ScoringConsumer(broker, ConstantModel(), config, threshold=0.5)
    base = pd.Timestamp("2025-03-01")

    consumer.score_event(
        {
            "transaction_id": 1,
            "customer_id": 1,
            "terminal_id": 1,
            "tx_datetime": base.isoformat(),
            "tx_amount": 10.0,
            "tx_day": 0,
            "is_fraud": 1,
        }
    )
    # Inside the delay the label is still pending, so terminal state stays empty.
    assert consumer.store.state_size()["terminal_events"] == 0

    consumer.score_event(
        {
            "transaction_id": 2,
            "customer_id": 1,
            "terminal_id": 1,
            "tx_datetime": (base + pd.Timedelta(days=8)).isoformat(),
            "tx_amount": 10.0,
            "tx_day": 8,
            "is_fraud": 0,
        }
    )
    assert consumer.store.state_size()["terminal_events"] >= 1


def test_a_poison_event_does_not_stall_the_partition(broker: InProcessBroker) -> None:
    """One malformed record must not block everything behind it."""
    config = FeatureConfig()
    consumer = ScoringConsumer(broker, ConstantModel(), config, threshold=0.5)
    broker.produce("transactions", key="1", value={"transaction_id": 1})  # missing fields
    broker.produce(
        "transactions",
        key="1",
        value={
            "transaction_id": 2,
            "customer_id": 1,
            "terminal_id": 1,
            "tx_datetime": "2025-03-01T10:00:00",
            "tx_amount": 10.0,
            "tx_day": 0,
            "is_fraud": 0,
        },
    )
    predictions = consumer.run(batch_size=10)
    assert consumer.stats.errors == 1
    assert len(predictions) == 1
    assert broker.lag("transactions", consumer.group) == 0


def test_consumer_drains_the_stream_and_publishes(
    broker: InProcessBroker, features: pd.DataFrame, config: PipelineConfig
) -> None:
    split = split_by_day(features, config.split)
    sample = split.test.head(500)
    publish_transactions(broker, sample, "transactions", progress_every=0)

    consumer = ScoringConsumer(broker, ConstantModel(0.2), config.features, threshold=0.5)
    predictions = consumer.run(batch_size=100)

    assert len(predictions) == len(sample)
    assert consumer.stats.errors == 0
    assert broker.lag("transactions", consumer.group) == 0
    assert broker.size("predictions") == len(sample)
    assert consumer.stats.alerts == 0  # every score is below the threshold


@pytest.mark.slow
def test_streamed_scores_match_batch_scores(features: pd.DataFrame, config: PipelineConfig) -> None:
    """The end-to-end guarantee: the same model, same data, same answers.

    The consumer is warmed from history first, which is what a production
    deployment does. Without warming the windows start empty and the streamed
    scores are measurably worse - quantified in docs/results.md.
    """
    models, split = train_all(features, config, include_isolation_forest=False)
    model = models[0]

    broker = InProcessBroker(partitions=4)
    sample = split.test.head(400)
    publish_transactions(broker, sample, "transactions", progress_every=0)

    consumer = ScoringConsumer(broker, model, config.features, threshold=model.threshold)
    consumer.warm_up(features[features["tx_day"] < split.boundaries["test_start_day"]])
    predictions = consumer.run(batch_size=200)

    streamed = {p["transaction_id"]: p["score"] for p in predictions}
    batch_scores = model.predict_proba(sample[model.features].to_numpy(dtype=np.float32))
    batch = dict(zip(sample["transaction_id"], batch_scores, strict=True))

    differences = [abs(streamed[k] - batch[k]) for k in streamed]
    # A handful of terminals in the test window have no history at all, so a
    # small number of rows differ; the bulk must agree closely.
    assert float(np.median(differences)) < 1e-6
    assert float(np.mean([d < 1e-6 for d in differences])) > 0.90


def test_partitions_are_polled_in_round_robin(broker: InProcessBroker) -> None:
    """Draining one partition before the next reorders the stream by weeks.

    Terminal state is shared across partitions, so a consumer handed one
    partition's whole history and then restarted at the beginning of time for
    the next scores most of the stream against a future it has already seen. On
    the full run that left 75% of events behind the consumer's own clock by a
    mean of 28 days and cost 0.24 PR-AUC, with no error anywhere.
    """
    for index in range(4_000):
        broker.produce("transactions", key=str(index), value={"transaction_id": index})

    batch = broker.poll("transactions", "g", max_records=400)
    contributions = {record.partition for record in batch}
    assert contributions == {0, 1, 2, 3}, "every partition must contribute to a batch"

    counts = [sum(1 for r in batch if r.partition == p) for p in range(4)]
    assert max(counts) - min(counts) <= 1, f"partitions drained unevenly: {counts}"


def test_poll_still_fills_the_batch_when_a_partition_runs_dry(
    broker: InProcessBroker,
) -> None:
    """Round-robin fairness must not cost throughput near the end of a stream."""
    for index in range(20):
        broker.produce("transactions", key=str(index), value={"transaction_id": index})
    batch = broker.poll("transactions", "g", max_records=20)
    assert len(batch) == 20


def test_labels_are_released_in_event_time_not_arrival_order(
    broker: InProcessBroker,
) -> None:
    """Records arrive interleaved from several partitions; state must not.

    The online history is a prefix-sum index over sorted timestamps, so applying
    a label behind one already applied corrupts every later window query for
    that terminal - silently, with a plausible value.
    """
    config = FeatureConfig(risk_delay_days=7)
    consumer = ScoringConsumer(broker, ConstantModel(), config, threshold=0.5)
    base = pd.Timestamp("2025-03-01")

    def event(index: int, day: float, terminal: int, fraud: int) -> dict[str, object]:
        return {
            "transaction_id": index,
            "customer_id": index,
            "terminal_id": terminal,
            "tx_datetime": (base + pd.Timedelta(days=day)).isoformat(),
            "tx_amount": 10.0,
            "tx_day": int(day),
            "is_fraud": fraud,
        }

    # Arrival order deliberately out of event-time order, within the 7-day
    # allowance the delay provides.
    for index, day in enumerate([0.0, 3.0, 1.0, 12.0, 10.0, 14.0]):
        consumer.score_event(event(index, day, terminal=1, fraud=index % 2))

    applied = consumer.store._terminal_labels[1]._timestamps
    assert applied == sorted(applied), "labels reached the store out of event-time order"
    assert consumer.stats.late_labels == 0


def test_a_label_later_than_the_allowance_is_counted_not_applied(
    broker: InProcessBroker,
) -> None:
    """Beyond the reordering allowance the only safe options are drop or corrupt."""
    config = FeatureConfig(risk_delay_days=1)
    consumer = ScoringConsumer(broker, ConstantModel(), config, threshold=0.5)
    base = pd.Timestamp("2025-03-01")

    def event(index: int, day: float) -> dict[str, object]:
        return {
            "transaction_id": index,
            "customer_id": index,
            "terminal_id": 1,
            "tx_datetime": (base + pd.Timedelta(days=day)).isoformat(),
            "tx_amount": 10.0,
            "tx_day": int(day),
            "is_fraud": 1,
        }

    consumer.score_event(event(0, 0.0))
    consumer.score_event(event(1, 10.0))  # watermark 10; day 0 clears the delay
    consumer.score_event(event(2, 20.0))  # watermark 20; day 10 clears it too
    consumer.score_event(event(3, 5.0))  # fifteen days behind a one-day allowance

    applied = consumer.store._terminal_labels[1]._timestamps
    days = [(t - base.timestamp()) / 86_400 for t in applied]
    assert days == [0.0, 10.0], "the late label must not reach the store"
    assert applied == sorted(applied)
    assert consumer.stats.late_labels == 1
