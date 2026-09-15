"""Feature engineering: correctness, leakage and batch/online agreement.

These are the highest-value tests in the repository. A feature bug does not
raise - it produces a plausible-looking number attached to the wrong row, and
every metric downstream quietly becomes meaningless. An early version of this
pipeline did exactly that, and the only symptom was a PR-AUC of 0.18 instead of
0.73.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rtml.config import FeatureConfig
from rtml.errors import FeatureError
from rtml.features.engineering import build_features, feature_names
from rtml.features.online import OnlineFeatureStore, _EntityHistory


def brute_force_windows(
    frame: pd.DataFrame, window_days: int, delay_days: int
) -> dict[str, np.ndarray]:
    """An obviously-correct reference: for each row, scan every other row.

    Far too slow for production and exactly right for a test. If the fast
    implementation and this disagree, the fast one is wrong.
    """
    ordered = frame.sort_values("transaction_id").reset_index(drop=True)
    times = ordered["tx_datetime"].to_numpy()
    customers = ordered["customer_id"].to_numpy()
    terminals = ordered["terminal_id"].to_numpy()
    amounts = ordered["tx_amount"].to_numpy()
    frauds = ordered["is_fraud"].to_numpy()

    window = np.timedelta64(window_days, "D")
    delay = np.timedelta64(delay_days, "D")
    n = len(ordered)
    result = {key: np.zeros(n) for key in ("cust_n", "cust_avg", "term_n", "term_risk")}

    for i in range(n):
        # Customer windows include the transaction being scored.
        mask = (customers == customers[i]) & (times > times[i] - window) & (times <= times[i])
        result["cust_n"][i] = mask.sum()
        result["cust_avg"][i] = amounts[mask].mean() if mask.any() else 0.0
        # Terminal windows end `delay` before it.
        delayed = (
            (terminals == terminals[i])
            & (times > times[i] - delay - window)
            & (times <= times[i] - delay)
        )
        result["term_n"][i] = delayed.sum()
        result["term_risk"][i] = frauds[delayed].mean() if delayed.any() else 0.0
    return result


@pytest.fixture
def scattered_frame() -> pd.DataFrame:
    """Many entities interleaved in time - the case a grouped rolling gets wrong."""
    rng = np.random.default_rng(3)
    n = 2500
    frame = pd.DataFrame(
        {
            "transaction_id": np.arange(n),
            "customer_id": rng.integers(0, 35, n),
            "terminal_id": rng.integers(0, 20, n),
            "tx_datetime": pd.Timestamp("2025-01-01")
            + pd.to_timedelta(np.sort(rng.integers(0, 60 * 86400, n)), unit="s"),
            "tx_amount": np.round(rng.uniform(1, 200, n), 2),
        }
    )
    frame["is_fraud"] = (rng.random(n) < 0.06).astype(int)
    return frame


def test_features_match_a_brute_force_reference(scattered_frame: pd.DataFrame) -> None:
    config = FeatureConfig(customer_windows=[7], terminal_windows=[7], risk_delay_days=7)
    produced = (
        build_features(scattered_frame, config).sort_values("transaction_id").reset_index(drop=True)
    )
    expected = brute_force_windows(scattered_frame, 7, 7)

    assert np.allclose(produced["customer_nb_tx_7d"], expected["cust_n"], atol=1e-4)
    assert np.allclose(produced["customer_avg_amount_7d"], expected["cust_avg"], atol=1e-3)
    assert np.allclose(produced["terminal_nb_tx_7d"], expected["term_n"], atol=1e-4)
    assert np.allclose(produced["terminal_risk_7d"], expected["term_risk"], atol=1e-4)


def test_grouped_windows_do_not_bleed_between_entities() -> None:
    """Two customers interleaved in time must each see only their own history."""
    frame = pd.DataFrame(
        {
            "transaction_id": range(6),
            "customer_id": [0, 1, 0, 1, 0, 1],
            "terminal_id": [0] * 6,
            "tx_datetime": pd.date_range("2025-01-01", periods=6, freq="D"),
            "tx_amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "is_fraud": [0] * 6,
        }
    )
    config = FeatureConfig(customer_windows=[30], terminal_windows=[7], risk_delay_days=7)
    produced = build_features(frame, config).sort_values("transaction_id").reset_index(drop=True)
    # Each customer's own first, second and third transaction.
    assert produced["customer_nb_tx_30d"].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


@pytest.mark.parametrize("delay", [0, 3, 7, 14])
def test_a_label_cannot_influence_features_before_the_delay(
    linear_frame: pd.DataFrame, delay: int
) -> None:
    """The central anti-leakage property.

    A fraud confirmed on day D must not change any feature computed before
    D + delay, because the label does not exist yet.
    """
    config = FeatureConfig(customer_windows=[7], terminal_windows=[7], risk_delay_days=delay)
    baseline = build_features(linear_frame, config)["terminal_risk_7d"].to_numpy()

    marked = linear_frame.copy()
    marked.loc[20, "is_fraud"] = 1
    after = build_features(marked, config)["terminal_risk_7d"].to_numpy()

    changed = np.flatnonzero(np.abs(after - baseline) > 1e-9)
    assert len(changed) > 0, "the label never becomes visible, so the feature is useless"
    assert int(changed.min()) >= 20 + delay, "a label leaked into the delay window"


def test_customer_features_never_depend_on_labels(linear_frame: pd.DataFrame) -> None:
    """They use no labels, so flipping every label must change nothing."""
    config = FeatureConfig(customer_windows=[7], terminal_windows=[7], risk_delay_days=7)
    without = build_features(linear_frame, config)
    flipped = linear_frame.copy()
    flipped["is_fraud"] = 1
    with_labels = build_features(flipped, config)
    assert np.allclose(without["customer_nb_tx_7d"], with_labels["customer_nb_tx_7d"])
    assert np.allclose(without["customer_avg_amount_7d"], with_labels["customer_avg_amount_7d"])


def test_window_is_half_open(linear_frame: pd.DataFrame) -> None:
    """A transaction exactly one window old is outside the window."""
    config = FeatureConfig(customer_windows=[7], terminal_windows=[7], risk_delay_days=0)
    produced = build_features(linear_frame, config)
    # One transaction per day: a 7-day window holds today plus the six before.
    assert produced["customer_nb_tx_7d"].iloc[10] == 7.0


def test_unseen_terminal_scores_zero_risk_not_nan() -> None:
    """A NaN would force every model to carry an imputation rule."""
    frame = pd.DataFrame(
        {
            "transaction_id": [0],
            "customer_id": [0],
            "terminal_id": [999],
            "tx_datetime": [pd.Timestamp("2025-01-01")],
            "tx_amount": [10.0],
            "is_fraud": [0],
        }
    )
    produced = build_features(frame, FeatureConfig())
    assert produced["terminal_risk_7d"].iloc[0] == 0.0
    assert produced.notna().all().all()


def test_feature_order_is_stable() -> None:
    """Serving builds the vector positionally, so the order is part of the contract."""
    config = FeatureConfig()
    assert feature_names(config) == feature_names(config)
    assert feature_names(config)[0] == "tx_amount"


def test_night_window_wrapping_midnight() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": [0, 1, 2],
            "customer_id": [0, 0, 0],
            "terminal_id": [0, 0, 0],
            "tx_datetime": pd.to_datetime(
                ["2025-01-01T23:30", "2025-01-02T03:00", "2025-01-02T12:00"]
            ),
            "tx_amount": [10.0, 10.0, 10.0],
            "is_fraud": [0, 0, 0],
        }
    )
    config = FeatureConfig(night_start_hour=22, night_end_hour=6)
    produced = build_features(frame, config).sort_values("transaction_id")
    assert produced["tx_during_night"].tolist() == [1, 1, 0]


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(FeatureError, match="missing required columns"):
        build_features(pd.DataFrame({"transaction_id": [1]}), FeatureConfig())


def test_terminal_risk_requires_labels() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": [0],
            "customer_id": [0],
            "terminal_id": [0],
            "tx_datetime": [pd.Timestamp("2025-01-01")],
            "tx_amount": [1.0],
        }
    )
    with pytest.raises(FeatureError, match="is_fraud"):
        build_features(frame, FeatureConfig())


# ------------------------------------------------------- online == batch


def test_online_features_match_batch_exactly(scattered_frame: pd.DataFrame) -> None:
    """Training/serving skew check.

    The batch pipeline trains the model; the online store serves it. If they
    disagree the model degrades in production in a way that looks like drift and
    gets misdiagnosed for weeks.
    """
    config = FeatureConfig()
    batch = (
        build_features(scattered_frame, config).sort_values("transaction_id").reset_index(drop=True)
    )

    store = OnlineFeatureStore(config)
    computed: list[tuple[int, dict[str, float]]] = []
    for row in scattered_frame.sort_values("tx_datetime").itertuples():
        timestamp = row.tx_datetime.timestamp()
        computed.append(
            (
                row.transaction_id,
                store.compute(
                    row.customer_id, row.terminal_id, timestamp, row.tx_amount, row.tx_datetime
                ),
            )
        )
        store.observe_transaction(row.customer_id, timestamp, row.tx_amount)
        store.observe_label(row.terminal_id, timestamp, row.is_fraud)

    online = pd.DataFrame([f for _, f in computed], index=[i for i, _ in computed]).sort_index()
    for name in feature_names(config):
        difference = np.abs(
            batch[name].to_numpy(dtype=np.float64) - online[name].to_numpy(dtype=np.float64)
        ).max()
        assert difference < 1e-4, f"{name} differs between batch and online by {difference}"


def test_online_vector_order_matches_feature_names() -> None:
    config = FeatureConfig()
    store = OnlineFeatureStore(config)
    features = store.compute(
        1, 1, pd.Timestamp("2025-01-01").timestamp(), 42.0, pd.Timestamp("2025-01-01")
    )
    vector = store.vector(features)
    assert vector.shape == (1, len(feature_names(config)))
    assert vector[0][0] == pytest.approx(42.0)


def test_online_store_evicts_beyond_the_longest_window() -> None:
    """Memory must track live events, not total events ever seen."""
    config = FeatureConfig(customer_windows=[1], terminal_windows=[1], risk_delay_days=0)
    store = OnlineFeatureStore(config)
    base = pd.Timestamp("2025-01-01")
    for day in range(400):
        store.observe_transaction(1, (base + pd.Timedelta(days=day)).timestamp(), 10.0)
    assert store.state_size()["customer_events"] < 10


def test_warm_up_populates_state(scattered_frame: pd.DataFrame) -> None:
    store = OnlineFeatureStore(FeatureConfig())
    assert store.state_size()["customers_tracked"] == 0
    deferred = store.warm_up(scattered_frame)
    assert deferred == []  # no cutoff given, so every label is applied
    assert store.state_size()["customers_tracked"] > 0
    assert store.state_size()["terminals_tracked"] > 0


def test_warm_up_holds_back_labels_that_had_not_arrived(scattered_frame: pd.DataFrame) -> None:
    """A snapshot cannot contain labels the investigation queue has not produced."""
    timestamps = pd.to_datetime(scattered_frame["tx_datetime"]).map(pd.Timestamp.timestamp)
    cutoff = float(timestamps.median())

    store = OnlineFeatureStore(FeatureConfig())
    deferred = store.warm_up(scattered_frame, label_cutoff=cutoff)

    assert deferred, "labels newer than the cutoff should be held back"
    assert all(timestamp > cutoff for timestamp, _, _ in deferred)
    assert deferred == sorted(deferred), "deferred labels must stay in timestamp order"
    # Every row still primes the customer side; only labels wait.
    assert store.state_size()["customers_tracked"] > 0


def test_out_of_order_history_is_rejected_rather_than_silently_wrong() -> None:
    """The prefix-sum index is only valid over sorted timestamps.

    Before this check existed, an out-of-order append produced no error and no
    obviously wrong value - just window aggregates computed against a corrupted
    index. It reached production behaviour as a 0.24 drop in streamed PR-AUC
    and took a feature-by-feature comparison against the batch table to find.
    """
    history = _EntityHistory()
    history.append(100.0 * 86_400, 1.0)
    history.append(118.0 * 86_400, 1.0)

    with pytest.raises(FeatureError, match="non-decreasing"):
        history.append(112.0 * 86_400, 1.0)


def test_equal_timestamps_are_still_accepted() -> None:
    """Two transactions in the same second are ordinary, not an error."""
    history = _EntityHistory()
    history.append(1_000.0, 1.0)
    history.append(1_000.0, 1.0)
    assert history.window_sum_count(1_000.0, 60.0) == (2.0, 2)
