"""Persistence for transactions, predictions, drift and deployments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rtml.data.store import PredictionStore, postgres_schema_sql, redact_url


@pytest.fixture
def store() -> PredictionStore:
    created = PredictionStore("sqlite:///:memory:")
    created.create_all()
    yield created
    created.close()


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "transaction_id": np.arange(n),
            "customer_id": rng.integers(0, 30, n),
            "terminal_id": rng.integers(0, 20, n),
            "tx_datetime": pd.date_range("2025-01-01", periods=n, freq="h"),
            "tx_day": np.arange(n) // 24,
            "tx_amount": np.round(rng.uniform(1, 200, n), 2),
            "is_fraud": (rng.random(n) < 0.05).astype(int),
            "fraud_scenario": np.zeros(n, dtype=int),
        }
    )


def test_transactions_round_trip(store: PredictionStore, sample_transactions: pd.DataFrame) -> None:
    assert store.write_transactions(sample_transactions) == len(sample_transactions)


def test_rewriting_replaces_rather_than_duplicates(
    store: PredictionStore, sample_transactions: pd.DataFrame
) -> None:
    store.write_transactions(sample_transactions)
    store.write_transactions(sample_transactions)
    with store.engine.connect() as connection:
        from sqlalchemy import func, select

        from rtml.data.store import transactions_table

        count = connection.execute(
            select(func.count()).select_from(transactions_table)
        ).scalar_one()
    assert count == len(sample_transactions)


def test_writing_nothing_is_a_no_op(store: PredictionStore) -> None:
    assert store.write_transactions(pd.DataFrame()) == 0
    assert store.write_predictions([]) == 0


def test_predictions_and_alert_counts(store: PredictionStore) -> None:
    predictions = [
        {
            "transaction_id": i,
            "customer_id": i % 10,
            "terminal_id": i % 5,
            "tx_datetime": "2025-01-01T00:00:00",
            "tx_day": 0,
            "tx_amount": 10.0,
            "score": 0.95 if i % 4 == 0 else 0.05,
            "threshold": 0.5,
            "is_alert": i % 4 == 0,
            "scoring_latency_ms": 0.8,
            "is_fraud": int(i % 8 == 0),
        }
        for i in range(40)
    ]
    assert store.write_predictions(predictions, model_name="lightgbm", model_version="3") == 40
    assert store.prediction_count() == 40
    assert store.alert_count() == 10

    frame = store.read_predictions()
    assert frame["model_name"].unique().tolist() == ["lightgbm"]
    assert frame["model_version"].unique().tolist() == ["3"]


def test_a_prediction_may_have_no_label_yet(store: PredictionStore) -> None:
    """The label does not exist when the prediction is written."""
    store.write_predictions(
        [
            {
                "transaction_id": 1,
                "customer_id": 1,
                "terminal_id": 1,
                "tx_datetime": "2025-01-01T00:00:00",
                "tx_day": 0,
                "tx_amount": 5.0,
                "score": 0.2,
                "threshold": 0.5,
                "is_alert": False,
            }
        ]
    )
    assert pd.isna(store.read_predictions()["is_fraud"].iloc[0])


def test_deployment_history_is_recorded(store: PredictionStore) -> None:
    store.record_deployment(
        "fraud-scorer", "1", "champion", 0.8, pr_auc=0.73, reason="all gates passed"
    )
    store.record_deployment(
        "fraud-scorer", "2", "champion", 0.82, pr_auc=0.75, reason="all gates passed"
    )
    history = store.deployment_history()
    assert len(history) == 2
    # Newest first, so "what is deployed now" is the first row.
    assert history.iloc[0]["model_version"] == "2"


def test_drift_observations_are_persisted(store: PredictionStore) -> None:
    from rtml.config import DriftConfig
    from rtml.monitoring.drift import compare_distributions

    rng = np.random.default_rng(1)
    report = compare_distributions(
        pd.DataFrame({"f": rng.normal(size=2000)}),
        pd.DataFrame({"f": rng.normal(1.0, 1, 2000)}),
        ["f"],
        DriftConfig(),
    )
    report.reference_window, report.comparison_window = (0, 13), (14, 20)
    assert store.write_drift(report) == 1
    stored = store.read_drift()
    assert stored.iloc[0]["feature"] == "f"
    assert stored.iloc[0]["severity"] in {"stable", "moderate", "significant"}


def test_sqlite_parent_directory_is_created(tmp_path) -> None:
    nested = tmp_path / "a" / "b" / "rtml.sqlite3"
    created = PredictionStore(f"sqlite:///{nested}")
    created.create_all()
    assert nested.exists()
    created.close()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("sqlite:///data/rtml.sqlite3", "sqlite:///data/rtml.sqlite3"),
        (
            "postgresql+psycopg://rtml:hunter2@db:5432/rtml",
            "postgresql+psycopg://rtml:***@db:5432/rtml",
        ),
        ("postgresql://user@host/db", "postgresql://user@host/db"),
    ],
)
def test_redact_url_strips_passwords(url: str, expected: str) -> None:
    assert redact_url(url) == expected


def test_shipped_schema_covers_every_table() -> None:
    sql = postgres_schema_sql()
    for table in ("transactions", "predictions", "drift_observations", "model_deployments"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
