"""HTTP surface: scoring, feedback, health, drift and metrics."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rtml.config import FeatureConfig
from rtml.features.online import OnlineFeatureStore
from rtml.services import api as api_module


class StubModel:
    """Scores on amount alone, so expectations are easy to reason about."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X[:, 0] / 500.0, 0.0, 1.0)


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A service wired by hand rather than loaded from disk.

    Building state directly keeps the API tests independent of whether a
    training run has produced a champion bundle.

    The database is redirected through the environment, not just by assigning to
    ``state`` afterwards: entering ``TestClient`` runs the app's lifespan, which
    rebuilds state from disk and would otherwise hand these tests whatever the
    developer's last pipeline run happened to leave in the real prediction
    store. That is how ``/drift`` came to return 200 in a test asserting 400 -
    the test passed on a clean checkout and failed on any machine that had run
    the stream.
    """
    from rtml.config import load_settings
    from rtml.data.store import PredictionStore

    database_url = f"sqlite:///{tmp_path}/api.sqlite3"
    monkeypatch.setenv("RTML_DATABASE_URL", database_url)

    state = api_module.state
    state.settings = load_settings()
    state.config = api_module.PipelineConfig.from_yaml("configs/fast.yaml")
    state.store = OnlineFeatureStore(FeatureConfig())
    state.model = StubModel()
    state.model_name = "stub"
    state.model_version = "1"
    state.threshold = 0.5
    state.metrics = {"pr_auc": 0.73}
    state.error = None

    with TestClient(api_module.app, raise_server_exceptions=True) as test_client:
        # The lifespan has just rebuilt state from disk; put the test's own
        # model, feature store and database back over the top of it.
        if state.database is not None:
            state.database.close()
        state.database = PredictionStore(database_url)
        state.database.create_all()
        state.model = StubModel()
        state.store = OnlineFeatureStore(FeatureConfig())
        state.model_name = "stub"
        state.threshold = 0.5
        state.error = None
        yield test_client
    if state.database is not None:
        state.database.close()


def _transaction(**overrides: object) -> dict[str, object]:
    payload = {
        "transaction_id": 1,
        "customer_id": 7,
        "terminal_id": 3,
        "tx_datetime": "2025-05-01T14:23:00",
        "tx_amount": 100.0,
    }
    payload.update(overrides)
    return payload


def test_health_reports_the_loaded_model(client: TestClient) -> None:
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_name"] == "stub"


def test_health_redacts_database_credentials() -> None:
    from rtml.data.store import redact_url

    assert "hunter2" not in redact_url("postgresql+psycopg://u:hunter2@h:5432/d")


def test_scoring_returns_a_decision_and_its_features(client: TestClient) -> None:
    payload = client.post("/score", json=_transaction(tx_amount=400.0)).json()
    assert payload["transaction_id"] == 1
    assert payload["score"] == pytest.approx(0.8)
    assert payload["is_alert"] is True
    assert payload["latency_ms"] > 0
    # The feature vector is returned so an analyst can see why it scored.
    assert payload["features"]["tx_amount"] == pytest.approx(400.0)
    assert "terminal_risk_7d" in payload["features"]


def test_a_low_amount_does_not_alert(client: TestClient) -> None:
    payload = client.post("/score", json=_transaction(tx_amount=50.0)).json()
    assert payload["is_alert"] is False


def test_scoring_advances_feature_state(client: TestClient) -> None:
    """The second transaction for a customer must see the first."""
    client.post("/score", json=_transaction(transaction_id=1, tx_amount=100.0))
    second = client.post(
        "/score",
        json=_transaction(transaction_id=2, tx_amount=100.0, tx_datetime="2025-05-01T15:00:00"),
    ).json()
    assert second["features"]["customer_nb_tx_1d"] == pytest.approx(2.0)


def test_batch_scoring_is_order_dependent(client: TestClient) -> None:
    body = {
        "transactions": [
            _transaction(transaction_id=i, tx_datetime=f"2025-05-01T1{i}:00:00") for i in range(3)
        ]
    }
    payload = client.post("/score/batch", json=body).json()
    assert payload["count"] == 3
    counts = [p["features"]["customer_nb_tx_1d"] for p in payload["predictions"]]
    assert counts == [1.0, 2.0, 3.0]


def test_invalid_amount_is_rejected(client: TestClient) -> None:
    assert client.post("/score", json=_transaction(tx_amount=-5.0)).status_code == 422


def test_empty_batch_is_rejected(client: TestClient) -> None:
    assert client.post("/score/batch", json={"transactions": []}).status_code == 422


def test_label_feedback_updates_terminal_risk(client: TestClient) -> None:
    """The investigation loop is a separate endpoint because the label arrives late."""
    response = client.post(
        "/labels",
        json={
            "transaction_id": 99,
            "terminal_id": 3,
            "tx_datetime": "2025-04-01T10:00:00",
            "is_fraud": 1,
        },
    )
    assert response.status_code == 202
    scored = client.post("/score", json=_transaction(terminal_id=3, tx_amount=10.0)).json()
    assert scored["features"]["terminal_risk_30d"] > 0.0


def test_model_endpoint_lists_the_feature_contract(client: TestClient) -> None:
    payload = client.get("/model").json()
    assert payload["name"] == "stub"
    assert payload["features"][0] == "tx_amount"
    assert len(payload["features"]) == 15


def test_drift_without_data_explains_itself(client: TestClient) -> None:
    response = client.get("/drift")
    assert response.status_code == 400
    assert "no predictions" in response.json()["detail"].lower()


def test_metrics_exposes_the_monitoring_counters(client: TestClient) -> None:
    client.post("/score", json=_transaction(tx_amount=400.0))
    body = client.get("/metrics").text
    assert "rtml_transactions_scored_total" in body
    assert "rtml_alerts_total" in body
    assert "rtml_scoring_latency_seconds" in body
    assert "rtml_score_distribution" in body


def test_openapi_document_is_served(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/score", "/score/batch", "/labels", "/model", "/drift", "/metrics"} <= set(
        paths
    )
