"""Scoring API.

Exposes the same model the streaming consumer runs, through the same online
feature store, so a synchronous caller and the stream cannot disagree about a
transaction's features.

The service holds feature state in process. That is the right shape for this
design - the online store is per-partition state keyed by customer - and it is
also the constraint to be aware of when scaling: several replicas each hold
their own view, so either route a customer consistently to one replica or move
the store into Redis. ``docs/architecture.md`` sets out both.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from .. import __version__
from ..config import PipelineConfig, Settings, load_settings
from ..data.store import PredictionStore, redact_url
from ..errors import RtmlError
from ..features.online import OnlineFeatureStore
from ..logging_utils import configure_logging, get_logger
from .schemas import (
    BatchRequest,
    BatchResponse,
    DriftResponse,
    HealthResponse,
    LabelRequest,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)

logger = get_logger(__name__)

SCORED = Counter("rtml_transactions_scored_total", "Transactions scored", ["model"])
ALERTS = Counter("rtml_alerts_total", "Transactions scored above the threshold", ["model"])
LATENCY = Histogram(
    "rtml_scoring_latency_seconds",
    "End-to-end scoring latency",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
SCORE_DISTRIBUTION = Histogram(
    "rtml_score_distribution",
    "Model score distribution, the leading indicator for prediction drift",
    buckets=(0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0),
)
LABELS_RECEIVED = Counter("rtml_labels_received_total", "Confirmed outcomes fed back")
FEATURE_STATE = Gauge("rtml_feature_state_entities", "Entities tracked online", ["entity"])


class ServiceState:
    """Objects shared by every request."""

    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.config: PipelineConfig | None = None
        self.model: Any = None
        self.model_name: str = ""
        self.model_version: str = ""
        self.threshold: float = 0.5
        self.store: OnlineFeatureStore | None = None
        self.database: PredictionStore | None = None
        self.metrics: dict[str, float] = {}
        self.error: str | None = None


state = ServiceState()

# Where scripts/train.py writes the serving bundle.
DEFAULT_MODEL_PATH = Path("artifacts/champion.joblib")


def load_champion(path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any] | None:
    """Load the serving bundle written by the training pipeline."""
    if not path.is_file():
        return None
    import joblib

    return joblib.load(path)


def build_state(config_path: str = "configs/default.yaml") -> ServiceState:
    """Load configuration, model and feature store."""
    settings = load_settings()
    configure_logging(settings.log_level, json_output=settings.is_production)
    state.settings = settings

    try:
        config = PipelineConfig.from_yaml(config_path)
        state.config = config
        state.store = OnlineFeatureStore(config.features)

        database = PredictionStore(settings.database_url)
        database.create_all()
        state.database = database

        bundle = load_champion()
        if bundle is None:
            # The service still starts so /health can explain why it cannot
            # score, rather than crash-looping with no diagnosis.
            state.error = f"no model at {DEFAULT_MODEL_PATH}; run `make train` to produce one"
            logger.warning("service_started_without_model", path=str(DEFAULT_MODEL_PATH))
        else:
            state.model = bundle["model"]
            state.model_name = bundle.get("name", "unknown")
            state.model_version = str(bundle.get("version", "0"))
            state.threshold = float(bundle.get("threshold", 0.5))
            state.metrics = bundle.get("metrics", {})
            state.error = None
            logger.info(
                "service_ready",
                model=state.model_name,
                version=state.model_version,
                threshold=round(state.threshold, 4),
            )
    except RtmlError as exc:
        state.error = str(exc)
        logger.error("service_startup_failed", error=str(exc))
    return state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    build_state()
    yield
    if state.database is not None:
        state.database.close()


app = FastAPI(
    title="Transaction risk scoring",
    description=(
        "Real-time fraud scoring over streaming transactions, with online "
        "features, drift monitoring and a champion/challenger model registry."
    ),
    version=__version__,
    lifespan=lifespan,
)


def require_model() -> Any:
    if state.model is None or state.store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.error or "no model is loaded",
        )
    return state.model


def _score(request: TransactionRequest) -> PredictionResponse:
    """Score one transaction through the shared online feature store."""
    assert state.store is not None
    when = pd.Timestamp(request.tx_datetime)
    timestamp = when.timestamp()

    start = time.perf_counter()
    features = state.store.compute(
        request.customer_id, request.terminal_id, timestamp, request.tx_amount, when
    )
    score = float(state.model.predict_proba(state.store.vector(features))[0])
    latency_ms = (time.perf_counter() - start) * 1000.0

    # State advances only after scoring, exactly as in the streaming consumer.
    state.store.observe_transaction(request.customer_id, timestamp, request.tx_amount)

    SCORED.labels(model=state.model_name).inc()
    LATENCY.observe(latency_ms / 1000.0)
    SCORE_DISTRIBUTION.observe(score)
    is_alert = score >= state.threshold
    if is_alert:
        ALERTS.labels(model=state.model_name).inc()

    return PredictionResponse(
        transaction_id=request.transaction_id,
        score=score,
        is_alert=is_alert,
        threshold=state.threshold,
        model_name=state.model_name,
        model_version=state.model_version,
        latency_ms=latency_ms,
        features={k: float(v) for k, v in features.items()},
    )


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Readiness, the loaded model and how much online state is retained."""
    feature_state = state.store.state_size() if state.store else {}
    for entity in ("customers_tracked", "terminals_tracked"):
        if entity in feature_state:
            FEATURE_STATE.labels(entity=entity).set(feature_state[entity])
    return HealthResponse(
        status="ok" if state.model is not None else "degraded",
        version=__version__,
        model_loaded=state.model is not None,
        model_name=state.model_name,
        model_version=state.model_version,
        threshold=state.threshold,
        feature_state=feature_state,
        database=redact_url(state.settings.database_url) if state.settings else "",
        detail=state.error or "",
    )


@app.get("/metrics", tags=["operations"])
def metrics() -> Response:
    """Prometheus exposition."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/score", response_model=PredictionResponse, tags=["scoring"])
def score(request: TransactionRequest, _: Any = Depends(require_model)) -> PredictionResponse:
    """Score a single transaction."""
    prediction = _score(request)
    if state.database is not None:
        state.database.write_predictions(
            [
                {
                    "transaction_id": prediction.transaction_id,
                    "customer_id": request.customer_id,
                    "terminal_id": request.terminal_id,
                    "tx_datetime": request.tx_datetime,
                    "tx_day": 0,
                    "tx_amount": request.tx_amount,
                    "score": prediction.score,
                    "threshold": prediction.threshold,
                    "is_alert": prediction.is_alert,
                    "scoring_latency_ms": prediction.latency_ms,
                }
            ],
            model_name=state.model_name,
            model_version=state.model_version,
        )
    return prediction


@app.post("/score/batch", response_model=BatchResponse, tags=["scoring"])
def score_batch(request: BatchRequest, _: Any = Depends(require_model)) -> BatchResponse:
    """Score several transactions in one call.

    Order matters: transactions are scored in the order supplied, and each
    updates the feature state for those after it.
    """
    start = time.perf_counter()
    predictions = [_score(transaction) for transaction in request.transactions]
    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        total_latency_ms=(time.perf_counter() - start) * 1000.0,
    )


@app.post("/labels", status_code=status.HTTP_202_ACCEPTED, tags=["feedback"])
def submit_label(request: LabelRequest) -> dict[str, Any]:
    """Feed a confirmed outcome back into the terminal risk features."""
    if state.store is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="not ready")
    state.store.observe_label(
        request.terminal_id, pd.Timestamp(request.tx_datetime).timestamp(), request.is_fraud
    )
    LABELS_RECEIVED.inc()
    return {"accepted": True, "transaction_id": request.transaction_id}


@app.get("/model", response_model=ModelInfoResponse, tags=["operations"])
def model_info(_: Any = Depends(require_model)) -> ModelInfoResponse:
    """What is deployed, and what the registry holds."""
    assert state.store is not None
    versions: list[dict[str, Any]] = []
    if state.settings is not None:
        try:
            from ..models.registry import ModelRegistry

            versions = ModelRegistry(state.settings.mlflow_tracking_uri).versions("fraud-scorer")
        except Exception as exc:  # the registry is optional at serving time
            logger.warning("registry_unavailable", error=str(exc))
    return ModelInfoResponse(
        name=state.model_name,
        version=state.model_version,
        threshold=state.threshold,
        features=state.store.names,
        metrics=state.metrics,
        registry_versions=versions,
    )


@app.get("/drift", response_model=DriftResponse, tags=["monitoring"])
def drift() -> DriftResponse:
    """Drift between the two most recent windows of stored predictions."""
    if state.database is None or state.config is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="not ready")

    frame = state.database.read_predictions()
    if frame.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="no predictions stored yet; run the stream first",
        )
    from ..monitoring.drift import drift_over_time

    try:
        report = drift_over_time(frame, ["score", "tx_amount"], state.config.drift)
    except RtmlError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return DriftResponse.model_validate(report.to_dict())
