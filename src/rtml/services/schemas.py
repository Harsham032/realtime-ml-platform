"""Request and response models for the scoring API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """One transaction to score."""

    transaction_id: int
    customer_id: int
    terminal_id: int
    tx_datetime: datetime
    tx_amount: float = Field(gt=0, description="Transaction amount in account currency")

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": 1,
                "customer_id": 42,
                "terminal_id": 7,
                "tx_datetime": "2025-05-01T14:23:00",
                "tx_amount": 129.99,
            }
        }
    }


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """A risk score and the decision it implies."""

    transaction_id: int
    score: float = Field(ge=0.0, le=1.0)
    is_alert: bool
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float
    # Returned so an analyst can see why a transaction scored the way it did.
    features: dict[str, float] = Field(default_factory=dict)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    total_latency_ms: float


class LabelRequest(BaseModel):
    """A confirmed outcome fed back from the investigation queue.

    This is the feedback loop that keeps terminal risk features current. It is
    a separate endpoint from scoring because in production the label arrives
    days later, through a different system.
    """

    transaction_id: int
    terminal_id: int
    tx_datetime: datetime
    is_fraud: int = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    model_name: str
    model_version: str
    threshold: float
    feature_state: dict[str, int] = Field(default_factory=dict)
    database: str = ""
    detail: str = ""


class DriftResponse(BaseModel):
    reference_window: list[int]
    comparison_window: list[int]
    features_monitored: int
    features_drifted: int
    features_significant: int
    max_psi: float
    alert: bool
    features: list[dict[str, Any]]


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    threshold: float
    features: list[str]
    metrics: dict[str, float] = Field(default_factory=dict)
    registry_versions: list[dict[str, Any]] = Field(default_factory=list)
