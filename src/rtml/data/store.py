"""Persistence for transactions, predictions, drift and deployments.

One SQLAlchemy Core schema runs on SQLite and PostgreSQL, so the database the
tests exercise cannot drift from the one that gets deployed. ``sql/schema.sql``
is the PostgreSQL-native form with the index definitions Core cannot express
portably.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from ..logging_utils import get_logger

logger = get_logger(__name__)

metadata = MetaData()

transactions_table = Table(
    "transactions",
    metadata,
    Column("transaction_id", BigInteger, primary_key=True),
    Column("customer_id", BigInteger, nullable=False, index=True),
    Column("terminal_id", BigInteger, nullable=False, index=True),
    Column("tx_datetime", DateTime, nullable=False),
    Column("tx_day", Integer, nullable=False, index=True),
    Column("tx_amount", Float, nullable=False),
    Column("is_fraud", SmallInteger, nullable=False, default=0),
    Column("fraud_scenario", SmallInteger, nullable=False, default=0),
)

predictions_table = Table(
    "predictions",
    metadata,
    Column("prediction_id", Integer, primary_key=True, autoincrement=True),
    Column("transaction_id", BigInteger, nullable=False, index=True),
    Column("customer_id", BigInteger, nullable=False),
    Column("terminal_id", BigInteger, nullable=False),
    Column("tx_datetime", DateTime, nullable=False),
    Column("tx_day", Integer, nullable=False, index=True),
    Column("tx_amount", Float, nullable=False),
    Column("score", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("is_alert", Boolean, nullable=False),
    Column("model_name", String(64), nullable=False, default=""),
    Column("model_version", String(32), nullable=False, default=""),
    Column("latency_ms", Float, nullable=False, default=0.0),
    Column("is_fraud", SmallInteger, nullable=True),
)

drift_table = Table(
    "drift_observations",
    metadata,
    Column("observation_id", Integer, primary_key=True, autoincrement=True),
    Column("feature", String(64), nullable=False, index=True),
    Column("psi", Float, nullable=False),
    Column("ks_statistic", Float, nullable=False),
    Column("ks_pvalue", Float, nullable=False),
    Column("severity", String(16), nullable=False),
    Column("reference_start", Integer, nullable=False),
    Column("reference_end", Integer, nullable=False),
    Column("comparison_start", Integer, nullable=False),
    Column("comparison_end", Integer, nullable=False),
)

deployments_table = Table(
    "model_deployments",
    metadata,
    Column("deployment_id", Integer, primary_key=True, autoincrement=True),
    Column("model_name", String(64), nullable=False, index=True),
    Column("model_version", String(32), nullable=False),
    Column("alias", String(32), nullable=False),
    Column("threshold", Float, nullable=False),
    Column("pr_auc", Float, nullable=True),
    Column("promotion_reason", Text, nullable=False, default=""),
)


def redact_url(url: str) -> str:
    """Strip credentials from a database URL before displaying or logging it.

    The health endpoint reports which database is configured; without this it
    would publish the password to anyone who can reach the endpoint.
    """
    import re

    return re.sub(r"(//[^:/@]+):[^@]*@", r"\1:***@", url)


class PredictionStore:
    """Read/write access to every table."""

    def __init__(self, database_url: str, *, echo: bool = False) -> None:
        if database_url.startswith("sqlite:///"):
            db_path = Path(database_url.removeprefix("sqlite:///"))
            if str(db_path) != ":memory:":
                db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(database_url, echo=echo, future=True)
        self.database_url = database_url

    def create_all(self) -> None:
        metadata.create_all(self.engine)

    def drop_all(self) -> None:
        metadata.drop_all(self.engine)

    # ---------------------------------------------------------- transactions

    def write_transactions(self, frame: pd.DataFrame, *, chunk_size: int = 20_000) -> int:
        """Insert transactions, replacing any rows with the same ids."""
        if frame.empty:
            return 0
        columns = [c.name for c in transactions_table.columns if c.name in frame.columns]
        # to_dict("records") is typed with Hashable keys because a DataFrame may
        # be keyed by anything; here the keys are `columns`, which are the
        # table's column names and therefore str.
        payload = cast(list[dict[str, Any]], frame[columns].to_dict("records"))
        ids = [int(row["transaction_id"]) for row in payload]

        with self.engine.begin() as connection:
            for start in range(0, len(ids), chunk_size):
                connection.execute(
                    delete(transactions_table).where(
                        transactions_table.c.transaction_id.in_(ids[start : start + chunk_size])
                    )
                )
            for start in range(0, len(payload), chunk_size):
                connection.execute(insert(transactions_table), payload[start : start + chunk_size])
        return len(payload)

    # ----------------------------------------------------------- predictions

    def write_predictions(
        self,
        predictions: Iterable[dict[str, Any]],
        *,
        model_name: str = "",
        model_version: str = "",
    ) -> int:
        rows = []
        for prediction in predictions:
            rows.append(
                {
                    "transaction_id": int(prediction["transaction_id"]),
                    "customer_id": int(prediction["customer_id"]),
                    "terminal_id": int(prediction["terminal_id"]),
                    "tx_datetime": pd.Timestamp(prediction["tx_datetime"]).to_pydatetime(),
                    "tx_day": int(prediction.get("tx_day", 0)),
                    "tx_amount": float(prediction["tx_amount"]),
                    "score": float(prediction["score"]),
                    "threshold": float(prediction.get("threshold", 0.5)),
                    "is_alert": bool(prediction.get("is_alert", False)),
                    "model_name": prediction.get("model_name", model_name),
                    "model_version": prediction.get("model_version", model_version),
                    "latency_ms": float(prediction.get("scoring_latency_ms", 0.0)),
                    "is_fraud": prediction.get("is_fraud"),
                }
            )
        if not rows:
            return 0
        with self.engine.begin() as connection:
            connection.execute(insert(predictions_table), rows)
        return len(rows)

    def read_predictions(self, *, limit: int | None = None, day: int | None = None) -> pd.DataFrame:
        statement = select(predictions_table)
        if day is not None:
            statement = statement.where(predictions_table.c.tx_day == day)
        statement = statement.order_by(predictions_table.c.tx_datetime)
        if limit:
            statement = statement.limit(limit)
        with self.engine.connect() as connection:
            return pd.DataFrame([dict(row) for row in connection.execute(statement).mappings()])

    def prediction_count(self) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.execute(select(func.count()).select_from(predictions_table)).scalar_one()
            )

    def alert_count(self) -> int:
        with self.engine.connect() as connection:
            return int(
                connection.execute(
                    select(func.count())
                    .select_from(predictions_table)
                    .where(predictions_table.c.is_alert.is_(True))
                ).scalar_one()
            )

    # ----------------------------------------------------------------- drift

    def write_drift(self, report: Any) -> int:
        rows = [
            {
                "feature": result.feature,
                "psi": result.psi,
                "ks_statistic": result.ks_statistic,
                "ks_pvalue": result.ks_pvalue,
                "severity": result.severity,
                "reference_start": report.reference_window[0],
                "reference_end": report.reference_window[1],
                "comparison_start": report.comparison_window[0],
                "comparison_end": report.comparison_window[1],
            }
            for result in report.results
        ]
        if not rows:
            return 0
        with self.engine.begin() as connection:
            connection.execute(insert(drift_table), rows)
        return len(rows)

    def read_drift(self, *, limit: int = 100) -> pd.DataFrame:
        statement = select(drift_table).order_by(drift_table.c.observation_id.desc()).limit(limit)
        with self.engine.connect() as connection:
            return pd.DataFrame([dict(row) for row in connection.execute(statement).mappings()])

    # ----------------------------------------------------------- deployments

    def record_deployment(
        self,
        model_name: str,
        model_version: str,
        alias: str,
        threshold: float,
        *,
        pr_auc: float | None = None,
        reason: str = "",
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                insert(deployments_table),
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "alias": alias,
                    "threshold": threshold,
                    "pr_auc": pr_auc,
                    "promotion_reason": reason,
                },
            )

    def deployment_history(self, *, limit: int = 50) -> pd.DataFrame:
        statement = (
            select(deployments_table)
            .order_by(deployments_table.c.deployment_id.desc())
            .limit(limit)
        )
        with self.engine.connect() as connection:
            return pd.DataFrame([dict(row) for row in connection.execute(statement).mappings()])

    def close(self) -> None:
        self.engine.dispose()


def postgres_schema_sql() -> str:
    """The PostgreSQL-native DDL shipped with the package."""
    return (Path(__file__).parent / "sql" / "schema.sql").read_text(encoding="utf-8")
