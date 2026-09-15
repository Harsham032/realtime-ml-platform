"""Configuration.

Two layers, in increasing precedence: a YAML file describing the pipeline
(simulation, features, models, evaluation), and ``RTML_*`` environment variables
describing the deployment (databases, brokers, tracking URIs). Experiment
settings stay reviewable in version control; credentials stay out of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigurationError

StreamBackend = Literal["inprocess", "kafka"]


class RunConfig(BaseModel):
    name: str = "default"
    seed: int = 20260101
    output_dir: Path = Path("reports")


class SimulationSettings(BaseModel):
    n_customers: int = Field(default=5_000, gt=0)
    n_terminals: int = Field(default=10_000, gt=0)
    n_days: int = Field(default=183, gt=0)
    start_date: str = "2025-01-01"
    radius: float = Field(default=5.0, gt=0)


class SplitConfig(BaseModel):
    """Temporal split with a label-availability delay.

    Fraud labels arrive only after investigation, so a model deployed on day *d*
    can only have been trained on labels from before *d - delay*. Training right
    up to the test boundary would use labels that would not have existed, which
    inflates every metric. The delay period is dropped from both sides.
    """

    train_days: int = Field(default=112, gt=0)
    delay_days: int = Field(default=7, ge=0)
    test_days: int = Field(default=56, gt=0)
    validation_days: int = Field(default=14, gt=0)

    @model_validator(mode="after")
    def _fits_in_the_simulation(self) -> SplitConfig:
        if self.validation_days >= self.train_days:
            raise ValueError("validation_days must be smaller than train_days")
        return self

    @property
    def total_days(self) -> int:
        return self.train_days + self.delay_days + self.test_days


class FeatureConfig(BaseModel):
    customer_windows: list[int] = Field(default_factory=lambda: [1, 7, 30])
    terminal_windows: list[int] = Field(default_factory=lambda: [1, 7, 30])
    # Applied to terminal risk features only: they depend on labels, which the
    # customer count and amount features do not.
    risk_delay_days: int = Field(default=7, ge=0)
    night_start_hour: int = Field(default=0, ge=0, le=23)
    night_end_hour: int = Field(default=6, ge=0, le=23)

    @field_validator("customer_windows", "terminal_windows")
    @classmethod
    def _sorted_positive(cls, value: list[int]) -> list[int]:
        if not value or any(window <= 0 for window in value):
            raise ValueError("windows must be a non-empty list of positive day counts")
        return sorted(set(value))


class ModelConfig(BaseModel):
    """Which model families to train, and their hyperparameters."""

    enabled: list[str] = Field(
        default_factory=lambda: ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    )
    class_weight: Literal["balanced", "none"] = "balanced"
    logistic_regression: dict[str, Any] = Field(
        default_factory=lambda: {"C": 1.0, "max_iter": 2000}
    )
    random_forest: dict[str, Any] = Field(
        default_factory=lambda: {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 20}
    )
    xgboost: dict[str, Any] = Field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    )
    lightgbm: dict[str, Any] = Field(
        default_factory=lambda: {
            "n_estimators": 300,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "min_child_samples": 50,
        }
    )
    isolation_forest: dict[str, Any] = Field(
        default_factory=lambda: {"n_estimators": 150, "contamination": "auto"}
    )


class EvaluationConfig(BaseModel):
    """What to measure and how to pick an operating point."""

    # Fraud is heavily imbalanced, so PR-AUC is the headline and ROC-AUC is
    # reported alongside it rather than instead of it.
    primary_metric: Literal["pr_auc", "roc_auc", "card_precision_at_k"] = "pr_auc"
    # Cards an investigation team can review per day. Precision at this depth is
    # the metric an operations team actually feels.
    top_k_per_day: int = Field(default=100, gt=0)
    threshold_objective: Literal["f1", "recall_at_precision", "precision_at_recall"] = "f1"
    target_precision: float = Field(default=0.60, gt=0.0, lt=1.0)
    target_recall: float = Field(default=0.60, gt=0.0, lt=1.0)
    bootstrap_resamples: int = Field(default=1000, ge=0)


class DriftConfig(BaseModel):
    """Thresholds for the drift monitors."""

    # Population Stability Index convention: <0.1 stable, 0.1-0.25 moderate,
    # >0.25 significant. These are the widely used operational bands.
    psi_warn: float = Field(default=0.10, gt=0)
    psi_alert: float = Field(default=0.25, gt=0)
    ks_alert_pvalue: float = Field(default=0.01, gt=0, lt=1)
    bins: int = Field(default=10, gt=1)
    reference_days: int = Field(default=14, gt=0)
    comparison_days: int = Field(default=7, gt=0)

    @model_validator(mode="after")
    def _bands_ordered(self) -> DriftConfig:
        if self.psi_warn >= self.psi_alert:
            raise ValueError("psi_warn must be below psi_alert")
        return self


class PromotionConfig(BaseModel):
    """Gates a challenger must pass before it replaces the champion.

    Retraining on a schedule without gates is how a pipeline quietly ships a
    worse model. Every gate here is a measured comparison, not a heuristic.
    """

    min_pr_auc: float = Field(default=0.30, ge=0.0, le=1.0)
    min_pr_auc_improvement: float = Field(default=0.005, ge=0.0)
    max_latency_regression_ms: float = Field(default=5.0, ge=0.0)
    min_evaluation_frauds: int = Field(default=50, gt=0)
    require_calibration: bool = True
    max_brier_regression: float = Field(default=0.002, ge=0.0)


class StreamingConfig(BaseModel):
    backend: StreamBackend = "inprocess"
    partitions: int = Field(default=4, gt=0)
    batch_size: int = Field(default=500, gt=0)
    poll_timeout_seconds: float = Field(default=1.0, gt=0)
    max_events: int | None = None
    speedup: float = Field(default=0.0, ge=0.0)


class PipelineConfig(BaseModel):
    """Experiment configuration loaded from YAML."""

    run: RunConfig = Field(default_factory=RunConfig)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    split: SplitConfig = Field(default_factory=SplitConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)

    @model_validator(mode="after")
    def _split_fits_simulation(self) -> PipelineConfig:
        if self.split.total_days > self.simulation.n_days:
            raise ValueError(
                f"the split needs {self.split.total_days} days but the simulation "
                f"generates only {self.simulation.n_days}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigurationError(f"configuration file not found: {config_path}")
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - depends on malformed input
            raise ConfigurationError(f"could not parse {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError(f"{config_path} must contain a YAML mapping")
        try:
            return cls.model_validate(raw)
        except Exception as exc:
            raise ConfigurationError(f"invalid configuration in {config_path}: {exc}") from exc

    def with_overrides(self, overrides: dict[str, Any]) -> PipelineConfig:
        """Return a copy with ``dotted.key=value`` overrides applied."""
        if not overrides:
            return self
        merged = self.model_dump(mode="python")
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            cursor: Any = merged
            for part in parts[:-1]:
                if not isinstance(cursor, dict) or part not in cursor:
                    raise ConfigurationError(f"unknown configuration key: {dotted_key}")
                cursor = cursor[part]
            leaf = parts[-1]
            if not isinstance(cursor, dict) or leaf not in cursor:
                raise ConfigurationError(f"unknown configuration key: {dotted_key}")
            cursor[leaf] = yaml.safe_load(value) if isinstance(value, str) else value
        try:
            return PipelineConfig.model_validate(merged)
        except Exception as exc:
            raise ConfigurationError(f"invalid override: {exc}") from exc


class Settings(BaseSettings):
    """Deployment settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="RTML_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    database_url: str = "sqlite:///data/processed/rtml.sqlite3"

    # MLflow's filesystem backend is deprecated and refuses to start in recent
    # releases; SQLite runs offline with no server and supports the registry.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_artifact_root: str = "./mlartifacts"
    mlflow_experiment: str = "transaction-risk"

    stream_backend: StreamBackend = "inprocess"
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_transactions_topic: str = "transactions"
    kafka_predictions_topic: str = "predictions"
    kafka_consumer_group: str = "scoring-workers"

    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = False

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"prod", "production"}


def load_settings() -> Settings:
    """Read deployment settings from the environment."""
    return Settings()
