"""Shared fixtures.

Every fixture builds from the simulator rather than a committed data file: the
generator is seeded, so the data is identical on every machine and no test needs
a download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rtml.config import PipelineConfig  # noqa: E402
from rtml.data.simulator import SimulationConfig, SimulationResult, simulate  # noqa: E402
from rtml.features.engineering import build_features  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def config() -> PipelineConfig:
    """A small configuration whose splits still contain enough fraud to score."""
    return PipelineConfig.from_yaml(REPO_ROOT / "configs" / "fast.yaml").with_overrides(
        {
            "simulation.n_customers": 400,
            "simulation.n_terminals": 800,
            "simulation.n_days": 70,
            "split.train_days": 42,
            "split.delay_days": 7,
            "split.test_days": 14,
            "split.validation_days": 10,
            "evaluation.bootstrap_resamples": 0,
            "models.enabled": ["lightgbm"],
        }
    )


@pytest.fixture(scope="session")
def simulation(config: PipelineConfig) -> SimulationResult:
    return simulate(SimulationConfig(**config.simulation.model_dump(), seed=config.run.seed))


@pytest.fixture(scope="session")
def transactions(simulation: SimulationResult) -> pd.DataFrame:
    return simulation.transactions


@pytest.fixture(scope="session")
def features(transactions: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    return build_features(transactions, config.features)


@pytest.fixture
def linear_frame() -> pd.DataFrame:
    """One customer and one terminal, one transaction per day.

    Deliberately degenerate so window arithmetic can be reasoned about by hand.
    """
    days = 40
    base = pd.Timestamp("2025-01-01")
    return pd.DataFrame(
        {
            "transaction_id": range(days),
            "customer_id": [0] * days,
            "terminal_id": [0] * days,
            "tx_datetime": [base + pd.Timedelta(days=i) for i in range(days)],
            "tx_amount": [50.0] * days,
            "tx_day": list(range(days)),
            "is_fraud": [0] * days,
        }
    )
