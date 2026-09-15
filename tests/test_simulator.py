"""The transaction generator."""

from __future__ import annotations

import numpy as np
import pytest

from rtml.data.simulator import (
    SimulationConfig,
    associate_terminals,
    generate_customer_profiles,
    generate_terminal_profiles,
    simulate,
)
from rtml.errors import DataGenerationError


def test_generation_is_deterministic() -> None:
    """The seed fixes the data, which is what makes every metric reproducible."""
    config = SimulationConfig(n_customers=150, n_terminals=300, n_days=20, seed=7)
    first, second = simulate(config).transactions, simulate(config).transactions
    assert first.equals(second)


def test_a_different_seed_gives_different_data() -> None:
    a = simulate(SimulationConfig(n_customers=150, n_terminals=300, n_days=20, seed=1)).transactions
    b = simulate(SimulationConfig(n_customers=150, n_terminals=300, n_days=20, seed=2)).transactions
    assert not a.equals(b)


def test_customer_profiles_respect_their_bounds() -> None:
    rng = np.random.default_rng(0)
    config = SimulationConfig(n_customers=500)
    customers = generate_customer_profiles(config, rng)
    assert len(customers) == 500
    assert customers["mean_amount"].between(config.amount_low, config.amount_high).all()
    # The handbook sets the spread to half the mean.
    assert np.allclose(customers["std_amount"], customers["mean_amount"] / 2)
    assert customers["mean_nb_tx_per_day"].between(config.tx_rate_low, config.tx_rate_high).all()


def test_terminal_association_respects_the_radius() -> None:
    rng = np.random.default_rng(0)
    config = SimulationConfig(n_customers=60, n_terminals=200, radius=5.0)
    customers = generate_customer_profiles(config, rng)
    terminals = generate_terminal_profiles(config, rng)
    flat, offsets = associate_terminals(customers, terminals, config.radius)

    for index in range(len(customers)):
        assigned = flat[offsets[index] : offsets[index + 1]]
        if len(assigned) == 0:
            continue
        dx = terminals.loc[assigned, "x_terminal"].to_numpy() - customers.loc[index, "x_customer"]
        dy = terminals.loc[assigned, "y_terminal"].to_numpy() - customers.loc[index, "y_customer"]
        assert np.all(dx**2 + dy**2 < config.radius**2)


def test_transactions_fall_inside_the_simulated_window() -> None:
    result = simulate(SimulationConfig(n_customers=200, n_terminals=400, n_days=15, seed=3))
    frame = result.transactions
    assert frame["tx_day"].between(0, 14).all()
    assert frame["tx_time_seconds"].between(0, 86_399).all()
    assert (frame["tx_amount"] > 0).all()
    assert frame["tx_datetime"].is_monotonic_increasing


def test_every_fraud_scenario_is_represented() -> None:
    result = simulate(SimulationConfig(n_customers=600, n_terminals=1200, n_days=60, seed=5))
    scenarios = result.transactions.loc[result.transactions["is_fraud"] == 1, "fraud_scenario"]
    assert set(scenarios.unique()) == {1, 2, 3}


def test_scenario_one_is_the_amount_rule() -> None:
    """Every transaction above the threshold must be fraudulent."""
    config = SimulationConfig(n_customers=400, n_terminals=800, n_days=40, seed=9)
    frame = simulate(config).transactions
    above = frame[frame["tx_amount"] > config.scenario1_amount]
    assert len(above) > 0
    assert (above["is_fraud"] == 1).all()


def test_fraud_rate_is_realistically_small() -> None:
    result = simulate(SimulationConfig(n_customers=2000, n_terminals=4000, n_days=60, seed=11))
    rate = result.transactions["is_fraud"].mean()
    assert 0.001 < rate < 0.05, f"fraud rate {rate:.4%} is outside a plausible range"


def test_non_fraud_transactions_are_the_vast_majority() -> None:
    result = simulate(SimulationConfig(n_customers=500, n_terminals=1000, n_days=30, seed=13))
    assert result.transactions["is_fraud"].mean() < 0.5


def test_ids_are_unique_and_contiguous() -> None:
    frame = simulate(
        SimulationConfig(n_customers=200, n_terminals=400, n_days=15, seed=4)
    ).transactions
    assert frame["transaction_id"].is_unique
    assert frame["transaction_id"].min() == 0
    assert frame["transaction_id"].max() == len(frame) - 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_customers": 0},
        {"n_terminals": 0},
        {"n_days": 0},
        {"radius": 0.0},
        {"scenario3_fraction": 1.5},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(DataGenerationError):
        SimulationConfig(**kwargs)  # type: ignore[arg-type]


def test_an_unreachable_radius_fails_loudly() -> None:
    """Silently producing zero transactions would be far worse than an error."""
    with pytest.raises(DataGenerationError):
        simulate(SimulationConfig(n_customers=20, n_terminals=20, n_days=5, radius=0.001, seed=1))


def test_reported_statistics_match_the_data() -> None:
    result = simulate(SimulationConfig(n_customers=300, n_terminals=600, n_days=20, seed=6))
    assert result.stats["transactions"] == len(result.transactions)
    assert result.stats["fraud_count"] == result.transactions["is_fraud"].sum()
    assert result.stats["fraud_rate"] == pytest.approx(result.transactions["is_fraud"].mean())
