"""Transaction simulator.

Implements the generation procedure described in *Reproducible Machine Learning
for Credit Card Fraud Detection* (Le Borgne, Siblini, Lebichot and Bontempi,
Université Libre de Bruxelles, 2022), available at
https://fraud-detection-handbook.github.io/fraud-detection-handbook/.

Using a simulator rather than a downloaded dataset is deliberate and is what the
handbook itself does: real card transaction data is not publicly redistributable,
and a generator makes the whole pipeline reproducible from a seed with no
credentialed download. What the simulator gives up is equally worth stating - it
produces the fraud patterns it was told to produce, so a model's score here
measures whether the pipeline can recover known injected structure, not whether
it would catch real fraud.

The reference implementation is a per-customer, per-day Python loop. This one is
vectorised over NumPy: the draws are the same distributions with the same
parameters, but all customers and days are drawn at once, which is roughly two
orders of magnitude faster and makes million-transaction runs practical.

Generation procedure
--------------------
*Customers* get a uniform position on a 100x100 grid, a mean spend drawn
uniformly from [5, 100] with standard deviation half the mean, and a mean
transaction rate drawn uniformly from [0, 4] per day.

*Terminals* get a uniform position on the same grid. A customer may transact at
any terminal within ``radius`` of their position, which produces a realistic
spatial locality: most customers use a handful of terminals.

*Transactions* per customer per day are Poisson distributed. Times are Gaussian
about midday and rejected outside the day. Amounts are Gaussian about the
customer's mean, resampled uniformly when negative.

*Fraud* is injected through three scenarios, each modelling a different real
compromise, and each requiring different features to detect:

1. Any amount above ``scenario1_amount`` is fraudulent. Detectable from the
   transaction alone.
2. Two terminals are drawn each day and every transaction on them for the next
   ``scenario2_days`` days is fraudulent - a compromised terminal. Requires
   terminal-level history.
3. Three customers are drawn each day and, for the next ``scenario3_days`` days,
   a third of their transactions have their amount multiplied by
   ``scenario3_multiplier`` and are fraudulent - stolen credentials spent at
   unusual size. Requires customer-level spending history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..errors import DataGenerationError
from ..logging_utils import get_logger

logger = get_logger(__name__)

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters of the generation procedure.

    Defaults follow the handbook except where noted.
    """

    n_customers: int = 5_000
    n_terminals: int = 10_000
    n_days: int = 183
    start_date: str = "2025-01-01"
    radius: float = 5.0
    seed: int = 20260101

    # Customer profile distributions.
    amount_low: float = 5.0
    amount_high: float = 100.0
    tx_rate_low: float = 0.0
    tx_rate_high: float = 4.0

    # Fraud scenarios.
    scenario1_amount: float = 220.0
    scenario2_terminals_per_day: int = 2
    scenario2_days: int = 28
    scenario3_customers_per_day: int = 3
    scenario3_days: int = 14
    scenario3_multiplier: float = 5.0
    scenario3_fraction: float = 1 / 3

    def __post_init__(self) -> None:
        if self.n_customers <= 0 or self.n_terminals <= 0 or self.n_days <= 0:
            raise DataGenerationError("n_customers, n_terminals and n_days must all be positive")
        if self.radius <= 0:
            raise DataGenerationError("radius must be positive")
        if not 0.0 <= self.scenario3_fraction <= 1.0:
            raise DataGenerationError("scenario3_fraction must lie in [0, 1]")


@dataclass
class SimulationResult:
    """Generated transactions plus the profiles that produced them."""

    transactions: pd.DataFrame
    customers: pd.DataFrame
    terminals: pd.DataFrame
    stats: dict[str, float] = field(default_factory=dict)


def generate_customer_profiles(config: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Draw customer positions, spend levels and transaction rates."""
    n = config.n_customers
    mean_amount = rng.uniform(config.amount_low, config.amount_high, n)
    return pd.DataFrame(
        {
            "customer_id": np.arange(n, dtype=np.int64),
            "x_customer": rng.uniform(0, 100, n),
            "y_customer": rng.uniform(0, 100, n),
            "mean_amount": mean_amount,
            "std_amount": mean_amount / 2.0,
            "mean_nb_tx_per_day": rng.uniform(config.tx_rate_low, config.tx_rate_high, n),
        }
    )


def generate_terminal_profiles(config: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Draw terminal positions."""
    n = config.n_terminals
    return pd.DataFrame(
        {
            "terminal_id": np.arange(n, dtype=np.int64),
            "x_terminal": rng.uniform(0, 100, n),
            "y_terminal": rng.uniform(0, 100, n),
        }
    )


def associate_terminals(
    customers: pd.DataFrame, terminals: pd.DataFrame, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map each customer to the terminals within ``radius`` of them.

    Returns a flat array of terminal ids and the offsets that slice it per
    customer, which is the ragged layout the vectorised draw needs. Computing
    the full pairwise distance matrix would be O(customers x terminals) in
    memory, so customers are processed in blocks.
    """
    terminal_xy = terminals[["x_terminal", "y_terminal"]].to_numpy()
    customer_xy = customers[["x_customer", "y_customer"]].to_numpy()
    squared_radius = radius * radius

    block = max(1, int(2_000_000 / max(len(terminal_xy), 1)))
    per_customer: list[np.ndarray] = []
    for start in range(0, len(customer_xy), block):
        chunk = customer_xy[start : start + block]
        squared = ((chunk[:, None, :] - terminal_xy[None, :, :]) ** 2).sum(axis=2)
        rows, cols = np.nonzero(squared < squared_radius)
        for index in range(len(chunk)):
            per_customer.append(cols[rows == index].astype(np.int64))

    counts = np.array([len(item) for item in per_customer], dtype=np.int64)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    flat = np.concatenate(per_customer) if counts.sum() else np.empty(0, dtype=np.int64)
    return flat, offsets


def _draw_transactions(
    config: SimulationConfig,
    customers: pd.DataFrame,
    flat_terminals: np.ndarray,
    offsets: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw every transaction in one vectorised pass."""
    n_customers, n_days = config.n_customers, config.n_days

    rates = customers["mean_nb_tx_per_day"].to_numpy()
    counts = rng.poisson(rates[:, None], size=(n_customers, n_days))

    # Customers with no terminal in range cannot transact.
    available = np.diff(offsets)
    counts[available == 0, :] = 0

    total = int(counts.sum())
    if total == 0:
        raise DataGenerationError(
            "the parameters produced no transactions; increase radius, n_terminals or n_days"
        )

    flat_counts = counts.reshape(-1)
    customer_ids = np.repeat(np.repeat(np.arange(n_customers), n_days), flat_counts)
    days = np.repeat(np.tile(np.arange(n_days), n_customers), flat_counts)

    # Times are Gaussian about midday; the handbook rejects draws outside the
    # day, which is done here by resampling uniformly rather than dropping rows
    # so that the Poisson counts are preserved exactly.
    seconds = rng.normal(SECONDS_PER_DAY / 2, 20_000, total)
    outside = (seconds < 0) | (seconds >= SECONDS_PER_DAY)
    seconds[outside] = rng.uniform(0, SECONDS_PER_DAY, int(outside.sum()))

    mean_amount = customers["mean_amount"].to_numpy()[customer_ids]
    std_amount = customers["std_amount"].to_numpy()[customer_ids]
    amounts = rng.normal(mean_amount, std_amount)
    negative = amounts < 0
    amounts[negative] = rng.uniform(0, mean_amount[negative] * 2)
    amounts = np.round(amounts, 2)

    # One uniform draw into each customer's own terminal segment.
    picks = (rng.random(total) * available[customer_ids]).astype(np.int64)
    terminal_ids = flat_terminals[offsets[customer_ids] + picks]

    frame = pd.DataFrame(
        {
            "customer_id": customer_ids.astype(np.int64),
            "terminal_id": terminal_ids.astype(np.int64),
            "tx_day": days.astype(np.int32),
            "tx_time_seconds": seconds.astype(np.int64),
            "tx_amount": amounts.astype(np.float64),
        }
    )
    frame["tx_datetime"] = pd.to_datetime(config.start_date) + pd.to_timedelta(
        frame["tx_day"].to_numpy() * SECONDS_PER_DAY + frame["tx_time_seconds"].to_numpy(),
        unit="s",
    )
    return frame.sort_values("tx_datetime", kind="stable", ignore_index=True)


def inject_fraud(
    transactions: pd.DataFrame, config: SimulationConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Apply the three fraud scenarios, recording which produced each label."""
    frame = transactions.copy()
    frame["is_fraud"] = 0
    frame["fraud_scenario"] = 0

    # Scenario 1: an amount above the threshold is always fraudulent.
    scenario1 = frame["tx_amount"] > config.scenario1_amount
    frame.loc[scenario1, ["is_fraud", "fraud_scenario"]] = [1, 1]

    days = frame["tx_day"].to_numpy()
    terminals = frame["terminal_id"].to_numpy()
    customers = frame["customer_id"].to_numpy()

    # Scenario 2: terminals compromised for a fixed window.
    compromised: dict[int, int] = {}
    for day in range(config.n_days):
        drawn = rng.choice(
            config.n_terminals, size=config.scenario2_terminals_per_day, replace=False
        )
        for terminal in drawn:
            # Keep the earliest compromise so overlapping draws do not extend
            # a window indefinitely.
            compromised.setdefault(int(terminal), day)

    if compromised:
        terminal_start = np.full(config.n_terminals, -1, dtype=np.int64)
        terminal_start[list(compromised)] = list(compromised.values())
        start = terminal_start[terminals]
        scenario2 = (start >= 0) & (days >= start) & (days < start + config.scenario2_days)
        newly = scenario2 & (frame["is_fraud"].to_numpy() == 0)
        frame.loc[newly, ["is_fraud", "fraud_scenario"]] = [1, 2]

    # Scenario 3: customer credentials stolen, a third of their transactions
    # inflated for a fixed window.
    stolen: dict[int, int] = {}
    for day in range(config.n_days):
        drawn = rng.choice(
            config.n_customers, size=config.scenario3_customers_per_day, replace=False
        )
        for customer in drawn:
            stolen.setdefault(int(customer), day)

    if stolen:
        customer_start = np.full(config.n_customers, -1, dtype=np.int64)
        customer_start[list(stolen)] = list(stolen.values())
        start = customer_start[customers]
        in_window = (start >= 0) & (days >= start) & (days < start + config.scenario3_days)
        selected = in_window & (rng.random(len(frame)) < config.scenario3_fraction)
        amounts = frame["tx_amount"].to_numpy().copy()
        amounts[selected] = np.round(amounts[selected] * config.scenario3_multiplier, 2)
        frame["tx_amount"] = amounts
        # Scenario 3 overwrites an earlier label: the inflated amount is the
        # reason this transaction is fraudulent, and attributing it to
        # scenario 1 would misreport which pattern the data contains.
        frame.loc[selected, ["is_fraud", "fraud_scenario"]] = [1, 3]

    return frame


def simulate(config: SimulationConfig | None = None) -> SimulationResult:
    """Generate a labelled transaction stream.

    Fully determined by ``config.seed``: the same configuration always produces
    the same transactions, which is what makes every downstream metric in this
    repository reproducible.
    """
    config = config or SimulationConfig()
    rng = np.random.default_rng(config.seed)

    customers = generate_customer_profiles(config, rng)
    terminals = generate_terminal_profiles(config, rng)
    flat_terminals, offsets = associate_terminals(customers, terminals, config.radius)

    available = np.diff(offsets)
    if available.sum() == 0:
        raise DataGenerationError(
            f"no customer has a terminal within radius {config.radius}; increase it or add terminals"
        )

    transactions = _draw_transactions(config, customers, flat_terminals, offsets, rng)
    transactions = inject_fraud(transactions, config, rng)
    transactions.insert(0, "transaction_id", np.arange(len(transactions), dtype=np.int64))

    scenario_counts = transactions["fraud_scenario"].value_counts().to_dict()
    stats = {
        "transactions": float(len(transactions)),
        "customers": float(config.n_customers),
        "terminals": float(config.n_terminals),
        "days": float(config.n_days),
        "fraud_count": float(transactions["is_fraud"].sum()),
        "fraud_rate": float(transactions["is_fraud"].mean()),
        "mean_terminals_per_customer": float(available.mean()),
        "customers_without_terminal": float((available == 0).sum()),
        "scenario_1": float(scenario_counts.get(1, 0)),
        "scenario_2": float(scenario_counts.get(2, 0)),
        "scenario_3": float(scenario_counts.get(3, 0)),
    }
    logger.info("simulation_complete", **{k: round(v, 4) for k, v in stats.items()})
    return SimulationResult(transactions, customers, terminals, stats)
