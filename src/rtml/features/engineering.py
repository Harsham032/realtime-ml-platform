"""Feature engineering.

Three families, following the Fraud Detection Handbook:

*Transaction features* are properties of the transaction itself - the amount,
and whether it fell at a weekend or at night.

*Customer features* summarise the cardholder's recent behaviour: how many
transactions and what average amount over trailing windows. These use no labels,
so the transaction being scored is included in its own window - its amount is
known the instant it arrives.

*Terminal features* summarise a terminal's recent fraud rate. These **do** use
labels, and that is where the care goes.

The delay period
----------------
A fraud label does not exist when the transaction happens. It exists once a
customer disputes the charge or an investigator confirms it, typically days
later. A terminal risk feature computed from labels up to the moment of scoring
would therefore use information the production system could not have had, and
every downstream metric would be inflated.

So terminal risk over a window is computed over
``[t - delay - window, t - delay]``: the window ends ``risk_delay_days`` before
the transaction being scored. Nothing inside the delay contributes. Customer
count and amount features carry no delay because they need no labels.

``tests/test_features.py`` asserts this directly: shifting a fraud label inside
the delay window must not change any feature value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from ..errors import FeatureError
from ..logging_utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ("transaction_id", "customer_id", "terminal_id", "tx_datetime", "tx_amount")

TRANSACTION_FEATURES = ("tx_amount", "tx_during_weekend", "tx_during_night")


def customer_feature_names(windows: list[int]) -> list[str]:
    names: list[str] = []
    for window in windows:
        names += [f"customer_nb_tx_{window}d", f"customer_avg_amount_{window}d"]
    return names


def terminal_feature_names(windows: list[int]) -> list[str]:
    names: list[str] = []
    for window in windows:
        names += [f"terminal_nb_tx_{window}d", f"terminal_risk_{window}d"]
    return names


def feature_names(config: FeatureConfig) -> list[str]:
    """Every model input, in a stable order.

    Order matters: the serving path builds a vector positionally, so a change
    here without a retrain would silently mis-assign features.
    """
    return [
        *TRANSACTION_FEATURES,
        *customer_feature_names(config.customer_windows),
        *terminal_feature_names(config.terminal_windows),
    ]


def add_time_features(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Flag weekend and night transactions."""
    timestamps = pd.to_datetime(frame["tx_datetime"])
    hour = timestamps.dt.hour
    if config.night_start_hour <= config.night_end_hour:
        night = (hour >= config.night_start_hour) & (hour < config.night_end_hour)
    else:
        # A window wrapping midnight, for example 22:00 to 06:00.
        night = (hour >= config.night_start_hour) | (hour < config.night_end_hour)
    frame["tx_during_weekend"] = (timestamps.dt.dayofweek >= 5).astype(np.int8)
    frame["tx_during_night"] = night.astype(np.int8)
    return frame


def _rolling_by_group(
    frame: pd.DataFrame,
    group_column: str,
    value_column: str,
    offset: str,
    how: str,
) -> np.ndarray:
    """Trailing time-window aggregate within each group, in original row order.

    Deliberately not ``groupby(...).rolling(...)``: that returns its values
    ordered by (group, timestamp), not by input row, and re-attaching them to a
    time-ordered frame silently assigns each row another row's history. The
    result looks entirely plausible - correct dtypes, correct range, no nulls -
    and quietly destroys the signal.

    ``groupby(...).indices`` gives the positional indices of each group. The
    frame arrives sorted by time, so each group's positions are already in that
    group's own time order, and scattering back through those positions is
    exact. ``tests/test_features.py`` pins the output against an independent
    brute-force implementation.
    """
    timestamps = pd.DatetimeIndex(frame["tx_datetime"])
    values = frame[value_column].to_numpy(dtype=np.float64)
    out = np.zeros(len(frame), dtype=np.float64)

    for positions in frame.groupby(group_column, sort=False).indices.values():
        window = pd.Series(values[positions], index=timestamps[positions])
        out[positions] = getattr(window.rolling(offset), how)().to_numpy()
    return out


def add_customer_features(frame: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Trailing transaction count and average amount per customer.

    No delay: these depend only on the customer's own transactions, which are
    known the moment they happen. The transaction being scored is included in
    its own window - its amount is available at scoring time.
    """
    for window in windows:
        offset = f"{window}D"
        counts = _rolling_by_group(frame, "customer_id", "tx_amount", offset, "count")
        sums = _rolling_by_group(frame, "customer_id", "tx_amount", offset, "sum")
        with np.errstate(divide="ignore", invalid="ignore"):
            averages = np.where(counts > 0, sums / np.maximum(counts, 1e-9), 0.0)
        frame[f"customer_nb_tx_{window}d"] = counts.astype(np.float32)
        frame[f"customer_avg_amount_{window}d"] = averages.astype(np.float32)
    return frame


def add_terminal_features(frame: pd.DataFrame, windows: list[int], delay_days: int) -> pd.DataFrame:
    """Trailing transaction count and fraud rate per terminal, behind the delay.

    Each window is the difference of two trailing sums - one over
    ``delay + window`` days, one over ``delay`` days - which leaves exactly the
    window ending ``delay`` days before the transaction being scored.
    """
    if "is_fraud" not in frame.columns:
        raise FeatureError("terminal risk features need an is_fraud column")

    if delay_days > 0:
        delay_offset = f"{delay_days}D"
        tx_in_delay = _rolling_by_group(frame, "terminal_id", "is_fraud", delay_offset, "count")
        fraud_in_delay = _rolling_by_group(frame, "terminal_id", "is_fraud", delay_offset, "sum")
    else:
        tx_in_delay = np.zeros(len(frame))
        fraud_in_delay = np.zeros(len(frame))

    for window in windows:
        total_offset = f"{delay_days + window}D"
        tx_total = _rolling_by_group(frame, "terminal_id", "is_fraud", total_offset, "count")
        fraud_total = _rolling_by_group(frame, "terminal_id", "is_fraud", total_offset, "sum")

        tx_window = tx_total - tx_in_delay
        fraud_window = fraud_total - fraud_in_delay

        # A terminal with no history in the window gets risk 0, not NaN: an
        # unseen terminal is not evidence of fraud, and a NaN would force every
        # model to carry an imputation rule.
        with np.errstate(divide="ignore", invalid="ignore"):
            risk = np.where(tx_window > 0, fraud_window / np.maximum(tx_window, 1e-9), 0.0)

        frame[f"terminal_nb_tx_{window}d"] = tx_window.astype(np.float32)
        frame[f"terminal_risk_{window}d"] = risk.astype(np.float32)
    return frame


def build_features(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Compute every feature for a labelled transaction frame."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FeatureError(f"missing required columns: {', '.join(missing)}")

    result = frame.copy()
    result["tx_datetime"] = pd.to_datetime(result["tx_datetime"])
    result = result.sort_values("tx_datetime", kind="stable", ignore_index=True)

    result = add_time_features(result, config)
    result = add_customer_features(result, config.customer_windows)
    result = add_terminal_features(result, config.terminal_windows, config.risk_delay_days)

    names = feature_names(config)
    produced = [name for name in names if name in result.columns]
    if len(produced) != len(names):
        raise FeatureError(f"expected {len(names)} features, produced {len(produced)}")

    logger.info("features_built", rows=len(result), features=len(names))
    return result
