"""Temporal splitting with a label-availability delay.

Fraud detection is a forecasting problem wearing a classification costume. A
random split lets a model learn from transactions that happen *after* the ones
it is scored on, and the resulting numbers are unreachable in production.

Every split here is chronological, and there is a gap between train and test:

    |<------- train ------->|<- delay ->|<------ test ------>|
              |<- val ->|

The gap models the fact that a model deployed at the start of the test window
could only have been trained on labels confirmed before it. Without the gap, the
last few days of training carry labels that in reality would not have arrived,
which inflates the terminal risk features and every metric downstream.

The validation window is carved out of the *end* of the training period, not
sampled from it, so operating points are chosen on the most recent data the
model is allowed to see - the closest available proxy for what it will meet.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import SplitConfig
from ..errors import EvaluationError
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TemporalSplit:
    """Three disjoint, chronologically ordered frames."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    boundaries: dict[str, int]

    def describe(self) -> dict[str, float]:
        return {
            "train_rows": float(len(self.train)),
            "validation_rows": float(len(self.validation)),
            "test_rows": float(len(self.test)),
            "train_frauds": float(self.train["is_fraud"].sum()),
            "validation_frauds": float(self.validation["is_fraud"].sum()),
            "test_frauds": float(self.test["is_fraud"].sum()),
            "train_fraud_rate": float(self.train["is_fraud"].mean()),
            "test_fraud_rate": float(self.test["is_fraud"].mean()),
            **{key: float(value) for key, value in self.boundaries.items()},
        }


def split_by_day(
    frame: pd.DataFrame, config: SplitConfig, *, day_column: str = "tx_day"
) -> TemporalSplit:
    """Split ``frame`` chronologically according to ``config``."""
    if day_column not in frame.columns:
        raise EvaluationError(f"the frame has no {day_column} column to split on")

    first_day = int(frame[day_column].min())
    train_end = first_day + config.train_days
    validation_start = train_end - config.validation_days
    test_start = train_end + config.delay_days
    test_end = test_start + config.test_days

    available_end = int(frame[day_column].max()) + 1
    if test_end > available_end:
        raise EvaluationError(
            f"the split needs days up to {test_end} but the data ends at {available_end}"
        )

    days = frame[day_column]
    train = frame[(days >= first_day) & (days < validation_start)].reset_index(drop=True)
    validation = frame[(days >= validation_start) & (days < train_end)].reset_index(drop=True)
    test = frame[(days >= test_start) & (days < test_end)].reset_index(drop=True)

    for name, part in (("train", train), ("validation", validation), ("test", test)):
        if len(part) == 0:
            raise EvaluationError(
                f"the {name} split is empty; widen the simulation or narrow the split"
            )
        if int(part["is_fraud"].sum()) == 0:
            raise EvaluationError(f"the {name} split contains no fraud; it cannot be evaluated")

    boundaries = {
        "first_day": first_day,
        "validation_start_day": validation_start,
        "train_end_day": train_end,
        "test_start_day": test_start,
        "test_end_day": test_end,
        "delay_days": config.delay_days,
    }
    split = TemporalSplit(train, validation, test, boundaries)
    logger.info("split_built", **{k: round(v, 4) for k, v in split.describe().items()})
    return split
