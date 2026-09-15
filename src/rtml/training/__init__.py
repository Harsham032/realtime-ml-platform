"""Training pipeline, splitting and promotion gates."""

from .pipeline import TrainedModel, train_all, train_one
from .promotion import Gate, PromotionDecision, evaluate_promotion, should_retrain
from .splitting import TemporalSplit, split_by_day

__all__ = [
    "Gate",
    "PromotionDecision",
    "TemporalSplit",
    "TrainedModel",
    "evaluate_promotion",
    "should_retrain",
    "split_by_day",
    "train_all",
    "train_one",
]
