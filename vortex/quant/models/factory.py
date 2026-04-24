"""Model construction helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .boosting import LightGBMRegressor
from .linear import RidgeRegressor


class Regressor(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Regressor":
        ...

    def predict(self, x: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ModelConfig:
    name: str = "ridge"
    alpha: float = 5.0
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    random_state: int = 42


def build_regressor(config: ModelConfig | None = None) -> Regressor:
    """Build a regression model from a small serializable config."""
    cfg = config or ModelConfig()
    name = cfg.name.strip().lower()
    if name == "ridge":
        return RidgeRegressor(alpha=cfg.alpha)
    if name in {"lightgbm", "lgbm"}:
        return LightGBMRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            random_state=cfg.random_state,
        )
    raise ValueError(f"unknown model: {cfg.name}")
