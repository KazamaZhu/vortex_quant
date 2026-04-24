"""树模型封装（优先 LightGBM）。"""
from __future__ import annotations

import numpy as np


class LightGBMRegressor:
    """轻封装：存在 lightgbm 时可训练，否则给出可读报错。"""

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.random_state = int(random_state)
        self._model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LightGBMRegressor":
        try:
            import lightgbm as lgb  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("未安装 lightgbm，无法训练 LightGBMRegressor。") from exc
        self._model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
        )
        self._model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("模型尚未 fit")
        return np.asarray(self._model.predict(x), dtype=float)
