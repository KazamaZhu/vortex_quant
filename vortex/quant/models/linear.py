"""线性模型（Ridge）。"""
from __future__ import annotations

import numpy as np


class RidgeRegressor:
    """轻量 Ridge 回归，避免额外依赖。"""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        if x.ndim != 2:
            raise ValueError("x 必须是二维矩阵")
        if y.ndim != 1:
            raise ValueError("y 必须是一维向量")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x/y 样本量不一致")

        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        x_center = x - x_mean
        y_center = y - y_mean

        n_features = x.shape[1]
        eye = np.eye(n_features, dtype=float)
        beta = np.linalg.solve(x_center.T @ x_center + self.alpha * eye, x_center.T @ y_center)
        self.coef_ = beta
        self.intercept_ = y_mean - float(x_mean @ beta)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("模型尚未 fit")
        return x @ self.coef_ + self.intercept_
