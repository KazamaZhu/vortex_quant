"""量化常用评估指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    valid = a.notna() & b.notna()
    if valid.sum() < 2:
        return float("nan")
    return float(a[valid].corr(b[valid], method=method))


def ic_by_date(df: pd.DataFrame, *, pred_col: str, label_col: str) -> pd.Series:
    """按日期计算 Pearson IC。"""
    return df.groupby("date").apply(lambda g: _safe_corr(g[pred_col], g[label_col]))


def rank_ic_by_date(df: pd.DataFrame, *, pred_col: str, label_col: str) -> pd.Series:
    """按日期计算 Spearman Rank IC。"""
    return df.groupby("date").apply(lambda g: _safe_corr(g[pred_col], g[label_col], method="spearman"))


def long_short_spread_by_date(
    df: pd.DataFrame,
    *,
    pred_col: str,
    label_col: str,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
) -> pd.Series:
    """按日期计算 top-bottom 分位收益差。"""
    if not (0 < top_quantile < 1 and 0 < bottom_quantile < 1):
        raise ValueError("quantile 参数必须在 (0, 1)")

    def _spread(g: pd.DataFrame) -> float:
        g = g[[pred_col, label_col]].dropna()
        if g.empty:
            return float("nan")
        upper = g[pred_col].quantile(1 - top_quantile)
        lower = g[pred_col].quantile(bottom_quantile)
        long_ret = g.loc[g[pred_col] >= upper, label_col].mean()
        short_ret = g.loc[g[pred_col] <= lower, label_col].mean()
        if np.isnan(long_ret) or np.isnan(short_ret):
            return float("nan")
        return float(long_ret - short_ret)

    return df.groupby("date").apply(_spread)
