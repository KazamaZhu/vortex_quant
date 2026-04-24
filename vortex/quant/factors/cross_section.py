"""截面标准化与去极值。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_by_mad(df: pd.DataFrame, factor_cols: list[str], *, n_mad: float = 5.0) -> pd.DataFrame:
    """按日期做 MAD 去极值。"""
    out = df.copy()
    for col in factor_cols:
        med = out.groupby("date")[col].transform("median")
        mad = (out[col] - med).abs().groupby(out["date"]).transform("median")
        scale = 1.4826 * mad.replace(0, np.nan)
        lower = med - n_mad * scale
        upper = med + n_mad * scale
        out[col] = out[col].clip(lower, upper)
    return out


def zscore_by_date(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """按日期截面 z-score。"""
    out = df.copy()
    for col in factor_cols:
        mean = out.groupby("date")[col].transform("mean")
        std = out.groupby("date")[col].transform("std").replace(0, np.nan)
        out[col] = (out[col] - mean) / std
    return out


def zscore_by_date_industry(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    industry_col: str = "industry",
) -> pd.DataFrame:
    """按日期+行业做截面 z-score（行业内标准化）。"""
    if industry_col not in df.columns:
        raise ValueError(f"缺少行业列: {industry_col}")
    out = df.copy()
    group_keys = ["date", industry_col]
    for col in factor_cols:
        mean = out.groupby(group_keys)[col].transform("mean")
        std = out.groupby(group_keys)[col].transform("std").replace(0, np.nan)
        out[col] = (out[col] - mean) / std
    return out


def rank_to_uniform(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """按日期做分位映射到 [0, 1]，可选用于鲁棒化。"""
    out = df.copy()
    for col in factor_cols:
        out[col] = out.groupby("date")[col].rank(pct=True, method="average")
    return out
