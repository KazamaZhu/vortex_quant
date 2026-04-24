"""经典截面因子构建。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context} 缺少必要列: {missing}")


def build_momentum_factors(
    bars: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (20, 60, 120),
    skip_recent: int = 5,
) -> pd.DataFrame:
    """构建动量因子（跳过最近若干交易日，降低短期反转干扰）。"""
    _require_columns(bars, ["date", "symbol", "close"], "build_momentum_factors")
    out = bars[["date", "symbol"]].copy()
    ordered = bars.sort_values(["symbol", "date"]).copy()
    grp = ordered.groupby("symbol", group_keys=False)["close"]
    base = grp.shift(skip_recent)
    for window in windows:
        past = grp.shift(skip_recent + window)
        out[f"mom_{window}"] = (base / past) - 1.0
    return out


def build_value_factors(fundamental: pd.DataFrame) -> pd.DataFrame:
    """构建价值因子（EP/BP）。"""
    _require_columns(fundamental, ["date", "symbol"], "build_value_factors")
    out = fundamental[["date", "symbol"]].copy()
    if "pe_ttm" in fundamental.columns:
        pe = fundamental["pe_ttm"].replace(0, np.nan)
        out["value_ep"] = 1.0 / pe
    if "pb" in fundamental.columns:
        pb = fundamental["pb"].replace(0, np.nan)
        out["value_bp"] = 1.0 / pb
    return out


def build_quality_factors(fundamental: pd.DataFrame) -> pd.DataFrame:
    """构建质量因子（盈利能力与杠杆）。"""
    _require_columns(fundamental, ["date", "symbol"], "build_quality_factors")
    out = fundamental[["date", "symbol"]].copy()
    if "roe" in fundamental.columns:
        out["quality_roe"] = fundamental["roe"]
    if "gross_margin" in fundamental.columns:
        out["quality_gross_margin"] = fundamental["gross_margin"]
    if "debt_to_assets" in fundamental.columns:
        out["quality_low_leverage"] = -fundamental["debt_to_assets"]
    return out


def build_volatility_factors(
    bars: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.DataFrame:
    """构建波动因子（滚动收益波动率）。"""
    _require_columns(bars, ["date", "symbol", "close"], "build_volatility_factors")
    out = bars[["date", "symbol"]].copy()
    ordered = bars.sort_values(["symbol", "date"]).copy()
    ordered["ret_1d"] = ordered.groupby("symbol")["close"].pct_change()
    out["volatility_20"] = (
        ordered.groupby("symbol")["ret_1d"].rolling(window).std().reset_index(level=0, drop=True)
    )
    return out


def build_microstructure_factors(
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """构建日频可用微观结构代理（非流动性与换手强度代理）。"""
    _require_columns(bars, ["date", "symbol", "close", "amount", "volume"], "build_microstructure_factors")
    out = bars[["date", "symbol"]].copy()
    ordered = bars.sort_values(["symbol", "date"]).copy()
    ret = ordered.groupby("symbol")["close"].pct_change().abs()
    amount = ordered["amount"].replace(0, np.nan)
    out["micro_amihud"] = ret / amount
    out["micro_turnover_proxy"] = ordered["volume"] / amount
    return out
