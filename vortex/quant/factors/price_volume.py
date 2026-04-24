"""Price-volume factor builders."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Return a trailing z-score without looking forward."""
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return (series - mean) / std


def build_price_volume_features(
    df: pd.DataFrame,
    *,
    label_horizon: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """Build single-stock price-volume features and a forward-return label."""
    required = {"date", "close_adj", "volume", "amount"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"price-volume features missing columns: {missing}")

    out = df.sort_values("date").copy()
    out["ret_1d"] = out["close_adj"].pct_change()
    out["ret_5d"] = out["close_adj"].pct_change(5)
    out["ret_10d"] = out["close_adj"].pct_change(10)
    out["ret_20d"] = out["close_adj"].pct_change(20)
    out["mom_20"] = out["close_adj"].shift(5) / out["close_adj"].shift(25) - 1.0
    out["mom_60"] = out["close_adj"].shift(5) / out["close_adj"].shift(65) - 1.0
    out["volatility_20"] = out["ret_1d"].rolling(20, min_periods=15).std()

    out["volume_z_20"] = rolling_zscore(out["volume"].astype(float), 20, 15)
    out["amount_z_20"] = rolling_zscore(out["amount"].astype(float), 20, 15)
    if "turnover_rate" in out.columns:
        out["turnover_z_20"] = rolling_zscore(out["turnover_rate"].astype(float), 20, 15)
        out["turnover_trend_5"] = out["turnover_rate"].pct_change(5)
    else:
        out["turnover_z_20"] = np.nan
        out["turnover_trend_5"] = np.nan

    out["volume_price_corr_20"] = out["ret_1d"].rolling(20, min_periods=15).corr(
        out["volume"].pct_change()
    )
    up_volume = out["volume"].where(out["ret_1d"] > 0, 0.0).rolling(20, min_periods=15).sum()
    total_volume = out["volume"].rolling(20, min_periods=15).sum().replace(0, np.nan)
    out["up_volume_ratio_20"] = up_volume / total_volume
    out["amihud_20"] = (
        out["ret_1d"].abs() / out["amount"].replace(0, np.nan)
    ).rolling(20, min_periods=15).mean()
    out["volume_trend_5"] = out["volume"].pct_change(5)
    out["amount_trend_5"] = out["amount"].pct_change(5)
    out["trend_strength"] = (
        0.5 * out["ret_20d"].fillna(0.0)
        + 0.3 * out["mom_20"].fillna(0.0)
        + 0.2 * out["mom_60"].fillna(0.0)
    )
    out["ma20_gap"] = out["close_adj"] / out["close_adj"].rolling(20, min_periods=15).mean() - 1.0
    out["ma60_gap"] = out["close_adj"] / out["close_adj"].rolling(60, min_periods=40).mean() - 1.0
    out["vol_ma5"] = out["volume"].rolling(5, min_periods=3).mean()
    out["vol_ma20"] = out["volume"].rolling(20, min_periods=15).mean()
    out["vol_ratio_5_20"] = out["vol_ma5"] / out["vol_ma20"].replace(0, np.nan)

    low_60 = out["close_adj"].rolling(60, min_periods=40).min()
    high_60 = out["close_adj"].rolling(60, min_periods=40).max()
    out["price_position_60"] = (out["close_adj"] - low_60) / (high_60 - low_60).replace(0, np.nan)

    optional_cols: list[str] = []
    if "net_mf_amount" in out.columns:
        out["net_mf_amount_z_20"] = rolling_zscore(out["net_mf_amount"].astype(float), 20, 15)
        optional_cols.append("net_mf_amount_z_20")
    if {"buy_lg_amount", "sell_lg_amount"}.issubset(out.columns):
        denom = (out["buy_lg_amount"] + out["sell_lg_amount"]).replace(0, np.nan)
        out["large_order_pressure_20"] = (
            (out["buy_lg_amount"] - out["sell_lg_amount"]) / denom
        ).rolling(20, min_periods=15).mean()
        optional_cols.append("large_order_pressure_20")
    if {"net_mf_amount", "amount"}.issubset(out.columns):
        out["main_control_ratio"] = out["net_mf_amount"] / out["amount"].replace(0, np.nan)
        out["main_control_strength_5"] = out["main_control_ratio"].rolling(5, min_periods=3).mean()
        optional_cols.extend(["main_control_ratio", "main_control_strength_5"])

    # Auction-based expectation (if stk_auction is available locally).
    auction_price_col = "auction_price" if "auction_price" in out.columns else None
    auction_amt_col = "auction_amount" if "auction_amount" in out.columns else None
    if auction_price_col is not None:
        out["auction_gap"] = out[auction_price_col] / out["close_adj"].shift(1).replace(0, np.nan) - 1.0
    else:
        out["auction_gap"] = np.nan
    if auction_amt_col is not None:
        out["auction_amt_ratio"] = out[auction_amt_col] / out["amount"].replace(0, np.nan)
    else:
        out["auction_amt_ratio"] = np.nan
    out["auction_trend_3"] = out["auction_gap"].rolling(3, min_periods=2).mean()

    # Multi-timeframe trend: derive weekly/monthly closes from daily and map back.
    wk = (
        out.set_index("date")["close_adj"]
        .resample("W-FRI")
        .last()
        .rename("wk_close")
        .to_frame()
    )
    wk["wk_ret_4"] = wk["wk_close"].pct_change(4)
    wk["wk_ma8_gap"] = wk["wk_close"] / wk["wk_close"].rolling(8, min_periods=5).mean() - 1.0
    out = out.merge(wk[["wk_ret_4", "wk_ma8_gap"]], left_on="date", right_index=True, how="left")
    out["wk_ret_4"] = out["wk_ret_4"].ffill()
    out["wk_ma8_gap"] = out["wk_ma8_gap"].ffill()

    mk = (
        out.set_index("date")["close_adj"]
        .resample("ME")
        .last()
        .rename("mk_close")
        .to_frame()
    )
    mk["mk_ret_3"] = mk["mk_close"].pct_change(3)
    mk["mk_ma6_gap"] = mk["mk_close"] / mk["mk_close"].rolling(6, min_periods=4).mean() - 1.0
    out = out.merge(mk[["mk_ret_3", "mk_ma6_gap"]], left_on="date", right_index=True, how="left")
    out["mk_ret_3"] = out["mk_ret_3"].ffill()
    out["mk_ma6_gap"] = out["mk_ma6_gap"].ffill()

    out["bull_bear_score"] = (
        0.25 * out["trend_strength"].fillna(0.0)
        + 0.20 * out["ma20_gap"].fillna(0.0)
        + 0.10 * out["ma60_gap"].fillna(0.0)
        + 0.15 * out["wk_ret_4"].fillna(0.0)
        + 0.10 * out["mk_ret_3"].fillna(0.0)
        + 0.10 * out["vol_ratio_5_20"].fillna(1.0).sub(1.0)
    )

    out["label_fwd_ret"] = out["close_adj"].shift(-label_horizon) / out["close_adj"] - 1.0
    factor_cols = [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "mom_20",
        "mom_60",
        "volatility_20",
        "volume_z_20",
        "amount_z_20",
        "turnover_z_20",
        "volume_trend_5",
        "amount_trend_5",
        "turnover_trend_5",
        "trend_strength",
        "ma20_gap",
        "ma60_gap",
        "vol_ratio_5_20",
        "volume_price_corr_20",
        "up_volume_ratio_20",
        "amihud_20",
        "price_position_60",
        "wk_ret_4",
        "wk_ma8_gap",
        "mk_ret_3",
        "mk_ma6_gap",
        "bull_bear_score",
        "auction_gap",
        "auction_amt_ratio",
        "auction_trend_3",
        *optional_cols,
    ]
    return out, factor_cols
