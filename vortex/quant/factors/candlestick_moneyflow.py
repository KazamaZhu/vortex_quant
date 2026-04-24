"""Candlestick and money-flow features for T+0 style timing research."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .price_volume import rolling_zscore


def build_candlestick_moneyflow_features(
    df: pd.DataFrame,
    *,
    high_threshold: float = 0.02,
    low_threshold: float = 0.02,
) -> tuple[pd.DataFrame, list[str]]:
    """Build daily candlestick + main-money features and next-day T labels."""
    required = {"date", "open", "high", "low", "close", "close_adj", "volume", "amount"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"T-model features missing columns: {missing}")

    out = df.sort_values("date").copy()
    prev_close = out["close"].shift(1).replace(0, np.nan)
    day_range = (out["high"] - out["low"]).replace(0, np.nan)
    body = (out["close"] - out["open"]).abs()

    out["ret_1d"] = out["close_adj"].pct_change()
    out["gap_open"] = out["open"] / prev_close - 1.0
    out["amplitude"] = (out["high"] - out["low"]) / prev_close
    out["body_ratio"] = body / day_range
    out["upper_shadow_ratio"] = (out["high"] - out[["open", "close"]].max(axis=1)) / day_range
    out["lower_shadow_ratio"] = (out[["open", "close"]].min(axis=1) - out["low"]) / day_range
    out["close_position"] = (out["close"] - out["low"]) / day_range
    out["is_up_day"] = (out["close"] > out["open"]).astype(float)
    out["consecutive_up_5"] = out["is_up_day"].rolling(5, min_periods=3).sum()
    out["consecutive_down_5"] = (1.0 - out["is_up_day"]).rolling(5, min_periods=3).sum()

    low_20 = out["close_adj"].rolling(20, min_periods=15).min()
    high_20 = out["close_adj"].rolling(20, min_periods=15).max()
    out["price_position_20"] = (out["close_adj"] - low_20) / (high_20 - low_20).replace(0, np.nan)
    out["volume_z_20"] = rolling_zscore(out["volume"].astype(float), 20, 15)
    out["amount_z_20"] = rolling_zscore(out["amount"].astype(float), 20, 15)
    out["ma20_gap"] = out["close_adj"] / out["close_adj"].rolling(20, min_periods=15).mean() - 1.0
    out["ma60_gap"] = out["close_adj"] / out["close_adj"].rolling(60, min_periods=40).mean() - 1.0
    out["vol_ma5"] = out["volume"].rolling(5, min_periods=3).mean()
    out["vol_ma20"] = out["volume"].rolling(20, min_periods=15).mean()
    out["vol_ratio_5_20"] = out["vol_ma5"] / out["vol_ma20"].replace(0, np.nan)
    out["ret_20d"] = out["close_adj"].pct_change(20)
    out["trend_strength"] = (
        0.50 * out["ret_20d"].fillna(0.0)
        + 0.30 * out["ma20_gap"].fillna(0.0)
        + 0.20 * out["ma60_gap"].fillna(0.0)
    )

    wk = out.set_index("date")["close_adj"].resample("W-FRI").last().to_frame("wk_close")
    wk["wk_ret_4"] = wk["wk_close"].pct_change(4)
    out = out.merge(wk[["wk_ret_4"]], left_on="date", right_index=True, how="left")
    out["wk_ret_4"] = out["wk_ret_4"].ffill()

    mk = out.set_index("date")["close_adj"].resample("ME").last().to_frame("mk_close")
    mk["mk_ret_3"] = mk["mk_close"].pct_change(3)
    out = out.merge(mk[["mk_ret_3"]], left_on="date", right_index=True, how="left")
    out["mk_ret_3"] = out["mk_ret_3"].ffill()

    out["bull_bear_score"] = (
        0.35 * out["trend_strength"].fillna(0.0)
        + 0.20 * out["wk_ret_4"].fillna(0.0)
        + 0.15 * out["mk_ret_3"].fillna(0.0)
        + 0.10 * out["vol_ratio_5_20"].fillna(1.0).sub(1.0)
    )

    optional_cols: list[str] = []
    if "net_mf_amount" in out.columns:
        out["main_net_inflow_ratio"] = out["net_mf_amount"] / out["amount"].replace(0, np.nan)
        out["main_net_inflow_z_20"] = rolling_zscore(out["net_mf_amount"].astype(float), 20, 15)
        optional_cols.extend(["main_net_inflow_ratio", "main_net_inflow_z_20"])
    if {"buy_lg_amount", "sell_lg_amount"}.issubset(out.columns):
        out["large_order_pressure"] = (
            out["buy_lg_amount"] - out["sell_lg_amount"]
        ) / out["amount"].replace(0, np.nan)
        optional_cols.append("large_order_pressure")
    if {"buy_md_amount", "sell_md_amount"}.issubset(out.columns):
        out["middle_order_pressure"] = (
            out["buy_md_amount"] - out["sell_md_amount"]
        ) / out["amount"].replace(0, np.nan)
        optional_cols.append("middle_order_pressure")
    if {"buy_elg_amount", "sell_elg_amount"}.issubset(out.columns):
        out["super_large_order_pressure"] = (
            out["buy_elg_amount"] - out["sell_elg_amount"]
        ) / out["amount"].replace(0, np.nan)
        optional_cols.append("super_large_order_pressure")
    if {"buy_sm_amount", "sell_sm_amount"}.issubset(out.columns):
        out["small_order_contra"] = (
            out["sell_sm_amount"] - out["buy_sm_amount"]
        ) / out["amount"].replace(0, np.nan)
        optional_cols.append("small_order_contra")

    main_buy_parts: list[pd.Series] = []
    main_sell_parts: list[pd.Series] = []
    for col in ["buy_lg_amount", "buy_elg_amount"]:
        if col in out.columns:
            main_buy_parts.append(pd.to_numeric(out[col], errors="coerce").fillna(0.0))
    for col in ["sell_lg_amount", "sell_elg_amount"]:
        if col in out.columns:
            main_sell_parts.append(pd.to_numeric(out[col], errors="coerce").fillna(0.0))
    if main_buy_parts and main_sell_parts:
        out["main_force_pressure"] = (
            sum(main_buy_parts) - sum(main_sell_parts)
        ) / out["amount"].replace(0, np.nan)
        out["main_force_persist_3"] = out["main_force_pressure"].rolling(3, min_periods=2).mean()
        out["main_force_persist_5"] = out["main_force_pressure"].rolling(5, min_periods=3).mean()
        out["main_force_accel"] = out["main_force_persist_3"] - out["main_force_persist_5"]
        optional_cols.extend(
            ["main_force_pressure", "main_force_persist_3", "main_force_persist_5", "main_force_accel"]
        )
    if {"main_force_pressure", "small_order_contra"}.issubset(out.columns):
        out["main_retail_divergence"] = out["main_force_pressure"] - out["small_order_contra"]
        optional_cols.append("main_retail_divergence")

    if "lhb_net_amount" in out.columns:
        out["lhb_net_ratio"] = out["lhb_net_amount"] / out["amount"].replace(0, np.nan)
        out["lhb_net_ratio_5"] = out["lhb_net_ratio"].rolling(5, min_periods=2).mean()
        optional_cols.extend(["lhb_net_ratio", "lhb_net_ratio_5"])
    if "inst_net_buy" in out.columns:
        out["inst_net_ratio"] = out["inst_net_buy"] / out["amount"].replace(0, np.nan)
        out["inst_net_ratio_5"] = out["inst_net_ratio"].rolling(5, min_periods=2).mean()
        optional_cols.extend(["inst_net_ratio", "inst_net_ratio_5"])

    out["main_action_score"] = (
        0.35 * out.get("main_force_persist_3", pd.Series(np.nan, index=out.index)).fillna(0.0)
        + 0.20 * out.get("main_force_accel", pd.Series(np.nan, index=out.index)).fillna(0.0)
        + 0.15 * out.get("large_order_pressure", pd.Series(np.nan, index=out.index)).fillna(0.0)
        + 0.10 * out.get("super_large_order_pressure", pd.Series(np.nan, index=out.index)).fillna(0.0)
        + 0.10 * out.get("inst_net_ratio_5", pd.Series(np.nan, index=out.index)).fillna(0.0)
        + 0.10 * out.get("lhb_net_ratio_5", pd.Series(np.nan, index=out.index)).fillna(0.0)
    )
    optional_cols.append("main_action_score")

    # Auction fields are optional (depends on account permissions / availability).
    auction_price_col = "auction_price" if "auction_price" in out.columns else None
    auction_amt_col = "auction_amount" if "auction_amount" in out.columns else None
    if auction_price_col is not None:
        out["auction_gap"] = out[auction_price_col] / prev_close - 1.0
    else:
        out["auction_gap"] = np.nan
    if auction_amt_col is not None:
        out["auction_amt_ratio"] = out[auction_amt_col] / out["amount"].replace(0, np.nan)
    else:
        out["auction_amt_ratio"] = np.nan
    out["auction_trend_3"] = out["auction_gap"].rolling(3, min_periods=2).mean()

    next_open = out["open"].shift(-1).replace(0, np.nan)
    next_high = out["high"].shift(-1)
    next_low = out["low"].shift(-1)
    out["label_sell_opportunity"] = next_high / next_open - 1.0
    out["label_buy_opportunity"] = next_open / next_low.replace(0, np.nan) - 1.0
    out["label_sell_hit"] = (out["label_sell_opportunity"] >= high_threshold).astype(float)
    out["label_buy_hit"] = (out["label_buy_opportunity"] >= low_threshold).astype(float)

    factor_cols = [
        "ret_1d",
        "gap_open",
        "amplitude",
        "body_ratio",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "close_position",
        "consecutive_up_5",
        "consecutive_down_5",
        "price_position_20",
        "volume_z_20",
        "amount_z_20",
        "ma20_gap",
        "ma60_gap",
        "vol_ratio_5_20",
        "trend_strength",
        "wk_ret_4",
        "mk_ret_3",
        "bull_bear_score",
        "auction_gap",
        "auction_amt_ratio",
        "auction_trend_3",
        *optional_cols,
    ]
    return out, factor_cols
