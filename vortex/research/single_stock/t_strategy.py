"""Candlestick + money-flow T-strategy research."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vortex.quant.factors.candlestick_moneyflow import build_candlestick_moneyflow_features
from vortex.research.single_stock.backtest import max_drawdown


@dataclass(frozen=True)
class TStrategyConfig:
    code: str
    train_start: str
    test_start: str
    test_end: str | None = None
    high_threshold: float = 0.02
    low_threshold: float = 0.02
    score_threshold: float = 0.012
    min_train_samples: int = 160
    min_test_samples: int = 60


@dataclass
class TStrategyStats:
    code: str
    train_start: str
    test_start: str
    test_end: str
    data_start: str
    data_end: str
    model: str
    factor_count: int
    train_samples: int
    test_days: int
    action_days: int
    sell_signal_days: int
    buy_signal_days: int
    signal_rate: float
    hit_rate: float
    sell_hit_rate: float
    buy_hit_rate: float
    avg_sell_opportunity: float
    avg_buy_opportunity: float
    benchmark_total_return: float
    benchmark_max_drawdown: float


@dataclass
class TStrategyResult:
    stats: TStrategyStats
    signals: pd.DataFrame
    factor_columns: list[str]


def run_t_strategy_backtest(
    frame: pd.DataFrame,
    *,
    config: TStrategyConfig,
    out_dir: Path | None = None,
) -> TStrategyResult:
    features, factor_cols = build_candlestick_moneyflow_features(
        frame,
        high_threshold=config.high_threshold,
        low_threshold=config.low_threshold,
    )
    data_start = pd.to_datetime(features["date"].min())
    data_end = pd.to_datetime(features["date"].max())
    test_start_dt = pd.to_datetime(config.test_start, format="%Y%m%d")
    test_end_dt = (
        pd.to_datetime(config.test_end, format="%Y%m%d")
        if config.test_end
        else features["date"].max()
    )
    train_start_dt = pd.to_datetime(config.train_start, format="%Y%m%d")

    cols = list(
        dict.fromkeys(
            [
                "date",
                "ret_1d",
                "label_sell_opportunity",
                "label_buy_opportunity",
                "label_sell_hit",
                "label_buy_hit",
                *factor_cols,
            ]
        )
    )
    sample = features[cols].copy()
    sample = sample[(sample["date"] >= train_start_dt) & (sample["date"] <= test_end_dt)]
    used_factor_cols = [c for c in factor_cols if sample[c].notna().mean() >= 0.20]
    if not used_factor_cols:
        raise ValueError("No usable T-strategy factor columns after null filtering.")
    sample = sample.dropna(subset=["label_sell_opportunity", "label_buy_opportunity", *used_factor_cols]).reset_index(drop=True)
    train = sample[sample["date"] < test_start_dt].copy()
    test_mask = sample["date"] >= test_start_dt

    if len(train) < config.min_train_samples:
        raise ValueError(f"Not enough T-strategy training samples: {len(train)}")
    if int(test_mask.sum()) < config.min_test_samples:
        raise ValueError(f"Not enough T-strategy test samples: {int(test_mask.sum())}")

    train_stats = train[used_factor_cols].agg(["mean", "std"]).to_dict()
    z_cols: list[pd.Series] = []
    for col in used_factor_cols:
        mu = float(train_stats[col]["mean"])
        sigma = float(train_stats[col]["std"])
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = 1.0
        z_cols.append((sample[col] - mu) / sigma)
    z_df = pd.concat(z_cols, axis=1)
    z_df.columns = used_factor_cols

    sell_weights = {
        "upper_shadow_ratio": 0.30,
        "amplitude": 0.20,
        "close_position": 0.20,
        "consecutive_up_5": 0.12,
        "volume_z_20": 0.08,
        "amount_z_20": 0.08,
        "gap_open": 0.05,
        "auction_gap": 0.10,
        "auction_amt_ratio": 0.08,
        "auction_trend_3": 0.10,
        "vol_ratio_5_20": 0.08,
        "bull_bear_score": 0.10,
    }
    buy_weights = {
        "lower_shadow_ratio": 0.30,
        "amplitude": 0.20,
        "consecutive_down_5": 0.15,
        "volume_z_20": 0.08,
        "amount_z_20": 0.08,
        "price_position_20": -0.20,
        "gap_open": -0.05,
        "auction_gap": -0.08,
        "auction_amt_ratio": 0.08,
        "auction_trend_3": -0.06,
        "vol_ratio_5_20": 0.08,
        "bull_bear_score": -0.08,
    }
    if "main_net_inflow_ratio" in z_df.columns:
        buy_weights["main_net_inflow_ratio"] = 0.12
        sell_weights["main_net_inflow_ratio"] = -0.10
    if "large_order_pressure" in z_df.columns:
        buy_weights["large_order_pressure"] = 0.12
        sell_weights["large_order_pressure"] = -0.10
    if "small_order_contra" in z_df.columns:
        buy_weights["small_order_contra"] = 0.06
        sell_weights["small_order_contra"] = 0.06

    sample["sell_score"] = 0.0
    sample["buy_score"] = 0.0
    for col, weight in sell_weights.items():
        if col in z_df.columns:
            sample["sell_score"] += weight * z_df[col]
    for col, weight in buy_weights.items():
        if col in z_df.columns:
            sample["buy_score"] += weight * z_df[col]

    sample["action"] = "hold"
    bull = sample["bull_bear_score"].fillna(0.0) >= 0.08
    bear = sample["bull_bear_score"].fillna(0.0) <= -0.08
    sell_thr = np.where(bull, config.score_threshold * 1.15, config.score_threshold)
    buy_thr = np.where(bear, config.score_threshold * 1.15, config.score_threshold)

    sell_mask = (sample["sell_score"] >= sell_thr) & (sample["sell_score"] > sample["buy_score"])
    buy_mask = (sample["buy_score"] >= buy_thr) & (sample["buy_score"] >= sample["sell_score"])

    # In uptrend, avoid over-trading T: only high-sell when price is relatively extended.
    if "price_position_20" in sample.columns:
        sell_mask = sell_mask & (~bull | (sample["price_position_20"] >= 0.75))
    # In downtrend, avoid aggressive buy-low unless oversold.
    if "price_position_20" in sample.columns:
        buy_mask = buy_mask & (~bear | (sample["price_position_20"] <= 0.35))

    sample.loc[sell_mask, "action"] = "sell_high"
    sample.loc[buy_mask, "action"] = "buy_low"

    test = sample[test_mask].copy()
    test["action_hit"] = np.where(
        test["action"] == "sell_high",
        test["label_sell_hit"],
        np.where(test["action"] == "buy_low", test["label_buy_hit"], np.nan),
    )
    action_rows = test[test["action"] != "hold"].copy()
    sell_rows = test[test["action"] == "sell_high"].copy()
    buy_rows = test[test["action"] == "buy_low"].copy()
    benchmark_curve = (1.0 + test["ret_1d"].fillna(0.0)).cumprod()

    def _hit_rate(df: pd.DataFrame) -> float:
        if df.empty:
            return float("nan")
        return float(df["action_hit"].mean())

    stats = TStrategyStats(
        code=config.code,
        train_start=config.train_start,
        test_start=test_start_dt.strftime("%Y%m%d"),
        test_end=test["date"].max().strftime("%Y%m%d"),
        data_start=data_start.strftime("%Y%m%d"),
        data_end=data_end.strftime("%Y%m%d"),
        model="rule_v1",
        factor_count=len(used_factor_cols),
        train_samples=len(train),
        test_days=len(test),
        action_days=len(action_rows),
        sell_signal_days=len(sell_rows),
        buy_signal_days=len(buy_rows),
        signal_rate=float(len(action_rows) / max(1, len(test))),
        hit_rate=_hit_rate(action_rows),
        sell_hit_rate=_hit_rate(sell_rows),
        buy_hit_rate=_hit_rate(buy_rows),
        avg_sell_opportunity=float(sell_rows["label_sell_opportunity"].mean()) if not sell_rows.empty else float("nan"),
        avg_buy_opportunity=float(buy_rows["label_buy_opportunity"].mean()) if not buy_rows.empty else float("nan"),
        benchmark_total_return=float(benchmark_curve.iloc[-1] - 1.0),
        benchmark_max_drawdown=max_drawdown(benchmark_curve),
    )

    signal_cols = [
        "date",
        "action",
        "sell_score",
        "buy_score",
        "label_sell_opportunity",
        "label_buy_opportunity",
        "action_hit",
        "bull_bear_score",
    ]
    signals = test[signal_cols].copy()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        signals.to_csv(out_dir / "t_strategy_signals.csv", index=False)
        signals.to_parquet(out_dir / "t_strategy_signals.parquet", index=False)

    return TStrategyResult(stats=stats, signals=signals, factor_columns=used_factor_cols)
