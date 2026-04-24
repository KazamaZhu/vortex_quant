"""Single-stock backtest engines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vortex.quant.factors.price_volume import build_price_volume_features


def max_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


@dataclass(frozen=True)
class PriceVolumeBacktestConfig:
    code: str
    train_start: str
    test_start: str
    test_end: str | None = None
    label_horizon: int = 5
    min_train_samples: int = 160
    min_test_samples: int = 60


@dataclass
class BacktestStats:
    code: str
    train_start: str
    test_start: str
    test_end: str
    data_start: str
    data_end: str
    dropped_tail_days: int
    label_horizon: int
    model: str
    factor_count: int
    train_samples: int
    test_days: int
    strategy_total_return: float
    benchmark_total_return: float
    excess_total_return: float
    strategy_annual_return: float
    strategy_annual_vol: float
    strategy_sharpe: float
    strategy_max_drawdown: float
    benchmark_max_drawdown: float
    win_rate: float
    avg_position: float
    turnover: float
    prediction_ic: float


@dataclass
class BacktestResult:
    stats: BacktestStats
    timeseries: pd.DataFrame
    factor_columns: list[str]


def run_price_volume_backtest(
    frame: pd.DataFrame,
    *,
    config: PriceVolumeBacktestConfig,
    out_dir: Path | None = None,
) -> BacktestResult:
    """Evaluate a deterministic price-volume timing rule before/after test_start."""
    features, factor_cols = build_price_volume_features(
        frame,
        label_horizon=config.label_horizon,
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

    sample_cols = list(dict.fromkeys(["date", "ret_1d", "label_fwd_ret", *factor_cols]))
    sample = features[sample_cols].copy()
    sample = sample[(sample["date"] >= train_start_dt) & (sample["date"] <= test_end_dt)]
    used_factor_cols = [c for c in factor_cols if sample[c].notna().mean() >= 0.20]
    if not used_factor_cols:
        raise ValueError("No usable factor columns after null filtering.")
    labeled_sample = sample.dropna(subset=["label_fwd_ret", *used_factor_cols]).reset_index(drop=True)
    dropped_tail_days = max(0, int((sample["date"] >= labeled_sample["date"].max()).sum() - 1))
    sample = labeled_sample
    train = sample[sample["date"] < test_start_dt].copy()
    test_mask = sample["date"] >= test_start_dt

    if len(train) < config.min_train_samples:
        raise ValueError(
            f"Not enough training samples before {config.test_start}: {len(train)}"
        )
    if int(test_mask.sum()) < config.min_test_samples:
        raise ValueError(
            f"Not enough test samples from {config.test_start}: {int(test_mask.sum())}"
        )

    # Rule-based composite score: trend + auction expectation + volume/turnover continuity + main-control.
    rule_weights = {
        "trend_strength": 0.32,
        "bull_bear_score": 0.35,
        "ret_20d": 0.08,
        "mom_20": 0.08,
        "volume_trend_5": 0.12,
        "amount_trend_5": 0.10,
        "turnover_trend_5": 0.12,
        "vol_ratio_5_20": 0.10,
        "volume_price_corr_20": 0.08,
        "up_volume_ratio_20": 0.10,
        "price_position_60": 0.10,
        "wk_ret_4": 0.12,
        "mk_ret_3": 0.10,
        "auction_gap": 0.10,
        "auction_amt_ratio": 0.08,
        "auction_trend_3": 0.12,
        "volatility_20": -0.10,
        "amihud_20": -0.10,
    }
    if "net_mf_amount_z_20" in used_factor_cols:
        rule_weights["net_mf_amount_z_20"] = 0.12
    if "large_order_pressure_20" in used_factor_cols:
        rule_weights["large_order_pressure_20"] = 0.10
    if "main_control_ratio" in used_factor_cols:
        rule_weights["main_control_ratio"] = 0.12
    if "main_control_strength_5" in used_factor_cols:
        rule_weights["main_control_strength_5"] = 0.16

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
    sample["pred"] = 0.0
    for col, weight in rule_weights.items():
        if col in z_df.columns:
            sample["pred"] += weight * z_df[col]

    train_pred = sample.loc[sample["date"] < test_start_dt, "pred"]
    pred_mean = float(train_pred.mean())
    pred_std = float(train_pred.std(ddof=0))
    if pred_std <= 1e-12:
        sample["position"] = 0.0
    else:
        z = (sample["pred"] - pred_mean) / pred_std
        sample["position"] = (0.45 + 0.30 * z).clip(0.0, 1.0)

    # Trend regime filter:
    # - In bull regime, avoid over-trading (hold higher base position, focus on high-sell only when score weakens).
    # - In bear regime, defensive positioning.
    bull = sample["bull_bear_score"] >= 0.08
    bear = sample["bull_bear_score"] <= -0.08
    sample.loc[bull, "position"] = sample.loc[bull, "position"].clip(0.55, 1.0)
    sample.loc[bear, "position"] = sample.loc[bear, "position"].clip(0.0, 0.45)

    # A signal formed after close can only be held from the next trading day.
    sample["position"] = sample["position"].shift(1).fillna(0.0)
    test = sample[test_mask].copy()
    test["strategy_ret"] = test["position"] * test["ret_1d"]
    test["benchmark_ret"] = test["ret_1d"]
    test["equity"] = (1.0 + test["strategy_ret"].fillna(0.0)).cumprod()
    test["benchmark_equity"] = (1.0 + test["benchmark_ret"].fillna(0.0)).cumprod()
    test["turnover"] = test["position"].diff().abs().fillna(test["position"].abs())

    total = float(test["equity"].iloc[-1] - 1.0)
    benchmark_total = float(test["benchmark_equity"].iloc[-1] - 1.0)
    n = len(test)
    ann_return = float((1.0 + total) ** (252 / max(1, n)) - 1.0)
    ann_vol = float(test["strategy_ret"].std(ddof=0) * np.sqrt(252))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")
    ic = float(test[["pred", "label_fwd_ret"]].corr(method="spearman").iloc[0, 1])

    stats = BacktestStats(
        code=config.code,
        train_start=config.train_start,
        test_start=test_start_dt.strftime("%Y%m%d"),
        test_end=test["date"].max().strftime("%Y%m%d"),
        data_start=data_start.strftime("%Y%m%d"),
        data_end=data_end.strftime("%Y%m%d"),
        dropped_tail_days=dropped_tail_days,
        label_horizon=config.label_horizon,
        model="rule_v1",
        factor_count=len(used_factor_cols),
        train_samples=len(train),
        test_days=n,
        strategy_total_return=total,
        benchmark_total_return=benchmark_total,
        excess_total_return=total - benchmark_total,
        strategy_annual_return=ann_return,
        strategy_annual_vol=ann_vol,
        strategy_sharpe=sharpe,
        strategy_max_drawdown=max_drawdown(test["equity"]),
        benchmark_max_drawdown=max_drawdown(test["benchmark_equity"]),
        win_rate=float((test["strategy_ret"] > 0).mean()),
        avg_position=float(test["position"].mean()),
        turnover=float(test["turnover"].sum()),
        prediction_ic=ic,
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        test.to_parquet(out_dir / "price_volume_backtest_timeseries.parquet", index=False)
        test.to_csv(out_dir / "price_volume_backtest_timeseries.csv", index=False)

    return BacktestResult(stats=stats, timeseries=test, factor_columns=used_factor_cols)
