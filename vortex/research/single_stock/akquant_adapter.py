"""AKQuant integration for single-stock research backtests."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from vortex.research.single_stock.backtest import PriceVolumeBacktestConfig, run_price_volume_backtest


@dataclass
class AKQuantBacktestStats:
    code: str
    engine: str
    strategy: str
    train_start: str
    test_start: str
    test_end: str
    data_start: str
    data_end: str
    label_horizon: int
    strategy_total_return: float | None
    strategy_sharpe: float | None
    strategy_max_drawdown: float | None
    trades: int
    report_html: str | None


@dataclass
class AKQuantBacktestResult:
    stats: AKQuantBacktestStats
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: pd.DataFrame


def _adjusted_ohlcv(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    df = frame.sort_values("date").copy()
    factor = df["close_adj"] / df["close"].replace(0, pd.NA)
    out = pd.DataFrame(
        {
            "date": df["date"],
            "symbol": code,
            "open": df["open"] * factor,
            "high": df["high"] * factor,
            "low": df["low"] * factor,
            "close": df["close_adj"],
            "volume": df["volume"],
        }
    )
    return out.dropna(subset=["open", "high", "low", "close", "volume"])


def _metric_value(metrics: pd.DataFrame, *names: str) -> float | None:
    if metrics is None or metrics.empty:
        return None
    for name in names:
        if name in metrics.index:
            try:
                return float(metrics.loc[name, "value"])
            except Exception:
                return None
    lower_names = {x.lower() for x in names}
    for col in metrics.columns:
        if str(col).lower() in {"value", "值"}:
            value_col = col
            break
    else:
        value_col = metrics.columns[-1]
    for _, row in metrics.iterrows():
        label = " ".join(str(x).lower() for x in row.values)
        if any(name in label for name in lower_names):
            try:
                return float(row[value_col])
            except Exception:
                return None
    return None


def run_akquant_price_volume_backtest(
    frame: pd.DataFrame,
    *,
    config: PriceVolumeBacktestConfig,
    out_dir: Path | None = None,
    initial_cash: float = 100_000.0,
) -> AKQuantBacktestResult:
    """Use Tushare-normalized data and AKQuant as the execution/backtest engine."""
    try:
        from akquant import Strategy, run_backtest
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("AKQuant is not installed. Install project dependencies first.") from exc

    class TargetWeightStrategy(Strategy):  # type: ignore[misc, valid-type]
        """AKQuant strategy that rebalances to bar.extra['target_weight']."""

        def on_bar(self, bar):  # type: ignore[no-untyped-def]
            target = 0.0
            if getattr(bar, "extra", None):
                target = float(bar.extra.get("target_weight") or 0.0)
            self.order_target_percent(target, symbol=bar.symbol)

    signal_result = run_price_volume_backtest(frame, config=config, out_dir=None)
    prices = _adjusted_ohlcv(frame, config.code)
    signals = signal_result.timeseries[["date", "position"]].copy()
    signals["date"] = pd.to_datetime(signals["date"])
    data = prices.merge(signals.rename(columns={"position": "target_weight"}), on="date", how="inner")
    data["target_weight"] = data["target_weight"].fillna(0.0)
    benchmark_returns = (
        data.sort_values("date").set_index("date")["close"].pct_change().fillna(0.0).rename("SIMPLE_BENCH")
    )

    result = run_backtest(
        data=data,
        strategy=TargetWeightStrategy,
        symbols=config.code,
        initial_cash=initial_cash,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
        transfer_fee_rate=0.00001,
        min_commission=5.0,
        t_plus_one=True,
        show_progress=False,
    )

    report_path: Path | None = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "akquant_price_volume_report.html"
        result.report(
            title=f"{config.code} AKQuant Price-Volume Backtest",
            filename=str(report_path),
            show=False,
            market_data=data[["date", "symbol", "open", "high", "low", "close", "volume"]].copy(),
            plot_symbol=config.code,
            include_trade_kline=True,
            benchmark=benchmark_returns,
            curve_freq="D",
        )

    equity = result.equity_curve.reset_index()
    equity.columns = ["date", "equity"]
    trades = result.trades_df.copy()
    metrics = result.metrics_df.copy()
    if out_dir is not None:
        equity.to_csv(out_dir / "akquant_price_volume_equity.csv", index=False)
        trades.to_csv(out_dir / "akquant_price_volume_trades.csv", index=False)
        metrics.to_csv(out_dir / "akquant_price_volume_metrics.csv", index=True)
        benchmark_returns.reset_index().to_csv(out_dir / "akquant_price_volume_benchmark.csv", index=False)

    start_equity = float(result.equity_curve.iloc[0]) if not result.equity_curve.empty else initial_cash
    end_equity = float(result.equity_curve.iloc[-1]) if not result.equity_curve.empty else initial_cash
    total_return = end_equity / start_equity - 1.0 if start_equity else None

    stats = AKQuantBacktestStats(
        code=config.code,
        engine="akquant",
        strategy="price_volume",
        train_start=config.train_start,
        test_start=config.test_start,
        test_end=signal_result.stats.test_end,
        data_start=signal_result.stats.data_start,
        data_end=signal_result.stats.data_end,
        label_horizon=config.label_horizon,
        strategy_total_return=total_return,
        strategy_sharpe=_metric_value(metrics, "sharpe_ratio", "sharpe", "夏普"),
        strategy_max_drawdown=(
            -abs(_metric_value(metrics, "max_drawdown_pct") or 0.0) / 100.0
        ),
        trades=len(trades),
        report_html=str(report_path) if report_path else None,
    )
    if out_dir is not None:
        (out_dir / "akquant_price_volume_stats.json").write_text(
            __import__("json").dumps(asdict(stats), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return AKQuantBacktestResult(stats=stats, equity_curve=equity, trades=trades, metrics=metrics)
