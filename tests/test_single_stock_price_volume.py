from __future__ import annotations

import numpy as np
import pandas as pd

from vortex.quant.factors.price_volume import build_price_volume_features
from vortex.quant.models import ModelConfig, build_regressor
from vortex.research.single_stock.backtest import PriceVolumeBacktestConfig, run_price_volume_backtest
from vortex.research.single_stock.akquant_adapter import run_akquant_price_volume_backtest
from vortex.research.single_stock.data import to_ts_code
from vortex.research.single_stock.t_strategy import TStrategyConfig, run_t_strategy_backtest


def _mock_price_volume_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=420, freq="B")
    rng = np.random.default_rng(7)
    close = 10.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.015, size=len(dates)))
    volume = 1_000_000 + rng.normal(0, 80_000, size=len(dates))
    volume = np.maximum(volume, 100_000)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * (1.0 + rng.normal(0, 0.004, size=len(dates))),
            "high": close * (1.0 + np.abs(rng.normal(0.012, 0.006, size=len(dates)))),
            "low": close * (1.0 - np.abs(rng.normal(0.012, 0.006, size=len(dates)))),
            "close": close,
            "close_adj": close,
            "volume": volume,
            "amount": close * volume,
            "turnover_rate": 1.5 + rng.normal(0, 0.2, size=len(dates)),
        }
    )


def test_to_ts_code_infers_exchange() -> None:
    assert to_ts_code("600396") == "600396.SH"
    assert to_ts_code("000001") == "000001.SZ"


def test_build_price_volume_features_returns_factor_columns() -> None:
    features, factor_cols = build_price_volume_features(_mock_price_volume_frame())

    assert "volume_z_20" in factor_cols
    assert "price_position_60" in factor_cols
    assert "label_fwd_ret" in features.columns


def test_model_factory_builds_ridge() -> None:
    model = build_regressor(ModelConfig(name="ridge", alpha=3.0))

    x = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    y = np.array([0.1, 0.2, 0.4])
    pred = model.fit(x, y).predict(x)

    assert pred.shape == (3,)


def test_run_price_volume_backtest_outputs_stats(tmp_path) -> None:
    frame = _mock_price_volume_frame()
    config = PriceVolumeBacktestConfig(
        code="600396.SH",
        train_start="20240102",
        test_start="20250102",
    )

    result = run_price_volume_backtest(frame, config=config, out_dir=tmp_path)

    assert result.stats.code == "600396.SH"
    assert result.stats.train_samples >= 160
    assert result.stats.test_days >= 60
    assert result.factor_columns
    assert (tmp_path / "price_volume_backtest_timeseries.parquet").exists()


def test_run_t_strategy_backtest_outputs_signals(tmp_path) -> None:
    frame = _mock_price_volume_frame()
    config = TStrategyConfig(
        code="600396.SH",
        train_start="20240102",
        test_start="20250102",
        score_threshold=0.0,
    )

    result = run_t_strategy_backtest(frame, config=config, out_dir=tmp_path)

    assert result.stats.code == "600396.SH"
    assert result.stats.test_days >= 60
    assert "action" in result.signals.columns
    assert (tmp_path / "t_strategy_signals.parquet").exists()


def test_run_akquant_price_volume_backtest_outputs_report(tmp_path) -> None:
    frame = _mock_price_volume_frame()
    config = PriceVolumeBacktestConfig(
        code="600396.SH",
        train_start="20240102",
        test_start="20250102",
    )

    result = run_akquant_price_volume_backtest(frame, config=config, out_dir=tmp_path)

    assert result.stats.engine == "akquant"
    assert result.stats.strategy_total_return is not None
    assert (tmp_path / "akquant_price_volume_report.html").exists()
