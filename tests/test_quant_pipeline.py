from __future__ import annotations

import numpy as np
import pandas as pd

from vortex.quant.pipeline import QuantPipelineConfig, build_factor_frame, run_quant_pipeline


def _mock_bars() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=160, freq="B")
    symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600519.SH", "300750.SZ", "688001.SH"]
    rows = []
    rng = np.random.default_rng(42)
    base_price = {
        "000001.SZ": 10.0,
        "000002.SZ": 15.0,
        "600000.SH": 8.0,
        "600519.SH": 1500.0,
        "300750.SZ": 200.0,
        "688001.SH": 60.0,
    }
    for dt in dates:
        ds = dt.strftime("%Y%m%d")
        for idx, sym in enumerate(symbols):
            drift = 0.0002 * (idx + 1)
            noise = rng.normal(0.0, 0.01)
            base_price[sym] *= 1.0 + drift + noise
            close = max(1.0, base_price[sym])
            volume = 1e6 + (idx + 1) * 2e5 + rng.normal(0.0, 5e4)
            amount = close * max(1.0, volume)
            rows.append(
                {
                    "date": ds,
                    "symbol": sym,
                    "close": close,
                    "volume": float(max(1.0, volume)),
                    "amount": float(max(1.0, amount)),
                }
            )
    return pd.DataFrame(rows)


def _mock_fundamental(dates: pd.Series, symbols: list[str]) -> pd.DataFrame:
    industry_map = {
        "000001.SZ": "bank",
        "000002.SZ": "real_estate",
        "600000.SH": "bank",
        "600519.SH": "food",
        "300750.SZ": "new_energy",
        "688001.SH": "semi",
    }
    rows = []
    for ds in dates.unique():
        for i, sym in enumerate(symbols):
            rows.append(
                {
                    "date": ds,
                    "symbol": sym,
                    "industry": industry_map[sym],
                    "pe_ttm": 8.0 + i * 3.0,
                    "pb": 0.8 + i * 0.4,
                    "roe": 0.06 + i * 0.01,
                    "gross_margin": 0.2 + i * 0.03,
                    "debt_to_assets": 0.7 - i * 0.06,
                }
            )
    return pd.DataFrame(rows)


def test_build_factor_frame_with_industry_neutral() -> None:
    bars = _mock_bars()
    symbols = sorted(bars["symbol"].unique())
    fundamental = _mock_fundamental(bars["date"], symbols)
    factors, factor_cols = build_factor_frame(bars, fundamental, do_industry_neutral=True)
    assert factor_cols
    assert "mom_20" in factor_cols
    assert "value_ep" in factor_cols
    assert "quality_roe" in factor_cols
    assert "volatility_20" in factor_cols
    assert "micro_amihud" in factor_cols
    assert {"date", "symbol"}.issubset(factors.columns)


def test_run_quant_pipeline_returns_metrics() -> None:
    bars = _mock_bars()
    symbols = sorted(bars["symbol"].unique())
    fundamental = _mock_fundamental(bars["date"], symbols)

    result = run_quant_pipeline(
        bars,
        fundamental,
        config=QuantPipelineConfig(label_horizon=5, do_industry_neutral=True),
    )
    assert result.factor_columns
    assert {"date", "symbol", "label_fwd_ret", "pred"} == set(result.prediction_frame.columns)
    assert "test_ic_mean" in result.metrics
    assert np.isfinite(result.metrics["train_samples"])
