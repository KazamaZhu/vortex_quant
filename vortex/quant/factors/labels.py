"""标签构建。"""
from __future__ import annotations

import pandas as pd


def make_forward_return_label(
    bars: pd.DataFrame,
    *,
    horizon: int = 5,
    price_col: str = "close",
    label_col: str = "label_fwd_ret",
) -> pd.DataFrame:
    """构建未来 N 日收益标签。"""
    if not {"date", "symbol", price_col}.issubset(bars.columns):
        raise ValueError("make_forward_return_label 需要 date/symbol/price 列")
    ordered = bars.sort_values(["symbol", "date"]).copy()
    fut = ordered.groupby("symbol")[price_col].shift(-horizon)
    ordered[label_col] = (fut / ordered[price_col]) - 1.0
    return ordered[["date", "symbol", label_col]]
