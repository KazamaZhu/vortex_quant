"""时间序列切分。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_split(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按时间顺序切分 train/valid/test 的索引。"""
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio 必须在 (0,1)")
    if not (0 < valid_ratio < 1):
        raise ValueError("valid_ratio 必须在 (0,1)")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio 必须小于 1")

    uniq_dates = np.array(sorted(df[date_col].dropna().unique()))
    n_dates = len(uniq_dates)
    if n_dates < 3:
        raise ValueError("至少需要 3 个交易日用于切分")

    train_end = max(1, int(n_dates * train_ratio))
    valid_end = max(train_end + 1, int(n_dates * (train_ratio + valid_ratio)))
    valid_end = min(valid_end, n_dates - 1)

    train_dates = set(uniq_dates[:train_end])
    valid_dates = set(uniq_dates[train_end:valid_end])
    test_dates = set(uniq_dates[valid_end:])

    idx = np.arange(len(df))
    train_idx = idx[df[date_col].isin(train_dates)]
    valid_idx = idx[df[date_col].isin(valid_dates)]
    test_idx = idx[df[date_col].isin(test_dates)]
    return train_idx, valid_idx, test_idx
