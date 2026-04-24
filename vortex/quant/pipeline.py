"""量化因子与模型流水线。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factors import (
    build_microstructure_factors,
    build_momentum_factors,
    build_quality_factors,
    build_value_factors,
    build_volatility_factors,
    make_forward_return_label,
    winsorize_by_mad,
    zscore_by_date,
    zscore_by_date_industry,
)
from .models import RidgeRegressor, ic_by_date, long_short_spread_by_date, rank_ic_by_date, walk_forward_split


@dataclass
class QuantPipelineConfig:
    """量化管线参数。"""

    label_horizon: int = 5
    winsor_n_mad: float = 5.0
    do_industry_neutral: bool = True
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    ridge_alpha: float = 2.0


@dataclass
class QuantPipelineResult:
    """量化管线输出。"""

    feature_frame: pd.DataFrame
    prediction_frame: pd.DataFrame
    factor_columns: list[str]
    metrics: dict[str, float]


def _merge_feature_blocks(blocks: list[pd.DataFrame]) -> pd.DataFrame:
    merged = blocks[0]
    for blk in blocks[1:]:
        merged = merged.merge(blk, on=["date", "symbol"], how="left")
    return merged


def build_factor_frame(
    bars: pd.DataFrame,
    fundamental: pd.DataFrame,
    *,
    do_industry_neutral: bool = True,
    industry_col: str = "industry",
    winsor_n_mad: float = 5.0,
) -> tuple[pd.DataFrame, list[str]]:
    """构建五大类因子并做截面标准化。"""
    mom = build_momentum_factors(bars)
    val = build_value_factors(fundamental)
    qua = build_quality_factors(fundamental)
    vol = build_volatility_factors(bars)
    mic = build_microstructure_factors(bars)

    features = _merge_feature_blocks([mom, val, qua, vol, mic])
    factor_cols = [c for c in features.columns if c not in {"date", "symbol", industry_col}]

    features = winsorize_by_mad(features, factor_cols, n_mad=winsor_n_mad)
    if do_industry_neutral and industry_col in features.columns:
        features = zscore_by_date_industry(features, factor_cols, industry_col=industry_col)
    else:
        features = zscore_by_date(features, factor_cols)
    return features, factor_cols


def run_quant_pipeline(
    bars: pd.DataFrame,
    fundamental: pd.DataFrame,
    *,
    config: QuantPipelineConfig | None = None,
) -> QuantPipelineResult:
    """执行：因子构建 -> 标签 -> 时间切分 -> Ridge 训练 -> 评估。"""
    cfg = config or QuantPipelineConfig()
    label_df = make_forward_return_label(bars, horizon=cfg.label_horizon)
    features, factor_cols = build_factor_frame(
        bars,
        fundamental,
        do_industry_neutral=cfg.do_industry_neutral,
        winsor_n_mad=cfg.winsor_n_mad,
    )

    dataset = features.merge(label_df, on=["date", "symbol"], how="inner")
    dataset = dataset.dropna(subset=factor_cols + ["label_fwd_ret"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    if dataset.empty:
        raise ValueError("可训练样本为空，请检查输入数据与标签窗口。")

    train_idx, _valid_idx, test_idx = walk_forward_split(
        dataset,
        train_ratio=cfg.train_ratio,
        valid_ratio=cfg.valid_ratio,
    )

    x = dataset[factor_cols].to_numpy(dtype=float)
    y = dataset["label_fwd_ret"].to_numpy(dtype=float)

    model = RidgeRegressor(alpha=cfg.ridge_alpha).fit(x[train_idx], y[train_idx])
    dataset["pred"] = model.predict(x)

    test_df = dataset.iloc[test_idx].copy()
    ic = ic_by_date(test_df, pred_col="pred", label_col="label_fwd_ret")
    ric = rank_ic_by_date(test_df, pred_col="pred", label_col="label_fwd_ret")
    ls = long_short_spread_by_date(test_df, pred_col="pred", label_col="label_fwd_ret")

    metrics = {
        "test_ic_mean": float(ic.mean()),
        "test_rank_ic_mean": float(ric.mean()),
        "test_long_short_mean": float(ls.mean()),
        "train_samples": float(len(train_idx)),
        "test_samples": float(len(test_idx)),
    }
    return QuantPipelineResult(
        feature_frame=features,
        prediction_frame=dataset[["date", "symbol", "label_fwd_ret", "pred"]],
        factor_columns=factor_cols,
        metrics=metrics,
    )
