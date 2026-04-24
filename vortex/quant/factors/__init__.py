"""因子模块。"""

from .classic import (
    build_microstructure_factors,
    build_momentum_factors,
    build_quality_factors,
    build_value_factors,
    build_volatility_factors,
)
from .candlestick_moneyflow import build_candlestick_moneyflow_features
from .cross_section import rank_to_uniform, winsorize_by_mad, zscore_by_date, zscore_by_date_industry
from .labels import make_forward_return_label
from .price_volume import build_price_volume_features, rolling_zscore

__all__ = [
    "build_microstructure_factors",
    "build_momentum_factors",
    "build_quality_factors",
    "build_value_factors",
    "build_volatility_factors",
    "build_candlestick_moneyflow_features",
    "build_price_volume_features",
    "rolling_zscore",
    "winsorize_by_mad",
    "zscore_by_date",
    "zscore_by_date_industry",
    "rank_to_uniform",
    "make_forward_return_label",
]
