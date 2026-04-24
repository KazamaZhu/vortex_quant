#!/usr/bin/env python3
"""单标的量化回测（以 600821 为例）。

流程：
1) 先调用 fetch_single_stock_research_pack.py 拉取最新数据；
2) 构建动量/价值/质量/波动/微观结构特征；
3) 时间切分训练 Ridge；
4) 生成信号并做长仓择时回测。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vortex.quant.models import RidgeRegressor


def _to_ts_code(code: str) -> str:
    s = code.strip().upper()
    if s.endswith((".SH", ".SZ")):
        return s
    if s.startswith("6"):
        return f"{s}.SH"
    if s.startswith(("0", "3")):
        return f"{s}.SZ"
    raise ValueError(f"无法推断交易所后缀: {code!r}")


def _run_fetch(root: Path, code: str, start: str) -> None:
    cmd = [
        sys.executable,
        "scripts/fetch_single_stock_research_pack.py",
        "--code",
        code,
        "--root",
        str(root),
        "--start",
        start,
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError("数据拉取失败，请检查 TUSHARE_TOKEN / 网络 / 积分权限。")


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _prepare_daily_frame(pack_dir: Path) -> pd.DataFrame:
    daily = _load_parquet(pack_dir / "daily.parquet")
    adj = _load_parquet(pack_dir / "adj_factor.parquet")
    daily_basic = _load_parquet(pack_dir / "daily_basic.parquet")
    fina = _load_parquet(pack_dir / "fina_indicator.parquet")

    if daily.empty:
        raise ValueError("daily 数据为空，无法回测。")

    df = daily.copy()
    if "vol" in df.columns and "volume" not in df.columns:
        df["volume"] = df["vol"]
    if "amount" not in df.columns:
        df["amount"] = np.nan
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)

    if not adj.empty and "adj_factor" in adj.columns:
        adj2 = adj[["trade_date", "adj_factor"]].copy()
        adj2["date"] = pd.to_datetime(adj2["trade_date"], format="%Y%m%d")
        df = df.merge(adj2[["date", "adj_factor"]], on="date", how="left")
        df["adj_factor"] = df["adj_factor"].ffill().bfill()
        # 对收益率而言只需相对变化，比例常数不影响结果。
        df["close_adj"] = df["close"] * df["adj_factor"]
    else:
        df["close_adj"] = df["close"]

    if not daily_basic.empty:
        db = daily_basic.copy()
        db["date"] = pd.to_datetime(db["trade_date"], format="%Y%m%d")
        keep_cols = [c for c in ["date", "pe_ttm", "pb", "turnover_rate"] if c in db.columns]
        df = df.merge(db[keep_cols], on="date", how="left")

    if not fina.empty and "ann_date" in fina.columns:
        fin = fina.copy()
        fin["ann_date"] = pd.to_datetime(fin["ann_date"], format="%Y%m%d", errors="coerce")
        fin = fin.sort_values("ann_date")
        fin_keep = [c for c in ["ann_date", "roe", "grossprofit_margin", "debt_to_assets"] if c in fin.columns]
        if len(fin_keep) > 1:
            df = pd.merge_asof(
                df.sort_values("date"),
                fin[fin_keep].rename(columns={"ann_date": "date"}).sort_values("date"),
                on="date",
                direction="backward",
            )

    # 缺失时给出可用退化项，保证流程可跑
    if "pe_ttm" not in df.columns:
        df["pe_ttm"] = np.nan
    if "pb" not in df.columns:
        df["pb"] = np.nan
    if "roe" not in df.columns:
        df["roe"] = np.nan
    if "grossprofit_margin" not in df.columns:
        df["grossprofit_margin"] = np.nan
    if "debt_to_assets" not in df.columns:
        df["debt_to_assets"] = np.nan
    if "turnover_rate" not in df.columns:
        df["turnover_rate"] = np.nan

    return df


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["close_adj"].pct_change()

    # 动量（跳过最近 5 日）
    for w in (20, 60, 120):
        out[f"mom_{w}"] = out["close_adj"].shift(5) / out["close_adj"].shift(5 + w) - 1.0

    # 价值
    out["value_ep"] = 1.0 / out["pe_ttm"].replace(0, np.nan)
    out["value_bp"] = 1.0 / out["pb"].replace(0, np.nan)

    # 质量
    out["quality_roe"] = out["roe"]
    out["quality_gross_margin"] = out["grossprofit_margin"]
    out["quality_low_leverage"] = -out["debt_to_assets"]

    # 波动
    out["volatility_20"] = out["ret_1d"].rolling(20).std()

    # 微观结构（日频代理）
    out["micro_amihud"] = out["ret_1d"].abs() / out["amount"].replace(0, np.nan)
    out["micro_turnover_proxy"] = np.where(
        out["turnover_rate"].notna(),
        out["turnover_rate"],
        out.get("volume", pd.Series(np.nan, index=out.index))
        / out["amount"].replace(0, np.nan),
    )

    # 标签：未来 5 日收益
    out["label_fwd_ret"] = out["close_adj"].shift(-5) / out["close_adj"] - 1.0
    return out


def _time_split(n: int, train_ratio: float = 0.7, valid_ratio: float = 0.15) -> tuple[slice, slice, slice]:
    train_end = max(1, int(n * train_ratio))
    valid_end = max(train_end + 1, int(n * (train_ratio + valid_ratio)))
    valid_end = min(valid_end, n - 1)
    return slice(0, train_end), slice(train_end, valid_end), slice(valid_end, n)


@dataclass
class BacktestStats:
    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    num_days: int


def _max_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def run_backtest(df: pd.DataFrame, out_dir: Path) -> BacktestStats:
    factor_cols = [
        "mom_20",
        "mom_60",
        "mom_120",
        "value_ep",
        "value_bp",
        "quality_roe",
        "quality_gross_margin",
        "quality_low_leverage",
        "volatility_20",
        "micro_amihud",
        "micro_turnover_proxy",
    ]
    sample = df[["date", "ret_1d", "label_fwd_ret", *factor_cols]].copy()
    # 单标的场景改用时间维度标准化，避免“按日截面”在 1 只股票上退化。
    for col in factor_cols:
        rolling_mean = sample[col].rolling(252, min_periods=60).mean()
        rolling_std = sample[col].rolling(252, min_periods=60).std().replace(0, np.nan)
        sample[col] = (sample[col] - rolling_mean) / rolling_std

    sample = sample.dropna().reset_index(drop=True)
    if len(sample) < 120:
        raise ValueError("可用样本过少，建议把 --start 提前。")

    tr, _va, te = _time_split(len(sample))
    x = sample[factor_cols].to_numpy(dtype=float)
    y = sample["label_fwd_ret"].to_numpy(dtype=float)
    model = RidgeRegressor(alpha=2.0).fit(x[tr], y[tr])
    sample["pred"] = model.predict(x)

    # 单标的使用连续仓位：基于训练集预测分布映射到 [0,1]。
    train_pred = sample.iloc[tr]["pred"]
    pred_mean = float(train_pred.mean())
    pred_std = float(train_pred.std(ddof=0))
    if pred_std <= 1e-12:
        sample["signal"] = 0.0
    else:
        z = (sample["pred"] - pred_mean) / pred_std
        sample["signal"] = (0.5 + 0.25 * z).clip(0.0, 1.0).shift(1).fillna(0.0)
    test = sample.iloc[te].copy()
    test["strategy_ret"] = test["signal"] * test["ret_1d"]
    test["equity"] = (1.0 + test["strategy_ret"].fillna(0.0)).cumprod()
    test["bh_equity"] = (1.0 + test["ret_1d"].fillna(0.0)).cumprod()

    total_return = float(test["equity"].iloc[-1] - 1.0)
    n = len(test)
    ann_return = float((1.0 + total_return) ** (252 / max(1, n)) - 1.0)
    ann_vol = float(test["strategy_ret"].std(ddof=0) * np.sqrt(252))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")
    mdd = _max_drawdown(test["equity"])
    win_rate = float((test["strategy_ret"] > 0).mean())

    out_dir.mkdir(parents=True, exist_ok=True)
    test.to_parquet(out_dir / "backtest_timeseries.parquet", index=False)
    return BacktestStats(
        total_return=total_return,
        annual_return=ann_return,
        annual_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=mdd,
        win_rate=win_rate,
        num_days=n,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="单标的量化模型回测")
    p.add_argument("--code", default="600821", help="股票代码，如 600821 或 600821.SH")
    p.add_argument("--start", default="20150101", help="起始日期 YYYYMMDD")
    p.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="项目根目录（默认当前目录）",
    )
    args = p.parse_args()

    root = args.root.resolve()
    ts_code = _to_ts_code(args.code)
    print(f"[1/3] 拉取最新数据: {ts_code}")
    _run_fetch(root, ts_code, args.start)

    pack_dir = root / "research" / "single_stock" / ts_code.replace(".", "_")
    if not pack_dir.exists():
        raise FileNotFoundError(f"未找到数据目录: {pack_dir}")

    print("[2/3] 构建特征与样本")
    frame = _prepare_daily_frame(pack_dir)
    feat = _build_features(frame)

    print("[3/3] 训练并回测")
    stats = run_backtest(feat, pack_dir)
    stats_path = pack_dir / "backtest_stats.json"
    stats_path.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
    print("完成。")
    print(f"- 输出目录: {pack_dir}")
    print(f"- 指标文件: {stats_path.name}")
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
