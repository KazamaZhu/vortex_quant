#!/usr/bin/env python3
"""Run a single-stock price-volume backtest."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from vortex.research.single_stock import (
    PriceVolumeBacktestConfig,
    export_kline_views,
    fetch_research_pack,
    load_price_volume_frame,
    run_price_volume_backtest,
    stock_pack_dir,
    to_ts_code,
)


DEFAULT_TRAIN_START = "20180101"


def one_year_ago_yyyymmdd(today: date | None = None) -> str:
    today = today or date.today()
    try:
        return today.replace(year=today.year - 1).strftime("%Y%m%d")
    except ValueError:
        return today.replace(year=today.year - 1, day=28).strftime("%Y%m%d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest a single-stock price-volume rule strategy.")
    parser.add_argument("--code", default="600396", help="Stock code, e.g. 600396 or 600396.SH")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project/workspace root")
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START, help="Training data start date YYYYMMDD")
    parser.add_argument("--test-start", default=one_year_ago_yyyymmdd(), help="Out-of-sample start date YYYYMMDD")
    parser.add_argument("--test-end", default=None, help="Out-of-sample end date YYYYMMDD")
    parser.add_argument("--label-horizon", type=int, default=5, help="Forward return horizon in trading days")
    parser.add_argument("--no-fetch", action="store_true", help="Reuse local parquet files without fetching")
    args = parser.parse_args()

    root = args.root.resolve()
    ts_code = to_ts_code(args.code)
    if not args.no_fetch:
        print(f"[1/3] Fetching data for {ts_code} from {args.train_start}")
        fetch_research_pack(root, ts_code, args.train_start)

    pack_dir = stock_pack_dir(root, ts_code)
    kline_paths = export_kline_views(pack_dir)
    if kline_paths:
        print(f"[kline] exported views: {kline_paths}")
    print("[2/3] Loading data and building price-volume features")
    frame = load_price_volume_frame(pack_dir)

    print(f"[3/3] Running out-of-sample backtest from {args.test_start}")
    config = PriceVolumeBacktestConfig(
        code=ts_code,
        train_start=args.train_start,
        test_start=args.test_start,
        test_end=args.test_end,
        label_horizon=args.label_horizon,
    )
    result = run_price_volume_backtest(frame, config=config, out_dir=pack_dir)

    stats_path = pack_dir / "price_volume_backtest_stats.json"
    stats_path.write_text(
        json.dumps(asdict(result.stats), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(asdict(result.stats), ensure_ascii=False, indent=2))
    print(f"Saved stats: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
