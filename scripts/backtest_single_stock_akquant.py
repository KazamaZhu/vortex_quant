#!/usr/bin/env python3
"""Run a Tushare-data + AKQuant-engine single-stock backtest."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vortex.research.single_stock import (
    PriceVolumeBacktestConfig,
    export_kline_views,
    fetch_research_pack,
    load_price_volume_frame,
    run_akquant_price_volume_backtest,
    stock_pack_dir,
    to_ts_code,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest price-volume strategy with AKQuant engine.")
    parser.add_argument("--code", default="600396")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train-start", default="20180101")
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--no-fetch", action="store_true")
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
    print("[2/3] Loading Tushare research pack")
    frame = load_price_volume_frame(pack_dir)
    print("[3/3] Running AKQuant engine")
    config = PriceVolumeBacktestConfig(
        code=ts_code,
        train_start=args.train_start,
        test_start=args.test_start,
        test_end=args.test_end,
        label_horizon=args.label_horizon,
    )
    result = run_akquant_price_volume_backtest(frame, config=config, out_dir=pack_dir)
    print(json.dumps(asdict(result.stats), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
