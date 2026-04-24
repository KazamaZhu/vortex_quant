#!/usr/bin/env python3
"""Fetch single-stock research pack from Tushare and store parquet locally."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def _to_ts_code(code: str) -> str:
    s = code.strip().upper().replace('.SH', '').replace('.SZ', '')
    if s.startswith('6'):
        return f"{s}.SH"
    if s.startswith(('0', '3')):
        return f"{s}.SZ"
    raise ValueError(f"Cannot infer exchange suffix, please pass ts_code directly: {code!r}")


def _safe_fetch(name: str, fn) -> pd.DataFrame | None:
    try:
        df = fn()
        if df is not None and not df.empty:
            return df
        return pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {name}: {exc}", file=sys.stderr)
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch single-stock research pack (Tushare).")
    p.add_argument('--code', default='600821', help='Stock code, e.g. 600821 or 600821.SH')
    p.add_argument('--root', type=Path, default=None, help='Workspace root, default ~/Documents/vortex_workspace')
    p.add_argument('--start', default='20100101', help='Start date YYYYMMDD')
    p.add_argument('--minute-days', type=int, default=30, help='Recent N days minute bars; 0 disables')
    args = p.parse_args()

    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        print('ERROR: TUSHARE_TOKEN is not set', file=sys.stderr)
        return 1

    ts_code = _to_ts_code(args.code)
    end = datetime.now().strftime('%Y%m%d')
    root = (args.root or Path.home() / 'Documents/vortex_workspace').expanduser().resolve()
    out = root / 'research' / 'single_stock' / ts_code.replace('.', '_')
    out.mkdir(parents=True, exist_ok=True)

    import tushare as ts

    ts.set_token(token)
    pro = ts.pro_api()

    print(f"output_dir: {out}")
    print(f"ts_code={ts_code} range {args.start} ~ {end}")

    tasks: list[tuple[str, Path, object]] = [
        ('daily', out / 'daily.parquet', lambda: pro.daily(ts_code=ts_code, start_date=args.start, end_date=end)),
        ('adj_factor', out / 'adj_factor.parquet', lambda: pro.adj_factor(ts_code=ts_code, start_date=args.start, end_date=end)),
        ('daily_basic', out / 'daily_basic.parquet', lambda: pro.daily_basic(ts_code=ts_code, start_date=args.start, end_date=end)),
        ('dividend', out / 'dividend.parquet', lambda: pro.dividend(ts_code=ts_code, fields='ts_code,ann_date,end_date,div_proc,stk_div,stk_bo_rate,stk_co_rate,cash_div_tax,cash_div')),
        ('moneyflow', out / 'moneyflow.parquet', lambda: pro.moneyflow(ts_code=ts_code, start_date=args.start, end_date=end)),
        ('stk_auction', out / 'stk_auction.parquet', lambda: pro.stk_auction(ts_code=ts_code, start_date=args.start, end_date=end)),
        ('fina_indicator', out / 'fina_indicator.parquet', lambda: pro.fina_indicator(ts_code=ts_code, start_date=args.start, end_date=end)),
        ('top_list', out / 'top_list.parquet', lambda: pro.top_list(ts_code=ts_code, start_date=args.start, end_date=end)),
        ('top_inst', out / 'top_inst.parquet', lambda: pro.top_inst(ts_code=ts_code, start_date=args.start, end_date=end)),
    ]

    if args.minute_days > 0:
        minute_start = (datetime.now() - timedelta(days=args.minute_days)).strftime('%Y%m%d 09:30:00')
        minute_end = datetime.now().strftime('%Y%m%d 15:00:00')
        tasks.append(
            ('stk_mins_recent', out / 'stk_mins_recent.parquet', lambda: pro.stk_mins(ts_code=ts_code, start_date=minute_start, end_date=minute_end))
        )

    for name, path, fn in tasks:
        df = _safe_fetch(name, fn)
        if df is None:
            continue
        df.to_parquet(path, index=False)
        print(f"  OK {name}: {len(df)} rows -> {path.name}")

    basic = _safe_fetch('stock_basic', lambda: pro.stock_basic(ts_code=ts_code, fields='ts_code,symbol,name,area,industry,market,list_date,delist_date,is_hs'))
    if basic is not None and not basic.empty:
        basic.to_parquet(out / 'stock_basic.parquet', index=False)
        print(f"  OK stock_basic: {len(basic)} rows")

    index_df = _safe_fetch('index_daily_000001', lambda: pro.index_daily(ts_code='000001.SH', start_date=args.start, end_date=end))
    if index_df is not None and not index_df.empty:
        index_df.to_parquet(out / 'index_daily_000001_SH.parquet', index=False)
        print(f"  OK index_daily (000001.SH): {len(index_df)} rows")

    readme = out / 'README.txt'
    readme.write_text(
        f"ts_code={ts_code}\n"
        f"range: {args.start} ~ {end}\n\n"
        "Files:\n"
        "- daily.parquet\n"
        "- adj_factor.parquet\n"
        "- daily_basic.parquet\n"
        "- dividend.parquet\n"
        "- moneyflow.parquet\n"
        "- stk_auction.parquet\n"
        "- fina_indicator.parquet\n"
        "- top_list.parquet (optional)\n"
        "- top_inst.parquet (optional)\n"
        "- stk_mins_recent.parquet (optional)\n"
        "- stock_basic.parquet\n"
        "- index_daily_000001_SH.parquet\n",
        encoding='utf-8',
    )
    print(f"done. readme: {readme}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
