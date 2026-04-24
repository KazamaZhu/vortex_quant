"""Single-stock research data helpers."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def to_ts_code(code: str) -> str:
    raw = code.strip().upper()
    if raw.endswith((".SH", ".SZ")):
        return raw
    compact = raw.replace(".SH", "").replace(".SZ", "")
    if compact.startswith("6"):
        return f"{compact}.SH"
    if compact.startswith(("0", "3")):
        return f"{compact}.SZ"
    raise ValueError(f"Cannot infer exchange suffix for code: {code!r}")


def stock_pack_dir(root: Path, code: str) -> Path:
    ts_code = to_ts_code(code)
    return root / "research" / "single_stock" / ts_code.replace(".", "_")


def fetch_research_pack(root: Path, code: str, start: str) -> None:
    """Fetch parquet files through the existing Tushare helper script."""
    cmd = [
        sys.executable,
        "scripts/fetch_single_stock_research_pack.py",
        "--code",
        to_ts_code(code),
        "--root",
        str(root),
        "--start",
        start,
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError("Data fetch failed. Check TUSHARE_TOKEN, network, and Tushare permissions.")


def read_parquet_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_price_volume_frame(pack_dir: Path) -> pd.DataFrame:
    """Load and normalize daily price-volume data from a single-stock pack."""
    daily = read_parquet_or_empty(pack_dir / "daily.parquet")
    adj = read_parquet_or_empty(pack_dir / "adj_factor.parquet")
    daily_basic = read_parquet_or_empty(pack_dir / "daily_basic.parquet")
    moneyflow = read_parquet_or_empty(pack_dir / "moneyflow.parquet")
    auction = read_parquet_or_empty(pack_dir / "stk_auction.parquet")
    top_list = read_parquet_or_empty(pack_dir / "top_list.parquet")
    top_inst = read_parquet_or_empty(pack_dir / "top_inst.parquet")

    if daily.empty:
        raise ValueError(f"daily.parquet is empty or missing under {pack_dir}")

    df = daily.copy()
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    if "vol" in df.columns and "volume" not in df.columns:
        df["volume"] = df["vol"]
    if "amount" not in df.columns:
        df["amount"] = np.nan
    df = df.sort_values("date").reset_index(drop=True)

    if not adj.empty and "adj_factor" in adj.columns:
        adj2 = adj[["trade_date", "adj_factor"]].copy()
        adj2["date"] = pd.to_datetime(adj2["trade_date"], format="%Y%m%d")
        df = df.merge(adj2[["date", "adj_factor"]], on="date", how="left")
        df["adj_factor"] = df["adj_factor"].ffill().bfill()
        df["close_adj"] = df["close"] * df["adj_factor"]
    else:
        df["close_adj"] = df["close"]

    if not daily_basic.empty:
        db = daily_basic.copy()
        db["date"] = pd.to_datetime(db["trade_date"], format="%Y%m%d")
        keep = [
            c
            for c in ["date", "turnover_rate", "volume_ratio", "total_mv", "circ_mv"]
            if c in db.columns
        ]
        df = df.merge(db[keep], on="date", how="left")

    if not moneyflow.empty:
        mf = moneyflow.copy()
        mf["date"] = pd.to_datetime(mf["trade_date"], format="%Y%m%d")
        keep = [
            c
            for c in [
                "date",
                "net_mf_amount",
                "net_mf_vol",
                "buy_sm_amount",
                "sell_sm_amount",
                "buy_md_amount",
                "sell_md_amount",
                "buy_lg_amount",
                "sell_lg_amount",
                "buy_elg_amount",
                "sell_elg_amount",
                "buy_sm_vol",
                "sell_sm_vol",
                "buy_md_vol",
                "sell_md_vol",
                "buy_lg_vol",
                "sell_lg_vol",
                "buy_elg_vol",
                "sell_elg_vol",
            ]
            if c in mf.columns
        ]
        if len(keep) > 1:
            df = df.merge(mf[keep], on="date", how="left")

    if not auction.empty:
        ac = auction.copy()
        if "trade_date" in ac.columns:
            ac["date"] = pd.to_datetime(ac["trade_date"], format="%Y%m%d", errors="coerce")
        elif "date" in ac.columns:
            ac["date"] = pd.to_datetime(ac["date"], errors="coerce")
        else:
            ac["date"] = pd.NaT

        # Keep generic auction columns if available; not all accounts expose all fields.
        keep = [c for c in ["date", "price", "vol", "amount", "v", "p", "current"] if c in ac.columns]
        if len(keep) > 1:
            ac2 = ac[keep].copy()
            if "vol" in ac2.columns and "auction_vol" not in ac2.columns:
                ac2 = ac2.rename(columns={"vol": "auction_vol"})
            if "amount" in ac2.columns and "auction_amount" not in ac2.columns:
                ac2 = ac2.rename(columns={"amount": "auction_amount"})
            if "price" in ac2.columns and "auction_price" not in ac2.columns:
                ac2 = ac2.rename(columns={"price": "auction_price"})
            if "p" in ac2.columns and "auction_price" not in ac2.columns:
                ac2 = ac2.rename(columns={"p": "auction_price"})
            if "v" in ac2.columns and "auction_vol" not in ac2.columns:
                ac2 = ac2.rename(columns={"v": "auction_vol"})
            if "current" in ac2.columns and "auction_price" not in ac2.columns:
                ac2 = ac2.rename(columns={"current": "auction_price"})
            df = df.merge(ac2, on="date", how="left")

    if not top_list.empty:
        tl = top_list.copy()
        if "trade_date" in tl.columns:
            tl["date"] = pd.to_datetime(tl["trade_date"], format="%Y%m%d", errors="coerce")
        elif "date" in tl.columns:
            tl["date"] = pd.to_datetime(tl["date"], errors="coerce")
        else:
            tl["date"] = pd.NaT
        tl = tl.dropna(subset=["date"])
        if not tl.empty:
            if "net_amount" in tl.columns:
                tl["lhb_net_amount"] = pd.to_numeric(tl["net_amount"], errors="coerce")
            elif {"buy", "sell"}.issubset(tl.columns):
                tl["lhb_net_amount"] = pd.to_numeric(tl["buy"], errors="coerce") - pd.to_numeric(tl["sell"], errors="coerce")
            else:
                tl["lhb_net_amount"] = np.nan
            if "amount" in tl.columns:
                tl["lhb_amount"] = pd.to_numeric(tl["amount"], errors="coerce")
            else:
                tl["lhb_amount"] = np.nan
            tl_agg = (
                tl.groupby("date", as_index=False)
                .agg({"lhb_net_amount": "sum", "lhb_amount": "sum"})
            )
            tl_agg["lhb_event"] = (tl_agg["lhb_amount"].fillna(0.0) > 0).astype(float)
            df = df.merge(tl_agg, on="date", how="left")

    if not top_inst.empty:
        ti = top_inst.copy()
        if "trade_date" in ti.columns:
            ti["date"] = pd.to_datetime(ti["trade_date"], format="%Y%m%d", errors="coerce")
        elif "date" in ti.columns:
            ti["date"] = pd.to_datetime(ti["date"], errors="coerce")
        else:
            ti["date"] = pd.NaT
        ti = ti.dropna(subset=["date"])
        if not ti.empty:
            if "net_buy" in ti.columns:
                ti["inst_net_buy"] = pd.to_numeric(ti["net_buy"], errors="coerce")
            elif {"buy", "sell"}.issubset(ti.columns):
                ti["inst_net_buy"] = pd.to_numeric(ti["buy"], errors="coerce") - pd.to_numeric(ti["sell"], errors="coerce")
            else:
                ti["inst_net_buy"] = np.nan
            if "buy" in ti.columns:
                ti["inst_buy"] = pd.to_numeric(ti["buy"], errors="coerce")
            else:
                ti["inst_buy"] = np.nan
            if "sell" in ti.columns:
                ti["inst_sell"] = pd.to_numeric(ti["sell"], errors="coerce")
            else:
                ti["inst_sell"] = np.nan
            ti_agg = (
                ti.groupby("date", as_index=False)
                .agg({"inst_net_buy": "sum", "inst_buy": "sum", "inst_sell": "sum"})
            )
            ti_agg["inst_event"] = ((ti_agg["inst_buy"].fillna(0.0) + ti_agg["inst_sell"].fillna(0.0)) > 0).astype(float)
            df = df.merge(ti_agg, on="date", how="left")

    return df.sort_values("date").reset_index(drop=True)


def export_kline_views(pack_dir: Path) -> dict[str, str]:
    """Export intraday/day/week/month kline views (with volume bars) for visualization."""
    out: dict[str, str] = {}
    daily = read_parquet_or_empty(pack_dir / "daily.parquet")
    if daily.empty:
        return out

    d = daily.copy()
    d["date"] = pd.to_datetime(d["trade_date"], format="%Y%m%d", errors="coerce")
    if "vol" in d.columns and "volume" not in d.columns:
        d["volume"] = d["vol"]
    if "amount" not in d.columns:
        d["amount"] = np.nan
    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in d.columns]
    day = d[keep].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    day_csv = pack_dir / "kline_day.csv"
    day.to_csv(day_csv, index=False)
    out["day_csv"] = str(day_csv)

    wk = (
        day.set_index("date")
        .resample("W-FRI")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    wk_csv = pack_dir / "kline_week.csv"
    wk.to_csv(wk_csv, index=False)
    out["week_csv"] = str(wk_csv)

    mk = (
        day.set_index("date")
        .resample("ME")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    mk_csv = pack_dir / "kline_month.csv"
    mk.to_csv(mk_csv, index=False)
    out["month_csv"] = str(mk_csv)

    mins = read_parquet_or_empty(pack_dir / "stk_mins_recent.parquet")
    if not mins.empty:
        m = mins.copy()
        time_col = "trade_time" if "trade_time" in m.columns else ("datetime" if "datetime" in m.columns else None)
        if time_col is not None:
            m["datetime"] = pd.to_datetime(m[time_col], errors="coerce")
            if "vol" in m.columns and "volume" not in m.columns:
                m["volume"] = m["vol"]
            if "amount" not in m.columns:
                m["amount"] = np.nan
            m_keep = [c for c in ["datetime", "open", "high", "low", "close", "volume", "amount"] if c in m.columns]
            m2 = m[m_keep].dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            minute_csv = pack_dir / "kline_intraday_recent.csv"
            m2.to_csv(minute_csv, index=False)
            out["intraday_csv"] = str(minute_csv)
    return out
