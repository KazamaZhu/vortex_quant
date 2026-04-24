from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _default_workspace() -> Path:
    return Path(os.path.expanduser("~/Documents/vortex_workspace")).resolve()


def _resolve_root(raw: str | None) -> Path:
    if not raw or not str(raw).strip():
        return _default_workspace()
    return Path(raw).expanduser().resolve()


def _tushare_token_status(root: Path) -> dict[str, Any]:
    env = os.environ.get("TUSHARE_TOKEN")
    if env:
        return {
            "configured": True,
            "source": "environment",
            "masked_prefix": (env[:8] + "...") if len(env) > 8 else "***",
        }
    env_file = root / ".env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "TUSHARE_TOKEN":
                val = value.strip().strip('"').strip("'")
                if val:
                    return {
                        "configured": True,
                        "source": "workspace",
                        "masked_prefix": (val[:8] + "...") if len(val) > 8 else "***",
                    }
    return {"configured": False, "source": None, "masked_prefix": None}


def _run_vortex_data_subprocess(args: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "vortex", "data", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0 and not proc.stdout.strip():
        raise HTTPException(status_code=400, detail=out)
    return out


def _run_vortex_init_subprocess(args: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "vortex", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        raise HTTPException(status_code=400, detail=out)
    return out


def _run_script_subprocess(args: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        raise HTTPException(status_code=400, detail=out)
    return out


def _backtest_result_dirs(root: Path) -> list[Path]:
    base = root / "research" / "single_stock"
    if not base.exists():
        return []
    return [
        p
        for p in base.iterdir()
        if p.is_dir()
        and (
            (p / "price_volume_backtest_stats.json").exists()
            or (p / "akquant_price_volume_stats.json").exists()
            or (p / "t_strategy_stats.json").exists()
        )
    ]


def _load_backtest_stats(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read {path}: {exc}") from exc


def _load_backtest_timeseries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pandas as pd

        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        keep = [
            c
            for c in [
                "date",
                "equity",
                "benchmark_equity",
                "equity_curve",
                "position",
                "pred",
                "strategy_ret",
                "benchmark_ret",
                "label_fwd_ret",
                "action",
                "sell_score",
                "buy_score",
                "label_sell_opportunity",
                "label_buy_opportunity",
                "action_hit",
            ]
            if c in df.columns
        ]
        return df[keep].replace({float("nan"): None}).to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read {path}: {exc}") from exc


def _load_kline_csv(path: Path, time_col: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pandas as pd

        df = pd.read_csv(path)
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M:%S" if "time" in time_col else "%Y-%m-%d"
            )
        keep = [c for c in [time_col, "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
        return df[keep].replace({float("nan"): None}).to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read {path}: {exc}") from exc


def _trend_snapshot_from_day(day_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not day_rows:
        return {"regime": "unknown", "score": None}
    try:
        import pandas as pd

        df = pd.DataFrame(day_rows)
        df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        if df.empty:
            return {"regime": "unknown", "score": None}
        ma20 = df["close"].rolling(20, min_periods=15).mean()
        ma60 = df["close"].rolling(60, min_periods=40).mean()
        vol5 = df["volume"].rolling(5, min_periods=3).mean()
        vol20 = df["volume"].rolling(20, min_periods=15).mean()
        ret20 = df["close"].pct_change(20)
        r = len(df) - 1
        ma20_gap = float(df.loc[r, "close"] / ma20.loc[r] - 1.0) if pd.notna(ma20.loc[r]) else 0.0
        ma60_gap = float(df.loc[r, "close"] / ma60.loc[r] - 1.0) if pd.notna(ma60.loc[r]) else 0.0
        vol_ratio = float(vol5.loc[r] / vol20.loc[r] - 1.0) if pd.notna(vol5.loc[r]) and pd.notna(vol20.loc[r]) and vol20.loc[r] else 0.0
        trend_strength = float(ret20.loc[r]) if pd.notna(ret20.loc[r]) else 0.0
        score = 0.5 * trend_strength + 0.25 * ma20_gap + 0.15 * ma60_gap + 0.10 * vol_ratio
        regime = "bull" if score >= 0.08 else ("bear" if score <= -0.08 else "neutral")
        return {
            "regime": regime,
            "score": score,
            "ma20_gap": ma20_gap,
            "ma60_gap": ma60_gap,
            "vol_ratio_5_20_minus_1": vol_ratio,
            "ret_20d": trend_strength,
        }
    except Exception:
        return {"regime": "unknown", "score": None}


def _parse_fetch_summary(stdout: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"tables": []}
    for raw in stdout.splitlines():
        line = raw.strip()
        m = re.search(r"ts_code=([0-9A-Z.]+)\s+区间\s+(\d{8})\s*~\s*(\d{8})", line)
        if m:
            summary["code"] = m.group(1)
            summary["fetch_start"] = m.group(2)
            summary["fetch_end"] = m.group(3)
            continue
        m2 = re.search(r"OK\s+([a-zA-Z0-9_]+):\s*(\d+)\s*行", line)
        if m2:
            summary["tables"].append({"name": m2.group(1), "rows": int(m2.group(2))})
    return summary


app = FastAPI(
    title="Vortex Web Console",
    description="Web wrapper around existing CLI/data/backtest features.",
    version="0.1.0",
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/workspace/summary")
def workspace_summary(root: str | None = Query(None, description="Workspace root")) -> dict[str, Any]:
    r = _resolve_root(root)
    db = r / "state" / "control.db"
    return {
        "root": str(r),
        "initialized": db.exists(),
        "data_dir": str(r / "data"),
        "state_dir": str(r / "state"),
        "profiles_dir": str(r / "profiles"),
        "tushare": _tushare_token_status(r),
    }


@app.get("/api/profile/resolve")
def profile_resolve(
    name: str = Query("default"),
    typ: str = Query("data", alias="type"),
    root: str | None = Query(None),
) -> dict[str, Any]:
    from vortex.config.profile.resolver import ProfileResolver
    from vortex.config.profile.store import ProfileStore
    from vortex.runtime.workspace import Workspace

    r = _resolve_root(root)
    try:
        ws = Workspace(r)
        ws.ensure_initialized()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = ProfileStore(ws.profiles_dir)
    resolver = ProfileResolver(store)
    try:
        _profile, sources = resolver.resolve(name, typ)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {k: {"value": rf.value, "source": rf.source} for k, rf in sources.items()}


@app.get("/api/profile/explain")
def profile_explain(
    name: str = Query("default"),
    typ: str = Query("data", alias="type"),
    root: str | None = Query(None),
) -> dict[str, str]:
    from vortex.config.profile.resolver import ProfileResolver
    from vortex.config.profile.store import ProfileStore
    from vortex.runtime.workspace import Workspace

    r = _resolve_root(root)
    try:
        ws = Workspace(r)
        ws.ensure_initialized()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = ProfileStore(ws.profiles_dir)
    resolver = ProfileResolver(store)
    try:
        text = resolver.explain(name, typ)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"text": text}


@app.get("/api/data/dataset-catalog")
def dataset_catalog() -> dict[str, Any]:
    from vortex.data.provider.tushare_registry import (
        DEFAULT_TUSHARE_PRIORITY_DATASETS,
        TUSHARE_DATASET_REGISTRY,
        get_default_tushare_datasets,
    )

    default_ids = set(get_default_tushare_datasets())
    items: list[dict[str, Any]] = []
    for name, meta in sorted(
        TUSHARE_DATASET_REGISTRY.items(),
        key=lambda x: (str(x[1].get("phase", "")), x[0]),
    ):
        items.append(
            {
                "id": name,
                "description": str(meta.get("description", "")),
                "phase": str(meta.get("phase", "")),
                "api": str(meta.get("api") or name),
                "in_default_for_current_account": name in default_ids,
            }
        )

    priority = [x for x in DEFAULT_TUSHARE_PRIORITY_DATASETS if x in TUSHARE_DATASET_REGISTRY]

    return {
        "datasets": items,
        "presets": {
            "minimal": ["instruments", "calendar", "bars"],
            "priority_bootstrap": priority,
            "default_account_filtered": sorted(default_ids),
        },
        "hint": (
            "Use profile defaults by omitting --datasets; custom selection maps to --datasets a,b,c."
        ),
    }


@app.get("/api/data/status")
def data_status(
    root: str | None = Query(None),
    profile: str | None = Query(None),
) -> dict[str, Any]:
    from vortex.cli import _collect_data_status, _load_workspace_env, _resolve_data_profile_name

    r = _resolve_root(root)
    _load_workspace_env(r)
    profile_name = _resolve_data_profile_name(profile)
    try:
        return _collect_data_status(r, profile_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/backtest/results")
def backtest_results(root: str | None = Query(None)) -> dict[str, Any]:
    r = _resolve_root(root)
    items: list[dict[str, Any]] = []
    for result_dir in _backtest_result_dirs(r):
        candidate_stats = [
            result_dir / "price_volume_backtest_stats.json",
            result_dir / "akquant_price_volume_stats.json",
            result_dir / "t_strategy_stats.json",
        ]
        for stats_path in candidate_stats:
            if not stats_path.exists():
                continue
            stats = _load_backtest_stats(stats_path)
            if stats_path.name == "akquant_price_volume_stats.json":
                default_engine = "akquant"
                default_strategy = "price_volume"
            elif stats_path.name == "t_strategy_stats.json":
                default_engine = "vortex"
                default_strategy = "t_strategy"
            else:
                default_engine = "vortex"
                default_strategy = "price_volume"
            engine = str(stats.get("engine", default_engine))
            strategy = str(stats.get("strategy", default_strategy))
            items.append(
                {
                    "id": f"{result_dir.name}:{engine}:{strategy}",
                    "result_dir_id": result_dir.name,
                    "code": stats.get("code") or result_dir.name.replace("_", "."),
                    "model": stats.get("model"),
                    "engine": engine,
                    "strategy": strategy,
                    "test_start": stats.get("test_start"),
                    "test_end": stats.get("test_end"),
                    "strategy_total_return": stats.get("strategy_total_return"),
                    "benchmark_total_return": stats.get("benchmark_total_return"),
                    "strategy_sharpe": stats.get("strategy_sharpe"),
                    "strategy_max_drawdown": stats.get("strategy_max_drawdown"),
                    "prediction_ic": stats.get("prediction_ic"),
                    "data_end": stats.get("data_end"),
                    "dropped_tail_days": stats.get("dropped_tail_days"),
                    "updated_at": stats_path.stat().st_mtime,
                    "stats_path": str(stats_path),
                }
            )
    items.sort(key=lambda x: float(x.get("updated_at") or 0), reverse=True)
    return {"root": str(r), "results": items}


@app.delete("/api/backtest/result/{result_id}")
def delete_backtest_result(
    result_id: str,
    root: str | None = Query(None),
    engine: str | None = Query(None),
    strategy: str | None = Query(None),
) -> dict[str, Any]:
    r = _resolve_root(root)
    safe_id = result_id.replace("/", "").replace("\\", "")
    base_dir = (r / "research" / "single_stock").resolve()
    target = (base_dir / safe_id).resolve()
    if not str(target).startswith(str(base_dir)):
        raise HTTPException(status_code=400, detail="Invalid result id.")
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Backtest result not found: {safe_id}")

    norm_engine = (engine or "").strip().lower()
    norm_strategy = (strategy or "").strip().lower()
    deleted_files: list[str] = []

    # Precise delete by card (engine + strategy). If not provided, fallback to deleting the whole folder.
    if norm_engine and norm_strategy:
        candidates: list[Path] = []
        if norm_engine == "akquant" and norm_strategy == "price_volume":
            candidates = [
                target / "akquant_price_volume_stats.json",
                target / "akquant_price_volume_equity.csv",
                target / "akquant_price_volume_trades.csv",
                target / "akquant_price_volume_metrics.csv",
                target / "akquant_price_volume_benchmark.csv",
                target / "akquant_price_volume_report.html",
            ]
        elif norm_engine == "vortex" and norm_strategy == "price_volume":
            candidates = [
                target / "price_volume_backtest_stats.json",
                target / "price_volume_backtest_timeseries.csv",
                target / "price_volume_backtest_timeseries.parquet",
            ]
        elif norm_engine == "vortex" and norm_strategy == "t_strategy":
            candidates = [
                target / "t_strategy_stats.json",
                target / "t_strategy_signals.csv",
                target / "t_strategy_signals.parquet",
            ]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported delete target: engine={norm_engine}, strategy={norm_strategy}")

        for p in candidates:
            if p.exists():
                p.unlink(missing_ok=True)
                deleted_files.append(str(p))
        if not deleted_files:
            raise HTTPException(status_code=404, detail=f"No files found for engine={norm_engine}, strategy={norm_strategy} under {safe_id}")
        return {
            "deleted": True,
            "mode": "card",
            "result_id": safe_id,
            "engine": norm_engine,
            "strategy": norm_strategy,
            "files": deleted_files,
        }

    # Backward compatibility: delete whole result directory when engine/strategy are not supplied.
    for child in list(target.iterdir()):
        if child.is_file():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            for nested in child.rglob("*"):
                if nested.is_file():
                    nested.unlink(missing_ok=True)
            for nested_dir in sorted([d for d in child.rglob("*") if d.is_dir()], reverse=True):
                nested_dir.rmdir()
            child.rmdir()
    target.rmdir()
    return {"deleted": True, "mode": "folder", "result_id": safe_id, "path": str(target)}


@app.get("/api/backtest/result/{result_id}")
def backtest_result_detail(result_id: str, root: str | None = Query(None)) -> dict[str, Any]:
    r = _resolve_root(root)
    safe_id = result_id.replace("/", "").replace("\\", "")
    result_dir = r / "research" / "single_stock" / safe_id
    stats_path = result_dir / "price_volume_backtest_stats.json"
    if not stats_path.exists():
        raise HTTPException(status_code=404, detail=f"Backtest result not found: {safe_id}")
    ts_path = result_dir / "price_volume_backtest_timeseries.csv"
    day_csv = result_dir / "kline_day.csv"
    week_csv = result_dir / "kline_week.csv"
    month_csv = result_dir / "kline_month.csv"
    intraday_csv = result_dir / "kline_intraday_recent.csv"
    day_rows = _load_kline_csv(day_csv, "date")
    return {
        "id": safe_id,
        "result_dir_id": safe_id,
        "stats": _load_backtest_stats(stats_path),
        "timeseries": _load_backtest_timeseries(ts_path),
        "trend": _trend_snapshot_from_day(day_rows),
        "paths": {
            "data_dir": str(result_dir),
            "stats": str(stats_path),
            "timeseries_csv": str(ts_path),
            "timeseries_parquet": str(result_dir / "price_volume_backtest_timeseries.parquet"),
            "kline_day_csv": str(day_csv),
            "kline_week_csv": str(week_csv),
            "kline_month_csv": str(month_csv),
            "kline_intraday_csv": str(intraday_csv),
        },
        "note": "Evaluation end date is usually earlier than data_end due to forward labels.",
    }


@app.get("/api/backtest/akquant-price-volume/{result_id}")
def akquant_price_volume_detail(result_id: str, root: str | None = Query(None)) -> dict[str, Any]:
    r = _resolve_root(root)
    safe_id = result_id.replace("/", "").replace("\\", "")
    result_dir = r / "research" / "single_stock" / safe_id
    stats_path = result_dir / "akquant_price_volume_stats.json"
    if not stats_path.exists():
        raise HTTPException(status_code=404, detail=f"AKQuant result not found: {safe_id}")
    equity_path = result_dir / "akquant_price_volume_equity.csv"
    day_csv = result_dir / "kline_day.csv"
    week_csv = result_dir / "kline_week.csv"
    month_csv = result_dir / "kline_month.csv"
    intraday_csv = result_dir / "kline_intraday_recent.csv"
    day_rows = _load_kline_csv(day_csv, "date")
    return {
        "id": safe_id,
        "result_dir_id": safe_id,
        "stats": _load_backtest_stats(stats_path),
        "timeseries": _load_backtest_timeseries(equity_path),
        "trend": _trend_snapshot_from_day(day_rows),
        "paths": {
            "data_dir": str(result_dir),
            "stats": str(stats_path),
            "equity_csv": str(equity_path),
            "trades_csv": str(result_dir / "akquant_price_volume_trades.csv"),
            "metrics_csv": str(result_dir / "akquant_price_volume_metrics.csv"),
            "report_html": str(result_dir / "akquant_price_volume_report.html"),
            "kline_day_csv": str(day_csv),
            "kline_week_csv": str(week_csv),
            "kline_month_csv": str(month_csv),
            "kline_intraday_csv": str(intraday_csv),
        },
        "note": "AKQuant report generated from Tushare data and AKQuant engine.",
    }


@app.get("/report/akquant/{result_id}")
def akquant_report_page(result_id: str, root: str | None = Query(None)) -> FileResponse:
    r = _resolve_root(root)
    safe_id = result_id.replace("/", "").replace("\\", "")
    report_path = r / "research" / "single_stock" / safe_id / "akquant_price_volume_report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"AKQuant report not found: {safe_id}")
    return FileResponse(
        str(report_path),
        media_type="text/html; charset=utf-8",
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/backtest/kline/{result_id}")
def backtest_kline_data(result_id: str, root: str | None = Query(None)) -> dict[str, Any]:
    r = _resolve_root(root)
    safe_id = result_id.replace("/", "").replace("\\", "")
    result_dir = r / "research" / "single_stock" / safe_id
    day_rows = _load_kline_csv(result_dir / "kline_day.csv", "date")
    week_rows = _load_kline_csv(result_dir / "kline_week.csv", "date")
    month_rows = _load_kline_csv(result_dir / "kline_month.csv", "date")
    intraday_rows = _load_kline_csv(result_dir / "kline_intraday_recent.csv", "datetime")
    if not day_rows:
        raise HTTPException(status_code=404, detail=f"Kline views not found: {safe_id}")
    return {
        "id": safe_id,
        "trend": _trend_snapshot_from_day(day_rows),
        "day": day_rows,
        "week": week_rows,
        "month": month_rows,
        "intraday": intraday_rows,
    }


@app.get("/report/kline/{result_id}")
def kline_report_page(result_id: str, root: str | None = Query(None)) -> HTMLResponse:
    r = _resolve_root(root)
    safe_id = result_id.replace("/", "").replace("\\", "")
    result_dir = r / "research" / "single_stock" / safe_id
    day_rows = _load_kline_csv(result_dir / "kline_day.csv", "date")
    week_rows = _load_kline_csv(result_dir / "kline_week.csv", "date")
    month_rows = _load_kline_csv(result_dir / "kline_month.csv", "date")
    intraday_rows = _load_kline_csv(result_dir / "kline_intraday_recent.csv", "datetime")
    if not day_rows:
        raise HTTPException(status_code=404, detail=f"Kline views not found: {safe_id}")
    trend = _trend_snapshot_from_day(day_rows)
    payload = {
        "day": day_rows[-400:],
        "week": week_rows[-200:],
        "month": month_rows[-120:],
        "intraday": intraday_rows[-600:],
    }
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_id} K线趋势</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ background: #0a1221; color: #d8e6ff; font-family: "Segoe UI", "PingFang SC", sans-serif; margin: 0; padding: 16px; }}
    .meta {{ margin: 0 0 16px 0; color: #9fb6da; }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); }}
    .card {{ background: #10203a; border: 1px solid #26426d; border-radius: 10px; padding: 10px; }}
    .title {{ font-size: 15px; font-weight: 600; margin: 0 0 8px 2px; }}
    .chart {{ width: 100%; height: 420px; }}
  </style>
</head>
<body>
  <h2>{safe_id} K线 + 成交量趋势面板</h2>
  <p class="meta">趋势判定: <strong>{trend.get("regime")}</strong> | score={trend.get("score")} | ma20_gap={trend.get("ma20_gap")} | ma60_gap={trend.get("ma60_gap")} | vol_ratio_5_20-1={trend.get("vol_ratio_5_20_minus_1")}</p>
  <div class="grid">
    <div class="card"><div class="title">分时K（最近）+ 成交量</div><div id="intraday" class="chart"></div></div>
    <div class="card"><div class="title">日K + 成交量</div><div id="day" class="chart"></div></div>
    <div class="card"><div class="title">周K + 成交量</div><div id="week" class="chart"></div></div>
    <div class="card"><div class="title">月K + 成交量</div><div id="month" class="chart"></div></div>
  </div>
  <script>
    const payload = {json.dumps(payload, ensure_ascii=False)};
    function draw(id, rows, timeCol) {{
      const node = document.getElementById(id);
      if (!rows || rows.length === 0) {{
        node.textContent = "暂无数据";
        return;
      }}
      const x = rows.map(r => r[timeCol]);
      const open = rows.map(r => Number(r.open));
      const high = rows.map(r => Number(r.high));
      const low = rows.map(r => Number(r.low));
      const close = rows.map(r => Number(r.close));
      const vol = rows.map(r => Number(r.volume || 0));
      const candle = {{ x, open, high, low, close, type: "candlestick", name: "K线", yaxis: "y", increasing: {{line: {{color: "#2ad47b"}}}}, decreasing: {{line: {{color: "#ff6a80"}}}} }};
      const bars = {{ x, y: vol, type: "bar", name: "成交量", yaxis: "y2", marker: {{color: "rgba(120,170,255,0.4)"}} }};
      Plotly.newPlot(node, [candle, bars], {{
        margin: {{l: 52, r: 52, t: 10, b: 30}},
        paper_bgcolor: "#10203a",
        plot_bgcolor: "#10203a",
        font: {{color: "#d8e6ff"}},
        xaxis: {{rangeslider: {{visible: false}}, gridcolor: "#1f3659"}},
        yaxis: {{domain: [0.30, 1], gridcolor: "#1f3659"}},
        yaxis2: {{domain: [0, 0.23], gridcolor: "#1f3659"}},
        legend: {{orientation: "h"}},
      }}, {{displaylogo: false, responsive: true}});
    }}
    draw("intraday", payload.intraday, "datetime");
    draw("day", payload.day, "date");
    draw("week", payload.week, "date");
    draw("month", payload.month, "date");
  </script>
</body>
</html>"""
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")


class BacktestRunBody(BaseModel):
    root: str | None = None
    code: str = "600396"
    train_start: str = "20180101"
    test_start: str
    test_end: str | None = None
    label_horizon: int = 5
    no_fetch: bool = False


@app.post("/api/backtest/single-stock-price-volume")
def run_single_stock_price_volume_backtest(body: BacktestRunBody) -> dict[str, Any]:
    r = _resolve_root(body.root)
    args = [
        "scripts/backtest_single_stock_price_volume.py",
        "--code",
        body.code,
        "--root",
        str(r),
        "--train-start",
        body.train_start,
        "--test-start",
        body.test_start,
        "--label-horizon",
        str(body.label_horizon),
    ]
    if body.test_end:
        args.extend(["--test-end", body.test_end])
    if body.no_fetch:
        args.append("--no-fetch")
    result = _run_script_subprocess(args)

    from vortex.research.single_stock import stock_pack_dir, to_ts_code

    result_id = stock_pack_dir(r, to_ts_code(body.code)).name
    detail = backtest_result_detail(result_id, root=str(r))
    return {"run": result, "fetch_summary": _parse_fetch_summary(result.get("stdout", "")), "result": detail}


@app.post("/api/backtest/single-stock-akquant-price-volume")
def run_single_stock_akquant_price_volume(body: BacktestRunBody) -> dict[str, Any]:
    r = _resolve_root(body.root)
    args = [
        "scripts/backtest_single_stock_akquant.py",
        "--code",
        body.code,
        "--root",
        str(r),
        "--train-start",
        body.train_start,
        "--test-start",
        body.test_start,
        "--label-horizon",
        str(body.label_horizon),
    ]
    if body.test_end:
        args.extend(["--test-end", body.test_end])
    if body.no_fetch:
        args.append("--no-fetch")
    result = _run_script_subprocess(args)

    from vortex.research.single_stock import stock_pack_dir, to_ts_code

    result_id = stock_pack_dir(r, to_ts_code(body.code)).name
    detail = akquant_price_volume_detail(result_id, root=str(r))
    return {"run": result, "fetch_summary": _parse_fetch_summary(result.get("stdout", "")), "result": detail}


@app.get("/api/backtest/t-strategy/{result_id}")
def t_strategy_detail(result_id: str, root: str | None = Query(None)) -> dict[str, Any]:
    r = _resolve_root(root)
    safe_id = result_id.replace("/", "").replace("\\", "")
    result_dir = r / "research" / "single_stock" / safe_id
    stats_path = result_dir / "t_strategy_stats.json"
    if not stats_path.exists():
        raise HTTPException(status_code=404, detail=f"T-strategy result not found: {safe_id}")
    signals_path = result_dir / "t_strategy_signals.csv"
    return {
        "id": safe_id,
        "result_dir_id": safe_id,
        "stats": _load_backtest_stats(stats_path),
        "signals": _load_backtest_timeseries(signals_path),
        "paths": {
            "data_dir": str(result_dir),
            "stats": str(stats_path),
            "signals_csv": str(signals_path),
            "signals_parquet": str(result_dir / "t_strategy_signals.parquet"),
        },
        "note": "Daily T-strategy opportunity report (rule-based).",
    }


@app.post("/api/backtest/single-stock-t-strategy")
def run_single_stock_t_strategy(body: BacktestRunBody) -> dict[str, Any]:
    r = _resolve_root(body.root)
    args = [
        "scripts/backtest_single_stock_t_strategy.py",
        "--code",
        body.code,
        "--root",
        str(r),
        "--train-start",
        body.train_start,
        "--test-start",
        body.test_start,
    ]
    if body.test_end:
        args.extend(["--test-end", body.test_end])
    if body.no_fetch:
        args.append("--no-fetch")
    result = _run_script_subprocess(args)

    from vortex.research.single_stock import stock_pack_dir, to_ts_code

    result_id = stock_pack_dir(r, to_ts_code(body.code)).name
    detail = t_strategy_detail(result_id, root=str(r))
    return {"run": result, "fetch_summary": _parse_fetch_summary(result.get("stdout", "")), "result": detail}


class DataActionBody(BaseModel):
    root: str | None = Field(None, description="Workspace root")
    profile: str | None = None
    datasets: str | None = Field(None, description="Comma-separated datasets")
    foreground: bool = Field(False, description="Run in foreground process")
    dry_run: bool = False
    start: str | None = None
    end: str | None = None
    as_of: str | None = None
    task_id: str | None = Field(None, description="Task id for logs/cancel")


@app.post("/api/data/{action}")
def data_action(action: str, body: DataActionBody) -> dict[str, Any]:
    from vortex.cli import (
        _load_workspace_env,
        _parse_dataset_override,
        _resolve_data_profile_name,
        _submit_data_background_task,
    )

    allowed = {"bootstrap", "update", "publish", "backfill", "repair", "logs", "cancel"}
    if action not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported action: {action}")

    r = _resolve_root(body.root)
    _load_workspace_env(r)
    profile_name = _resolve_data_profile_name(body.profile)
    ds_list = _parse_dataset_override(body.datasets)
    ds_csv = body.datasets.strip() if body.datasets else None

    if action == "logs":
        sp_args = ["logs", "--root", str(r), "--format", "json", "--lines", "120"]
        if body.profile:
            sp_args.extend(["--profile", body.profile])
        if body.task_id:
            sp_args.extend(["--task-id", body.task_id])
        return _run_vortex_data_subprocess(sp_args)

    if action == "cancel":
        sp_args = ["cancel", "--root", str(r), "--format", "json"]
        if body.profile:
            sp_args.extend(["--profile", body.profile])
        if body.task_id:
            sp_args.extend(["--task-id", body.task_id])
        return _run_vortex_data_subprocess(sp_args)

    if body.foreground:
        sp_args: list[str] = [action, "--root", str(r), "--format", "json"]
        if body.profile:
            sp_args.extend(["--profile", body.profile])
        if ds_csv:
            sp_args.extend(["--datasets", ds_csv])
        if action in {"bootstrap", "update"}:
            if body.dry_run:
                sp_args.append("--dry-run")
            sp_args.append("--foreground")
        elif action in {"backfill", "repair"}:
            if not body.start or not body.end:
                raise HTTPException(status_code=400, detail="backfill/repair requires start and end")
            sp_args.extend(["--start", body.start, "--end", body.end, "--foreground"])
        elif action == "publish":
            sp_args.append("--foreground")
            if body.as_of:
                sp_args.extend(["--as-of", body.as_of])
        return _run_vortex_data_subprocess(sp_args)

    if action in {"backfill", "repair"}:
        if not body.start or not body.end:
            raise HTTPException(status_code=400, detail="backfill/repair requires start and end")
        payload = _submit_data_background_task(
            root=r,
            profile_name=profile_name,
            action=action,
            fmt="json",
            datasets=ds_list or None,
            start=body.start,
            end=body.end,
            as_of=None,
            verbose=False,
        )
        return {"mode": "background", "result": payload}

    if action == "publish":
        as_of = body.as_of.replace("-", "")[:8] if body.as_of else None
        payload = _submit_data_background_task(
            root=r,
            profile_name=profile_name,
            action="publish",
            fmt="json",
            datasets=None,
            start=None,
            end=None,
            as_of=as_of,
            verbose=False,
        )
        return {"mode": "background", "result": payload}

    payload = _submit_data_background_task(
        root=r,
        profile_name=profile_name,
        action=action,
        fmt="json",
        datasets=ds_list or None,
        verbose=False,
    )
    return {"mode": "background", "result": payload}


class InitBody(BaseModel):
    root: str | None = None
    non_interactive: bool = True


@app.post("/api/init")
def init_workspace(body: InitBody) -> dict[str, Any]:
    r = _resolve_root(body.root)
    args = ["init", "--root", str(r)]
    if body.non_interactive:
        args.append("--non-interactive")
    return _run_vortex_init_subprocess(args)


if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
