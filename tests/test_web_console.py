"""Web 控制台最小冒烟测试（需 optional web 依赖）。"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi", reason="需要安装 web 可选依赖: uv sync --extra web")

from fastapi.testclient import TestClient

from web.server.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_dataset_catalog(client: TestClient) -> None:
    r = client.get("/api/data/dataset-catalog")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data.get("datasets"), list)
    assert len(data["datasets"]) >= 1
    presets = data.get("presets") or {}
    assert "minimal" in presets
    assert isinstance(presets["minimal"], list)


def test_backtest_results_and_detail(client: TestClient, tmp_path) -> None:
    result_dir = tmp_path / "research" / "single_stock" / "600396_SH"
    result_dir.mkdir(parents=True)
    (result_dir / "price_volume_backtest_stats.json").write_text(
        json.dumps(
            {
                "code": "600396.SH",
                "model": "ridge",
                "test_start": "20250423",
                "test_end": "20260415",
                "strategy_total_return": 0.35,
                "benchmark_total_return": 1.99,
                "strategy_sharpe": 1.78,
                "strategy_max_drawdown": -0.11,
                "prediction_ic": 0.12,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "price_volume_backtest_timeseries.csv").write_text(
        "date,equity,benchmark_equity,position,pred\n"
        "2025-04-23,1.0,1.0,0.4,0.01\n"
        "2025-04-24,1.02,1.03,0.5,0.02\n",
        encoding="utf-8",
    )

    r = client.get("/api/backtest/results", params={"root": str(tmp_path)})
    assert r.status_code == 200
    rows = r.json()["results"]
    assert rows[0]["id"] == "600396_SH:vortex:price_volume"
    assert rows[0]["result_dir_id"] == "600396_SH"
    assert rows[0]["code"] == "600396.SH"

    detail = client.get("/api/backtest/result/600396_SH", params={"root": str(tmp_path)})
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["stats"]["strategy_total_return"] == 0.35
    assert payload["timeseries"][1]["equity"] == 1.02


def test_t_strategy_detail(client: TestClient, tmp_path) -> None:
    result_dir = tmp_path / "research" / "single_stock" / "600396_SH"
    result_dir.mkdir(parents=True)
    (result_dir / "t_strategy_stats.json").write_text(
        json.dumps(
            {
                "code": "600396.SH",
                "model": "ridge",
                "test_start": "20250423",
                "test_end": "20260415",
                "data_end": "20260423",
                "action_days": 10,
                "hit_rate": 0.6,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "t_strategy_signals.csv").write_text(
        "date,action,sell_score,buy_score,action_hit\n"
        "2025-04-23,sell_high,0.03,0.01,1\n",
        encoding="utf-8",
    )

    detail = client.get("/api/backtest/t-strategy/600396_SH", params={"root": str(tmp_path)})
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["stats"]["hit_rate"] == 0.6
    assert payload["signals"][0]["action"] == "sell_high"


def test_akquant_price_volume_detail(client: TestClient, tmp_path) -> None:
    result_dir = tmp_path / "research" / "single_stock" / "600396_SH"
    result_dir.mkdir(parents=True)
    (result_dir / "akquant_price_volume_stats.json").write_text(
        json.dumps(
            {
                "code": "600396.SH",
                "engine": "akquant",
                "strategy": "price_volume",
                "test_start": "20250423",
                "test_end": "20260415",
                "strategy_total_return": 0.38,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "akquant_price_volume_equity.csv").write_text(
        "date,equity\n2025-04-23,100000\n2025-04-24,101000\n",
        encoding="utf-8",
    )

    detail = client.get("/api/backtest/akquant-price-volume/600396_SH", params={"root": str(tmp_path)})
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["stats"]["engine"] == "akquant"
    assert payload["result_dir_id"] == "600396_SH"
    assert payload["timeseries"][1]["equity"] == 101000


def test_akquant_report_page(client: TestClient, tmp_path) -> None:
    result_dir = tmp_path / "research" / "single_stock" / "600396_SH"
    result_dir.mkdir(parents=True)
    (result_dir / "akquant_price_volume_report.html").write_text(
        "<html><body><h1>AKQuant Report</h1></body></html>",
        encoding="utf-8",
    )

    detail = client.get("/report/akquant/600396_SH", params={"root": str(tmp_path)})
    assert detail.status_code == 200
    assert "text/html" in detail.headers["content-type"]
    assert "AKQuant Report" in detail.text
