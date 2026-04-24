const LS_KEY = "vortex_web_root";
const LS_KEY_PROFILE = "vortex_web_data_profile";

function $(id) {
  return document.getElementById(id);
}

function rootValue() {
  const el = $("root");
  const v = el ? el.value.trim() : "";
  return v || null;
}

function dataProfileValue() {
  const el = $("data-profile");
  const v = el ? el.value.trim() : "";
  return v || null;
}

function setOut(obj) {
  const el = $("out");
  if (!el) return;
  el.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

function fmtPct(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(2)}%`;
}

function fmtNum(v, digits = 2) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(digits);
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = String(s ?? "");
  return d.innerHTML;
}

function btInput(id) {
  const el = $(id);
  return el ? el.value.trim() : "";
}

function selectedStrategyType() {
  return btInput("bt-strategy") || "price_volume";
}

function selectedBacktestEngine() {
  return btInput("bt-engine") || "vortex";
}

function syncBacktestEngineByStrategy() {
  const strategy = selectedStrategyType();
  const engineEl = $("bt-engine");
  const status = $("bt-status");
  if (!engineEl) return;
  const akOpt = Array.from(engineEl.options || []).find((o) => String(o.value) === "akquant");
  if (strategy === "t_strategy") {
    if (akOpt) akOpt.disabled = true;
    if (engineEl.value === "akquant") {
      engineEl.value = "vortex";
      if (status) status.textContent = "做T策略当前仅支持 Vortex，引擎已自动切换。";
    }
  } else if (akOpt) {
    akOpt.disabled = false;
  }
}

async function apiGet(path, includeRoot = true) {
  const r = rootValue();
  const url = includeRoot && r
    ? `${path}${path.includes("?") ? "&" : "?"}root=${encodeURIComponent(r)}`
    : path;
  const res = await fetch(url);
  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = text;
  }
  if (!res.ok) {
    throw new Error(typeof data === "object" ? JSON.stringify(data, null, 2) : text);
  }
  return data;
}

async function apiPost(path, body) {
  const payload = { ...(body || {}) };
  const r = rootValue();
  if (r) payload.root = r;
  if (path.startsWith("/api/data/")) {
    const p = dataProfileValue();
    if (p) payload.profile = p;
  }
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = text;
  }
  if (!res.ok) {
    throw new Error(typeof data === "object" ? JSON.stringify(data, null, 2) : text);
  }
  return data;
}

function makeMetric(label, value) {
  const box = document.createElement("div");
  box.className = "bt-metric";
  const l = document.createElement("div");
  l.className = "bt-metric-label";
  l.textContent = label;
  const v = document.createElement("div");
  v.className = "bt-metric-value";
  v.textContent = value;
  box.appendChild(l);
  box.appendChild(v);
  return box;
}

function renderFetchSummary(summary) {
  const box = $("bt-fetch-summary");
  if (!box) return;
  if (!summary || !Array.isArray(summary.tables)) {
    box.classList.add("hidden");
    return;
  }
  const rows = summary.tables
    .map((x) => `${escapeHtml(x.name)}: <strong>${escapeHtml(x.rows)}</strong> 行`)
    .join(" | ");
  box.innerHTML = `拉取摘要：<strong>${escapeHtml(summary.code || "")}</strong> 区间 <strong>${escapeHtml(summary.fetch_start || "-")}</strong> ~ <strong>${escapeHtml(summary.fetch_end || "-")}</strong><br>${rows}`;
  box.classList.remove("hidden");
}

function renderBacktestDetail(data) {
  const detail = $("bt-detail");
  const metrics = $("bt-metrics");
  const paths = $("bt-paths");
  const note = $("bt-note");
  const reportRow = $("bt-report-row");
  const reportBtn = $("btn-open-akquant-report");
  const reportLink = $("bt-report-link");
  const klineBtn = $("btn-open-kline-report");
  const klineLink = $("bt-kline-link");
  if (!detail || !metrics || !paths) return;

  const stats = data.stats || {};
  metrics.innerHTML = "";
  [
    ["回测引擎", String(stats.engine || "vortex")],
    ["策略收益", fmtPct(stats.strategy_total_return)],
    ["买入持有", fmtPct(stats.benchmark_total_return)],
    ["超额", fmtPct(stats.excess_total_return)],
    ["Sharpe", fmtNum(stats.strategy_sharpe, 2)],
    ["最大回撤", fmtPct(stats.strategy_max_drawdown)],
    ["RankIC", fmtNum(stats.prediction_ic, 4)],
    ["平均仓位", fmtPct(stats.avg_position)],
    ["交易日", fmtNum(stats.test_days, 0)],
    ["数据最新日", String(stats.data_end || "-")],
    ["标签截断", `${fmtNum(stats.dropped_tail_days, 0)} 天`],
  ].forEach(([label, value]) => metrics.appendChild(makeMetric(label, value)));

  if (data.trend && data.trend.regime) {
    metrics.appendChild(
      makeMetric("趋势判定", `${String(data.trend.regime)} (${fmtNum(data.trend.score, 4)})`),
    );
  }

  const p = data.paths || {};
  const csvPath = p.timeseries_csv || p.equity_csv || "";
  const htmlPath = p.report_html || "";
  const kd = p.kline_day_csv || "";
  const kw = p.kline_week_csv || "";
  const km = p.kline_month_csv || "";
  const ki = p.kline_intraday_csv || "";
  paths.textContent = `data: ${p.data_dir || ""} | stats: ${p.stats || ""} | csv: ${csvPath}${htmlPath ? ` | html: ${htmlPath}` : ""} | day: ${kd} | week: ${kw} | month: ${km} | intraday: ${ki}`;

  if (reportRow && reportBtn) {
    const hasAkReport = Boolean(p.report_html);
    const hasKlineReport = Boolean(p.kline_day_csv);
    reportRow.classList.toggle("hidden", !(hasAkReport || hasKlineReport));

    if (hasAkReport) {
      const targetId = String(data.result_dir_id || data.id || "");
      const params = new URLSearchParams();
      const root = rootValue();
      if (root) params.set("root", root);
      params.set("_ts", `${Date.now()}`);
      const reportUrl = `/report/akquant/${encodeURIComponent(targetId)}?${params.toString()}`;
      reportBtn.onclick = () => window.location.assign(reportUrl);
      if (reportLink) reportLink.href = reportUrl;
    } else {
      reportBtn.onclick = null;
      if (reportLink) reportLink.href = "#";
    }

    if (klineBtn) {
      if (hasKlineReport) {
        const targetId = String(data.result_dir_id || data.id || "");
        const params = new URLSearchParams();
        const root = rootValue();
        if (root) params.set("root", root);
        params.set("_ts", `${Date.now()}`);
        const klineUrl = `/report/kline/${encodeURIComponent(targetId)}?${params.toString()}`;
        klineBtn.onclick = () => window.location.assign(klineUrl);
        if (klineLink) klineLink.href = klineUrl;
      } else {
        klineBtn.onclick = null;
        if (klineLink) klineLink.href = "#";
      }
    }
  }

  if (note) {
    note.textContent = data.note
      ? `${data.note} 当前：最新数据日 ${stats.data_end || "-"}，有效评估截止 ${stats.test_end || "-"}，label_horizon=${stats.label_horizon || "-"}。`
      : "";
  }

  detail.classList.remove("hidden");
}

function renderTStrategyDetail(data) {
  const detail = $("t-detail");
  const metrics = $("t-metrics");
  const signals = $("t-signals");
  const paths = $("t-paths");
  if (!detail || !metrics || !signals || !paths) return;

  const stats = data.stats || {};
  metrics.innerHTML = "";
  [
    ["信号天数", fmtNum(stats.action_days, 0)],
    ["信号率", fmtPct(stats.signal_rate)],
    ["总命中率", fmtPct(stats.hit_rate)],
    ["高抛命中", fmtPct(stats.sell_hit_rate)],
    ["低吸命中", fmtPct(stats.buy_hit_rate)],
    ["高抛均值", fmtPct(stats.avg_sell_opportunity)],
    ["低吸均值", fmtPct(stats.avg_buy_opportunity)],
    ["数据最新日", String(stats.data_end || "-")],
  ].forEach(([label, value]) => metrics.appendChild(makeMetric(label, value)));

  const rows = (data.signals || []).filter((x) => x.action && x.action !== "hold").slice(-30).reverse();
  signals.innerHTML = '<div class="t-signal-row t-signal-head"><span>日期</span><span>动作</span><span>高抛分</span><span>低吸分</span><span>命中</span></div>';
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "t-signal-row";
    item.innerHTML = `<span>${escapeHtml(row.date || "")}</span><span>${escapeHtml(row.action || "")}</span><span>${escapeHtml(fmtPct(row.sell_score))}</span><span>${escapeHtml(fmtPct(row.buy_score))}</span><span>${row.action_hit === 1 || row.action_hit === true ? "yes" : "no"}</span>`;
    signals.appendChild(item);
  });
  if (rows.length === 0) {
    signals.innerHTML += '<div class="t-signal-row"><span>暂无触发信号</span></div>';
  }

  const p = data.paths || {};
  paths.textContent = `${data.note || ""} data: ${p.data_dir || ""} | stats: ${p.stats || ""}`;
  detail.classList.remove("hidden");
}

async function loadBacktestDetail(id, engine = "vortex", strategy = "price_volume", activeKey = "") {
  let path = `/api/backtest/result/${encodeURIComponent(id)}`;
  if (strategy === "t_strategy") {
    path = `/api/backtest/t-strategy/${encodeURIComponent(id)}`;
  } else if (engine === "akquant") {
    path = `/api/backtest/akquant-price-volume/${encodeURIComponent(id)}`;
  }
  const data = await apiGet(path);
  if (strategy === "t_strategy") {
    renderTStrategyDetail(data);
  } else {
    renderBacktestDetail(data);
  }
  const status = $("bt-status");
  if (status) {
    status.textContent = strategy === "t_strategy"
      ? "已加载做T结果"
      : (engine === "akquant" ? "已加载 AKQuant 结果" : "已加载回测结果");
  }

  document.querySelectorAll(".bt-result-card").forEach((el) => {
    el.classList.toggle("active", el.getAttribute("data-key") === activeKey);
  });
}

async function refreshBacktestResults() {
  const status = $("bt-status");
  const list = $("bt-results");
  if (!status || !list) return;
  status.textContent = "加载中...";
  try {
    const data = await apiGet("/api/backtest/results");
    const rows = data.results || [];
    list.innerHTML = "";

    if (rows.length === 0) {
      list.innerHTML = '<p class="empty-hint">还没有回测结果，运行一次后这里会自动出现。</p>';
      const detail = $("bt-detail");
      if (detail) detail.classList.add("hidden");
      status.textContent = "暂无结果";
      return;
    }

    let preferredIndex = -1;
    const wantEngine = selectedBacktestEngine();
    const wantStrategy = selectedStrategyType();
    rows.forEach((row, idx) => {
      if (preferredIndex < 0) {
        const eng = String(row.engine || "vortex");
        const strat = String(row.strategy || "price_volume");
        if (eng === wantEngine && strat === wantStrategy) {
          preferredIndex = idx;
        }
      }
    });
    if (preferredIndex < 0) preferredIndex = 0;

    rows.forEach((row, idx) => {
      const item = document.createElement("div");
      item.className = "bt-result-item";
      const card = document.createElement("button");
      card.type = "button";
      card.className = "bt-result-card";
      card.setAttribute("data-id", row.result_dir_id || row.id);
      card.setAttribute("data-engine", row.engine || "vortex");
      card.setAttribute("data-strategy", row.strategy || "price_volume");
      const activeKey = `${String(row.result_dir_id || row.id)}|${String(row.engine || "vortex")}|${String(row.strategy || "price_volume")}`;
      card.setAttribute("data-key", activeKey);
      card.innerHTML = `
        <div class="bt-card-title">${escapeHtml(row.code || row.id)}</div>
        <div class="bt-card-meta">
          ${escapeHtml(row.engine || "vortex")} / ${escapeHtml(row.strategy || "price_volume")}<br />
          ${escapeHtml(row.test_start || "")} - ${escapeHtml(row.test_end || "")}<br />
          策略 ${escapeHtml(fmtPct(row.strategy_total_return))} | 持有 ${escapeHtml(fmtPct(row.benchmark_total_return))}<br />
          Sharpe ${escapeHtml(fmtNum(row.strategy_sharpe, 2))} | IC ${escapeHtml(fmtNum(row.prediction_ic, 4))}
        </div>`;
      card.addEventListener("click", () => {
        void loadBacktestDetail(
          String(row.result_dir_id || row.id),
          String(row.engine || "vortex"),
          String(row.strategy || "price_volume"),
          activeKey,
        );
      });
      item.appendChild(card);

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "btn-sm";
      delBtn.textContent = "删除";
      delBtn.style.marginLeft = "8px";
      delBtn.addEventListener("click", async () => {
        const rid = String(row.result_dir_id || row.id);
        const eng = String(row.engine || "vortex");
        const strat = String(row.strategy || "price_volume");
        if (!window.confirm(`确认删除该回测卡片？\\n${rid}\\n${eng} / ${strat}`)) return;
        try {
          const params = new URLSearchParams();
          const rv = rootValue();
          if (rv) params.set("root", rv);
          params.set("engine", eng);
          params.set("strategy", strat);
          await fetch(`/api/backtest/result/${encodeURIComponent(rid)}?${params.toString()}`, {
            method: "DELETE",
          }).then(async (res) => {
            if (!res.ok) {
              const text = await res.text();
              throw new Error(text);
            }
          });
          await refreshBacktestResults();
        } catch (e) {
          setOut(String(e.message || e));
        }
      });
      item.appendChild(delBtn);
      list.appendChild(item);
      if (idx === preferredIndex) {
        void loadBacktestDetail(
          String(row.result_dir_id || row.id),
          String(row.engine || "vortex"),
          String(row.strategy || "price_volume"),
          activeKey,
        );
      }
    });

    status.textContent = `共 ${rows.length} 条结果`;
  } catch (e) {
    status.textContent = "加载失败";
    setOut(String(e.message || e));
  }
}

function wireSimpleAction(buttonId, endpoint, bodyFactory = () => ({})) {
  const btn = $(buttonId);
  if (!btn) return;
  btn.onclick = async () => {
    try {
      const data = await apiPost(endpoint, bodyFactory());
      setOut(data);
    } catch (e) {
      setOut(String(e.message || e));
    }
  };
}

function wireDataAction(buttonId, action, bodyFactory = () => ({})) {
  const btn = $(buttonId);
  if (!btn) return;
  btn.onclick = async () => {
    try {
      const data = await apiPost(`/api/data/${action}`, bodyFactory());
      setOut(data);
    } catch (e) {
      setOut(String(e.message || e));
    }
  };
}

function ymdFromDateInput(id) {
  const el = $(id);
  if (!el || !el.value) return "";
  const [y, m, d] = el.value.split("-");
  return `${y}${m}${d}`;
}

window.addEventListener("DOMContentLoaded", () => {
  const savedRoot = localStorage.getItem(LS_KEY);
  if (savedRoot && $("root")) $("root").value = savedRoot;
  const savedProfile = localStorage.getItem(LS_KEY_PROFILE);
  if (savedProfile && $("data-profile")) $("data-profile").value = savedProfile;

  if ($("root")) {
    $("root").addEventListener("change", () => {
      localStorage.setItem(LS_KEY, $("root").value);
      void refreshBacktestResults();
    });
  }
  if ($("data-profile")) {
    $("data-profile").addEventListener("change", () => {
      localStorage.setItem(LS_KEY_PROFILE, $("data-profile").value);
    });
  }
  if ($("bt-strategy")) {
    $("bt-strategy").addEventListener("change", () => {
      syncBacktestEngineByStrategy();
      void refreshBacktestResults();
    });
  }
  syncBacktestEngineByStrategy();

  wireSimpleAction("btn-summary", "/api/workspace/summary");
  wireSimpleAction("btn-profile", "/api/profile/resolve?name=default&type=data", undefined);
  wireSimpleAction("btn-explain", "/api/profile/explain?name=default&type=data", undefined);

  const btnStatus = $("btn-status");
  if (btnStatus) {
    btnStatus.onclick = async () => {
      try {
        setOut(await apiGet("/api/data/status"));
      } catch (e) {
        setOut(String(e.message || e));
      }
    };
  }

  wireDataAction("btn-bootstrap", "bootstrap", () => ({ foreground: false }));
  wireDataAction("btn-update", "update", () => ({ foreground: false }));
  wireDataAction("btn-bootstrap-dry", "bootstrap", () => ({ foreground: true, dry_run: true }));
  wireDataAction("btn-publish", "publish", () => ({ foreground: !!$("foreground-publish")?.checked }));
  wireDataAction("btn-logs", "logs", () => ({ task_id: btInput("task-id") || null }));
  wireDataAction("btn-cancel", "cancel", () => ({ task_id: btInput("task-id") || null }));
  wireSimpleAction("btn-init", "/api/init", () => ({ non_interactive: true }));

  const btnBackfill = $("btn-backfill");
  if (btnBackfill) {
    btnBackfill.onclick = async () => {
      try {
        const start = ymdFromDateInput("start-date");
        const end = ymdFromDateInput("end-date");
        if (!start || !end) throw new Error("请先选择开始和结束日期");
        setOut(await apiPost("/api/data/backfill", { foreground: false, start, end }));
      } catch (e) {
        setOut(String(e.message || e));
      }
    };
  }

  const btnRepair = $("btn-repair");
  if (btnRepair) {
    btnRepair.onclick = async () => {
      try {
        const start = ymdFromDateInput("start-date");
        const end = ymdFromDateInput("end-date");
        if (!start || !end) throw new Error("请先选择开始和结束日期");
        setOut(await apiPost("/api/data/repair", { foreground: false, start, end }));
      } catch (e) {
        setOut(String(e.message || e));
      }
    };
  }

  const btRefresh = $("btn-bt-refresh");
  if (btRefresh) {
    btRefresh.onclick = async () => {
      await refreshBacktestResults();
    };
  }

  const btRun = $("btn-bt-run");
  if (btRun) {
    btRun.onclick = async () => {
      const status = $("bt-status");
      try {
        const strategyType = selectedStrategyType();
        syncBacktestEngineByStrategy();
        const engine = selectedBacktestEngine();
        const isTStrategy = strategyType === "t_strategy";
        const useAkquant = engine === "akquant" && !isTStrategy;
        if (status) status.textContent = "回测运行中...";

        const payload = {
          code: btInput("bt-code") || "600396",
          train_start: btInput("bt-train-start") || "20180101",
          test_start: btInput("bt-test-start") || "20250423",
          no_fetch: !!$("bt-no-fetch")?.checked,
        };
        if (!isTStrategy) payload.label_horizon = 5;

        const endpoint = useAkquant
          ? "/api/backtest/single-stock-akquant-price-volume"
          : isTStrategy
            ? "/api/backtest/single-stock-t-strategy"
            : "/api/backtest/single-stock-price-volume";

        const data = await apiPost(endpoint, payload);
        renderFetchSummary(data.fetch_summary);
        if (isTStrategy) {
          renderTStrategyDetail(data.result);
        } else {
          renderBacktestDetail(data.result);
        }
        await refreshBacktestResults();
        setOut({ stdout: data.run?.stdout, stderr: data.run?.stderr, stats: data.result?.stats || null });
        if (status) status.textContent = "策略回测完成";
      } catch (e) {
        if (status) status.textContent = "回测失败";
        setOut(String(e.message || e));
      }
    };
  }

  void refreshBacktestResults();
});
