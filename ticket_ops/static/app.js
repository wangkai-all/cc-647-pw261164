/* Ticket-ops occurrence estimation page.
   Charts are Chart.js 4 (vendored locally).  Labels come from the API and are
   inserted with textContent only. */

(function () {
  "use strict";

  const state = {
    meta: null,
    data: null,
    channel: "__all__",
    window: 28,
    smooth: 0.3,
    selectedDay: null,
    replay: null,
    loading: false,
    requestSeq: 0,
  };

  const charts = {};
  const SERIES = {
    observed: "#2a78d6",
    estimate: "#eb6834",
    predicted: "#1baf7a",
  };
  const ORDINAL = ["#86b6ef", "#5598e7", "#3987e5", "#1c5cab", "#0d366b"];
  const STATUS = { critical: "#d03b3b", serious: "#ec835a", warning: "#fab219" };

  function cssVar(name) {
    return getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(name).trim();
  }

  function fmtInt(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return Math.round(value).toLocaleString("zh-CN");
  }

  function fmtNum(value, digits) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return value.toLocaleString("zh-CN", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function fmtCompact(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    const abs = Math.abs(value);
    if (abs >= 10000) return (value / 10000).toFixed(1) + "万";
    if (abs >= 1000) return (value / 1000).toFixed(1) + "k";
    return String(Math.round(value));
  }

  function fmtPct(value, digits) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return (value * 100).toFixed(digits === undefined ? 1 : digits) + "%";
  }

  function fmtSmooth(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return value.toFixed(2);
  }

  function shortDay(iso) {
    const [, m, d] = iso.split("-");
    return `${Number(m)}/${Number(d)}`;
  }

  function weekdayName(iso) {
    const names = ["日", "一", "二", "三", "四", "五", "六"];
    return "周" + names[new Date(iso + "T00:00:00").getDay()];
  }

  function el(tag, props, children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [key, value] of Object.entries(props)) {
        if (key === "text") node.textContent = value;
        else if (key === "className") node.className = value;
        else if (key === "dataset") Object.assign(node.dataset, value);
        else node.setAttribute(key, value);
      }
    }
    if (children) for (const child of children) if (child) node.appendChild(child);
    return node;
  }

  function currentChannelData() {
    if (!state.data) return null;
    if (state.channel === "__all__") {
      return { aggregate: true, name: "全部渠道", days: state.data.aggregate.days, observed_reports: state.data.aggregate.observed_reports, predicted_reports: state.data.aggregate.predicted_reports, warnings: collectWarnings(), profile: null, status: "ok" };
    }
    return state.data.channels[state.channel];
  }

  function collectWarnings() {
    const out = [];
    for (const [name, ch] of Object.entries(state.data.channels)) {
      for (const w of ch.warnings || []) out.push(Object.assign({ channel: name }, w));
    }
    return out;
  }

  // ---------------------------------------------------------------- fetch
  async function loadMeta() {
    const res = await fetch("/api/meta");
    state.meta = await res.json();
    document.getElementById("cutoff-date").textContent = state.meta.cutoff;
    document.getElementById("cutoff-note").textContent =
      `数据自 ${state.meta.start}，共 ${Object.keys(state.meta.channels).length} 个渠道`;
    const select = document.getElementById("channel-select");
    for (const [name, info] of Object.entries(state.meta.channels)) {
      const option = el("option", { value: name, text: `${name}（${fmtCompact(info.tickets)} 单）` });
      select.appendChild(option);
    }
    document.getElementById("smooth-range").max = String(state.meta.smooth_max);
    document.getElementById("smooth-range").value = String(state.meta.defaults.smooth);
    document.getElementById("window-range").value = String(state.meta.defaults.window);
    state.window = state.meta.defaults.window;
    state.smooth = state.meta.defaults.smooth;
    document.getElementById("window-value").textContent = String(state.window);
    document.getElementById("smooth-value").textContent = fmtSmooth(state.smooth);
  }

  async function loadEstimates() {
    if (state.loading) return;
    state.loading = true;
    const seq = ++state.requestSeq;
    document.getElementById("root").classList.add("loading");
    try {
      const url = `/api/estimates?window=${state.window}&smooth=${state.smooth}&draws=200`;
      const res = await fetch(url);
      const payload = await res.json();
      if (seq !== state.requestSeq) return;
      if (payload.error) throw new Error(payload.error);
      state.data = payload;
      document.getElementById("foot-meta").textContent =
        `截止 ${payload.cutoff} · 拟合窗口 ${payload.params.window} 天 · 平滑 ${payload.params.smooth} · 汇总口径为各渠道独立拟合后相加`;
      renderAll();
    } catch (err) {
      showError(err.message);
    } finally {
      if (seq === state.requestSeq) {
        state.loading = false;
        document.getElementById("root").classList.remove("loading");
      }
    }
  }

  async function loadReplay() {
    const btn = document.getElementById("replay-button");
    const status = document.getElementById("replay-status");
    btn.disabled = true;
    status.textContent = "回放中…";
    try {
      const res = await fetch(`/api/replays?window=${state.window}&smooth=${state.smooth}`);
      const payload = await res.json();
      if (payload.error) throw new Error(payload.error);
      state.replay = payload;
      renderReplay();
      status.textContent = `已完成 ${payload.replays.length} 个回放截止日`;
    } catch (err) {
      status.textContent = "";
      showError(err.message);
    } finally {
      btn.disabled = false;
    }
  }

  function showError(message) {
    let banner = document.querySelector(".error-banner");
    if (!banner) {
      banner = el("div", { className: "error-banner", role: "alert" });
      document.querySelector(".controls").after(banner);
    }
    banner.textContent = "加载失败：" + message;
  }

  // ---------------------------------------------------------------- tiles
  function renderTiles() {
    const host = document.getElementById("tiles");
    host.replaceChildren();
    if (!state.data) return;
    const ch = currentChannelData();
    const days = (ch && ch.days) || state.data.aggregate.days;
    if (!days.length) return;
    const last = days[days.length - 1];
    const last7 = days.slice(-7);
    const obs7 = last7.reduce((s, d) => s + d.observed, 0);
    const est7 = last7.reduce((s, d) => s + d.estimated, 0);
    const unreportedTotal = days.reduce((s, d) => s + d.unreported, 0);
    const firstMatured = days.find((d) => d.matured && !d.lower_bound);

    const tiles = [
      {
        label: "截止日估计发生量",
        value: fmtInt(last.estimated),
        delta: `${shortDay(last.day)} ${weekdayName(last.day)} · 已上报 ${fmtInt(last.observed)}`,
      },
      {
        label: "近 7 日估计总量",
        value: fmtInt(est7),
        delta: `其中待报 ${fmtInt(est7 - obs7)}（${fmtPct(est7 ? (est7 - obs7) / est7 : 0)}）`,
      },
      {
        label: "窗口内待补总量",
        value: fmtInt(unreportedTotal),
        delta: `${state.channel === "__all__" ? "全部渠道" : state.channel} · 截至 ${shortDay(state.data.cutoff)}`,
      },
      {
        label: "已成熟日占比",
        value: fmtPct(firstMatured ? days.filter((d) => d.matured).length / days.length : 0, 0),
        delta: firstMatured ? `最早未成熟日 ${shortDay(days.find((d) => !d.matured).day)}` : "窗口内全部已成熟",
      },
    ];
    for (const t of tiles) {
      host.appendChild(
        el("div", { className: "tile" }, [
          el("span", { className: "tile-label", text: t.label }),
          el("span", { className: "tile-value", text: t.value }),
          el("span", { className: "tile-delta", text: t.delta }),
        ])
      );
    }
  }

  // ---------------------------------------------------------------- main chart
  function buildReportSeries() {
    const ch = currentChannelData();
    if (!ch) return { labels: [], observed: [], predicted: [] };
    const cutoff = state.data.cutoff;
    const obs = ch.observed_reports || {};
    const pred = ch.predicted_reports || {};
    const labels = [];
    const start = Object.keys(pred).length ? Object.keys(pred).sort()[0] : Object.keys(obs).sort()[0];
    let cursor = new Date(start + "T00:00:00");
    // The back-projected curve is only compared with real reports up to the
    // cutoff.  Beyond it the forecast covers in-window occurrence days only,
    // so it decays to zero by construction and would swamp the scale.
    const end = new Date(cutoff + "T00:00:00");
    while (cursor <= end) {
      labels.push(cursor.toISOString().slice(0, 10));
      cursor.setDate(cursor.getDate() + 1);
    }
    const observed = labels.map((d) => (d in obs ? obs[d] : null));
    const predicted = labels.map((d) => (d in pred ? pred[d] : null));
    return { labels, observed, predicted };
  }

  function renderTilesAndMain() {
    renderTiles();
    const ch = currentChannelData();
    if (!ch) return;
    const cutoff = state.data.cutoff;
    const days = ch.days || [];
    const report = buildReportSeries();
    document.getElementById("main-subtitle").textContent =
      state.channel === "__all__"
        ? `各渠道分别反推后按发生日相加。橙色为估计发生量，蓝色为实际收到的上报，绿色为用估计值回推的上报；蓝绿两条都只画到截止日 ${cutoff}，可直接比对拟合是否成立。`
        : `${state.channel}：橙色为估计发生量，蓝色为实际收到的上报，绿色为用估计值回推的上报；蓝绿两条都只画到截止日 ${cutoff}，可直接比对拟合是否成立。`;

    const labels = report.labels.concat(days.map((d) => d.day));
    // Deduplicate while preserving order.
    const seen = new Set();
    const allLabels = [];
    for (const label of labels) if (!seen.has(label)) { seen.add(label); allLabels.push(label); }
    const occByDay = new Map(days.map((d) => [d.day, d]));
    const observedOcc = allLabels.map((d) => (occByDay.has(d) ? occByDay.get(d).observed : null));
    const estimated = allLabels.map((d) => (occByDay.has(d) ? occByDay.get(d).estimated : null));
    const lower = allLabels.map((d) => (occByDay.has(d) && !occByDay.get(d).matured ? occByDay.get(d).q05 : null));
    const upper = allLabels.map((d) => (occByDay.has(d) && !occByDay.get(d).matured ? occByDay.get(d).q95 : null));
    const observedReports = allLabels.map((d) => {
      const idx = report.labels.indexOf(d);
      return idx >= 0 ? report.observed[idx] : null;
    });
    const predictedReports = allLabels.map((d) => {
      const idx = report.labels.indexOf(d);
      return idx >= 0 ? report.predicted[idx] : null;
    });

    const ctx = document.getElementById("main-chart");
    if (charts.main) charts.main.destroy();
    charts.main = new Chart(ctx, {
      type: "line",
      data: {
        labels: allLabels,
        datasets: [
          {
            label: "估计发生量（90% 区间）",
            data: upper,
            borderWidth: 0,
            pointRadius: 0,
            fill: "+1",
            backgroundColor: "rgba(235,104,52,0.13)",
            spanGaps: false,
            order: 5,
          },
          {
            label: "区间下界",
            data: lower,
            borderWidth: 0,
            pointRadius: 0,
            fill: false,
            spanGaps: false,
            order: 6,
          },
          {
            label: "原始上报曲线",
            data: observedReports,
            borderColor: SERIES.observed,
            backgroundColor: SERIES.observed,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.15,
            spanGaps: false,
            order: 1,
          },
          {
            label: "回推上报曲线",
            data: predictedReports,
            borderColor: "rgba(27,175,122,0.55)",
            backgroundColor: "rgba(27,175,122,0.55)",
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.15,
            spanGaps: false,
            order: 2,
          },
          {
            label: "估计发生量",
            data: estimated,
            borderColor: SERIES.estimate,
            backgroundColor: SERIES.estimate,
            borderWidth: 2,
            pointRadius: allLabels.map((d) => {
              const day = occByDay.get(d);
              return day && day.day === state.selectedDay ? 5 : 0;
            }),
            pointBackgroundColor: SERIES.estimate,
            pointBorderColor: cssVar("--surface-1"),
            pointBorderWidth: 2,
            tension: 0.2,
            spanGaps: false,
            order: 0,
          },
          {
            label: "已上报（发生日）",
            data: observedOcc,
            borderColor: "rgba(42,120,214,0.35)",
            backgroundColor: "rgba(42,120,214,0.35)",
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.2,
            spanGaps: false,
            order: 3,
          },
        ],
      },
      options: baseOptions({
        onClick: (evt, elements, chart) => {
          const idx = nearestIndex(chart, evt);
          if (idx === null) return;
          // Snap to the nearest day that is inside the occurrence window, so a
          // click on the forecast tail still opens the closest explained day.
          const candidates = [idx, idx - 1, idx + 1, idx - 2, idx + 2, idx - 3, idx + 3];
          for (const i of candidates) {
            if (i >= 0 && i < allLabels.length && occByDay.has(allLabels[i])) {
              selectDay(allLabels[i]);
              return;
            }
          }
        },
        onHover: (evt, elements, chart) => {
          const idx = nearestIndex(chart, evt);
          chart._hoverIndex = idx;
          const canvas = chart.canvas;
          canvas.style.cursor = idx !== null && occByDay.has(allLabels[idx]) ? "pointer" : "default";
          showMainTooltip(chart, idx, evt, allLabels, occByDay, observedReports, predictedReports);
        },
        plugins: {
          annotation: null,
          tooltip: { enabled: false },
        },
        scales: {
          x: {
            ticks: { callback: (value, index) => shortDay(allLabels[index]), maxTicksLimit: 12 },
          },
          y: { beginAtZero: true, ticks: { callback: (v) => fmtCompact(v) } },
        },
      }),
    });
    bindTooltipHide("main-tooltip", charts.main);
    renderMainTable(allLabels, occByDay, observedReports, predictedReports);
    renderMainLegend();
    updateReadout();
  }

  function renderMainLegend() {
    const host = document.getElementById("main-legend");
    host.replaceChildren();
    const items = [
      { label: "原始上报曲线", cls: "line", color: SERIES.observed },
      { label: "回推上报曲线", cls: "line dashed", color: "rgba(27,175,122,0.75)" },
      { label: "估计发生量", cls: "line", color: SERIES.estimate },
      { label: "已上报（发生日）", cls: "line", color: "rgba(42,120,214,0.5)" },
      { label: "90% 区间", cls: "band", color: "rgba(235,104,52,0.22)" },
    ];
    for (const item of items) {
      const key = el("span", { className: "key " + item.cls });
      if (item.cls === "line") key.style.borderTopColor = item.color;
      else key.style.background = item.color;
      host.appendChild(el("li", {}, [key, el("span", { text: item.label })]));
    }
  }

  function nearestIndex(chart, evt) {
    const points = chart.getElementsAtEventForMode(evt, "index", { intersect: false }, true);
    if (!points.length) return null;
    return points[0].index;
  }

  function showMainTooltip(chart, idx, evt, labels, occByDay, observedReports, predictedReports) {
    const tip = document.getElementById("main-tooltip");
    if (idx === null || idx === undefined) { tip.hidden = true; return; }
    const day = labels[idx];
    const occ = occByDay.get(day);
    tip.replaceChildren();
    tip.appendChild(el("div", { className: "tt-title", text: `${day} ${weekdayName(day)}` }));
    if (occ) {
      addTipRow(tip, "估计发生量", fmtInt(occ.estimated), SERIES.estimate, "line");
      addTipRow(tip, "已上报", fmtInt(occ.observed), SERIES.observed, "line");
      if (!occ.matured) {
        addTipRow(tip, "90% 区间", `${fmtInt(occ.q05)} – ${fmtInt(occ.q95)}`, "rgba(235,104,52,0.6)", "band");
        addTipRow(tip, "待报", fmtInt(occ.unreported), "var(--text-muted)", "band");
      }
      if (occ.lower_bound) tip.appendChild(el("div", { className: "tt-note", text: "下界：受停报或超长延迟影响" }));
    } else {
      tip.appendChild(el("div", { className: "tt-note", text: "该日不在反推窗口内" }));
    }
    const ridx = labels.indexOf(day);
    const pred = predictedReports[ridx];
    if (pred !== null && pred !== undefined) addTipRow(tip, "回推上报", fmtInt(pred), "rgba(27,175,122,0.85)", "line");
    const obs = observedReports[ridx];
    if (obs !== null && obs !== undefined) addTipRow(tip, "当日实际上报", fmtInt(obs), SERIES.observed, "line");
    positionTooltip(tip, chart, evt);
    tip.hidden = false;
  }

  function addTipRow(host, label, value, color, shape) {
    const key = el("i", {});
    if (shape === "line") key.style.borderTopColor = color;
    else { key.className = "bar"; key.style.background = color; }
    host.appendChild(
      el("div", { className: "tt-row" }, [
        el("span", { className: "tt-key" }, [key, el("span", { text: label })]),
        el("span", { className: "tt-val", text: value }),
      ])
    );
  }

  function positionTooltip(tip, chart, evt) {
    const rect = chart.canvas.getBoundingClientRect();
    const wrap = chart.canvas.parentElement.getBoundingClientRect();
    let x = evt.x + 16;
    let y = evt.y + 12;
    const tipRect = tip.getBoundingClientRect();
    if (x + tipRect.width > wrap.width - 6) x = evt.x - tipRect.width - 16;
    if (y + tipRect.height > wrap.height - 6) y = Math.max(6, evt.y - tipRect.height - 12);
    tip.style.left = Math.max(6, x) + "px";
    tip.style.top = Math.max(6, y) + "px";
  }

  function bindTooltipHide(id, chart) {
    const tip = document.getElementById(id);
    chart.canvas.addEventListener("mouseleave", () => { tip.hidden = true; portableTransient(chart, null); });
  }

  function portableTransient(chart, value) {
    if (!chart) return;
    chart._hoverIndex = value;
  }

  function baseOptions(extra) {
    const opts = {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      animation: false,
      layout: { padding: { top: 8, right: 8, bottom: 0, left: 0 } },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { color: cssVar("--baseline"), width: 1 },
          ticks: { color: cssVar("--text-muted"), font: { size: 11 }, maxRotation: 0, autoSkip: true },
        },
        y: {
          grid: { color: cssVar("--grid"), lineWidth: 1, drawTicks: false },
          border: { display: false },
          ticks: { color: cssVar("--text-muted"), font: { size: 11 }, padding: 6 },
          beginAtZero: true,
        },
      },
    };
    return deepMerge(opts, extra || {});
  }

  function deepMerge(a, b) {
    const out = Object.assign({}, a);
    for (const [key, value] of Object.entries(b)) {
      if (value && typeof value === "object" && !Array.isArray(value) && typeof out[key] === "object" && out[key] !== null && !Array.isArray(out[key])) {
        out[key] = deepMerge(out[key], value);
      } else {
        out[key] = value;
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- delay chart
  function renderDelay() {
    const ch = currentChannelData();
    const host = document.getElementById("delay-channel-chip");
    host.textContent = state.channel === "__all__" ? "全部渠道" : state.channel;
    if (!ch) return;
    let probs, horizon, label, meta;
    if (state.channel === "__all__") {
      const profiles = Object.values(state.data.channels).filter((c) => c.profile);
      horizon = Math.max(0, ...profiles.map((c) => c.profile.horizon));
      const weights = profiles.map((c) => c.profile.train_tickets);
      const totalWeight = weights.reduce((s, w) => s + w, 0) || 1;
      probs = [];
      for (let k = 0; k < horizon; k++) {
        let v = 0;
        profiles.forEach((c, i) => {
          const p = c.profile.probs[k];
          if (p !== undefined) v += p * (weights[i] / totalWeight);
        });
        probs.push(v);
      }
      label = "各渠道延迟分布按训练票数加权展示；拟合与反推仍是分开进行的";
      meta = profiles.map((c) => `${c.channel} H=${c.profile.horizon}`);
    } else {
      if (!ch.profile) {
        document.getElementById("delay-subtitle").textContent = "该渠道成熟历史不足，无法训练延迟分布。";
        return;
      }
      probs = ch.profile.probs;
      horizon = ch.profile.horizon;
      const pd = ch.profile;
      label = `训练自 ${pd.train_days} 个成熟发生日、${fmtInt(pd.train_tickets)} 单；时滞上限 ${pd.horizon - 1} 天，均值 ${fmtNum(pd.mean_delay, 2)} 天。`;
      meta = [];
    }
    document.getElementById("delay-subtitle").textContent =
      (state.channel === "__all__" ? label : label + " " + label2(ch)) + (meta.length ? " " + meta.join(" · ") : "");
    const labels = probs.map((_, k) => k === 0 ? "当日" : `+${k} 天`);
    const ctx = document.getElementById("delay-chart");
    if (charts.delay) charts.delay.destroy();
    charts.delay = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "延迟分布",
          data: probs.map((p) => p * 100),
          backgroundColor: ORDINAL[2],
          hoverBackgroundColor: ORDINAL[3],
          borderRadius: 4,
          borderSkipped: "start",
          maxBarThickness: 24,
          categoryPercentage: 0.82,
          barPercentage: 0.9,
        }],
      },
      options: baseOptions({
        onHover: (evt, elements, chart) => {
          const tip = document.getElementById("delay-tooltip");
          const idx = nearestIndex(chart, evt);
          if (idx === null) { tip.hidden = true; return; }
          tip.replaceChildren();
          tip.appendChild(el("div", { className: "tt-title", text: labels[idx] }));
          addTipRow(tip, "占比", fmtPct(probs[idx], 1), ORDINAL[2], "band");
          if (state.channel !== "__all__" && ch.profile) {
            addTipRow(tip, "训练票数", fmtInt(Math.round(probs[idx] * ch.profile.train_tickets)), ORDINAL[2], "band");
          }
          positionTooltip(tip, chart, evt);
          tip.hidden = false;
        },
        scales: {
          y: { ticks: { callback: (v) => v + "%" } },
          x: { ticks: { maxTicksLimit: 14 } },
        },
      }),
    });
    bindTooltipHide("delay-tooltip", charts.delay);
    const rows = probs.map((p, k) => ({
      delay: k,
      p,
      count: state.channel !== "__all__" && ch.profile ? Math.round(p * ch.profile.train_tickets) : null,
    }));
    renderSimpleTable("delay-table", ["时滞（天）", "占比", "训练票数"], rows.map((r) => [r.delay === 0 ? "当日" : "+" + r.delay, fmtPct(r.p, 2), r.count === null ? "—" : fmtInt(r.count)]));
  }

  function label2(ch) {
    return "";
  }

  // ---------------------------------------------------------------- day breakdown
  function dayBreakdown(day) {
    // Returns { labels, observed[], unreported[], total[] } for the selected day.
    // In aggregate mode the per-delay decomposition is rebuilt from each
    // channel's own profile and observed delays, then summed, because the
    // aggregate payload carries volumes only.
    let horizon = 0;
    let observed = [];
    let unreported = [];
    if (state.channel === "__all__") {
      for (const ch of Object.values(state.data.channels)) {
        if (!ch.profile) continue;
        const info = ch.days.find((d) => d.day === day);
        if (!info) continue;
        horizon = Math.max(horizon, ch.profile.horizon);
        for (let k = 0; k < ch.profile.horizon; k++) {
          observed[k] = (observed[k] || 0) + ((info.observed_by_delay || [])[k] || 0);
          const contribution = (info.contribution || [])[k] || 0;
          unreported[k] = (unreported[k] || 0) + Math.max(0, contribution - ((info.observed_by_delay || [])[k] || 0));
        }
      }
    } else {
      const info = ch_days(day);
      if (!info) return null;
      horizon = (info.contribution || []).length;
      const obs = info.observed_by_delay || [];
      observed = Array.from({ length: horizon }, (_, k) => obs[k] || 0);
      unreported = (info.contribution || []).map((v, k) => Math.max(0, v - (obs[k] || 0)));
    }
    observed = Array.from({ length: horizon }, (_, k) => observed[k] || 0);
    unreported = Array.from({ length: horizon }, (_, k) => unreported[k] || 0);
    return {
      labels: Array.from({ length: horizon }, (_, k) => (k === 0 ? "当日" : `+${k} 天`)),
      observed,
      unreported,
      total: observed.map((v, k) => v + unreported[k]),
    };
  }

  function ch_days(day) {
    const ch = state.data.channels[state.channel];
    return ch.days.find((d) => d.day === day);
  }

  function renderDay() {
    const ch = currentChannelData();
    const day = state.selectedDay;
    const subtitle = document.getElementById("day-subtitle");
    if (!ch || !day) {
      subtitle.textContent = "在总览图中点选一天，查看当天估计值如何按时滞构成。";
      if (charts.day) { charts.day.destroy(); charts.day = null; }
      renderSimpleTable("day-table", ["时滞（天）", "已上报", "估计未上报", "合计"], []);
      return;
    }
    const info = ch.days.find((d) => d.day === day);
    if (!info) { subtitle.textContent = "该日不在当前窗口内。"; return; }
    subtitle.textContent = `${day} ${weekdayName(day)}：估计发生 ${fmtInt(info.estimated)}，已上报 ${fmtInt(info.observed)}，待报 ${fmtInt(info.unreported)}${info.lower_bound ? "（下界）" : ""}。`;
    const breakdown = dayBreakdown(day);
    if (!breakdown || !breakdown.labels.length) {
      subtitle.textContent = "该日没有可用的时滞拆解。";
      if (charts.day) { charts.day.destroy(); charts.day = null; }
      return;
    }
    const observedSeries = breakdown.observed;
    const estimatedSeries = breakdown.unreported;
    const labels = breakdown.labels;
    const ctx = document.getElementById("day-chart");
    if (charts.day) charts.day.destroy();
    charts.day = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "已上报",
            data: labels.map((_, k) => observedSeries[k] || 0),
            backgroundColor: SERIES.observed,
            borderRadius: 0,
            borderSkipped: false,
            maxBarThickness: 24,
            stack: "s",
          },
          {
            label: "估计未上报",
            data: labels.map((_, k) => estimatedSeries[k] || 0),
            backgroundColor: "rgba(235,104,52,0.55)",
            borderRadius: 4,
            borderSkipped: false,
            maxBarThickness: 24,
            stack: "s",
          },
        ],
      },
      options: baseOptions({
        interaction: { mode: "index", intersect: false },
        plugins: {
          tooltip: { enabled: false },
        },
        scales: {
          x: { stacked: true, ticks: { maxTicksLimit: 14 } },
          y: { stacked: true, ticks: { callback: (v) => fmtCompact(v) } },
        },
        onHover: (evt, elements, chart) => {
          const tip = document.getElementById("day-tooltip");
          const idx = nearestIndex(chart, evt);
          if (idx === null) { tip.hidden = true; return; }
          tip.replaceChildren();
          tip.appendChild(el("div", { className: "tt-title", text: `${day} ${labels[idx]}` }));
          const obs = observedSeries[idx] || 0;
          addTipRow(tip, "已上报", fmtInt(obs), SERIES.observed, "band");
          addTipRow(tip, "估计未上报", fmtInt(estimatedSeries[idx] || 0), "rgba(235,104,52,0.8)", "band");
          addTipRow(tip, "合计", fmtInt(breakdown.total[idx] || 0), "var(--text-secondary)", "band");
          positionTooltip(tip, chart, evt);
          tip.hidden = false;
        },
      }),
    });
    bindTooltipHide("day-tooltip", charts.day);
    const rows = labels.map((label, k) => [
      label,
      fmtInt(observedSeries[k] || 0),
      fmtInt(estimatedSeries[k] || 0),
      fmtInt(breakdown.total[k] || 0),
    ]);
    renderSimpleTable("day-table", ["时滞（天）", "已上报", "估计未上报", "合计"], rows);
  }

  function selectDay(day) {
    const ch = currentChannelData();
    const info = ch.days.find((d) => d.day === day);
    if (!info) return; // only days inside the back-projection window are explained
    state.selectedDay = day;
    const readout = document.getElementById("main-readout");
    const parts = [];
    parts.push(`${day} ${weekdayName(day)}：估计发生 ${fmtInt(info.estimated)}，其中已上报 ${fmtInt(info.observed)}。`);
    if (!info.matured) parts.push(`90% 区间 ${fmtInt(info.q05)}–${fmtInt(info.q95)}。`);
    if (info.long_observed > 0) parts.push(`另有 ${fmtInt(info.long_observed)} 单延迟超出训练时滞，按已观察量计为下界。`);
    if (info.lower_bound) parts.push("该日标为下界。");
    readout.textContent = parts.join(" ");
    renderTilesAndMain();
    renderDay();
  }

  function updateReadout() {
    if (state.selectedDay) {
      const ch = currentChannelData();
      if (ch && ch.days.some((d) => d.day === state.selectedDay)) return;
    }
    const ch = currentChannelData();
    if (!ch || !ch.days.length) return;
    const last = ch.days[ch.days.length - 1];
    document.getElementById("main-readout").textContent =
      `点选图中任意一天，查看它受哪些延迟段影响。最后一天 ${shortDay(last.day)} 估计 ${fmtInt(last.estimated)} 单，已上报 ${fmtInt(last.observed)} 单。`;
  }

  // ---------------------------------------------------------------- tables
  function renderMainTable(labels, occByDay, observedReports, predictedReports) {
    const rows = labels.map((d, i) => {
      const occ = occByDay.get(d);
      return [
        d,
        occ ? fmtInt(occ.observed) : "—",
        occ ? fmtInt(occ.estimated) : "—",
        occ && !occ.matured ? `${fmtInt(occ.q05)}–${fmtInt(occ.q95)}` : "—",
        predictedReports[i] === null || predictedReports[i] === undefined ? "—" : fmtInt(predictedReports[i]),
        observedReports[i] === null || observedReports[i] === undefined ? "—" : fmtInt(observedReports[i]),
      ];
    });
    renderSimpleTable("main-table", ["日期", "发生日已上报", "估计发生量", "90% 区间", "回推上报", "实际上报"], rows);
  }

  function renderSimpleTable(id, headers, rows) {
    const host = document.getElementById(id);
    if (!rows.length) {
      host.replaceChildren(el("p", { className: "placeholder", text: "当前没有可列出的数据。" }));
      return;
    }
    const table = el("table");
    const thead = el("thead");
    const hr = el("tr");
    for (const h of headers) hr.appendChild(el("th", { text: h }));
    thead.appendChild(hr);
    table.appendChild(thead);
    const tbody = el("tbody");
    for (const row of rows) {
      const tr = el("tr");
      for (const cell of row) tr.appendChild(el("td", { text: String(cell) }));
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    host.replaceChildren(table);
  }

  // ---------------------------------------------------------------- warnings
  function renderWarnings() {
    const host = document.getElementById("warning-list");
    host.replaceChildren();
    const source = state.channel === "__all__" ? collectWarnings() : (currentChannelData().warnings || []).map((w) => Object.assign({ channel: state.channel }, w));
    if (!source.length) {
      host.appendChild(el("p", { className: "placeholder", text: "当前窗口没有稳定性警告。" }));
      return;
    }
    const order = { critical: 0, serious: 1, warning: 2 };
    source.sort((a, b) => (order[a.severity] ?? 9) - (order[b.severity] ?? 9));
    const titles = {
      slowdown: "上报变慢",
      outage: "疑似停报",
      uncovered_delay: "历史未覆盖的延迟",
      wide_interval: "区间偏宽",
      history_short: "成熟历史偏少",
      horizon_capped: "延迟触及上限",
      no_profile: "无法估计",
    };
    const icons = { critical: "!", serious: "!", warning: "i" };
    for (const w of source) {
      const icon = el("span", { className: "icon", text: icons[w.severity] || "i", "aria-hidden": "true" });
      const body = el("div", { className: "body" }, [
        el("span", { className: "w-title", text: titles[w.kind] || w.title || w.kind }),
        el("span", { className: "w-meta", text: `${w.channel} · ${w.title || w.kind}` }),
        el("span", { className: "w-detail", text: w.detail || "" }),
      ]);
      host.appendChild(el("div", { className: "warning sev-" + w.severity }, [icon, body]));
    }
  }

  // ---------------------------------------------------------------- replay
  function renderReplay() {
    const host = document.getElementById("replay-body");
    host.replaceChildren();
    const data = state.replay;
    if (!data || !data.replays.length) {
      host.appendChild(el("p", { className: "placeholder", text: "没有可用的回放结果。" }));
      return;
    }
    const channelNames = Object.keys(data.summary);
    for (const rep of data.replays) {
      const block = el("div", { className: "replay-cutoff" });
      block.appendChild(
        el("h3", {}, [
          el("span", { text: `截止 ${rep.cutoff}` }),
          el("span", { className: "sub", text: `当时未报齐、现已成熟的发生日` }),
        ])
      );
      const row = el("div", { className: "score-row" });
      for (const name of channelNames) {
        const info = rep.channels[name];
        if (!info) continue;
        const better = info.mape < info.naive_mape;
        const score = el("div", { className: "score" }, [
          el("span", { className: "s-name" }, [
            el("span", { className: "dot", style: `background:${channelColor(name)}` }),
            el("span", { text: name }),
          ]),
          el("span", { className: "s-main", text: fmtPct(info.mape, 1) }),
          el("span", { className: "s-sub " + (better ? "better" : "worse"), text: `对照“直接用已上报”：${fmtPct(info.naive_mape, 1)}` }),
          el("span", { className: "s-sub", text: `区间覆盖 ${fmtPct(info.coverage90, 0)} · ${info.scored_days} 天` }),
        ]);
        score.title = `${name} @ ${rep.cutoff}：MAPE ${fmtPct(info.mape, 1)}，偏差 ${fmtNum(info.bias, 1)}，90% 区间覆盖 ${fmtPct(info.coverage90, 0)}`;
        row.appendChild(score);
      }
      block.appendChild(row);
      host.appendChild(block);
    }
    host.appendChild(buildReplaySummary(data));
    renderReplayTable(data);
  }

  function channelColor(name) {
    const names = Object.keys(state.data ? state.data.channels : {});
    const idx = names.indexOf(name);
    const hues = [SERIES.observed, SERIES.estimate, SERIES.predicted, "#eda100", "#e87ba4", "#6250d6", "#008300"];
    return hues[(idx >= 0 ? idx : 0) % hues.length];
  }

  function buildReplaySummary(data) {
    const block = el("div", { className: "replay-cutoff" });
    block.appendChild(el("h3", { text: "按渠道汇总（各回放截止日平均）" }));
    const row = el("div", { className: "score-row" });
    for (const [name, info] of Object.entries(data.summary)) {
      const better = info.mape < info.naive_mape;
      row.appendChild(
        el("div", { className: "score" }, [
          el("span", { className: "s-name" }, [el("span", { className: "dot", style: `background:${channelColor(name)}` }), el("span", { text: name })]),
          el("span", { className: "s-main", text: fmtPct(info.mape, 1) }),
          el("span", { className: "s-sub " + (better ? "better" : "worse"), text: `对照直接用已上报：${fmtPct(info.naive_mape, 1)}` }),
          el("span", { className: "s-sub", text: `区间覆盖 ${fmtPct(info.coverage90, 0)}` }),
        ])
      );
    }
    block.appendChild(row);
    return block;
  }

  function renderReplayTable(data) {
    const rows = [];
    for (const rep of data.replays) {
      for (const [name, info] of Object.entries(rep.channels)) {
        for (const pair of info.pairs) {
          if (!pair.immature_then) continue;
          rows.push([
            rep.cutoff,
            name,
            pair.day,
            pair.age_then + " 天",
            fmtInt(pair.estimated),
            `${fmtInt(pair.q05)}–${fmtInt(pair.q95)}`,
            fmtInt(pair.observed_then),
            fmtInt(pair.matured_now),
            fmtInt(pair.later_arrived),
          ]);
        }
      }
    }
    renderSimpleTable("replay-table", ["回放截止日", "渠道", "发生日", "当时年龄", "当时估计", "90% 区间", "当时已上报", "后来成熟值", "后来补到"], rows);
  }

  // ---------------------------------------------------------------- wiring
  function bindTables() {
    for (const id of ["main", "delay", "day", "replay"]) {
      const btn = document.getElementById(id + "-table-btn");
      const view = document.getElementById(id + "-table");
      btn.addEventListener("click", () => {
        const open = btn.getAttribute("aria-expanded") === "true";
        btn.setAttribute("aria-expanded", String(!open));
        view.hidden = open;
        btn.textContent = open ? "查看数据表" : "收起数据表";
      });
    }
  }

  function debounce(fn, wait) {
    let t;
    return function (...args) {
      clearTimeout(t);
      t = setTimeout(() => fn.apply(this, args), wait);
    };
  }

  function wireControls() {
    document.getElementById("channel-select").addEventListener("change", (e) => {
      state.channel = e.target.value;
      state.selectedDay = null;
      renderTilesAndMain();
      renderDelay();
      renderDay();
      renderWarnings();
    });
    const win = document.getElementById("window-range");
    win.addEventListener("input", (e) => {
      state.window = Number(e.target.value);
      document.getElementById("window-value").textContent = String(state.window);
    });
    win.addEventListener("change", debounce(loadEstimates, 160));
    const sm = document.getElementById("smooth-range");
    sm.addEventListener("input", (e) => {
      state.smooth = Number(e.target.value);
      document.getElementById("smooth-value").textContent = fmtSmooth(state.smooth);
    });
    sm.addEventListener("change", debounce(loadEstimates, 160));
    document.getElementById("replay-button").addEventListener("click", loadReplay);
  }

  function renderAll() {
    renderTilesAndMain();
    renderDelay();
    renderDay();
    renderWarnings();
    if (state.selectedDay) {
      const ch = currentChannelData();
      if (!ch.days.some((d) => d.day === state.selectedDay)) state.selectedDay = null;
    }
  }

  async function boot() {
    bindTables();
    wireControls();
    try {
      await loadMeta();
      await loadEstimates();
    } catch (err) {
      showError(err.message);
    }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
