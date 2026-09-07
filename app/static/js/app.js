/**
 * Crop Disease Outbreak Risk Dashboard — real data only.
 * Scenario presets are the actual 77 extracted events (/api/historical_events),
 * not invented numbers. Predictions come from a real RandomForest trained on
 * real satellite/weather data, with real per-request SHAP attribution.
 * No client-side fallback calculation: if the API is unreachable, we say so.
 */

const FEATURES = [
  { key: "precip_mean_mm", slider: "precipMean", disp: "valPrecipMean", fmt: v => `${parseFloat(v).toFixed(2)} mm/day` },
  { key: "precip_max_mm", slider: "precipMax", disp: "valPrecipMax", fmt: v => `${parseFloat(v).toFixed(1)} mm` },
  { key: "precip_sum_mm", slider: "precipSum", disp: "valPrecipSum", fmt: v => `${parseFloat(v).toFixed(1)} mm` },
  { key: "precip_anomaly_mm", slider: "precipAnomaly", disp: "valPrecipAnomaly", fmt: v => `${parseFloat(v) >= 0 ? "+" : ""}${parseFloat(v).toFixed(2)} mm` },
  { key: "rh_mean_pct", slider: "rhMean", disp: "valRhMean", fmt: v => `${parseFloat(v).toFixed(1)}%` },
  { key: "rh_max_pct", slider: "rhMax", disp: "valRhMax", fmt: v => `${parseFloat(v).toFixed(1)}%` },
  { key: "temp_mean_c", slider: "tempMean", disp: "valTempMean", fmt: v => `${parseFloat(v).toFixed(1)} °C` },
  { key: "vpd_mean_kpa", slider: "vpdMean", disp: "valVpdMean", fmt: v => `${parseFloat(v).toFixed(2)} kPa` },
  { key: "wet_persistence_max_days", slider: "wetPersistence", disp: "valWetPersistence", fmt: v => `${parseFloat(v).toFixed(0)} days` },
  { key: "sar_vv_db_mean", slider: "sarVv", disp: "valSarVv", fmt: v => `${parseFloat(v).toFixed(1)} dB` },
  { key: "sar_vh_db_mean", slider: "sarVh", disp: "valSarVh", fmt: v => `${parseFloat(v).toFixed(1)} dB` },
  { key: "ndvi_mean", slider: "ndvi", disp: "valNdvi", fmt: v => parseFloat(v).toFixed(3) },
  { key: "ndwi_mean", slider: "ndwi", disp: "valNdwi", fmt: v => parseFloat(v).toFixed(3) },
  { key: "lst_celsius_mean", slider: "lst", disp: "valLst", fmt: v => `${parseFloat(v).toFixed(1)} °C` },
];

let HISTORICAL_EVENTS = [];

document.addEventListener("DOMContentLoaded", () => {
  initSliderDisplays();
  initInferenceTrigger();
  loadHistoricalEvents();
  loadDistricts();
  loadResults();
});

function initSliderDisplays() {
  FEATURES.forEach(f => {
    const el = document.getElementById(f.slider);
    const disp = document.getElementById(f.disp);
    if (el && disp) {
      el.addEventListener("input", e => { disp.textContent = f.fmt(e.target.value); });
    }
  });
}

function initInferenceTrigger() {
  const btn = document.getElementById("btnRunInference");
  if (btn) btn.addEventListener("click", runInference);
}

async function loadHistoricalEvents() {
  const sel = document.getElementById("scenarioSelect");
  if (!sel) return;
  try {
    const res = await fetch("/api/historical_events");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    HISTORICAL_EVENTS = data.events || [];

    sel.innerHTML = "";
    HISTORICAL_EVENTS.forEach((ev, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      const outcome = ev.label === 1 ? "outbreak" : "no outbreak";
      opt.textContent = `${ev.district} ${ev.year} — ${ev.disease.replace("_", " ")} (${outcome})`;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", e => {
      applyEvent(HISTORICAL_EVENTS[parseInt(e.target.value, 10)]);
      runInference();
    });

    if (HISTORICAL_EVENTS.length > 0) {
      // Default to a real positive (outbreak) event so the demo starts meaningfully.
      const firstPositive = HISTORICAL_EVENTS.findIndex(ev => ev.label === 1);
      const startIdx = firstPositive >= 0 ? firstPositive : 0;
      sel.value = startIdx;
      applyEvent(HISTORICAL_EVENTS[startIdx]);
      runInference();
    }
  } catch (err) {
    console.error("Failed to load historical events:", err);
    sel.innerHTML = '<option value="">Could not load real events from API</option>';
  }
}

function applyEvent(ev) {
  if (!ev) return;
  FEATURES.forEach(f => {
    const val = ev[f.key];
    if (val === null || val === undefined) return;
    const el = document.getElementById(f.slider);
    const disp = document.getElementById(f.disp);
    if (el) el.value = val;
    if (disp) disp.textContent = f.fmt(val);
  });
}

async function runInference() {
  const reqData = {};
  FEATURES.forEach(f => {
    const el = document.getElementById(f.slider);
    reqData[f.key] = el ? parseFloat(el.value) : 0;
  });

  const errorBox = document.getElementById("predictErrorBox");
  if (errorBox) errorBox.hidden = true;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqData),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderPredictionResults(data);
  } catch (err) {
    console.error("Prediction request failed:", err);
    if (errorBox) {
      errorBox.hidden = false;
      errorBox.textContent = `Model API unreachable (${err.message}). No fallback number is shown — this is a real model call or nothing.`;
    }
  }
}

function renderPredictionResults(data) {
  const banner = document.getElementById("riskAlertBanner");
  const levelText = document.getElementById("riskLevelText");
  const probVal = document.getElementById("riskProbVal");

  probVal.textContent = `${data.risk_percentage}%`;
  levelText.textContent = data.risk_level;

  banner.className = "risk-banner";
  if (data.risk_level === "HIGH RISK") banner.classList.add("alert-critical");
  else if (data.risk_level === "MODERATE RISK") banner.classList.add("alert-high");
  else banner.classList.add("alert-low");

  document.getElementById("modelProvenance").textContent = data.model;
  renderContributingFactors(data.top_contributing_factors);
}

function renderContributingFactors(factors) {
  const container = document.getElementById("attentionBarsContainer");
  if (!container || !factors) return;

  container.innerHTML = "";
  const maxAbs = Math.max(...factors.map(f => Math.abs(f.shap_value)), 0.001);

  factors.forEach(f => {
    const row = document.createElement("div");
    row.className = "factor-row";
    const widthPct = Math.max(4, (Math.abs(f.shap_value) / maxAbs) * 100);
    const barColor = f.shap_value > 0 ? "#ef4444" : "#22c55e";
    row.innerHTML = `
      <span class="factor-name">${f.feature}</span>
      <div class="factor-bar-track">
        <div class="factor-bar-fill" style="width:${widthPct}%; background:${barColor};"></div>
      </div>
      <span class="factor-dir">${f.direction}</span>
    `;
    container.appendChild(row);
  });
}

async function loadDistricts() {
  const grid = document.getElementById("districtsGrid");
  if (!grid) return;
  try {
    const res = await fetch("/api/districts");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    grid.innerHTML = "";
    (data.districts || []).forEach(d => {
      const card = document.createElement("div");
      card.className = "district-card";
      card.innerHTML = `
        <div class="district-header">
          <span class="district-name">${d.name}</span>
          <span class="district-zone">${d.n_events} real events</span>
        </div>
        <div class="district-meta-row"><strong>Positive (outbreak) events:</strong> ${d.n_positive} / ${d.n_events}</div>
        <div class="district-meta-row"><strong>Coordinates:</strong> ${d.lat?.toFixed(3)}, ${d.lon?.toFixed(3)}</div>
      `;
      grid.appendChild(card);
    });
  } catch (err) {
    console.error("Failed to load districts:", err);
    grid.innerHTML = "<p>Could not load district data from API.</p>";
  }
}

async function loadResults() {
  try {
    const res = await fetch("/api/results");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderResultsTable(data.classical_baselines, data.fusion_model);
    renderShapRanking(data.classical_baselines.shap_feature_importance_random_forest);
    renderDecisionText(data.fusion_model.stage_3_4_decision);
  } catch (err) {
    console.error("Failed to load results:", err);
  }
}

function renderResultsTable(classical, fusion) {
  const tbody = document.getElementById("resultsTableBody");
  if (!tbody) return;

  const rows = [
    { name: "RandomForest", mod: "Classical baseline — grouped-by-district CV", m: classical.RandomForest.grouped_by_district, cls: "emerald" },
    { name: "XGBoost", mod: "Classical baseline — grouped-by-district CV", m: classical.XGBoost.grouped_by_district, cls: "blue" },
    { name: "CNN-LSTM Fusion Model", mod: "Documented negative finding (see decision note)", m: { oof_auprc: fusion.grouped_by_district.oof_auprc, oof_roc_auc: fusion.grouped_by_district.oof_roc_auc }, cls: "purple" },
  ];

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><strong>${r.name}</strong></td>
      <td><span class="modality-tag ${r.cls}">${r.mod}</span></td>
      <td><strong>${r.m.oof_auprc?.toFixed(3) ?? "—"}</strong></td>
      <td><strong>${r.m.oof_roc_auc?.toFixed(3) ?? "—"}</strong></td>
    </tr>
  `).join("");
}

function renderShapRanking(shapRanking) {
  const container = document.getElementById("shapRankingGrid");
  if (!container || !shapRanking) return;

  const entries = Object.entries(shapRanking).slice(0, 6);
  container.innerHTML = entries.map(([name, val], idx) => `
    <div class="feature-item rank-${idx + 1}">
      <div class="feature-rank-num">#${idx + 1}</div>
      <div class="feature-info">
        <div class="feature-name">${name}</div>
        <div class="feature-desc">Mean |SHAP value| across all 77 real events (RandomForest)</div>
      </div>
      <div class="importance-score">${val.toFixed(4)}</div>
    </div>
  `).join("");
}

function renderDecisionText(decision) {
  const el = document.getElementById("decisionText");
  if (!el || !decision) return;
  el.textContent = decision.rationale;
}
