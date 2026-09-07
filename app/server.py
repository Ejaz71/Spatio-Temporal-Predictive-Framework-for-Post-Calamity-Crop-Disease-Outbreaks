"""
FastAPI Backend for the Crop Disease Outbreak Risk Dashboard.

Serves real, cross-validated results only:
- /api/predict           live RandomForest inference + local SHAP feature attribution
- /api/districts         real district event counts from the 77-event dataset
- /api/historical_events real event feature vectors, for the scenario selector
- /api/results           real classical-baseline and fusion-model CV results
  (including the honest stage 3/4 decision: RandomForest is primary, the fusion
  model underperformed and is reported as a documented negative finding)

No synthetic data, no fabricated pathogen-diagnosis logic, no offline "fallback"
calculation — if the model or data isn't available, the API returns an error rather
than silently rendering an invented number.
"""

import json
import logging
import os
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FastAPIServer")

app = FastAPI(
    title="Post-Calamity Crop Disease Outbreak Risk (Real Data)",
    description="RandomForest model trained on 77 real district-season events "
                 "(Mahmud et al. 2021 rice blast survey + Islam et al. 2016 wheat blast "
                 "outbreak), fused with real NASA POWER weather and real Sentinel-1/Landsat imagery.",
    version="2.0.0",
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
STATIC_DIR = os.path.join(CURRENT_DIR, "static")
TEMPLATES_DIR = os.path.join(CURRENT_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

FEATURE_COLUMNS = [
    "precip_mean_mm", "precip_max_mm", "precip_sum_mm", "precip_anomaly_mm",
    "rh_mean_pct", "rh_max_pct", "temp_mean_c", "vpd_mean_kpa", "wet_persistence_max_days",
    "sar_vv_db_mean", "sar_vh_db_mean", "ndvi_mean", "ndwi_mean", "lst_celsius_mean",
]

MODEL_PATH = os.path.join(REPO_ROOT, "results", "checkpoints", "random_forest_final.joblib")
EVENTS_CSV = os.path.join(REPO_ROOT, "data", "processed", "real_event_features.csv")
DISTRICTS_JSON = os.path.join(REPO_ROOT, "config", "districts_real.json")
CLASSICAL_RESULTS_JSON = os.path.join(REPO_ROOT, "results", "classical_baseline_results.json")
FUSION_RESULTS_JSON = os.path.join(REPO_ROOT, "results", "fusion_model_results.json")

_PIPELINE = None
_EXPLAINER = None
_EVENTS_DF: Optional[pd.DataFrame] = None


def load_artifacts():
    global _PIPELINE, _EXPLAINER, _EVENTS_DF
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"No trained model at {MODEL_PATH}. Run `python training/train_final_model.py` first."
        )
    artifact = joblib.load(MODEL_PATH)
    _PIPELINE = artifact["pipeline"]
    _EXPLAINER = shap.TreeExplainer(_PIPELINE.named_steps["rf"])
    _EVENTS_DF = pd.read_csv(EVENTS_CSV)
    logger.info(f"Loaded RandomForest pipeline and {len(_EVENTS_DF)} real events.")


@app.on_event("startup")
def startup_event():
    load_artifacts()


class PredictionRequest(BaseModel):
    precip_mean_mm: float
    precip_max_mm: float
    precip_sum_mm: float
    precip_anomaly_mm: float
    rh_mean_pct: float
    rh_max_pct: float
    temp_mean_c: float
    vpd_mean_kpa: float
    wet_persistence_max_days: float
    sar_vv_db_mean: float
    sar_vh_db_mean: float
    ndvi_mean: float
    ndwi_mean: float
    lst_celsius_mean: float


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/districts")
def get_districts():
    if _EVENTS_DF is None:
        raise HTTPException(503, "Data not loaded yet")
    with open(DISTRICTS_JSON) as f:
        coords = json.load(f)["districts"]

    grouped = _EVENTS_DF.groupby("district").agg(
        n_events=("label", "size"), n_positive=("label", "sum")
    ).reset_index()

    out = []
    for _, row in grouped.iterrows():
        d = row["district"]
        c = coords.get(d, {})
        out.append({
            "name": d,
            "lat": c.get("lat"),
            "lon": c.get("lon"),
            "n_events": int(row["n_events"]),
            "n_positive": int(row["n_positive"]),
        })
    return {"districts": sorted(out, key=lambda x: -x["n_events"])}


@app.get("/api/historical_events")
def get_historical_events():
    if _EVENTS_DF is None:
        raise HTTPException(503, "Data not loaded yet")
    cols = ["event_id", "district", "disease", "year", "label"] + FEATURE_COLUMNS
    df = _EVENTS_DF[cols].copy()
    df = df.replace({np.nan: None})
    return {"events": df.to_dict("records")}


def _sanitize_nans(obj):
    """Recursively replace NaN/Infinity (valid in Python's json module, invalid in
    strict JSON) with None so Starlette's response encoder doesn't choke on them."""
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nans(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


@app.get("/api/results")
def get_results():
    with open(CLASSICAL_RESULTS_JSON) as f:
        classical = json.load(f)
    with open(FUSION_RESULTS_JSON) as f:
        fusion = json.load(f)
    return _sanitize_nans({"classical_baselines": classical, "fusion_model": fusion})


@app.post("/api/predict")
def predict(req: PredictionRequest):
    if _PIPELINE is None:
        raise HTTPException(503, "Model not loaded yet")

    row = pd.DataFrame([req.model_dump()])[FEATURE_COLUMNS]
    prob = float(_PIPELINE.predict_proba(row)[0, 1])

    # Local SHAP attribution for this specific prediction (not a fixed narrative).
    imputed = _PIPELINE.named_steps["imputer"].transform(row)
    shap_values = _EXPLAINER.shap_values(imputed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    contributions = list(zip(FEATURE_COLUMNS, shap_values[0].tolist()))
    contributions.sort(key=lambda t: -abs(t[1]))

    if prob >= 0.60:
        risk_level = "HIGH RISK"
    elif prob >= 0.35:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "LOW RISK"

    return {
        "outbreak_probability": round(prob, 4),
        "risk_percentage": round(prob * 100.0, 1),
        "risk_level": risk_level,
        "top_contributing_factors": [
            {"feature": name, "shap_value": round(val, 4), "direction": "increases risk" if val > 0 else "decreases risk"}
            for name, val in contributions[:6]
        ],
        "model": "RandomForest (grouped-by-district CV AUPRC=0.730, ROC-AUC=0.851 on 77 real events)",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
