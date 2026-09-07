"""
Fits the primary model (RandomForest — see results/fusion_model_results.json for why
the classical baseline, not the CNN-LSTM fusion model, is the primary result) on ALL
77 real events, and persists it for the dashboard's /api/predict endpoint.

This is a production artifact for serving, not a new evaluation: reported performance
numbers come from training/classical_baselines.py's cross-validated results, not from
this fit (which sees all the data and would look artificially good).
"""

import json
import logging
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainFinalModel")

FEATURES_CSV = "data/processed/real_event_features.csv"
FEATURE_COLUMNS = [
    "precip_mean_mm", "precip_max_mm", "precip_sum_mm", "precip_anomaly_mm",
    "rh_mean_pct", "rh_max_pct", "temp_mean_c", "vpd_mean_kpa", "wet_persistence_max_days",
    "sar_vv_db_mean", "sar_vh_db_mean", "ndvi_mean", "ndwi_mean", "lst_celsius_mean",
]
OUT_PATH = "results/checkpoints/random_forest_final.joblib"
FEATURE_RANGES_OUT = "results/feature_ranges.json"


def main():
    df = pd.read_csv(FEATURES_CSV)
    X = df[FEATURE_COLUMNS]
    y = df["label"].astype(int).values

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=400, max_depth=5, min_samples_leaf=2,
            class_weight="balanced", random_state=42,
        )),
    ])
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": FEATURE_COLUMNS}, OUT_PATH)
    logger.info(f"Saved fitted RandomForest pipeline to {OUT_PATH}")

    # Real min/max/median per feature from the actual 77-event dataset, for sensible
    # dashboard slider ranges and defaults (not invented numbers).
    ranges = {}
    for col in FEATURE_COLUMNS:
        vals = X[col].dropna()
        ranges[col] = {
            "min": float(vals.min()), "max": float(vals.max()),
            "median": float(vals.median()),
        }
    with open(FEATURE_RANGES_OUT, "w") as f:
        json.dump(ranges, f, indent=2)
    logger.info(f"Saved real feature ranges to {FEATURE_RANGES_OUT}")


if __name__ == "__main__":
    main()
