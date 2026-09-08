"""
Retries real Sentinel-1/Landsat extraction for events that came back with zero
scenes on a prior real_feature_pipeline.py run. This matters when the network
connection was unstable during the original run (e.g. intermittent power/internet —
"load shedding") — a `n_s1_scenes==0` or `n_landsat_scenes==0` row does not
necessarily mean no real scene was available; it may mean the STAC search or the
raster read failed transiently. This script re-attempts ONLY those specific rows'
remote-sensing calls (meteorology, already real and already present, is left alone)
and overwrites the affected columns IN PLACE only if the retry succeeds — a value
that still comes back empty after retrying stays NaN (genuinely no real scene
available in that window), never filled with anything synthetic.

Safe to run repeatedly (idempotent: rows that already have scenes are skipped).
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

from real_feature_pipeline import (
    DISTRICTS_JSON,
    PER_CALL_TIMEOUT_SEC,
    _with_timeout,
    compute_optical_thermal_features,
    compute_sar_features,
    get_catalog,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RetryMissingScenes")

N_ATTEMPTS = 2  # each attempt is a fresh STAC search + read, useful against transient drops


def resolve_lat_lon(row, districts):
    if "lat" in row and pd.notna(row.get("lat")) and "lon" in row and pd.notna(row.get("lon")):
        return float(row["lat"]), float(row["lon"])
    district = row["district"]
    if district not in districts:
        return None, None
    return districts[district]["lat"], districts[district]["lon"]


def main(features_csv, districts_json=DISTRICTS_JSON):
    with open(districts_json) as f:
        districts = json.load(f)["districts"]

    df = pd.read_csv(features_csv)
    catalog = get_catalog()

    n_sar_recovered, n_sar_still_missing = 0, 0
    n_opt_recovered, n_opt_still_missing = 0, 0

    for idx, row in df.iterrows():
        lat, lon = resolve_lat_lon(row, districts)
        if lat is None:
            continue
        event_id = row["event_id"]
        window_start, window_end = row["window_start"], row["window_end"]

        if row.get("n_s1_scenes", 0) == 0 or pd.isna(row.get("sar_vv_db_mean")):
            for attempt in range(N_ATTEMPTS):
                result = _with_timeout(
                    compute_sar_features, catalog, lat, lon, window_start, window_end,
                    timeout=PER_CALL_TIMEOUT_SEC,
                    default={"sar_vv_db_mean": np.nan, "sar_vh_db_mean": np.nan, "n_s1_scenes": 0,
                             "water_extent_frac": np.nan, "water_extent_frac_strict": np.nan,
                             "water_extent_frac_max": np.nan},
                    label=f"SAR retry ({attempt+1}/{N_ATTEMPTS}) for {event_id}",
                )
                if result["n_s1_scenes"] > 0:
                    df.loc[idx, "sar_vv_db_mean"] = result["sar_vv_db_mean"]
                    df.loc[idx, "sar_vh_db_mean"] = result["sar_vh_db_mean"]
                    df.loc[idx, "n_s1_scenes"] = result["n_s1_scenes"]
                    for wcol in ("water_extent_frac", "water_extent_frac_strict", "water_extent_frac_max"):
                        df.loc[idx, wcol] = result[wcol]
                    n_sar_recovered += 1
                    logger.info(f"{event_id}: SAR recovered on retry ({result['n_s1_scenes']} real scenes)")
                    break
            else:
                n_sar_still_missing += 1

        if row.get("n_landsat_scenes", 0) == 0 or pd.isna(row.get("ndvi_mean")):
            for attempt in range(N_ATTEMPTS):
                result = _with_timeout(
                    compute_optical_thermal_features, catalog, lat, lon, window_start, window_end,
                    timeout=PER_CALL_TIMEOUT_SEC,
                    default={"ndvi_mean": np.nan, "ndwi_mean": np.nan, "lst_celsius_mean": np.nan, "n_landsat_scenes": 0},
                    label=f"Landsat retry ({attempt+1}/{N_ATTEMPTS}) for {event_id}",
                )
                if result["n_landsat_scenes"] > 0:
                    df.loc[idx, "ndvi_mean"] = result["ndvi_mean"]
                    df.loc[idx, "ndwi_mean"] = result["ndwi_mean"]
                    df.loc[idx, "lst_celsius_mean"] = result["lst_celsius_mean"]
                    df.loc[idx, "n_landsat_scenes"] = result["n_landsat_scenes"]
                    n_opt_recovered += 1
                    logger.info(f"{event_id}: Landsat recovered on retry ({result['n_landsat_scenes']} real scenes)")
                    break
            else:
                n_opt_still_missing += 1

        # Checkpoint after every event so an interrupted retry pass doesn't lose progress.
        df.to_csv(features_csv, index=False)

    logger.info(f"SAR: {n_sar_recovered} recovered, {n_sar_still_missing} still genuinely missing after "
                f"{N_ATTEMPTS} retries (no real scene available in that window)")
    logger.info(f"Landsat: {n_opt_recovered} recovered, {n_opt_still_missing} still genuinely missing after "
                f"{N_ATTEMPTS} retries")
    df.to_csv(features_csv, index=False)
    logger.info(f"Saved updated {features_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Feature CSV to retry missing scenes in (in place)")
    args = parser.parse_args()
    main(args.features)
