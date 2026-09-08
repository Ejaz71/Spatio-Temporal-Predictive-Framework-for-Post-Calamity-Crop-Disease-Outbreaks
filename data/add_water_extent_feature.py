"""
Backfills the SAR water-extent features onto an already-extracted feature CSV, by
re-reading ONLY the Sentinel-1 rasters (no NASA POWER, no Landsat re-fetching).

Why this feature exists
-----------------------
The existing SAR columns are `sar_vv_db_mean` / `sar_vh_db_mean`: the mean gamma0
backscatter over the event's ~13 km bounding box. A box mean is structurally incapable
of representing *partial* surface water — open water sits below about -15 dB while the
observed box means cluster around -7 dB, i.e. the mean is dominated by non-water land
and washes the water signal out entirely. `water_extent_frac` instead reports the
FRACTION of pixels below the water threshold, which measures something the mean
cannot.

This closes a real gap against the proposal, which specified SAR flood detection
(Section 6.1) but was only ever implemented as mean backscatter.

Interpretation caveat (important, and carried into the report)
-------------------------------------------------------------
Every observation window here is Dec-Mar — Bangladesh's DRY season. Monsoon flooding
is Jun-Sep. So what this feature detects is overwhelmingly irrigation / paddy standing
water, NOT calamity flooding. That is still epidemiologically meaningful (water-stressed
aerobic fields favour blast, continuously ponded paddies less so), and it is consistent
with the strongest remote-sensing signal already in the data: `sar_vv_db_mean` is the
#1 correlate of neck blast severity (Spearman rho=0.384, p=0.0013), positively — i.e.
drier, less-ponded fields associate with more severe blast. But the feature must be
named and reported as water/inundation extent, not flood extent.

Usage (run from repo root; resumable — skips rows already populated):
    python3 data/add_water_extent_feature.py --features data/processed/real_event_features.csv
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd

from real_feature_pipeline import (
    DISTRICTS_JSON,
    PER_CALL_TIMEOUT_SEC,
    WATER_THRESHOLD_DB,
    WATER_THRESHOLD_DB_STRICT,
    _with_timeout,
    compute_sar_features,
    get_catalog,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AddWaterExtent")

WATER_COLUMNS = ["water_extent_frac", "water_extent_frac_strict", "water_extent_frac_max"]
SAR_DEFAULT = {"sar_vv_db_mean": np.nan, "sar_vh_db_mean": np.nan, "n_s1_scenes": 0,
               "water_extent_frac": np.nan, "water_extent_frac_strict": np.nan,
               "water_extent_frac_max": np.nan}


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
    for col in WATER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    catalog = get_catalog()
    n_done, n_skipped, n_no_scene = 0, 0, 0

    for idx, row in df.iterrows():
        event_id = row["event_id"]

        # Resume: already populated on a previous run.
        if pd.notna(df.loc[idx, "water_extent_frac"]):
            n_skipped += 1
            continue

        # Events with genuinely no Sentinel-1 scene in the window can't have a water
        # fraction; leave them NaN (median-imputed per-fold downstream, same as the
        # other SAR columns) rather than inventing a value.
        if row.get("n_s1_scenes", 0) == 0:
            logger.info(f"{event_id}: no S1 scenes in window — leaving water extent as NaN")
            n_no_scene += 1
            continue

        lat, lon = resolve_lat_lon(row, districts)
        if lat is None:
            logger.error(f"{event_id}: no coordinates — skipping")
            continue

        # vv_only: the water fractions are derived purely from VV, and every row we
        # touch here already has a real sar_vh_db_mean from the main extraction run.
        # Skipping the VH reads halves the network work per event. Only the three
        # water columns are written back, so the VV/VH/scene-count columns already in
        # the CSV are never disturbed.
        result = _with_timeout(
            compute_sar_features, catalog, lat, lon, row["window_start"], row["window_end"],
            True,  # vv_only
            timeout=PER_CALL_TIMEOUT_SEC, default=SAR_DEFAULT,
            label=f"SAR water extent for {event_id}",
        )
        for col in WATER_COLUMNS:
            df.loc[idx, col] = result[col]

        if not np.isnan(result["water_extent_frac"]):
            n_done += 1
            logger.info(f"[{idx+1}/{len(df)}] {event_id}: water extent "
                        f"{result['water_extent_frac']:.4f} (mean, <{WATER_THRESHOLD_DB:g}dB), "
                        f"{result['water_extent_frac_strict']:.4f} (strict, <{WATER_THRESHOLD_DB_STRICT:g}dB), "
                        f"{result['water_extent_frac_max']:.4f} (wettest scene)")
        else:
            logger.warning(f"[{idx+1}/{len(df)}] {event_id}: water extent unavailable this attempt")

        # Checkpoint every event so an interrupted run resumes cleanly.
        df.to_csv(features_csv, index=False)

    df.to_csv(features_csv, index=False)
    logger.info(f"Done. {n_done} events populated this run, {n_skipped} already had values, "
                f"{n_no_scene} have no S1 scene at all. "
                f"Total populated: {int(df['water_extent_frac'].notna().sum())}/{len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/real_event_features.csv")
    args = parser.parse_args()
    main(args.features)
