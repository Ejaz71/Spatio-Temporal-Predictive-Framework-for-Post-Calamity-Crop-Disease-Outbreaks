"""
Extracts PRE-SEASON (preceding-monsoon) features for every event, as `monsoon_*`
columns added to the existing feature CSV.

Why this exists
---------------
Every feature currently in `real_event_features.csv` is measured over the Dec-Mar dry
season *in which the outbreak is observed*. But the project's framing is *post-calamity*
risk: the outbreak follows something. In Bangladesh the dominant antecedent climatic
event is the Jun-Sep monsoon of the preceding calendar year (flooding, waterlogging,
prolonged saturation, and the cropping intensity / inoculum carry-over that follow from
it). None of that is represented.

This script computes, for the monsoon window immediately BEFORE each event's dry-season
observation window, the same real quantities the main pipeline already knows how to
extract:
  - NASA POWER: monsoon rainfall total / peak / anomaly-vs-climatology, humidity,
    temperature, VPD, wet-spell persistence.
  - Sentinel-1 RTC: monsoon SAR backscatter and, crucially, `monsoon_water_extent_frac`
    -- the fraction of the box below the open-water threshold. Unlike the dry-season
    water-extent feature (which sees only irrigation ponding), THIS one is measured in
    the actual flood season, so it is a genuine inundation signal -- the thing proposal
    Section 6.1 ("SAR flood detection") was reaching for.
  - Landsat C2 L2: monsoon NDVI / NDWI / LST. Expect heavy missingness -- monsoon cloud
    cover over Bangladesh routinely defeats optical sensors, and a high monsoon-Landsat
    missing rate is itself a reportable result, not a bug.

Monsoon window derivation: an event's dry-season window starts 1 December of calendar
year Y (its `window_start`), so the preceding monsoon is 1 Jun - 30 Sep of that same
year Y. E.g. window_start 2016-12-01  ->  monsoon 2016-06-01 .. 2016-09-30.

No synthetic data anywhere. A quantity that cannot be obtained for an event stays NaN
(median-imputed per-fold downstream, exactly like the existing satellite columns) and
is logged -- never invented.

Resumable: re-running skips any event whose `monsoon_precip_sum_mm` is already set.
Checkpoints after every event.

Usage (from repo root, ideally under `caffeinate -i` on mains power):
    python3 data/extract_preseason_features.py --features data/processed/real_event_features.csv
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd

from real_feature_pipeline import (
    DISTRICTS_JSON,
    PER_CALL_TIMEOUT_SEC,
    _with_timeout,
    compute_optical_thermal_features,
    compute_sar_features,
    compute_temporal_features,
    get_catalog,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PreseasonFeatures")

# Monsoon SAR reads more scenes than a dry-season window; give each call more room
# than the pipeline default before abandoning it (later passes retry stragglers).
SAR_TIMEOUT_SEC = 150

SAR_DEFAULT = {"sar_vv_db_mean": np.nan, "sar_vh_db_mean": np.nan, "n_s1_scenes": 0,
               "water_extent_frac": np.nan, "water_extent_frac_strict": np.nan,
               "water_extent_frac_max": np.nan}
OPT_DEFAULT = {"ndvi_mean": np.nan, "ndwi_mean": np.nan, "lst_celsius_mean": np.nan,
               "n_landsat_scenes": 0}

# Which raw keys from each extractor to keep, and their `monsoon_`-prefixed names are
# just the same key with the prefix. Everything the extractors return is kept -- the
# decision about which columns actually enter the model feature set is made later, in
# analysis/preseason_exploration.py, on the same evidence basis as water_extent_frac.
MET_KEYS = ["precip_mean_mm", "precip_max_mm", "precip_sum_mm", "precip_climatology_normal_mm",
            "precip_anomaly_mm", "rh_mean_pct", "rh_max_pct", "temp_mean_c", "vpd_mean_kpa",
            "wet_persistence_max_days", "n_met_days"]
SAR_KEYS = ["sar_vv_db_mean", "sar_vh_db_mean", "n_s1_scenes",
            "water_extent_frac", "water_extent_frac_strict", "water_extent_frac_max"]
OPT_KEYS = ["ndvi_mean", "ndwi_mean", "lst_celsius_mean", "n_landsat_scenes"]
ALL_MONSOON_COLUMNS = ([f"monsoon_{k}" for k in MET_KEYS]
                       + [f"monsoon_{k}" for k in SAR_KEYS]
                       + [f"monsoon_{k}" for k in OPT_KEYS])


def monsoon_window(window_start):
    """1 Jun - 30 Sep of the calendar year the dry-season window starts in."""
    year = str(window_start)[:4]
    return f"{year}-06-01", f"{year}-09-30"


def resolve_location(row, districts):
    district = row["district"]
    upazila = row["upazila"] if "upazila" in row and pd.notna(row.get("upazila")) else None
    if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
        lat, lon = float(row["lat"]), float(row["lon"])
    elif district in districts:
        lat, lon = districts[district]["lat"], districts[district]["lon"]
    else:
        return None, None, None
    location_key = f"{district}_{upazila}" if upazila else district
    return lat, lon, location_key


def main(features_csv, districts_json=DISTRICTS_JSON):
    with open(districts_json) as f:
        districts = json.load(f)["districts"]

    df = pd.read_csv(features_csv)
    for col in ALL_MONSOON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    catalog = get_catalog()
    n_done = n_skipped = 0

    for idx, row in df.iterrows():
        event_id = row["event_id"]
        lat, lon, location_key = resolve_location(row, districts)
        if lat is None:
            logger.error(f"{event_id}: no coordinates -- skipping")
            continue

        # Three independent "is this piece still outstanding?" guards. Meteorology
        # succeeds or fails on its own; a met success must NOT suppress later retries
        # of a SAR/Landsat read that failed transiently. A read that has genuinely
        # returned zero scenes is retried on every pass -- Sentinel-1/Landsat coverage
        # of 2015-2018 Bangladesh is dense, so a monsoon zero is almost always a
        # transient drop, and the worst case is a few wasted retries on the rare real
        # gap. A read that succeeded (water frac set, or a positive scene count) is
        # never repeated.
        need_met = pd.isna(df.loc[idx, "monsoon_precip_sum_mm"])
        need_sar = (pd.isna(df.loc[idx, "monsoon_water_extent_frac"])
                    and (pd.isna(df.loc[idx, "monsoon_n_s1_scenes"])
                         or df.loc[idx, "monsoon_n_s1_scenes"] == 0))
        need_opt = (pd.isna(df.loc[idx, "monsoon_ndvi_mean"])
                    and (pd.isna(df.loc[idx, "monsoon_n_landsat_scenes"])
                         or df.loc[idx, "monsoon_n_landsat_scenes"] == 0))
        if not (need_met or need_sar or need_opt):
            n_skipped += 1
            continue

        m_start, m_end = monsoon_window(row["window_start"])
        logger.info(f"[{idx+1}/{len(df)}] {event_id}: monsoon {m_start}..{m_end} ({location_key}) "
                    f"[met={need_met} sar={need_sar} opt={need_opt}]")

        # --- Meteorology (NASA POWER; distinct cache key via the monsoon dates) ---
        if need_met:
            try:
                met = compute_temporal_features(location_key, lat, lon, m_start, m_end)
                for k in MET_KEYS:
                    df.loc[idx, f"monsoon_{k}"] = met.get(k, np.nan)
            except Exception as e:  # noqa: BLE001 -- logged, left NaN, never fabricated
                logger.warning(f"{event_id}: monsoon meteorology failed ({type(e).__name__}: {e}) -- left NaN")

        # --- SAR (this is the real flood-season inundation signal) ---
        # vv_only + max_scenes cap: the monsoon window holds ~2x the scenes of a
        # dry-season window, and reading VV+VH for all of them blew past the 100s
        # per-call budget. The water fraction and monsoon_sar_vv_db_mean both come
        # from VV alone, so VH is dropped here (monsoon_sar_vh_db_mean stays NaN and
        # is not a planned model feature). Longer per-call timeout for headroom.
        if need_sar:
            sar = _with_timeout(
                compute_sar_features, catalog, lat, lon, m_start, m_end, True, 5,
                timeout=SAR_TIMEOUT_SEC, default=SAR_DEFAULT,
                label=f"monsoon SAR for {event_id}",
            )
            for k in SAR_KEYS:
                df.loc[idx, f"monsoon_{k}"] = sar.get(k, np.nan)

        # --- Landsat (expect frequent NaN from monsoon cloud cover) ---
        if need_opt:
            opt = _with_timeout(
                compute_optical_thermal_features, catalog, lat, lon, m_start, m_end,
                timeout=PER_CALL_TIMEOUT_SEC, default=OPT_DEFAULT,
                label=f"monsoon Landsat for {event_id}",
            )
            for k in OPT_KEYS:
                df.loc[idx, f"monsoon_{k}"] = opt.get(k, np.nan)

        def _f(col):
            v = df.loc[idx, col]
            return f"{v:.3f}" if pd.notna(v) else "NaN"
        logger.info(f"    precip_sum={_f('monsoon_precip_sum_mm')}mm "
                    f"anomaly={_f('monsoon_precip_anomaly_mm')}mm "
                    f"water_frac={_f('monsoon_water_extent_frac')} "
                    f"n_s1={_f('monsoon_n_s1_scenes')} "
                    f"n_landsat={_f('monsoon_n_landsat_scenes')}")
        n_done += 1
        df.to_csv(features_csv, index=False)

    df.to_csv(features_csv, index=False)
    logger.info(f"Done. {n_done} events touched this run, {n_skipped} already complete. "
                f"monsoon meteorology: {int(df['monsoon_precip_sum_mm'].notna().sum())}/{len(df)}, "
                f"monsoon SAR non-null: {int(df['monsoon_water_extent_frac'].notna().sum())}/{len(df)}, "
                f"monsoon Landsat non-null: {int(df['monsoon_ndvi_mean'].notna().sum())}/{len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/real_event_features.csv")
    args = parser.parse_args()
    main(args.features)
