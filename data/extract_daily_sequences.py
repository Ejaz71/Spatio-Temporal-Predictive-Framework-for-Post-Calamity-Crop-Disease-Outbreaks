"""
Extracts the REAL day-by-day meteorological sequence for each event from the NASA
POWER responses already cached by real_feature_pipeline.py — no new API calls.
real_feature_pipeline.py computed these same per-day values internally but only
persisted their summary statistics (mean/max/sum); this script recovers and saves
the full daily series, which gives the fusion model's LSTM branch something real to
actually operate over instead of 9 pre-aggregated scalars reshaped into a pseudo-
sequence.

All events' windows are >= 90 real days long (wheat blast: 90 days, Dec1-Feb28; rice
blast: 105 days, Dec1-Mar15). Sequences are right-aligned and truncated to the last
SEQ_LEN=90 real days so every event contributes the same amount of genuine daily data,
closest in time to the survey/outcome window.

Per-day precip_anomaly_mm uses the SAME real 20-year (2001-2020) NASA POWER
climatology baseline as classical_baselines.py's precip_anomaly_mm (via
real_feature_pipeline.get_climatology_baseline, reading its own separately cached
"clim_*.json" file — no new API calls), rather than a self-referential median/mean of
the event's own ~90-105 day window. The earlier version of this script computed the
latter, which in Bangladesh's zero-inflated Dec-Mar dry season collapses to ~0 for
most events (same root cause as the classical-feature bug, fixed 2026-09-08 — see
CLAUDE.md "Known Issues").
"""

import json
import logging
import os

import numpy as np
import pandas as pd

from real_feature_pipeline import get_climatology_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExtractDailySequences")

EVENTS_CSV = "data/raw_literature/events.csv"
DISTRICTS_JSON = "config/districts_real.json"
CACHE_DIR = "data/raw_authentic/nasa_power"
OUT_PATH = "data/processed/real_daily_sequences.npz"
SEQ_LEN = 90
DAILY_FEATURES = ["precip_mm", "precip_anomaly_mm", "temp_c", "rh_pct", "vpd_kpa", "wet_persistence_days"]


def build_daily_series(location_key, lat, lon, window_start, window_end):
    cache_key = f"nasa_{location_key}_{window_start}_{window_end}".replace("-", "")
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if not os.path.exists(cache_file):
        return None

    with open(cache_file) as f:
        data = json.load(f)
    props = data["properties"]["parameter"]
    p_dict = props.get("PRECTOTCORR", {})
    rh_dict = props.get("RH2M", {})
    t_dict = props.get("T2M", {})
    td_dict = props.get("T2MDEW", {})

    dates = sorted(p_dict.keys())
    # Real 20-year (2001-2020) climatological mean for this location/window's
    # day-of-year range — same cached "clim_*.json" file classical_baselines'
    # precip_anomaly_mm reads, no new API call.
    clim_baseline = get_climatology_baseline(location_key, lat, lon, window_start, window_end)
    if np.isnan(clim_baseline):
        clim_baseline = 0.0

    series = np.zeros((len(dates), len(DAILY_FEATURES)), dtype=np.float32)
    wet_persistence = 0
    for i, d in enumerate(dates):
        precip = max(0.0, p_dict.get(d, 0.0) or 0.0)
        rh = rh_dict.get(d, np.nan)
        t_mean = t_dict.get(d, np.nan)
        t_dew = td_dict.get(d, np.nan)

        if t_mean is not None and t_dew is not None:
            es = 0.61078 * np.exp((17.27 * t_mean) / (t_mean + 237.3))
            ea = 0.61078 * np.exp((17.27 * t_dew) / (t_dew + 237.3))
            vpd = max(0.02, float(es - ea))
        else:
            vpd = np.nan

        anomaly = precip - clim_baseline
        if precip >= 5.0 or (rh is not None and rh >= 85.0):
            wet_persistence += 1
        else:
            wet_persistence = max(0, wet_persistence - 1)

        series[i] = [precip, anomaly, t_mean, rh, vpd, wet_persistence]

    return series


def main(events_csv=EVENTS_CSV, out_path=OUT_PATH, districts_json=DISTRICTS_JSON):
    with open(districts_json, "r") as f:
        districts = json.load(f)["districts"]

    events = pd.read_csv(events_csv)
    all_sequences = []
    event_ids = []
    skipped = []

    for _, ev in events.iterrows():
        district = ev["district"]
        has_own_coords = "lat" in ev and "lon" in ev and pd.notna(ev["lat"]) and pd.notna(ev["lon"])
        if has_own_coords:
            lat, lon = float(ev["lat"]), float(ev["lon"])
            upazila = ev["upazila"] if "upazila" in ev and pd.notna(ev["upazila"]) else None
            location_key = f"{district}_{upazila}" if upazila else district
        elif district in districts:
            lat, lon = districts[district]["lat"], districts[district]["lon"]
            location_key = district
        else:
            logger.error(f"No coordinates for district '{district}' — skipping {ev['event_id']}")
            skipped.append(ev["event_id"])
            continue

        series = build_daily_series(location_key, lat, lon, ev["window_start"], ev["window_end"])
        if series is None:
            skipped.append(ev["event_id"])
            continue
        if len(series) < SEQ_LEN:
            skipped.append(ev["event_id"])
            continue
        # Right-align: keep the SEQ_LEN days closest to window_end.
        truncated = series[-SEQ_LEN:]
        all_sequences.append(truncated)
        event_ids.append(ev["event_id"])

    if skipped:
        logger.warning(f"Skipped {len(skipped)} events (no cache or < {SEQ_LEN} days): {skipped}")

    sequences_arr = np.stack(all_sequences, axis=0)  # (n_events, SEQ_LEN, n_features)
    logger.info(f"Built real daily sequences: shape={sequences_arr.shape} for {len(event_ids)} events")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, sequences=sequences_arr, event_ids=np.array(event_ids))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=EVENTS_CSV)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()
    main(events_csv=args.events, out_path=args.out)
