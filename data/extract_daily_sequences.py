"""
Extracts the REAL day-by-day meteorological sequence for each of the 77 events from
the NASA POWER responses already cached by real_feature_pipeline.py — no new API
calls. real_feature_pipeline.py computed these same per-day values internally but
only persisted their summary statistics (mean/max/sum); this script recovers and
saves the full daily series, which gives the fusion model's LSTM branch something
real to actually operate over instead of 9 pre-aggregated scalars reshaped into a
pseudo-sequence.

All events' windows are >= 90 real days long (wheat blast: 90 days, Dec1-Feb28; rice
blast: 105 days, Dec1-Mar15). Sequences are right-aligned and truncated to the last
SEQ_LEN=90 real days so every event contributes the same amount of genuine daily data,
closest in time to the survey/outcome window.
"""

import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExtractDailySequences")

EVENTS_CSV = "data/raw_literature/events.csv"
CACHE_DIR = "data/raw_authentic/nasa_power"
OUT_PATH = "data/processed/real_daily_sequences.npz"
SEQ_LEN = 90
DAILY_FEATURES = ["precip_mm", "precip_anomaly_mm", "temp_c", "rh_pct", "vpd_kpa", "wet_persistence_days"]


def build_daily_series(district, window_start, window_end):
    cache_key = f"nasa_{district}_{window_start}_{window_end}".replace("-", "")
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
    valid_p = [v for v in p_dict.values() if v is not None and v >= 0]
    climatological_base_p = float(np.median(valid_p)) if valid_p else 0.0

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

        anomaly = precip - climatological_base_p
        if precip >= 5.0 or (rh is not None and rh >= 85.0):
            wet_persistence += 1
        else:
            wet_persistence = max(0, wet_persistence - 1)

        series[i] = [precip, anomaly, t_mean, rh, vpd, wet_persistence]

    return series


def main():
    events = pd.read_csv(EVENTS_CSV)
    all_sequences = []
    event_ids = []
    skipped = []

    for _, ev in events.iterrows():
        series = build_daily_series(ev["district"], ev["window_start"], ev["window_end"])
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

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(OUT_PATH, sequences=sequences_arr, event_ids=np.array(event_ids))
    logger.info(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
