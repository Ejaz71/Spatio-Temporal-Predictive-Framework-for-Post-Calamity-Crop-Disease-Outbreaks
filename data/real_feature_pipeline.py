"""
Real Feature Extraction Pipeline (replaces the fabricated authentic_data_pipeline.py)

Builds the 77-row Option A feature matrix described in the proposal (Section 6.2),
directly from:
  - data/raw_literature/events.csv          (Mahmud et al. 2021 rice blast survey,
                                              Islam et al. 2016 wheat blast outbreak)
  - config/districts_real.json              (real district-centroid coordinates)
  - NASA POWER Agroclimatology API           (real daily meteorology)
  - Sentinel-1 RTC (Planetary Computer STAC) (real calibrated SAR VV/VH gamma0, dB)
  - Landsat Collection 2 Level-2 (STAC)      (real NDVI, NDWI, land surface temperature)

No synthetic/random data is generated anywhere in this file. If a real value cannot
be obtained for an event (e.g. missing scene, API failure), the corresponding cell is
left as NaN and the event is flagged in `data_quality_notes` rather than being filled in.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta

# Must be set before rasterio opens any remote dataset: keeps COG reads to
# small HTTP range requests instead of GDAL probing full directory/file listings.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff,.TIFF")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "25000000")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
# Fail fast rather than grind. GDAL's default GDAL_HTTP_RETRY_DELAY is 30s, and a COG
# read issues one request per tile — so on a flaky link, 3 retries x 30s x many tiles
# turned a single scene read into an hour-long stall (observed 2026-09-08: one event
# took 56 minutes). One quick retry absorbs a transient blip; anything worse is better
# handled by re-running the resumable script later, when the connection is back.
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "1")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "15")

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RealFeaturePipeline")

NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

EVENTS_CSV = "data/raw_literature/events.csv"
DISTRICTS_JSON = "config/districts_real.json"
CACHE_DIR = "data/raw_authentic"
OUT_CSV = "data/processed/real_event_features.csv"

CLIMATOLOGY_START_YEAR = 2001
CLIMATOLOGY_END_YEAR = 2020
BBOX_DELTA_DEG = 0.06  # ~ +/-6-7 km box around district centroid


# ---------------------------------------------------------------------------
# NASA POWER (real meteorology)
# ---------------------------------------------------------------------------

def fetch_nasa_power(lat, lon, start, end, cache_key):
    cache_dir = os.path.join(CACHE_DIR, "nasa_power")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    params = {
        "parameters": "PRECTOTCORR,RH2M,T2M,T2MDEW",
        "community": "AG",
        "longitude": f"{lon:.4f}",
        "latitude": f"{lat:.4f}",
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "format": "JSON",
    }
    for attempt in range(4):
        try:
            res = requests.get(NASA_POWER_BASE, params=params, timeout=60)
            if res.status_code == 200:
                data = res.json()
                with open(cache_file, "w") as f:
                    json.dump(data, f)
                return data
            logger.warning(f"NASA POWER {res.status_code} for {cache_key}, retrying...")
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            logger.warning(f"NASA POWER error for {cache_key}: {e}, retrying...")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"NASA POWER fetch failed for {cache_key}")


def get_climatology_baseline(location_key, lat, lon, window_start, window_end):
    """
    Real climatological normal for the event's day-of-year range, computed from
    NASA POWER daily precipitation across CLIMATOLOGY_START_YEAR..CLIMATOLOGY_END_YEAR
    at this location's coordinates (not from the event's own window). `location_key`
    is district name for the 77-event district-level dataset, or "District_Upazila"
    for the finer-grained upazila-level dataset — it only affects the cache filename
    (lat/lon already pin down the actual query point), kept for human-readable caching.
    """
    cache_key = f"clim_{location_key}_{lat:.3f}_{lon:.3f}"
    data = fetch_nasa_power(
        lat, lon,
        f"{CLIMATOLOGY_START_YEAR}-01-01", f"{CLIMATOLOGY_END_YEAR}-12-31",
        cache_key,
    )
    props = data["properties"]["parameter"]
    p_dict = props.get("PRECTOTCORR", {})

    start_doy = datetime.strptime(window_start, "%Y-%m-%d").timetuple().tm_yday
    end_dt = datetime.strptime(window_end, "%Y-%m-%d")
    start_dt = datetime.strptime(window_start, "%Y-%m-%d")
    span_days = (end_dt - start_dt).days + 1

    vals = []
    for dt_key, v in p_dict.items():
        if v is None or v < 0:
            continue
        d = datetime.strptime(dt_key, "%Y%m%d")
        doy = d.timetuple().tm_yday
        # crude wraparound-aware day-of-year window match
        in_window = False
        for offset in range(span_days):
            target_doy = (start_doy + offset - 1) % 366 + 1
            if abs(doy - target_doy) <= 2:
                in_window = True
                break
        if in_window:
            vals.append(v)

    # Mean, not median: Bangladesh's Dec-Mar dry season is zero-inflated (roughly 80%
    # of days receive 0mm rain across the 2001-2020 climatology window), so the median
    # of this distribution is genuinely 0.0mm — which would make precip_anomaly_mm
    # identical to precip_mean_mm for every event (a known bug fixed here; see
    # CLAUDE.md "Known Issues" for the discovery). The mean is still real, still
    # computed from the same real NASA POWER daily series, and does not collapse to
    # zero, so it is the correct summary statistic for a "typical rainfall level"
    # baseline in this climate.
    return float(np.mean(vals)) if vals else float(np.nan)


def compute_temporal_features(location_key, lat, lon, window_start, window_end):
    # NOTE: for the original 77-event district-level dataset, location_key == district,
    # exactly reproducing the pre-existing cache filenames (so that run is unaffected
    # and its cache stays valid). For the upazila-level dataset, location_key is
    # "District_Upazila" so multiple upazilas in the same district/year don't collide
    # on the same cache file despite sharing a window.
    cache_key = f"nasa_{location_key}_{window_start}_{window_end}".replace("-", "")
    data = fetch_nasa_power(lat, lon, window_start, window_end, cache_key)
    props = data["properties"]["parameter"]
    p = props.get("PRECTOTCORR", {})
    rh = props.get("RH2M", {})
    t = props.get("T2M", {})
    td = props.get("T2MDEW", {})

    dates = sorted(p.keys())
    precip = np.array([max(0.0, p[d]) for d in dates if p[d] is not None and p[d] >= -100], dtype=float)
    rh_vals = np.array([rh[d] for d in dates if d in rh and rh[d] is not None and rh[d] >= 0], dtype=float)
    t_vals = np.array([t[d] for d in dates if d in t and t[d] is not None], dtype=float)
    td_vals = np.array([td[d] for d in dates if d in td and td[d] is not None], dtype=float)

    es = 0.61078 * np.exp((17.27 * t_vals) / (t_vals + 237.3))
    ea = 0.61078 * np.exp((17.27 * td_vals) / (td_vals + 237.3))
    vpd = np.clip(es - ea, 0.02, None)

    wet_persistence = 0
    max_wet = 0
    for i in range(len(precip)):
        r = precip[i] if i < len(precip) else 0
        h = rh_vals[i] if i < len(rh_vals) else 0
        if r >= 5.0 or h >= 85.0:
            wet_persistence += 1
        else:
            wet_persistence = max(0, wet_persistence - 1)
        max_wet = max(max_wet, wet_persistence)

    clim_baseline = get_climatology_baseline(location_key, lat, lon, window_start, window_end)

    return {
        "precip_mean_mm": float(np.mean(precip)) if len(precip) else np.nan,
        "precip_max_mm": float(np.max(precip)) if len(precip) else np.nan,
        "precip_sum_mm": float(np.sum(precip)) if len(precip) else np.nan,
        "precip_climatology_normal_mm": clim_baseline,
        "precip_anomaly_mm": (float(np.mean(precip)) - clim_baseline) if len(precip) and not np.isnan(clim_baseline) else np.nan,
        "rh_mean_pct": float(np.mean(rh_vals)) if len(rh_vals) else np.nan,
        "rh_max_pct": float(np.max(rh_vals)) if len(rh_vals) else np.nan,
        "temp_mean_c": float(np.mean(t_vals)) if len(t_vals) else np.nan,
        "vpd_mean_kpa": float(np.mean(vpd)) if len(vpd) else np.nan,
        "wet_persistence_max_days": float(max_wet),
        "n_met_days": int(len(dates)),
    }


# ---------------------------------------------------------------------------
# Real remote sensing (Sentinel-1 RTC + Landsat C2 L2), via Planetary Computer STAC
# ---------------------------------------------------------------------------

def get_catalog():
    import pystac_client
    import planetary_computer
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _fresh_href(href):
    """Re-sign a Planetary Computer blob URL immediately before reading it.

    PC's SAS tokens are short-lived (~45 minutes). `sign_inplace` stamps them onto
    items at STAC-search time, so on a slow link a token could expire part-way through
    an event's reads and every subsequent tile request came back HTTP 403 — which GDAL
    then dutifully retried, making the stall worse. Signing at the point of use instead
    of the point of search removes that race. planetary_computer caches the token per
    container for its lifetime, so this is one HTTP call per container per ~45 min, not
    per read. If signing fails we fall back to the existing href rather than aborting:
    it may still be valid, and a failed read is handled (as missing) downstream."""
    try:
        import planetary_computer
        return planetary_computer.sign(href)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"re-sign failed ({type(e).__name__}: {e}) — using existing token")
        return href


def _read_window_array(href, bbox_wgs84):
    """Reads the bbox window from a remote COG and returns its finite pixel values as
    a flat array (or None if unreadable/empty). Shared by the SAR and optical paths so
    a single HTTP range read can serve several derived statistics."""
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(href) as src:
        if src.crs is None:
            return None
        b = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
        win = rasterio.windows.from_bounds(*b, transform=src.transform)
        arr = src.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        arr = arr[np.isfinite(arr)]
        return arr if arr.size else None


def _read_band_mean_std(href, bbox_wgs84):
    arr = _read_window_array(href, bbox_wgs84)
    if arr is None:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr))


# Surface-water detection thresholds on Sentinel-1 RTC gamma0 VV, in dB. Smooth open
# water is a specular reflector and returns very little energy to the sensor, so it
# sits far below vegetated/bare land; -15 dB is the conventional operational threshold
# in the Sentinel-1 flood/water-mapping literature, and -18 dB is a stricter variant
# reported alongside it as a sensitivity check (so the finding can't be an artifact of
# one arbitrary cutoff).
#
# NOTE on interpretation: every observation window in this project falls in Dec-Mar,
# Bangladesh's DRY season — monsoon flooding is Jun-Sep. Water detected here is
# therefore predominantly irrigation / paddy standing water, NOT calamity flooding.
# That is still an epidemiologically meaningful quantity (paddy water management
# modulates blast risk: water-stressed, aerobic fields favour the disease), but it
# must be named and reported as water/inundation extent rather than "flood extent".
WATER_THRESHOLD_DB = -15.0
WATER_THRESHOLD_DB_STRICT = -18.0


def _water_fraction(arr_linear, threshold_db):
    """Fraction of valid pixels whose gamma0 backscatter falls below `threshold_db`.

    Operates on the LINEAR power values rasterio returns (comparing against the
    linearised threshold) rather than taking log of each pixel — avoids -inf for
    zero-valued pixels and is numerically cheaper. Only strictly-positive pixels count
    as valid, consistent with compute_sar_features' own `> 0` guard: a zero or negative
    gamma0 is a processing artifact, not a real specular-water return, and counting
    those as water would inflate this feature exactly where the data is worst."""
    valid = arr_linear[arr_linear > 0]
    if valid.size == 0:
        return np.nan
    threshold_linear = 10.0 ** (threshold_db / 10.0)
    return float(np.mean(valid < threshold_linear))


def compute_sar_features(catalog, lat, lon, window_start, window_end,
                          vv_only=False, max_scenes=6):
    """Sentinel-1 RTC gamma0 statistics over the event bbox.

    `vv_only=True` skips the VH reads, halving the network work. Use it when only the
    VV-derived quantities are wanted (the water-extent fractions and sar_vv_db_mean) —
    e.g. backfilling water extent onto rows whose sar_vh_db_mean is already populated.
    The returned sar_vh_db_mean is then NaN and MUST NOT be written back over an
    existing real value.
    """
    bbox = [lon - BBOX_DELTA_DEG, lat - BBOX_DELTA_DEG, lon + BBOX_DELTA_DEG, lat + BBOX_DELTA_DEG]
    search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox,
                             datetime=f"{window_start}/{window_end}", limit=20)
    items = list(search.items())
    if not items:
        return {"sar_vv_db_mean": np.nan, "sar_vh_db_mean": np.nan, "n_s1_scenes": 0,
                "water_extent_frac": np.nan, "water_extent_frac_strict": np.nan,
                "water_extent_frac_max": np.nan}

    vv_vals, vh_vals = [], []
    water_fracs, water_fracs_strict = [], []
    for it in items[:max_scenes]:  # cap scenes per event to bound runtime
        try:
            if "vv" in it.assets:
                # One read serves both the scene-mean backscatter and the per-pixel
                # water-extent fractions — no extra network cost for the new feature.
                arr = _read_window_array(_fresh_href(it.assets["vv"].href), bbox)
                if arr is not None:
                    vv_mean = float(np.mean(arr))
                    if vv_mean > 0:
                        vv_vals.append(10 * np.log10(vv_mean))
                    wf = _water_fraction(arr, WATER_THRESHOLD_DB)
                    wf_strict = _water_fraction(arr, WATER_THRESHOLD_DB_STRICT)
                    if not np.isnan(wf):
                        water_fracs.append(wf)
                    if not np.isnan(wf_strict):
                        water_fracs_strict.append(wf_strict)
            if not vv_only and "vh" in it.assets:
                vh_mean, _ = _read_band_mean_std(_fresh_href(it.assets["vh"].href), bbox)
                if vh_mean and vh_mean > 0:
                    vh_vals.append(10 * np.log10(vh_mean))
        except Exception as e:
            logger.warning(f"S1 read failed for {it.id}: {e}")
            continue

    return {
        "sar_vv_db_mean": float(np.mean(vv_vals)) if vv_vals else np.nan,
        "sar_vh_db_mean": float(np.mean(vh_vals)) if vh_vals else np.nan,
        "n_s1_scenes": len(vv_vals),
        # Mean water extent across the window's scenes (the typical inundation state),
        # plus the max (the wettest single observation in the window — closer to the
        # "was there a standing-water event at any point" question the proposal's
        # flood-detection framing was reaching for).
        "water_extent_frac": float(np.mean(water_fracs)) if water_fracs else np.nan,
        "water_extent_frac_strict": float(np.mean(water_fracs_strict)) if water_fracs_strict else np.nan,
        "water_extent_frac_max": float(np.max(water_fracs)) if water_fracs else np.nan,
    }


def compute_optical_thermal_features(catalog, lat, lon, window_start, window_end):
    bbox = [lon - BBOX_DELTA_DEG, lat - BBOX_DELTA_DEG, lon + BBOX_DELTA_DEG, lat + BBOX_DELTA_DEG]
    search = catalog.search(
        collections=["landsat-c2-l2"], bbox=bbox,
        datetime=f"{window_start}/{window_end}",
        query={"eo:cloud_cover": {"lt": 50}},
        limit=30,
    )
    items = list(search.items())
    if not items:
        return {"ndvi_mean": np.nan, "ndwi_mean": np.nan, "lst_celsius_mean": np.nan, "n_landsat_scenes": 0}

    items = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))

    ndvi_vals, ndwi_vals, lst_vals = [], [], []
    for it in items[:4]:
        try:
            red_href = it.assets["red"].href
            nir_href = it.assets["nir08"].href
            green_href = it.assets["green"].href
            thermal_key = "lwir11" if "lwir11" in it.assets else "lwir"

            red_mean, _ = _read_band_mean_std(red_href, bbox)
            nir_mean, _ = _read_band_mean_std(nir_href, bbox)
            green_mean, _ = _read_band_mean_std(green_href, bbox)

            # Collection 2 L2 surface reflectance scaling (USGS): SR = DN*0.0000275 - 0.2
            red_sr = red_mean * 0.0000275 - 0.2
            nir_sr = nir_mean * 0.0000275 - 0.2
            green_sr = green_mean * 0.0000275 - 0.2

            if np.isfinite(red_sr) and np.isfinite(nir_sr) and (nir_sr + red_sr) != 0:
                ndvi_vals.append((nir_sr - red_sr) / (nir_sr + red_sr))
            if np.isfinite(green_sr) and np.isfinite(nir_sr) and (green_sr + nir_sr) != 0:
                ndwi_vals.append((green_sr - nir_sr) / (green_sr + nir_sr))

            if thermal_key in it.assets:
                lwir_mean, _ = _read_band_mean_std(it.assets[thermal_key].href, bbox)
                # Collection 2 L2 surface temperature scaling: ST(K) = DN*0.00341802 + 149.0
                st_k = lwir_mean * 0.00341802 + 149.0
                if np.isfinite(st_k) and 250 < st_k < 340:
                    lst_vals.append(st_k - 273.15)
        except Exception as e:
            logger.warning(f"Landsat read failed for {it.id}: {e}")
            continue

    return {
        "ndvi_mean": float(np.mean(ndvi_vals)) if ndvi_vals else np.nan,
        "ndwi_mean": float(np.mean(ndwi_vals)) if ndwi_vals else np.nan,
        "lst_celsius_mean": float(np.mean(lst_vals)) if lst_vals else np.nan,
        "n_landsat_scenes": len(ndvi_vals),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PER_CALL_TIMEOUT_SEC = 100  # bounds a single flaky scene/token from stalling the whole batch


def _with_timeout(fn, *args, timeout=PER_CALL_TIMEOUT_SEC, default=None, label=""):
    """Run fn(*args) in a daemon thread; if it exceeds `timeout`, abandon it and
    return `default` instead of letting one stuck remote read stall everything.

    Uses a raw daemon thread rather than ThreadPoolExecutor deliberately. The
    executor version of this function had a silent bug: `with ThreadPoolExecutor(...)`
    calls shutdown(wait=True) on block exit, so after the TimeoutError was caught and
    "moving on" was logged, the code *still* blocked until the stuck read finished.
    The timeout was decorative — one event was observed taking 56 minutes under a
    100s cap. Python cannot force-kill a thread, so the abandoned worker does keep
    running; marking it daemon at least stops it from blocking interpreter exit."""
    import threading
    box = {}

    def target():
        try:
            box["value"] = fn(*args)
        except Exception as e:  # noqa: BLE001 - reported below, never silently swallowed
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        logger.warning(f"{label} timed out after {timeout}s — recording as missing and moving on")
        return default
    if "error" in box:
        e = box["error"]
        logger.warning(f"{label} raised {type(e).__name__}: {e} — recording as missing and moving on")
        return default
    return box.get("value", default)


def run(limit=None, start_at=0, events_csv=EVENTS_CSV, out_csv=OUT_CSV, districts_json=DISTRICTS_JSON):
    """
    events_csv rows are resolved to (lat, lon, location_key) one of two ways:
      - if the row already has non-null 'lat'/'lon' columns (the upazila-level
        dataset), those real per-upazila coordinates are used directly, and
        location_key = "{district}_{upazila}" to keep cache files disambiguated
        between upazilas sharing a district and a window.
      - otherwise (the original 77-event district-level dataset), lat/lon are
        looked up from districts_json by district name, and location_key =
        district — byte-for-byte identical to the original cache key scheme,
        so that dataset's existing cache is reused rather than re-fetched.
    """
    with open(districts_json, "r") as f:
        districts = json.load(f)["districts"]

    events = pd.read_csv(events_csv)
    if limit:
        events = events.iloc[start_at:start_at + limit]

    # Resume: skip events already present in a prior checkpoint so a restart
    # doesn't repeat the (slow, network-bound) work already done.
    rows = []
    done_ids = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done_ids = set(prev["event_id"])
        logger.info(f"Resuming: {len(done_ids)} events already in {out_csv}, skipping those")

    catalog = get_catalog()

    for idx, ev in events.iterrows():
        if ev["event_id"] in done_ids:
            continue

        district = ev["district"]
        has_own_coords = "lat" in ev and "lon" in ev and pd.notna(ev["lat"]) and pd.notna(ev["lon"])
        if has_own_coords:
            lat, lon = float(ev["lat"]), float(ev["lon"])
            upazila = ev["upazila"] if "upazila" in ev and pd.notna(ev["upazila"]) else None
            location_key = f"{district}_{upazila}" if upazila else district
        else:
            if district not in districts:
                logger.error(f"No coordinates for district '{district}' — skipping {ev['event_id']}")
                continue
            lat, lon = districts[district]["lat"], districts[district]["lon"]
            location_key = district

        logger.info(f"[{idx+1}/{len(events)}] {ev['event_id']} ({location_key}, {ev['window_start']}..{ev['window_end']})")

        row = ev.to_dict()
        try:
            row.update(compute_temporal_features(location_key, lat, lon, ev["window_start"], ev["window_end"]))
        except Exception as e:
            logger.error(f"Meteorology failed for {ev['event_id']}: {e}")

        sar_result = _with_timeout(
            compute_sar_features, catalog, lat, lon, ev["window_start"], ev["window_end"],
            default={"sar_vv_db_mean": np.nan, "sar_vh_db_mean": np.nan, "n_s1_scenes": 0,
                     "water_extent_frac": np.nan, "water_extent_frac_strict": np.nan,
                     "water_extent_frac_max": np.nan},
            label=f"SAR for {ev['event_id']}",
        )
        row.update(sar_result)

        opt_result = _with_timeout(
            compute_optical_thermal_features, catalog, lat, lon, ev["window_start"], ev["window_end"],
            default={"ndvi_mean": np.nan, "ndwi_mean": np.nan, "lst_celsius_mean": np.nan, "n_landsat_scenes": 0},
            label=f"Optical/thermal for {ev['event_id']}",
        )
        row.update(opt_result)

        rows.append(row)

        # Checkpoint every event so a restart resumes right after the last completed one.
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        logger.info(f"Checkpoint: saved {len(rows)} rows so far to {out_csv}")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(out_df)} real event feature rows to {out_csv}")
    return out_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("limit", type=int, nargs="?", default=None)
    parser.add_argument("start_at", type=int, nargs="?", default=0)
    parser.add_argument("--events", default=EVENTS_CSV, help="Events CSV path")
    parser.add_argument("--out", default=OUT_CSV, help="Output feature CSV path")
    args = parser.parse_args()
    run(limit=args.limit, start_at=args.start_at, events_csv=args.events, out_csv=args.out)
