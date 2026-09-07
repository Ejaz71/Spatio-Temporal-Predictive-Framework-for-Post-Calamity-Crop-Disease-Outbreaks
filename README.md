# Post-Calamity Crop Disease Outbreak Prediction (Bangladesh)

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)

A spatio-temporal predictive framework for post-calamity rice blast and wheat blast
outbreak risk in Bangladesh, built entirely on **real** data: real epidemiological
survey records, real NASA POWER weather, and real Sentinel-1/Landsat satellite imagery.

This is a rebuild of an earlier version of this repository whose data pipeline and
reported results turned out to be synthetically generated and misrepresented as
"authentic." That version has been removed. Everything described below was verified
against its own generating code and, where possible, against independent copies of
the source data before being reported here.

---

## What's real, and what it found

**Data.** 77 labeled district-season events: 24 rice blast district-years (Mahmud et
al., 2021, *South Asian Journal of Biological Research* 4(1):1–13, CC BY 4.0) and 53
wheat blast districts across the pre-emergence (2015), outbreak (2016), and recurrence
(2017) seasons (Islam et al., 2016, *BMC Biology* 14:84). 24 of the 77 events carry a
positive outbreak label. See [`data/raw_literature/events.csv`](data/raw_literature/events.csv).

**Features.** For every one of the 77 events, real values were fetched for its actual
district and season window:
- Meteorology: NASA POWER Agroclimatology API (precipitation, relative humidity,
  temperature, vapor pressure deficit, wet-day persistence), with rainfall anomaly
  computed against a real 20-year (2001–2020) climatological baseline for that
  district and day-of-year range — not a self-referential baseline.
- Radar: Sentinel-1 RTC (radiometrically terrain-corrected, calibrated gamma0 SAR
  VV/VH backscatter) via Microsoft's Planetary Computer STAC catalog.
- Optical/thermal: Landsat Collection 2 Level-2 (NDVI, NDWI, land surface
  temperature) — used instead of Sentinel-2 because Sentinel-2 has no coverage
  before mid-2015 and several of these events predate that.

See [`data/real_feature_pipeline.py`](data/real_feature_pipeline.py) and the extracted
result at `data/processed/real_event_features.csv`.

**Models and results.** Random Forest and XGBoost, evaluated with grouped-by-district
and leave-one-event-out cross-validation ([`training/classical_baselines.py`](training/classical_baselines.py)):

| Model | Split | OOF AUPRC | OOF ROC-AUC |
|---|---|---|---|
| RandomForest | grouped-by-district | 0.730 | 0.851 |
| RandomForest | leave-one-event-out | 0.686 | 0.819 |
| XGBoost | grouped-by-district | 0.669 | 0.811 |
| XGBoost | leave-one-event-out | 0.728 | 0.855 |

(Random-guess AUPRC at this class balance — 24/77 positive — is ~0.31.)

A CNN-LSTM-style fusion model was also built and evaluated on identical folds
([`models/fusion_model_tabular.py`](models/fusion_model_tabular.py),
[`training/fusion_model_eval.py`](training/fusion_model_eval.py)). Its temporal branch
runs an LSTM+attention over the REAL 90-day daily NASA POWER sequence per event
(recovered from already-cached API responses — [`data/extract_daily_sequences.py`](data/extract_daily_sequences.py)),
not a pseudo-sequence of summary statistics; hyperparameters were picked via a
held-out internal validation split, and each fold's prediction is averaged over 5
random seeds to separate real signal from the high run-to-run variance a small neural
net has on n=77:

| Model | Split | OOF AUPRC | OOF ROC-AUC |
|---|---|---|---|
| Fusion (CNN-LSTM + attention, real sequences) | grouped-by-district | 0.565 | 0.778 |
| Fusion (CNN-LSTM + attention, real sequences) | leave-one-event-out | 0.644 | 0.770 |

An earlier version of the fusion model — trained on 9 pre-aggregated summary scalars
reshaped into a pseudo-sequence, one seed, no tuning — scored decisively worse
(grouped AUPRC 0.442; bootstrap CI vs. RandomForest **[−0.351, −0.084]**, entirely
negative). After giving it real daily data, tuning, and seed-averaging, its per-fold
AUPRC gap to RandomForest narrowed to **−0.019** with a bootstrapped 95% CI of
**[−0.164, 0.104]** — now *includes* zero, i.e. the two models are no longer
statistically distinguishable, even though RandomForest (0.730) is still numerically
ahead. **RandomForest remains the primary result** per the proposal's decision rule
(the CI doesn't cross zero in the fusion model's favor either), and the fusion model
is reported as a real negative finding reached after a genuine improvement attempt —
not the original under-tuned architecture. Full numbers:
[`results/classical_baseline_results.json`](results/classical_baseline_results.json),
[`results/fusion_model_results.json`](results/fusion_model_results.json).

**Feature attribution (real SHAP values, RandomForest).** Meteorological variables
dominate, not remote sensing:

1. `precip_max_mm` (max daily rainfall in the pre-outbreak window) — 0.079
2. `temp_mean_c` (mean temperature) — 0.053
3. `precip_sum_mm` (total rainfall) — 0.037
4. `precip_mean_mm` — 0.036
5. `rh_mean_pct` (mean relative humidity) — 0.025

This is a genuine finding, not an assumption carried over from the proposal — it
differs from what the earlier (fabricated) version of this project claimed.

---

## Known limitations, stated plainly

- n = 77 is small. All results use grouped/leave-one-event-out cross-validation
  specifically to guard against inflated performance from correlated events, but the
  confidence intervals on any individual number are wide.
- SAR features are missing for ~13% of events and optical/thermal for ~9%, mostly
  where a cloud-free scene wasn't available in a given window; these are
  median-imputed, not invented.
- The 24 wheat-blast "negative" districts (2016) are inferred (not in the outbreak
  report ⇒ assumed unaffected), not independently confirmed negative — flagged as
  such in `events.csv`'s `source` column.
- A finer-grained upazila-level rice blast dataset (72 records vs. the 24 district-year
  aggregates used here) exists at `data/raw_literature/sajbr2021/upazila_x_year_raw.csv`
  and is a planned follow-up robustness check, not yet integrated.

---

## Repository layout

```text
config/
  districts_real.json          # Real lat/lon for every district in events.csv
data/
  raw_literature/events.csv    # The 77-event label table (source of truth)
  raw_literature/sajbr2021/    # Upazila-level raw data + source citation
  raw_authentic/nasa_power/    # Cached real NASA POWER API responses
  real_feature_pipeline.py     # Real NASA POWER + Sentinel-1 RTC + Landsat extraction
  extract_daily_sequences.py   # Recovers real 90-day daily sequences from cached NASA POWER responses
  processed/real_event_features.csv    # Output: 77 rows x real Option A summary features
  processed/real_daily_sequences.npz   # Output: 77 x 90 x 6 real daily sequences (fusion model input)
  gee_pipeline.py              # Alternative GEE-based extraction path (unused, real, needs GEE auth)
models/
  fusion_model_tabular.py      # CNN-LSTM-style fusion model (documented negative finding)
training/
  classical_baselines.py       # RandomForest/XGBoost + grouped CV + SHAP (primary result)
  fusion_model_eval.py         # Fusion model eval + bootstrap CI vs. classical baseline
  train_final_model.py         # Fits and persists the production RandomForest
results/
  classical_baseline_results.json
  fusion_model_results.json
  checkpoints/random_forest_final.joblib
app/                           # FastAPI dashboard serving the real model + real events
tests/test_real_pipeline.py    # Label/feature/leakage/sanity checks
```

---

## Running it

### 1. Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Extract real features (network-bound, ~2-3 hours for all 77 events)
```bash
python3 -m data.real_feature_pipeline
```
Resumable — safe to re-run if interrupted; it skips events already in
`data/processed/real_event_features.csv`.

### 3. Run the classical baselines (primary result)
```bash
python3 training/classical_baselines.py
```

### 4. Run the fusion model comparison
```bash
python3 data/extract_daily_sequences.py   # recovers real daily sequences from cached NASA POWER data
python3 -m training.fusion_model_eval
```

### 5. Train the production model and launch the dashboard
```bash
python3 training/train_final_model.py
python3 -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to `http://127.0.0.1:8000` — the scenario dropdown loads the 77 real events,
predictions come from the real trained model, and the results/SHAP sections are read
live from the JSON files above.

### 6. Tests
```bash
python3 -m unittest discover tests
```
