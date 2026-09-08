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

**Data — 125 labeled events.**
- **72 upazila-season rice blast events** — 24 upazilas across 8 districts, over three
  Boro seasons (2017–2019), from Mahmud et al., 2021, *South Asian Journal of
  Biological Research* 4(1):1–13 (CC BY 4.0). Upazila coordinates were looked up
  individually from each upazila's public geographic record, not estimated
  ([`config/upazila_gazetteer.json`](config/upazila_gazetteer.json) records a source
  URL per entry).
- **53 district-season wheat blast events** — pre-emergence (2015), outbreak (2016),
  and recurrence (2017) seasons, from Islam et al., 2016, *BMC Biology* 14:84.
- **46 of the 125 events carry a positive outbreak label** (36.8% base rate), applied
  via the same pre-registered threshold throughout (leaf blast incidence ≥ 15% OR neck
  blast incidence ≥ 8% for rice blast).

**Features.** For every event, real values were fetched for its actual location and
season window:
- Meteorology: NASA POWER Agroclimatology API (precipitation, relative humidity,
  temperature, vapor pressure deficit, wet-day persistence), with rainfall anomaly
  computed against a real 20-year (2001–2020) climatological baseline for that
  location and day-of-year range — not a self-referential baseline.
- Radar: Sentinel-1 RTC (radiometrically terrain-corrected, calibrated gamma0 SAR
  VV/VH backscatter) via Microsoft's Planetary Computer STAC catalog.
- Optical/thermal: Landsat Collection 2 Level-2 (NDVI, NDWI, land surface
  temperature) — used instead of Sentinel-2 because Sentinel-2 has no coverage
  before mid-2015 and several of these events predate that.

Missing values are left as `NaN` and median-imputed **within each cross-validation
fold** (imputer fit on training data only), never filled with invented numbers. After
a dedicated scene-recovery pass ([`data/retry_missing_scenes.py`](data/retry_missing_scenes.py),
which re-attempts reads that failed to transient network errors), missingness is
**4% for SAR and 0% for optical/thermal**.

### Primary result — classical baselines

Random Forest and XGBoost with **nested** hyperparameter tuning (selected independently
inside each outer fold, so no tuning information leaks into any test fold), evaluated
on three cross-validation schemes
([`training/classical_baselines.py`](training/classical_baselines.py)):

| Model | Split | OOF AUPRC | OOF ROC-AUC |
|---|---|---|---|
| **RandomForest** | **grouped-by-district** | **0.762** | **0.815** |
| RandomForest | leave-one-event-out | 0.784 | 0.846 |
| RandomForest | leave-one-year-out | 0.317 | 0.430 |
| XGBoost | grouped-by-district | 0.718 | 0.814 |
| XGBoost | leave-one-event-out | 0.766 | 0.848 |
| XGBoost | leave-one-year-out | 0.315 | 0.406 |

(Random-guess AUPRC at this class balance — 46/125 positive — is ~0.368.)

At the standard 0.5 threshold on real leave-one-event-out predictions, RandomForest
catches **32 of 46 real outbreaks (recall 0.70)** with 20 false alarms among 52 flagged
events (**precision 0.62**). We emphasize recall because a missed outbreak is more
costly than an unnecessary fungicide application; precision is reported alongside it so
the cost of that trade-off is visible.

### The neural fusion model — a documented negative finding

A CNN-LSTM-style fusion model was built and evaluated on identical folds
([`models/fusion_model_tabular.py`](models/fusion_model_tabular.py),
[`training/fusion_model_eval.py`](training/fusion_model_eval.py)). Its temporal branch
runs an LSTM+attention over the **real 90-day daily NASA POWER sequence** per event,
not a pseudo-sequence of summary statistics; hyperparameters are nested-tuned per fold
and each fold's prediction is averaged over 5 random seeds.

| Model | Split | OOF AUPRC | OOF ROC-AUC |
|---|---|---|---|
| Fusion (CNN-LSTM + attention) | grouped-by-district | 0.520 | 0.699 |
| Fusion (CNN-LSTM + attention) | leave-one-event-out | 0.645 | 0.765 |
| Fusion (CNN-LSTM + attention) | leave-one-year-out | 0.334 | 0.478 |

Two independent significance tests, both including zero:
- 5-fold AUPRC difference (fusion − RandomForest): **−0.102**, bootstrap 95% CI
  **[−0.297, 0.015]**.
- Paired per-event Brier score across all 125 leave-one-event-out predictions:
  **−0.027**, bootstrap 95% CI **[−0.057, 0.0007]**.

**RandomForest remains the primary result**, per the decision rule fixed in advance.
Notably, the gap *widened* when the dataset grew from 77 to 125 events — RandomForest
improved (0.692 → 0.762) while the fusion model did not (0.551 → 0.520). This
pre-empts the obvious objection: the neural model was not simply starved of data.
This is reported as a real negative finding reached after a genuine, methodologically
hardened improvement attempt.

### Which modality carries the signal? (ablation)

| Feature set | RandomForest AUPRC | XGBoost AUPRC |
|---|---|---|
| Remote sensing only (5 features) | 0.412 | 0.399 |
| **Meteorology only (9 features)** | **0.807** | **0.809** |
| Full (14 features) | 0.762 | 0.718 |

Meteorology alone *outperforms* the full model — the remote-sensing features dilute
rather than add signal at this sample size. This replicated and strengthened when the
dataset grew.

### Feature attribution

Four independent methods — SHAP, Mann-Whitney U with FDR correction, the modality
ablation above, and permutation importance on the fusion model — all agree that
**meteorological variables dominate over remote sensing**. Top SHAP values
(RandomForest): `precip_max_mm` (0.087), `precip_mean_mm` (0.045), `rh_max_pct`
(0.037), `precip_sum_mm` (0.035), `temp_mean_c` (0.027). The strongest single test:
`precip_max_mm` differs between outbreak and non-outbreak events at
**p = 1.0 × 10⁻⁸** (FDR-corrected q < 0.0001, rank-biserial r = 0.616, "very large").

This differs from what the earlier (fabricated) version of this project claimed — it
had reported SAR backscatter as the strongest predictor.

---

## Known limitations, stated plainly

- **Cross-season generalization is not supported by this data.** Under leave-one-year-out
  cross-validation every model falls *below* the random baseline (best AUPRC 0.334 vs.
  baseline 0.368). Five seasons is too few, and their composition is unbalanced. The
  headline numbers show the model generalizes to unseen *places*, not unseen *years*.
- **n = 125 is still small.** All results use grouped or leave-one-out cross-validation
  specifically to guard against inflated performance from correlated events, but
  confidence intervals on any individual number are wide, and mid-table feature
  rankings shift between sample sizes.
- **The fusion model's attention weights contradict the domain hypothesis.** The model
  attends away from high-humidity days for rice blast and toward them for wheat blast —
  significant (p = 7.3 × 10⁻⁷) but opposite to the epidemiological pairing the study
  hypothesized. This is reported as an open question and a reason for caution, not
  explained away.
- **SAR features are missing for 4% of events**, median-imputed within folds. This
  remaining missingness shows no significant association with the outcome label
  (Fisher's exact p = 0.157) or year (p = 0.482).
- **The wheat-blast "negative" districts (2016) are inferred** (not named in the
  outbreak report ⇒ assumed unaffected), not independently confirmed negative — flagged
  as such in `events.csv`'s `source` column.
- **Severity regression is only partly successful.** Leaf blast severity is predictable
  (held-out R² ≈ 0.31); neck blast severity is essentially not (held-out R² ≈ 0.05).

---

## Repository layout

```text
config/
  districts_real.json          # Real lat/lon for every district in events.csv
  upazila_gazetteer.json       # Real lat/lon for all 24 upazilas, with source URLs
data/
  raw_literature/events.csv    # Original 77-event district-level label table
  raw_literature/sajbr2021/    # Upazila-level survey data + source citation
  raw_authentic/nasa_power/    # Cached real NASA POWER API responses
  build_upazila_events.py      # Builds the 72-row upazila event table (reproducible)
  real_feature_pipeline.py     # Real NASA POWER + Sentinel-1 RTC + Landsat extraction
  retry_missing_scenes.py      # Re-attempts scene reads lost to transient network errors
  extract_daily_sequences.py   # Recovers real 90-day daily sequences from cached responses
  merge_upazila_dataset.py     # Merges upazila rice blast + wheat blast into 125 events
  processed/real_event_features.csv    # 125 rows x real Option A summary features
  processed/real_daily_sequences.npz   # 125 x 90 x 6 real daily sequences (fusion input)
models/
  fusion_model_tabular.py      # CNN-LSTM-style fusion model (documented negative finding)
training/
  classical_baselines.py       # RandomForest/XGBoost + nested CV + SHAP (primary result)
  fusion_model_eval.py         # Fusion model eval + bootstrap CIs vs. classical baseline
  rq2_ablations.py             # Remote-sensing-only vs. meteorology-only vs. full
  rq3_fusion_interpretability.py  # Permutation importance + attention-weight analysis
  severity_regression.py       # Continuous severity targets, classical stats + held-out CV
  train_final_model.py         # Fits and persists the production RandomForest
analysis/
  statistical_tests.py         # Descriptives, Mann-Whitney U + FDR, missingness, confusion matrix
  calibration_plot.py          # Calibration / reliability plot
results/                       # All result JSONs, figures, and the persisted model
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

### 2. (Optional) Rebuild the dataset from scratch
The processed dataset is already committed. Only do this to rebuild from source —
it is network-bound and takes several hours.
```bash
python3 data/build_upazila_events.py
python3 data/real_feature_pipeline.py \
  --events data/raw_literature/sajbr2021/upazila_events.csv \
  --out data/processed/real_upazila_features.csv
python3 data/retry_missing_scenes.py --features data/processed/real_upazila_features.csv
python3 data/extract_daily_sequences.py \
  --events data/raw_literature/sajbr2021/upazila_events.csv \
  --out data/processed/real_upazila_daily_sequences.npz
python3 data/merge_upazila_dataset.py
```
The feature pipeline is resumable — safe to re-run if interrupted.

### 3. Models and analysis
```bash
python3 -m training.classical_baselines          # primary result
python3 -m training.fusion_model_eval            # fusion comparison + decision rule
python3 -m training.rq2_ablations                # modality ablations
python3 -m training.rq3_fusion_interpretability  # permutation importance + attention
python3 -m training.severity_regression          # continuous severity targets
python3 -m analysis.statistical_tests            # classical statistics layer
python3 -m analysis.calibration_plot             # calibration / reliability plot
```

### 4. Dashboard
```bash
python3 -m training.train_final_model
python3 -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to `http://127.0.0.1:8000` — the scenario dropdown loads the real events,
predictions come from the real trained model, and the results/SHAP sections are read
live from the JSON files above.

### 5. Tests
```bash
python3 -m pytest tests/ -q
```

---

## Data and code availability

Rice blast data: Mahmud et al. (2021), *SAJBR* 4(1):1–13 (CC BY 4.0). Wheat blast
data: Islam et al. (2016), *BMC Biology* 14:84 (open access). Meteorology: NASA POWER
(<https://power.larc.nasa.gov/>, public domain). Imagery: Sentinel-1 RTC and Landsat
Collection 2 Level-2 via Microsoft Planetary Computer
(<https://planetarycomputer.microsoft.com/>). Cached raw NASA POWER responses are
included under `data/raw_authentic/`; satellite imagery is queried live rather than
redistributed.
