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
| **RandomForest** | **grouped-by-district** | **0.757** | **0.822** |
| RandomForest | leave-one-event-out | 0.804 | 0.861 |
| RandomForest | leave-one-year-out | 0.332 | 0.438 |
| XGBoost | grouped-by-district | 0.774 | 0.838 |
| XGBoost | leave-one-event-out | 0.803 | 0.868 |
| XGBoost | leave-one-year-out | 0.338 | 0.343 |

(Random-guess AUPRC at this class balance — 46/125 positive — is ~0.368.)

RandomForest and XGBoost are statistically indistinguishable here (0.757 vs 0.774
grouped, a gap well inside the fold-to-fold SD of ≈0.28). RandomForest is reported as
primary — it is the model carried through the calibration analysis, the robustness
checks, the persisted production model, and the dashboard. XGBoost is the secondary
baseline and the (stronger) bar the fusion model is measured against.

**Is 0.76 real, or luck with 125 events and 5 folds?** A label-permutation test (300
shuffles, [`analysis/robustness_checks.py`](analysis/robustness_checks.py)) gives
**p = 0.003** — observed AUPRC 0.81 against a null-distribution mean of 0.39. But the
point estimate is imprecise: the cluster bootstrap over districts puts the 95% CI at
**[0.48, 0.91]** (event-level bootstrap [0.63, 0.87]). The honest headline is "AUPRC
≈ 0.76, 95% CI [0.48, 0.91], permutation p = 0.003" — a real signal, loosely pinned.

At the standard 0.5 threshold on real leave-one-event-out predictions, RandomForest
catches **30 of 46 real outbreaks** with **16 false alarms among the 46 flagged**
events — recall, precision and F1 all ≈ **0.65**. We emphasize recall because a missed
outbreak is more costly than an unnecessary fungicide application; precision is reported
alongside it so the cost of that trade-off is visible. A lower decision threshold trades
precision for recall.

### The neural fusion model — a documented negative finding

A CNN-LSTM-style fusion model was built and evaluated on identical folds
([`models/fusion_model_tabular.py`](models/fusion_model_tabular.py),
[`training/fusion_model_eval.py`](training/fusion_model_eval.py)). Its temporal branch
runs an LSTM+attention over the **real 90-day daily NASA POWER sequence** per event,
not a pseudo-sequence of summary statistics; hyperparameters are nested-tuned per fold
and each fold's prediction is averaged over 5 random seeds.

| Model | Split | OOF AUPRC | OOF ROC-AUC |
|---|---|---|---|
| Fusion (CNN-LSTM + attention) | grouped-by-district | 0.661 | 0.771 |
| Fusion (CNN-LSTM + attention) | leave-one-event-out | 0.739 | 0.810 |
| Fusion (CNN-LSTM + attention) | leave-one-year-out | 0.362 | 0.412 |

Two independent significance tests, both including zero (compared against the *stronger*
classical baseline, XGBoost at 0.774 — the harder benchmark for the fusion model):
- 5-fold AUPRC difference (fusion − XGBoost): **−0.049**, bootstrap 95% CI
  **[−0.177, 0.031]**.
- Paired per-event Brier score across all 125 leave-one-event-out predictions:
  **−0.025**, bootstrap 95% CI **[−0.063, 0.014]**.

**The classical baseline remains the primary result**, per the decision rule fixed in
advance. The fusion model has now been given: real 90-day daily meteorological sequences
(6 channels over the Dec–Mar window) alongside its 6 remote-sensing scalars, true nested
per-fold tuning, 5-seed averaging, and 62% more data (77 → 125 events) — and still does
not beat the classical baseline with any statistical margin. (The 6 preceding-monsoon
features added in Phase D went into the *classical* feature set, not the fusion model's
inputs — see the ablation and preceding-monsoon sections below.) This is reported as a
real negative finding after a genuine, methodologically hardened improvement attempt, not
the original under-tuned one.

### Which modality carries the signal? (ablation)

| Feature set | RandomForest AUPRC | XGBoost AUPRC |
|---|---|---|
| Remote sensing only (6 features) | 0.414 | 0.369 |
| Meteorology only (15 features) | 0.733 | 0.827 |
| Full (21 features) | 0.757 | 0.774 |

(The "full" row equals the headline primary result exactly — same features, order, CV,
and hyperparameters.)

**Meteorology dominates remote sensing by a wide margin** (0.73–0.83 vs 0.37–0.41) —
this is the stable finding across SHAP, Mann-Whitney U, this ablation, and fusion-model
permutation importance. At the earlier 14-feature version, meteorology-only also
*beat the full model* (the remote-sensing features actively diluted the signal); after
6 preceding-monsoon features were added to the meteorology set that gap is no longer
significant (paired cluster-bootstrap of meteorology-only − full = −0.024, 95% CI
[−0.057, 0.042]), so the full RandomForest model is kept as primary.

#### We tested the obvious objection, and it did not hold

The natural rebuttal to "remote sensing doesn't help" is that the *representation* was
too crude: `sar_vv_db_mean` is a mean over a ~13 km box, and a box mean cannot express
partial surface water at all (open water sits below about −15 dB, while the observed
box means cluster near −7 dB, so any water signal is averaged away). The proposal had
specified SAR flood detection; only mean backscatter was ever implemented.

So we implemented it properly — `water_extent_frac`, the *fraction* of pixels below the
open-water backscatter threshold, extracted for all 120 events with Sentinel-1 coverage
— and re-ran the ablation. It measurably is not redundant with the existing SAR columns
(strongest rank correlation with any existing feature is ρ = −0.548 against
`sar_vh_db_mean`), so it does carry information the mean discarded.

**It made no difference to outbreak prediction.** Remote-sensing-only AUPRC moved
0.412 → 0.414 for RandomForest and 0.399 → 0.369 for XGBoost; full-model performance
changed by −0.004 ± 0.013 per fold, against fold-to-fold variability of 0.28 — i.e.
indistinguishable from zero in either direction. The feature was kept in the reported
feature set rather than removed, so that this negative result is a test of the best
SAR representation available to us and not of a straw man.

This is the difference between "remote sensing didn't help" and "remote sensing didn't
help, and we checked that this wasn't just a representation artifact."

### Feature attribution

Four independent methods — SHAP, Mann-Whitney U with FDR correction, the modality
ablation above, and permutation importance on the fusion model — all agree that
**meteorological variables dominate over remote sensing**. Top SHAP values
(RandomForest): `precip_max_mm` (0.073), `precip_mean_mm` (0.041), `precip_sum_mm`
(0.035), `precip_anomaly_mm` (0.030), `rh_max_pct` (0.026), then two preceding-monsoon
features (`monsoon_precip_anomaly_mm`, `monsoon_rh_mean_pct`) — all above the top
remote-sensing feature. The strongest single test:
`precip_max_mm` differs between outbreak and non-outbreak events at
**p = 1.0 × 10⁻⁸** (FDR-corrected q < 0.0001, rank-biserial r = −0.616, "very large").

The direction is worth stating explicitly, because it is the opposite of the intuition
that "more rain means more disease": **outbreak events have roughly half the peak
rainfall of non-outbreak events** (median 12.6 mm vs 24.0 mm). The same inverse
relationship holds for severity among the rice-blast events (`precip_max_mm` vs leaf
blast severity, Spearman ρ = −0.579, p = 1.0 × 10⁻⁷). Occurrence and severity agree.
This is consistent with blast epidemiology: the fungus needs leaf wetness to infect,
but heavy rain physically washes conidia off the leaf surface, and every observation
window here is the Dec–Mar dry season, when the disease is driven by dew and irrigation
rather than rainfall.

This differs from what the earlier (fabricated) version of this project claimed — it
had reported SAR backscatter as the strongest predictor.

### The preceding monsoon carries a real, year-independent signal

Every feature above describes the Dec–Mar dry season the outbreak is *observed* in. But
the project's framing is post-calamity risk, so we also extracted NASA POWER + Sentinel-1
+ Landsat features over the Jun–Sep monsoon window immediately *before* each observation
window ([`data/extract_preseason_features.py`](data/extract_preseason_features.py)).

Six preceding-monsoon meteorology features passed pre-registered evidence gates
(independent of their dry-season analogue, association survives FDR correction, and
survives partial-correlation control for calendar year) and were added to the feature
set. On the 72 rice-blast events their univariate association with outbreak is large
(rank-biserial |r| 0.30–0.60, all q < 0.01) and **survives controlling for year** —
`monsoon_rh_mean` raw ρ −0.51 → partial ρ|year −0.53; year explains only 10–24% of these
features' variance, so they are mostly spatial, not calendar proxies. Direction: a
**drier, hotter preceding monsoon precedes rice blast** — the same water-stress story as
the dry-season finding. All monsoon *SAR and optical* features were rejected: monsoon SAR
has a genuine Sentinel-1 archive gap for 2014 that falls entirely on label-0 events
(imputing it would leak), and monsoon Landsat showed no FDR-surviving signal.

Folding the 6 features into the classifier gives a small, consistent, within-noise lift
(RandomForest grouped AUPRC 0.724 → 0.757, XGBoost 0.720 → 0.774; per-fold +0.031 ± 0.040
against fold SD 0.28, 4/5 folds up). It does **not** fix cross-year generalization
(leave-one-year-out 0.318 → 0.332, still below baseline). In the severity regression the
monsoon features are non-redundant — `monsoon_temp_mean_c` and `monsoon_precip_anomaly_mm`
survive collinearity reduction and enter the leaf-blast-severity model — but they raise
only in-sample R² (0.408 → 0.428), not held-out R² (~0.31, unchanged). The consistent
reading across the classifier and the regression: a **real univariate signal with no
robust incremental predictive value**. The feature set was frozen by the pre-registered
gates *before* any model number was seen, which is what makes even that modest, hedged
result trustworthy rather than fished.

### Robustness checks

[`analysis/robustness_checks.py`](analysis/robustness_checks.py), on the primary model:

| Check | Result |
|---|---|
| Label-permutation test (300 shuffles) | observed AUPRC 0.81 vs null mean 0.39, **p = 0.003** |
| Bootstrap 95% CI on grouped AUPRC | **[0.48, 0.91]** cluster-over-districts; [0.63, 0.87] event-level |
| Meteorology-only vs full model (paired bootstrap) | −0.024, 95% CI [−0.057, 0.042] — not distinguishable |
| Rice label-threshold sensitivity | AUPRC 0.73–0.81 across the plausible LBI/NBI cutoff band |
| Decision-curve analysis | model net benefit beats "treat all" and "treat none" at every threshold 0.10–0.60 |

The permutation test is the reassuring one (the model has genuinely learned something);
the wide bootstrap CI is the honest caveat (125 events pin the point estimate loosely).

---

## Known limitations, stated plainly

- **Cross-season generalization is not supported by this data.** Under leave-one-year-out
  cross-validation every model falls *below* the random baseline (best AUPRC 0.334 vs.
  baseline 0.368). Five seasons is too few, and their composition is unbalanced. The
  headline numbers show the model generalizes to unseen *places*, not unseen *years*.
- **n = 125 is still small, and the confidence interval proves it.** The bootstrap 95%
  CI on the primary grouped AUPRC is [0.48, 0.91] (cluster-over-districts). The
  permutation test (p = 0.003) confirms the model beats chance, but any single
  three-decimal number should be read as "≈ 0.76", and mid-table feature rankings shift
  between sample sizes.
- **The preceding-monsoon signal is rice-blast-specific and correlational.** It does not
  survive FDR when rice and wheat events are pooled, and "controlling for year" rests on
  only three rice seasons. It is reported as a real but modest, hedged finding.
- **The fusion model's attention weights contradict the domain hypothesis.** The model
  attends away from high-humidity days for rice blast and toward them for wheat blast —
  significant (p = 3.8 × 10⁻⁴, rank-biserial r = −0.37) but opposite to the
  epidemiological pairing the study hypothesized. This is reported as an open question
  and a reason for caution, not explained away.
- **SAR features are missing for 4% of events**, median-imputed within folds. This
  remaining missingness shows no significant association with the outcome label
  (Fisher's exact p = 0.157) or year (p = 0.482).
- **The wheat-blast "negative" districts (2016) are inferred** (not named in the
  outbreak report ⇒ assumed unaffected), not independently confirmed negative — flagged
  as such in `events.csv`'s `source` column.
- **Severity regression is only partly successful.** Leaf blast severity is weakly
  predictable (held-out R² ≈ 0.31); neck blast severity is not (held-out R² ≈ −0.07,
  worse than predicting the mean).

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
  add_water_extent_feature.py  # Backfills the dry-season SAR water-extent fraction
  extract_preseason_features.py # Extracts preceding-monsoon (Jun-Sep) monsoon_* features
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
  water_extent_exploration.py  # Evidence gate for the dry-season SAR water feature
  preseason_exploration.py     # Evidence gate for the preceding-monsoon features
  robustness_checks.py         # Bootstrap CI, permutation test, threshold sensitivity, decision curve
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
python3 data/add_water_extent_feature.py --features data/processed/real_event_features.csv
python3 data/extract_preseason_features.py --features data/processed/real_event_features.csv
```
Every extraction step is resumable — safe to re-run if interrupted.

### 3. Models and analysis
Order matters: `statistical_tests` must run before `severity_regression` (which reads
its output), and `classical_baselines` before `fusion_model_eval`.
```bash
python3 -m training.classical_baselines          # primary result
python3 -m training.rq2_ablations                # modality ablations
python3 -m analysis.statistical_tests            # classical statistics layer
python3 -m training.severity_regression          # continuous severity targets
python3 -m training.fusion_model_eval            # fusion comparison + decision rule
python3 -m training.rq3_fusion_interpretability  # permutation importance + attention
python3 -m analysis.calibration_plot             # calibration / reliability plot
python3 -m analysis.robustness_checks            # bootstrap CI, permutation test, decision curve
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

---

## License

Code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)).
Third-party data products retain their own terms from their respective providers, as
noted above and in `LICENSE`.

## Authors

Ejaz Chowdhury and Md Abdur Rahaman — BRAC University, CSE791 (Research Methodology),
Group 13.
