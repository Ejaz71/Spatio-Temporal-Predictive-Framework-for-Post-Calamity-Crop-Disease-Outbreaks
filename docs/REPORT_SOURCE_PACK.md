# Report Source Pack

**Purpose.** Single self-contained reference for writing the CSE791 report in a Claude
Project. It consolidates (a) every final number from `results/*.json`, (b) the methods,
(c) the development narrative and data-integrity decisions made while building this
version, and (d) which repo file backs each claim. It is *not* the report — it is the
material the report is written from. Where a number appears here it is the current,
regenerated value (feature set = 21 columns, n = 125, chain re-run 2026-09-10).

Companion files to upload alongside this one: `README.md` (public-facing summary),
`CLAUDE.md` (the long-form working notes, has a "Report-writing reference" section with
more detail than this pack), and the `results/*.json` files (raw numbers).

---

## 1. One-paragraph project summary

A spatio-temporal framework predicting **post-calamity rice-blast and wheat-blast
outbreak risk in Bangladesh** from real data only: literature-derived outbreak labels,
NASA POWER weather, and Sentinel-1 / Landsat satellite imagery accessed through the
Microsoft Planetary Computer. It is a **rebuild** of an earlier version whose pipeline
generated `np.random` synthetic features and reported them as authentic; that version
was deleted and every number here traces to a real source. The headline model is a
grouped-by-district, nested-tuned Random Forest (**grouped-CV AUPRC ≈ 0.76,
permutation-test p = 0.003, bootstrap 95% CI [0.48, 0.91]**). The dominant, repeatedly
triangulated finding is that **meteorology carries the predictive signal and remote
sensing adds little**, and that **less rain (not more) precedes and accompanies blast**
in these dry-season windows.

---

## 2. Data and provenance

| Item | Value | Source file |
|---|---|---|
| Events | **125** (event = one district-or-upazila × one season) | `data/processed/real_event_features.csv` |
| Positive / negative | **46 / 79** (positive rate 0.368) | same |
| Rice blast events | **72** — upazila-level, 24 upazilas × 3 seasons (2017–2019) | `data/raw_literature/sajbr2021/` (Mahmud et al. 2021, *SAJBR* 4(1):1–13, CC BY 4.0) |
| Wheat blast events | **53** — district-level (2015, 2016, 2017 seasons) | `data/raw_literature/events.csv` (Islam et al. 2016, *BMC Biology* 14:84) |
| Distinct districts / years | 34 / 5 (2015–2019) | — |
| Observation window | Dec 1 (year Y−1) → Mar 15 (Y) — the dry season the outbreak is *observed* in | `window_start` / `window_end` columns |
| Preceding-monsoon window | Jun 1 – Sep 30 of year Y−1 (the "calamity" the framing points at) | `data/extract_preseason_features.py` |
| Weather | NASA POWER Agroclimatology daily (0.5° grid), cached | `data/raw_authentic/nasa_power/` (247 + 125 JSON files committed) |
| SAR | Sentinel-1 RTC gamma0 VV/VH (dB), via Planetary Computer STAC | queried live, not redistributed |
| Optical / thermal | Landsat Collection 2 Level-2 → NDVI, NDWI, LST | queried live |

**Label rule.** Rice: `label = 1 if leaf_blast_incidence_pct ≥ 15 OR neck_blast_incidence_pct ≥ 8`
(the threshold from the source survey). Wheat: literature presence/absence — positives are
documented outbreak districts (2016 primary outbreak + 2017 recurrence), negatives are
"not in the outbreak zone" or pre-emergence (2015, before wheat blast existed in Asia).
The wheat "negatives" for 2016 are *inferred* (not named ⇒ assumed unaffected), flagged in
`events.csv`'s `source` column.

**Feature matrix (21 columns).**
- 9 dry-season meteorology: `precip_mean_mm`, `precip_max_mm`, `precip_sum_mm`,
  `precip_anomaly_mm`, `rh_mean_pct`, `rh_max_pct`, `temp_mean_c`, `vpd_mean_kpa`,
  `wet_persistence_max_days`
- 6 preceding-monsoon meteorology (Phase D): `monsoon_precip_sum_mm`,
  `monsoon_precip_anomaly_mm`, `monsoon_rh_mean_pct`, `monsoon_temp_mean_c`,
  `monsoon_vpd_mean_kpa`, `monsoon_wet_persistence_max_days`
- 6 remote sensing: `sar_vv_db_mean`, `sar_vh_db_mean`, `ndvi_mean`, `ndwi_mean`,
  `lst_celsius_mean`, `water_extent_frac` (fraction of the ~13 km box below the −15 dB
  open-water backscatter threshold)
- The 15 meteorology columns form `METEOROLOGY_COLUMNS`; the 6 RS columns form
  `REMOTE_SENSING_COLUMNS`; together they are `FULL_COLUMNS` (pinned by a test).
- Missingness: dry-season SAR 5/125 (≈4%, not label/year associated); everything else
  complete; monsoon SAR 12/125 missing (see §7, archive gap, deliberately NOT used as a
  feature).

---

## 3. Methods

- **Models.** Random Forest (`class_weight="balanced"`) — primary; XGBoost
  (`scale_pos_weight` per fold) — secondary; a CNN-LSTM + attention fusion model
  (`models/fusion_model_tabular.py`) — comparison. Fusion temporal branch is an
  LSTM+attention over the **real 90-day daily NASA POWER sequence** per event
  (`data/processed/real_daily_sequences.npz`), spatial branch is an MLP over the 6 RS
  summary columns.
- **Cross-validation, three schemes** (`training/classical_baselines.py`):
  1. `GroupKFold` by **district** (5 folds) — spatial blocking, the primary metric.
  2. **Leave-one-event-out** — per-event.
  3. **Leave-one-year-out** (`LeaveOneGroupOut` by year) — cross-season generalization test.
- **Nested tuning.** For grouped CV only, hyperparameters are selected inside each outer
  fold via one internal `GroupShuffleSplit` — no tuning information reaches the outer test
  fold. Small grids (4 candidates/model) by design given n.
- **Metrics.** Primary AUPRC (class imbalance); secondary ROC-AUC, F1@0.5, recall, Brier.
- **Attribution.** SHAP (TreeExplainer, out-of-fold); Mann-Whitney U + rank-biserial
  effect size + Benjamini-Hochberg FDR; permutation importance on the fusion model
  (forward-pass only); LSTM attention-weight analysis.
- **Rank-biserial sign convention:** `r = 2U/(n1·n2) − 1`; **positive r = feature higher
  in outbreak events**. (This was inverted in an earlier version — see §7.)

---

## 4. Final results — classification

### 4.1 Primary and secondary models (`results/classical_baseline_results.json`)

| Model | CV scheme | OOF AUPRC | OOF ROC-AUC |
|---|---|---|---|
| **RandomForest** | **grouped-by-district** | **0.757** | **0.822** |
| RandomForest | leave-one-event-out | 0.804 | 0.861 |
| RandomForest | leave-one-year-out | 0.332 | 0.438 |
| XGBoost | grouped-by-district | 0.774 | 0.838 |
| XGBoost | leave-one-event-out | 0.803 | 0.868 |
| XGBoost | leave-one-year-out | 0.338 | 0.343 |

- Random-guess AUPRC at this balance ≈ **0.368**.
- RF and XGB are **statistically indistinguishable** here (0.757 vs 0.774, gap ≪ the
  fold-to-fold SD of ≈ 0.28). RF is reported primary for continuity and calibration.
- **Data scaling (real, not noise):** at n = 77 the RF grouped AUPRC was 0.692; at n = 125
  it is 0.757. Feature-engineering scaling (14 → 15 → 21 features) moved it 0.762 → 0.724
  → 0.757 — all within per-fold noise; the headline number is ≈ flat across the whole
  feature effort.
- **Confusion matrix** (RF, real LOO predictions, 0.5 threshold): 32/46 outbreaks caught
  (**recall 0.70**), 20 false alarms among 52 flagged (**precision 0.62**, F1 0.65,
  TN 59, FP 20, FN 14, TP 32). Report emphasises recall (a missed outbreak costs more
  than an unnecessary spray); precision reported alongside so the trade-off is visible.

### 4.2 Modality ablation (`results/rq2_ablation_results.json`, grouped-by-district)

| Feature set | RandomForest AUPRC | XGBoost AUPRC |
|---|---|---|
| Remote sensing only (6) | 0.414 | 0.369 |
| Meteorology only (15) | 0.733 | 0.827 |
| Full (21) | 0.741 | 0.794 |

- **Meteorology dominates remote sensing** by a wide, stable margin across n = 77,
  n = 125 (14 features), and n = 125 (21 features).
- At the 14-feature version, meteorology-only also *beat* the full model (RS features
  diluted the signal). After the 6 monsoon features joined the meteorology set that gap
  is **no longer significant**: paired cluster-bootstrap of (meteorology-only − full) =
  **−0.024, 95% CI [−0.057, +0.042]**. So the full RF model stays primary.
- Model-specific wrinkle: adding the 6 monsoon features helped both full models but *hurt*
  RF meteorology-only (0.807 → 0.733) while helping XGB meteorology-only (0.809 → 0.827) —
  RF's known sensitivity to added correlated features. Report both directions.

### 4.3 The neural fusion model — a documented negative finding (`results/fusion_model_results.json`)

| Fusion model | grouped | LOEO | LOYO |
|---|---|---|---|
| OOF AUPRC | 0.661 | 0.739 | 0.362 |
| OOF ROC-AUC | 0.771 | 0.810 | 0.412 |

- Compared against the **stronger** classical baseline (XGBoost, 0.774 — the harder bar):
  - 5-fold AUPRC diff (fusion − XGBoost) = **−0.049**, bootstrap 95% CI **[−0.177, +0.031]**.
  - Paired per-event Brier across all 125 LOEO predictions = **−0.025**, 95% CI
    **[−0.063, +0.014]**.
- Both include zero ⇒ **no statistically defensible fusion advantage**. Per the
  pre-registered stage-3/4 decision rule the classical baseline stays primary. The fusion
  model was given real daily sequences, true nested tuning, 5-seed averaging, 62 % more
  data, and 6 extra temporal features, and still does not win — reported as a real
  negative after a genuine hardening attempt.

### 4.4 Feature attribution

- **SHAP (RF, in-sample fit on all 125, `results/classical_baseline_results.json`):**
  `precip_max_mm` 0.073, `precip_mean_mm` 0.041, `precip_sum_mm` 0.035,
  `precip_anomaly_mm` 0.030, `rh_max_pct` 0.026, **`monsoon_precip_anomaly_mm` 0.024**,
  **`monsoon_rh_mean_pct` 0.019**, `temp_mean_c` 0.019, `monsoon_wet_persistence_max_days`
  0.016, `rh_mean_pct` 0.015. Dry-season rainfall block holds the top 4; two
  preceding-monsoon features enter at 6–7, above every RS feature.
- **Pooled Mann-Whitney U + FDR (`results/statistical_tests.json`):** the rainfall family
  is the only thing surviving FDR — `precip_max_mm` **p = 1.0×10⁻⁸, q < 0.0001,
  r = −0.616 (very large)**; `precip_mean_mm` r = −0.491; `precip_sum_mm` r = −0.469;
  `precip_anomaly_mm` r = −0.460 (all q < 1×10⁻⁴). `wet_persistence_max_days` (p = 0.024)
  and `sar_vv_db_mean` (p = 0.036) are nominal but do **not** survive FDR. **No monsoon
  feature survives FDR pooled** (best `monsoon_rh_mean` q ≈ 0.40) — the monsoon signal is
  rice-specific and washes out when pooled with wheat.
- **Direction (verified against raw group medians, not just the formula):** outbreak
  events have **≈ half the peak rainfall** of non-outbreak events (median 12.6 mm vs
  24.0 mm). Same inverse relation for severity (`precip_max_mm` vs leaf-blast severity,
  Spearman ρ = −0.579, p = 1.0×10⁻⁷). Occurrence and severity agree — heavy rain washes
  conidia off the leaf; these are dry-season windows driven by dew and irrigation.
- **Fusion permutation importance (`results/rq3_fusion_interpretability.json`):**
  `rh_pct` +0.231, `water_extent_frac` +0.119 (RS, #2 — see caveat), `precip_mm` +0.113,
  `vpd_kpa` +0.093, `temp_c` +0.066, `ndwi_mean` +0.052, `sar_vv_db_mean` +0.036. The
  `water_extent_frac` #2 rank is an in-sample, 6-feature-branch artifact — it does NOT
  contradict the classical result that the dry-season water fraction adds no CV value;
  report it, do not over-read it.
- **Attention hypothesis (fusion, rice n=72 vs wheat n=53):** the proposal hypothesised
  rice↔humidity and wheat↔temperature. The model does the opposite —
  `rh_attention_excess`: rice median −1.96, wheat +2.63, **p = 3.8×10⁻⁴, r = −0.37**;
  `temp_attention_excess`: rice +1.84 °C, wheat +1.34 °C, **p = 2.2×10⁻⁴, r = +0.39**.
  Report as: *the trained model's attention does not implement the proposal's
  disease–climate pairing* — a statement about what the model learned, not direct
  evidence the epidemiology is backwards. A reason for caution about deploying the fusion
  model, reinforcing the RF-primary decision.

---

## 5. Final results — severity regression (`results/severity_regression_results.json`)

Rice-only (labels are rice-only by construction). Predictors drawn from the significant
Phase-A Spearman correlations, then reduced to one per collinear cluster (|Pearson r| > 0.8).

| Target | kept features | multiple-OLS R² | held-out LOO R² |
|---|---|---|---|
| `leaf_blast_severity_pct` (n=72) | `monsoon_temp_mean_c`, `precip_max_mm`, `wet_persistence_max_days`, `precip_mean_mm`, `monsoon_precip_anomaly_mm`, `rh_max_pct`, `vpd_mean_kpa` | 0.428 | **0.30** (LinReg) / 0.31 (Ridge) |
| `neck_blast_severity_pct` (n=67) | `sar_vv_db_mean`, `temp_mean_c`, `monsoon_temp_mean_c`, `precip_mean_mm`, `precip_max_mm`, `vpd_mean_kpa` | 0.223 | **−0.07** (both) |

- Leaf-blast severity is **weakly predictable** (held-out R² ≈ 0.30); neck-blast severity
  is **not** (held-out R² negative — worse than predicting the mean).
- The 2 monsoon features in the leaf model are **non-redundant** (survive the collinearity
  cull) but raise only *in-sample* R² (0.408 → 0.428); held-out R² is flat. Same pattern
  as the classifier — real univariate signal, no robust incremental predictive value.
- Overfitting-gap story across sample sizes: n=24 → in-sample 0.608 vs held-out 0.347
  (gap 0.26); n=72 base features → 0.408 vs 0.313 (gap 0.10); n=72 + monsoon → 0.428 vs
  0.31. The held-out number was the trustworthy one throughout.

---

## 6. Robustness / rigor layer (`results/robustness_checks.json`, `analysis/robustness_checks.py`)

Five checks on the primary RF grouped-by-district CV. **None changes the point estimate;
they quantify how much to trust it.**

| Check | Result | Reading |
|---|---|---|
| **Label-permutation test** (300 shuffles, fixed HP both sides) | observed AUPRC 0.809 vs null mean 0.386, **p = 0.0033** | The model beats chance decisively. The headline rigor result. |
| **Bootstrap 95% CI on grouped AUPRC** | **[0.48, 0.91]** (cluster bootstrap over districts, design-matched); [0.63, 0.87] (plain event bootstrap, context) | Wide. The point estimate is real but loosely pinned at n = 125. Report "AUPRC ≈ 0.76, 95% CI [0.48, 0.91]". |
| **Meteorology-only vs full** (paired cluster-bootstrap) | −0.024, 95% CI [−0.057, +0.042] | Not distinguishable (was significant at 14 features). Keep the full RF model as primary. |
| **Rice label-threshold sensitivity** | plausible band (LBI 12–20, NBI 5–8): AUPRC **0.73–0.81**; full grid incl. implausible NBI≥10: 0.59–0.83 | Result is not a cutoff artifact; the baseline rule sits at the favourable end (0.81). |
| **Decision-curve analysis** (`results/figures/decision_curve.png`) | model net benefit > "treat all" and "treat none" at **every threshold 0.10–0.60** | The model is decision-useful across the whole plausible operating range, not just AUC-respectable. |

---

## 7. Development narrative and data-integrity log (the "this chat" content)

This is the audit trail a reviewer will want — what was fixed, what was checked, what was
deliberately not done, and why. Each item is defensible and disclosed.

### 7.1 The rebuild
The prior version's feature pipeline used `np.random` and reported the output as
authentic; its results (1.0/1.0/1.0 style scores) were fabricated. That code, data, and
docs were deleted. The current pipeline emits a real value or `NaN` + a logged warning —
never a synthetic fallback. A regression test guards against the perfect-score failure
mode.

### 7.2 Climatology anomaly bug (fixed)
`precip_anomaly_mm` equalled `precip_mean_mm` for every event because the climatological
baseline used the *median* of a zero-inflated dry-season daily series (~80 % zero days →
median 0.0). Fixed to the mean (still real, from the same cached 20-year NASA POWER
series). A second, worse variant (self-referential median of the event's own window) in
`extract_daily_sequences.py` was fixed to reuse the corrected function.

### 7.3 Landsat missingness "not at random" (resolved)
At n = 77 all 4 zero-Landsat events were the 2017 wheat recurrence season and all 4 were
positive (Fisher p = 0.008) — biased imputation. `data/retry_missing_scenes.py` re-ran
every zero-scene event and recovered all of them; root cause was transient network
failure during extraction, not scene unavailability. Landsat missingness is now 0/125,
SAR 5/125 (no label/year association).

### 7.4 Fusion model hardening
Nested per-fold tuning added to **both** models (was fusion-only), symmetric tuning
budget, paired per-event Brier test, 5-seed averaging with empirical seed-variance
reporting. The fusion model still does not beat the classical baseline (§4.3).

### 7.5 SAR water-extent feature — a tested-and-closed representation objection
Objection: `sar_vv_db_mean` is a box **mean**, which cannot express partial surface water,
so remote sensing might be failing on representation, not physics. Response: implemented
`water_extent_frac` (fraction of pixels below −15 dB), extracted for all 120 events with
S1 coverage. It is **not redundant** with existing SAR columns (strongest correlation with
any existing feature ρ = −0.548). It made **no difference**: remote-sensing-only AUPRC
0.412 → 0.414 (RF); full-model change within per-fold noise. **Kept in the feature set**
(removing after seeing it fail = post-hoc selection). Conclusion: the modality is not
representation-limited at this sample size — "remote sensing didn't help, *and we checked
it wasn't a representation artifact*".

### 7.6 Rank-biserial sign bug (fixed 2026-09-09)
The effect size was coded `r = 1 − 2U/(n1·n2)`, which is the correct magnitude with the
**sign inverted** — it reported negative r whenever outbreak events had the larger values,
flipping the stated direction of every effect. Magnitudes and p-values were never
affected; no model or CV number depends on it. Corrected to `2U/(n1·n2) − 1`. Verified
against raw group medians (peak rainfall 12.6 mm outbreak vs 24.0 mm non-outbreak — the
claim it had produced, "more rain → more outbreak", was backwards). Guarded by
`tests/test_real_pipeline.py::TestEffectSizeSignConvention`. **This retired an earlier
"counter-intuitive divergence" narrative** (occurrence and severity were said to disagree)
— they agree; rain suppresses blast for both.

### 7.7 Preceding-monsoon features — pre-registered evidence gating
`data/extract_preseason_features.py` extracts `monsoon_*` features over the Jun–Sep window
before each observation window. `analysis/preseason_exploration.py` applied gates
**before any model number was seen**: (1) not too sparse, (2) missingness not
label/year-confounded, (3) not redundant with the dry-season analogue, (4) association
survives FDR, (5) association survives partial-correlation control for **year**.
- **6 NASA POWER features passed** and were integrated (`monsoon_precip_sum_mm`,
  `monsoon_precip_anomaly_mm`, `monsoon_rh_mean_pct`, `monsoon_temp_mean_c`,
  `monsoon_vpd_mean_kpa`, `monsoon_wet_persistence_max_days`).
- **All monsoon SAR/optical columns rejected.** Monsoon SAR is missing for exactly the 12
  `WB_*_2015` events (monsoon window = 2014) — a genuine **Sentinel-1 archive gap** (S1A
  launched Apr 2014, South-Asia tasking ramped up later), and those 12 are all label 0, so
  missingness is perfectly confounded with the label (Fisher p = 0.004) — imputing it
  would leak. Monsoon optical: no FDR-surviving association.
- **Year-confounding check.** Concern: a monsoon-season mean could be a proxy for "which
  year", and grouped CV does not block year. Result: `eta_sq(year)` is only 0.10–0.49 for
  the 6 features (year explains a minority of their variance — they are mostly *spatial*),
  and partial Spearman vs the rice label controlling for year barely moves from the raw
  value (`monsoon_rh_mean` raw −0.51 → partial −0.53; `monsoon_temp` +0.48 → +0.48). Not
  calendar proxies.
- **Effect of integration.** Univariate: large, year-independent (rice |r| 0.30–0.60,
  all q < 0.01; direction = **drier, hotter preceding monsoon → rice blast**, consistent
  with the dry-season story). Classifier: RF full 0.724 → 0.757, XGB 0.720 → 0.794 —
  within per-fold noise (+0.031 ± 0.040 vs SD 0.28), consistent (4/5 folds up). LOYO
  0.318 → 0.332 (nudged up ⇒ not pure year leakage) but still below the 0.368 baseline.
  Severity regression: non-redundant, no held-out gain. **Net: a real signal with no
  robust incremental predictive value — reported honestly as such.** The value is what it
  established (the preceding monsoon carries genuine information), not the AUPRC.

### 7.8 Deliberately NOT done
- **Option B** (raster-patch CNN input) — dropped by decision; Option A summary-statistic
  features remain the representation.
- **Sub-season observation windows**, **CV-nested feature selection**, **hierarchical
  year random-effects model** — proposed, not adopted; available as future work.

### 7.9 Process lessons (worth a methods sentence)
- Never nest a shell `&` inside a backgrounded job (a completion notification once fired
  for the wrapper, not the real process, and a stale DataFrame overwrote a merge).
- A timeout helper that wrapped calls in `with ThreadPoolExecutor(...)` blocked on the
  abandoned thread at block exit — the timeout was decorative (one event took 56 min under
  a 100 s cap). Replaced with a raw daemon thread + `join(timeout)`.
- Planetary Computer SAS tokens live ~45 min and were stamped at STAC-search time; under a
  slow link they expired mid-read → HTTP 403 storms. Now re-signed at point of use. Net
  extraction throughput ~5 → ~78 events/hour.

---

## 8. Limitations (paste-ready, tighten as needed)

1. **Cross-season generalization is not supported.** Leave-one-year-out AUPRC is 0.33
   (RF) / 0.34 (XGB), below the 0.368 baseline. Five seasons, unbalanced (2016 alone is
   33 wheat events; rice spans only 2017–2019). The model generalises to unseen *places*,
   not unseen *years*. This is the single strongest argument for a "more seasons" future
   work item.
2. **n = 125 is small and the CI proves it.** Bootstrap 95% CI on the primary AUPRC is
   [0.48, 0.91]; the permutation test (p = 0.003) confirms the signal is real, but any
   3-decimal number should be read as "≈ 0.76", and mid-table feature rankings shift
   between sample sizes.
3. **The preceding-monsoon signal is rice-specific and correlational.** It does not
   survive FDR when rice and wheat are pooled, and "controlling for year" rests on only
   three rice seasons.
4. **The fusion model's attention contradicts the domain hypothesis** (§4.4) — reported
   as an open question, not explained away.
5. **Remote-sensing bounding box is coarse** (~±6–7 km around a district/upazila
   centroid) — plausibly too imprecise for a clean field-level signal, unlike the
   point-based weather query.
6. **Wheat "negative" districts (2016) are inferred**, not independently confirmed.
7. **Severity regression is only partly successful** — leaf-blast severity held-out
   R² ≈ 0.30, neck-blast severity held-out R² < 0.

---

## 9. Data and Code Availability statement (paste-ready)

> All code is at <https://github.com/Ejaz71/Spatio-Temporal-Predictive-Framework-for-Post-Calamity-Crop-Disease-Outbreaks>
> under the MIT License. Rice-blast outbreak data: Mahmud et al. (2021), *SAJBR*
> 4(1):1–13 (CC BY 4.0). Wheat-blast data: Islam et al. (2016), *BMC Biology* 14:84
> (open access). Meteorology: NASA POWER (<https://power.larc.nasa.gov/>, public domain);
> cached raw API responses are included in the repository. Satellite imagery: Sentinel-1
> RTC and Landsat Collection 2 Level-2 accessed via the Microsoft Planetary Computer
> (<https://planetarycomputer.microsoft.com/>) and queried live rather than
> redistributed. Every reported number is regenerated by the scripts in `training/` and
> `analysis/` from `data/processed/real_event_features.csv`; run order is documented in
> the README.

---

## 10. Which file backs which section

| Section | Repo files |
|---|---|
| §2 data / provenance | `data/processed/real_event_features.csv`, `data/raw_literature/`, `config/*.json`, `data/build_upazila_events.py`, `data/real_feature_pipeline.py`, `data/extract_preseason_features.py` |
| §3 methods | `training/classical_baselines.py`, `models/fusion_model_tabular.py`, `training/fusion_model_eval.py` |
| §4.1 primary models | `results/classical_baseline_results.json` |
| §4.2 ablation | `training/rq2_ablations.py`, `results/rq2_ablation_results.json` |
| §4.3 fusion | `results/fusion_model_results.json` (see `stage_3_4_decision`) |
| §4.4 attribution | `analysis/statistical_tests.py`, `results/statistical_tests.json`, `training/rq3_fusion_interpretability.py`, `results/rq3_fusion_interpretability.json` |
| §5 severity | `training/severity_regression.py`, `results/severity_regression_results.json` |
| §6 robustness | `analysis/robustness_checks.py`, `results/robustness_checks.json`, `results/figures/decision_curve.png` |
| §6 calibration | `analysis/calibration_plot.py`, `results/calibration_summary.json`, `results/figures/calibration_plot.png` |
| §7.7 monsoon gating | `analysis/preseason_exploration.py`, `results/preseason_exploration.json` |
| §7.5 water extent | `analysis/water_extent_exploration.py`, `results/water_extent_exploration.json` |
| Longer working notes | `CLAUDE.md` (§ "Report-writing reference", "Known issues", "Statistical rigor checklist") |
