"""
Exploratory analysis of the pre-season (preceding-monsoon) features, run BEFORE
deciding which `monsoon_*` columns enter the model feature set.

Motivating hypothesis
---------------------
Every feature currently in the model describes the Dec-Mar dry season in which the
outbreak is *observed*. The project's framing is *post-calamity* risk -- the outbreak
follows something. In Bangladesh the dominant antecedent is the Jun-Sep monsoon of the
preceding calendar year. `data/extract_preseason_features.py` extracts the same real
quantities over that monsoon window. The specific, falsifiable questions:

  1. Non-degenerate and physically plausible? (distribution, missingness -- monsoon
     Landsat is expected to be badly cloud-limited, and the missing RATE is itself a
     result to report.)
  2. New information, or just a lagged copy of the dry-season features? (rank
     correlation of each monsoon_* column with its dry-season analogue -- a |rho| near
     1 means the model already had it.)
  3. Associated with outbreak OCCURRENCE? (Mann-Whitney U + rank-biserial, POSITIVE r =
     higher in outbreak events; pooled AND within rice-only, because rice/wheat differ
     in region, season and crop and a pooled test partly measures that.)
  4. Associated with blast SEVERITY / INCIDENCE? (Spearman on the 72 rice events, and
     partial Spearman controlling for the dry-season analogue -- the sharpest test of
     incremental value.)
  5. The directional hypothesis: does a WETTER / more anomalous preceding monsoon go
     with higher outbreak risk? Report the sign explicitly, don't just report |effect|.

Multiplicity: Benjamini-Hochberg FDR within each family of tests. A monsoon_* column is
recommended for integration only if it is (a) not redundant with a dry-season feature
and (b) shows an association that survives FDR in at least one of the occurrence /
severity families.

Usage (from repo root, after data/extract_preseason_features.py has populated the
monsoon_* columns):
    python3 -m analysis.preseason_exploration
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.statistical_tests import benjamini_hochberg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PreseasonExploration")

FEATURES_CSV = "data/processed/real_event_features.csv"
RESULTS_DIR = "results"

# Candidate monsoon features and, for each, the dry-season column it should be checked
# against for redundancy (None = no direct dry-season analogue).
MONSOON_CANDIDATES = {
    "monsoon_precip_sum_mm": "precip_sum_mm",
    "monsoon_precip_max_mm": "precip_max_mm",
    "monsoon_precip_anomaly_mm": "precip_anomaly_mm",
    "monsoon_rh_mean_pct": "rh_mean_pct",
    "monsoon_temp_mean_c": "temp_mean_c",
    "monsoon_vpd_mean_kpa": "vpd_mean_kpa",
    "monsoon_wet_persistence_max_days": "wet_persistence_max_days",
    "monsoon_water_extent_frac": "water_extent_frac",
    "monsoon_sar_vv_db_mean": "sar_vv_db_mean",
    "monsoon_ndvi_mean": "ndvi_mean",
    "monsoon_ndwi_mean": "ndwi_mean",
    "monsoon_lst_celsius_mean": "lst_celsius_mean",
}
SEVERITY_COLUMNS = ["leaf_blast_severity_pct", "neck_blast_severity_pct",
                    "leaf_blast_incidence_pct", "neck_blast_incidence_pct"]
MIN_N = 12


def effect_size_note(effect):
    effect = abs(effect)
    if effect < 0.1:
        return "negligible"
    if effect < 0.3:
        return "small"
    if effect < 0.5:
        return "medium"
    return "large"


def _partial_spearman(x, y, z):
    """Spearman(x, y) controlling for z, via correlation of rank residuals. Returns
    (rho, p, n). Answers 'does x track y beyond what z already explains?'."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    zc = np.column_stack([np.ones_like(rz), rz])
    res_x = rx - zc @ np.linalg.lstsq(zc, rx, rcond=None)[0]
    res_y = ry - zc @ np.linalg.lstsq(zc, ry, rcond=None)[0]
    rho, p = stats.pearsonr(res_x, res_y)
    return float(rho), float(p), int(len(rx))


def distribution_and_missingness(df):
    out = {}
    for col in MONSOON_CANDIDATES:
        if col not in df.columns:
            out[col] = {"note": "column not present"}
            continue
        s = df[col].dropna()
        if s.empty:
            out[col] = {"n_non_missing": 0, "note": "no values extracted"}
            continue
        out[col] = {
            "n_non_missing": int(s.size),
            "missing_pct": round(float(df[col].isna().mean() * 100), 1),
            "mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4), "median": round(float(s.median()), 4),
            "max": round(float(s.max()), 4), "n_unique": int(s.nunique()),
        }
    return out


def missingness_mechanism(df):
    """Is each monsoon_* column's missingness associated with the outbreak label or the
    event year? Sentinel-1 has no usable archive over Bangladesh for the 2014 monsoon
    (S1A launched Apr 2014, systematic South-Asia tasking ramped up later), and those
    2014-monsoon events are exactly the 12 pre-emergence wheat NEGATIVES -- so monsoon
    SAR is missing NOT at random, in a way that lines up with the label. A column that
    fails this check must NOT be median-imputed into the model feature set (the
    imputed value would carry label information); it can still be described on the
    subset that has it, with the caveat stated.
    """
    out = {}
    for col in MONSOON_CANDIDATES:
        if col not in df.columns:
            continue
        miss = df[col].isna().astype(int)
        n_miss = int(miss.sum())
        entry = {"n_missing": n_miss, "missing_pct": round(100 * n_miss / len(df), 1)}
        if 0 < n_miss < len(df):
            tab = pd.crosstab(miss, df["label"])
            if tab.shape == (2, 2):
                _, p_label = stats.fisher_exact(tab.values)
                entry["fisher_p_vs_label"] = round(float(p_label), 5)
                entry["missing_rate_label1"] = round(float(miss[df["label"] == 1].mean()), 3)
                entry["missing_rate_label0"] = round(float(miss[df["label"] == 0].mean()), 3)
            try:
                chi2, p_year, _, exp = stats.chi2_contingency(pd.crosstab(miss, df["year"]))
                entry["chi2_p_vs_year"] = round(float(p_year), 5)
                entry["min_expected_cell"] = round(float(exp.min()), 2)
            except ValueError:
                entry["chi2_p_vs_year"] = None
            entry["missing_not_at_random"] = bool(
                (entry.get("fisher_p_vs_label", 1) < 0.05)
                or (entry.get("chi2_p_vs_year") is not None and entry["chi2_p_vs_year"] < 0.05)
            )
        else:
            entry["missing_not_at_random"] = False
        out[col] = entry
    return out


def redundancy_vs_dryseason(df):
    out = {}
    for col, analogue in MONSOON_CANDIDATES.items():
        if col not in df.columns or analogue is None or analogue not in df.columns:
            continue
        pair = df[[col, analogue]].dropna()
        if len(pair) < MIN_N:
            out[col] = {"analogue": analogue, "note": "insufficient overlap"}
            continue
        rho, p = stats.spearmanr(pair[col], pair[analogue])
        out[col] = {
            "dry_season_analogue": analogue,
            "spearman_rho": round(float(rho), 4), "p_value": round(float(p), 6),
            "n": int(len(pair)),
            "verdict": ("largely redundant" if abs(rho) >= 0.8
                        else "moderately correlated" if abs(rho) >= 0.5
                        else "largely independent"),
        }
    return out


def year_confounding(df):
    """The decisive check. Rice outbreak labels are strongly year-clustered (this is
    why leave-one-year-out CV collapses), and a whole monsoon season's mean RH / temp /
    VPD is essentially ONE number per year per POWER grid cell -- so a monsoon_*
    meteorology column can be little more than a proxy for 'which year'. Grouped-by-
    district CV does not block year, so adding a year proxy would inflate the grouped
    AUPRC spuriously (the model learns 'monsoon 2017 => outbreak year').

    For each candidate, on the 72 rice events:
      - eta_sq_year: fraction of the feature's variance explained by year (one-way
        ANOVA). Near 1.0 => the feature basically IS the year.
      - partial Spearman of feature vs rice label, controlling for year. If the raw
        association does not survive removing year, the feature carries no
        outbreak information beyond the calendar.
    """
    rice = df[df["disease"] == "rice_blast"].copy()
    yr = rice["year"].values
    lab = rice["label"].values
    out = {}
    for col in MONSOON_CANDIDATES:
        if col not in rice.columns:
            continue
        s = rice[[col]].join(rice["year"]).dropna()
        if s[col].nunique() < 3 or len(s) < MIN_N:
            continue
        groups = [g[col].values for _, g in s.groupby("year")]
        grand = s[col].mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = ((s[col] - grand) ** 2).sum()
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else np.nan

        pair = rice[[col]].assign(year=yr, label=lab).dropna()
        prho, pp, _ = _partial_spearman(pair[col].values, pair["label"].values,
                                        pair["year"].values.astype(float))
        raw_rho, raw_p = stats.spearmanr(pair[col], pair["label"])
        out[col] = {
            "eta_sq_year": round(eta_sq, 3),
            "raw_spearman_vs_label": round(float(raw_rho), 4),
            "raw_p": round(float(raw_p), 5),
            "partial_spearman_vs_label_given_year": round(prho, 4),
            "partial_p_given_year": round(pp, 5),
            "survives_year_control": bool(abs(prho) >= 0.2 and pp < 0.05),
            "verdict": ("mostly a year proxy" if eta_sq >= 0.7 and not (abs(prho) >= 0.2 and pp < 0.05)
                        else "year-confounded, weak residual" if not (abs(prho) >= 0.2 and pp < 0.05)
                        else "signal survives year control"),
        }
    return out


def outcome_association(df, subset_name):
    pos, neg = df[df["label"] == 1], df[df["label"] == 0]
    rows = []
    for col in MONSOON_CANDIDATES:
        if col not in df.columns:
            continue
        a, b = pos[col].dropna().values, neg[col].dropna().values
        if len(a) < MIN_N // 2 or len(b) < MIN_N // 2:
            rows.append({"feature": col, "subset": subset_name,
                         "note": f"insufficient data (n_pos={len(a)}, n_neg={len(b)})"})
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # POSITIVE r = feature HIGHER in outbreak events (see analysis/statistical_tests.py).
        r_rb = (2 * u) / (len(a) * len(b)) - 1
        rows.append({
            "feature": col, "subset": subset_name,
            "n_outbreak": int(len(a)), "n_no_outbreak": int(len(b)),
            "median_outbreak": round(float(np.median(a)), 4),
            "median_no_outbreak": round(float(np.median(b)), 4),
            "U": float(u), "p_value": float(p),
            "rank_biserial_r": round(float(r_rb), 4),
            "effect_size_note": effect_size_note(r_rb),
            "direction": "higher in outbreak" if r_rb > 0 else "lower in outbreak",
        })
    tested = [r for r in rows if "p_value" in r]
    if tested:
        q = benjamini_hochberg([r["p_value"] for r in tested])
        for r, qv in zip(tested, q):
            r["fdr_q_value"] = round(float(qv), 6)
            r["survives_fdr"] = bool(qv < 0.05)
    return sorted(rows, key=lambda r: r.get("p_value", 1.0))


def severity_association(df):
    rice = df[df["disease"] == "rice_blast"]
    rows = []
    for col, analogue in MONSOON_CANDIDATES.items():
        if col not in rice.columns:
            continue
        for sev in SEVERITY_COLUMNS:
            if sev not in rice.columns:
                continue
            cols = [col, sev] + ([analogue] if analogue and analogue in rice.columns else [])
            pair = rice[cols].dropna()
            if len(pair) < MIN_N:
                continue
            rho, p = stats.spearmanr(pair[col], pair[sev])
            entry = {"feature": col, "outcome": sev, "n": int(len(pair)),
                     "spearman_rho": round(float(rho), 4), "p_value": float(p)}
            if analogue and analogue in pair.columns:
                prho, pp, _ = _partial_spearman(pair[col].values, pair[sev].values,
                                                pair[analogue].values)
                entry["partial_rho_controlling_dry_analogue"] = round(prho, 4)
                entry["partial_p_value"] = round(pp, 6)
                entry["retains_signal_after_control"] = bool(abs(prho) >= 0.15 and pp < 0.05)
            rows.append(entry)
    if rows:
        q = benjamini_hochberg([r["p_value"] for r in rows])
        for r, qv in zip(rows, q):
            r["fdr_q_value"] = round(float(qv), 6)
            r["survives_fdr"] = bool(qv < 0.05)
    return sorted(rows, key=lambda r: r["p_value"])


def recommend(dist, redundancy, occ_pooled, occ_rice, sev, missing, yearconf):
    """A monsoon_* column is recommended for integration only if ALL hold:
      - not too sparse (>= 60 non-missing),
      - its missingness is NOT label/year-associated (else median-imputing it leaks),
      - not largely redundant with a dry-season feature,
      - at least one association survives FDR (occurrence or rice severity/incidence),
      - AND its rice-label association survives controlling for YEAR -- otherwise it is
        a calendar proxy, and since grouped-by-district CV does not block year, adding
        it would inflate the grouped metric without adding real predictive content.
    """
    fdr_occ = {r["feature"] for r in (occ_pooled + occ_rice) if r.get("survives_fdr")}
    fdr_sev = {r["feature"] for r in sev if r.get("survives_fdr")}
    recs = {}
    for col in MONSOON_CANDIDATES:
        d = dist.get(col, {})
        if d.get("n_non_missing", 0) < 60:
            recs[col] = {"recommend": False, "reason": f"too sparse (n={d.get('n_non_missing', 0)})"}
            continue
        if missing.get(col, {}).get("missing_not_at_random"):
            m = missing[col]
            recs[col] = {"recommend": False,
                         "reason": (f"missing NOT at random (Fisher p vs label "
                                    f"{m.get('fisher_p_vs_label')}, chi2 p vs year "
                                    f"{m.get('chi2_p_vs_year')}) -- imputing it would leak")}
            continue
        red = redundancy.get(col, {}).get("verdict", "unknown")
        has_signal = col in fdr_occ or col in fdr_sev
        yc = yearconf.get(col, {})
        if red == "largely redundant":
            recs[col] = {"recommend": False, "reason": "largely redundant with dry-season analogue"}
        elif not has_signal:
            recs[col] = {"recommend": False, "reason": "no association survives FDR"}
        elif yc and not yc.get("survives_year_control"):
            recs[col] = {"recommend": False,
                         "reason": (f"year proxy: eta_sq(year)={yc.get('eta_sq_year')}, "
                                    f"partial rho|year={yc.get('partial_spearman_vs_label_given_year')} "
                                    f"(p={yc.get('partial_p_given_year')}) -- association is the calendar, "
                                    f"not the monsoon; would inflate grouped CV")}
        else:
            where = []
            if col in fdr_occ:
                where.append("outbreak occurrence")
            if col in fdr_sev:
                where.append("rice severity/incidence")
            recs[col] = {"recommend": True,
                         "reason": (f"independent signal, survives FDR for {', '.join(where)} "
                                    f"AND survives year control (partial rho|year="
                                    f"{yc.get('partial_spearman_vs_label_given_year')})")}
    return recs


def main():
    df = pd.read_csv(FEATURES_CSV)
    present = [c for c in MONSOON_CANDIDATES if c in df.columns]
    if not present:
        raise SystemExit("no monsoon_* columns in the feature CSV -- run data/extract_preseason_features.py first")
    n_pop = int(df["monsoon_precip_sum_mm"].notna().sum()) if "monsoon_precip_sum_mm" in df.columns else 0
    logger.info(f"{len(present)} monsoon_* columns present; {n_pop}/{len(df)} events have monsoon meteorology")
    if n_pop < 100:
        logger.warning(f"Only {n_pop} events populated -- extraction incomplete, results are PRELIMINARY.")

    dist = distribution_and_missingness(df)
    missing = missingness_mechanism(df)
    yearconf = year_confounding(df)
    redundancy = redundancy_vs_dryseason(df)
    occ_pooled = outcome_association(df, "all_events")
    occ_rice = outcome_association(df[df["disease"] == "rice_blast"], "rice_blast_only")
    sev = severity_association(df)
    recs = recommend(dist, redundancy, occ_pooled, occ_rice, sev, missing, yearconf)

    out = {
        "n_events": int(len(df)), "n_events_with_monsoon_meteorology": n_pop,
        "distribution_and_missingness": dist,
        "missingness_mechanism": missing,
        "year_confounding_rice": yearconf,
        "redundancy_vs_dryseason": redundancy,
        "outcome_association_pooled": occ_pooled,
        "outcome_association_rice_only": occ_rice,
        "severity_association_rice": sev,
        "integration_recommendation": recs,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "preseason_exploration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    logger.info("=" * 72)
    logger.info("MISSINGNESS MECHANISM (monsoon_* columns with any missing):")
    for col, m in missing.items():
        if m["n_missing"] == 0:
            continue
        flag = "  *NOT AT RANDOM*" if m.get("missing_not_at_random") else ""
        logger.info(f"  {col:32} missing {m['n_missing']:3} ({m['missing_pct']}%)  "
                    f"Fisher p vs label={m.get('fisher_p_vs_label')}  "
                    f"chi2 p vs year={m.get('chi2_p_vs_year')}{flag}")
    logger.info("YEAR CONFOUNDING (rice; eta_sq = fraction of feature variance that IS year):")
    for col, yc in yearconf.items():
        logger.info(f"  {col:32} eta_sq(year)={yc['eta_sq_year']:.2f}  raw rho={yc['raw_spearman_vs_label']:+.3f}  "
                    f"partial rho|year={yc['partial_spearman_vs_label_given_year']:+.3f} "
                    f"(p={yc['partial_p_given_year']:.4f})  -> {yc['verdict']}")
    logger.info("REDUNDANCY vs dry-season analogue:")
    for col, r in redundancy.items():
        if "spearman_rho" in r:
            logger.info(f"  {col:32} vs {r['dry_season_analogue']:20} rho={r['spearman_rho']:+.3f} -> {r['verdict']}")
    for key, label in [("outcome_association_pooled", "OUTBREAK (pooled)"),
                       ("outcome_association_rice_only", "OUTBREAK (rice only)")]:
        logger.info(f"{label}:")
        for r in out[key]:
            if "note" in r:
                continue
            flag = "  *FDR*" if r.get("survives_fdr") else ""
            logger.info(f"  {r['feature']:32} p={r['p_value']:.4f} q={r.get('fdr_q_value', float('nan')):.4f} "
                        f"r={r['rank_biserial_r']:+.3f} ({r['direction']}){flag}")
    logger.info("SEVERITY / INCIDENCE (rice; top 8 by p):")
    for r in sev[:8]:
        pc = r.get("partial_rho_controlling_dry_analogue")
        flag = "  *FDR*" if r.get("survives_fdr") else ""
        logger.info(f"  {r['feature']:28} vs {r['outcome']:26} rho={r['spearman_rho']:+.3f} "
                    f"p={r['p_value']:.4f} q={r.get('fdr_q_value', float('nan')):.4f} "
                    f"partial={pc if pc is None else f'{pc:+.3f}'}{flag}")
    logger.info("RECOMMENDATION:")
    for col, r in recs.items():
        mark = "ADD " if r["recommend"] else "skip"
        logger.info(f"  [{mark}] {col:32} {r['reason']}")
    logger.info("=" * 72)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    main()
