"""
Phase A statistical rigor layer, per the CSE791 Week 6 methodology checklist
(descriptives -> hypothesis tests with effect sizes -> correlation -> ML evaluation,
held to a "no p-value without an effect size" discipline). Complements, rather than
replaces, the SHAP-based feature attribution in classical_baselines.py:

1. Descriptive statistics (mean/SD or median/IQR depending on skew) for all 14 real
   features, by outbreak label.
2. Missingness-mechanism check: is SAR/Landsat scene unavailability random, or
   associated with label/year? (Fisher's exact, small expected cell counts.)
3. Mann-Whitney U (not t-test — features are skewed, confirmed via Shapiro-Wilk below)
   + rank-biserial effect size, per feature, outbreak vs. non-outbreak. Benjamini-
   Hochberg FDR correction applied across the 14 tests (don't run 14 tests and report
   whichever has the smallest p-value).
4. Spearman correlation, each feature vs. rice blast severity % (the 24 rice-blast
   rows that carry a continuous severity value) — precursor to the severity
   regression task, not a replacement for it.
5. Confusion matrix at the standard 0.5 threshold for RandomForest's real
   leave-one-event-out predictions, with precision/recall/F1 and an explicit written
   justification for prioritizing recall in this domain.
"""

import json
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StatisticalTests")

FEATURES_CSV = "data/processed/real_event_features.csv"
CLASSICAL_RESULTS_JSON = "results/classical_baseline_results.json"
RESULTS_DIR = "results"

FEATURE_COLUMNS = [
    "precip_mean_mm", "precip_max_mm", "precip_sum_mm", "precip_anomaly_mm",
    "rh_mean_pct", "rh_max_pct", "temp_mean_c", "vpd_mean_kpa", "wet_persistence_max_days",
    "sar_vv_db_mean", "sar_vh_db_mean", "ndvi_mean", "ndwi_mean", "lst_celsius_mean",
]
SEVERITY_COLUMNS = ["leaf_blast_severity_pct", "neck_blast_severity_pct"]


def descriptive_statistics(df):
    rows = []
    for col in FEATURE_COLUMNS:
        vals = df[col].dropna().values
        skew = float(stats.skew(vals)) if len(vals) > 2 else np.nan
        if 3 <= len(vals) <= 5000:
            w_stat, w_p = stats.shapiro(vals)
        else:
            w_stat, w_p = np.nan, np.nan
        note = "approximately symmetric"
        if not np.isnan(skew):
            note = "right-skewed" if skew > 1 else ("left-skewed" if skew < -1 else note)
        rows.append({
            "feature": col, "n": int(len(vals)),
            "missing_pct": round(float(df[col].isna().mean() * 100), 1),
            "mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals, ddof=1)), 4),
            "median": round(float(np.median(vals)), 4),
            "iqr_low": round(float(np.percentile(vals, 25)), 4),
            "iqr_high": round(float(np.percentile(vals, 75)), 4),
            "skewness": round(skew, 3) if not np.isnan(skew) else None,
            "shapiro_w": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
            "shapiro_p": round(float(w_p), 6) if not np.isnan(w_p) else None,
            "distribution_note": note,
        })
    return rows


def missingness_mechanism_check(df):
    """Is SAR / Landsat scene unavailability associated with the outbreak label, or
    with which year an event falls in? Small expected cell counts throughout (this is
    n=77 data), so Fisher's exact rather than chi-square, per the lecture's own
    parametric/non-parametric decision rule."""
    results = {}
    for missing_col, feature_name in [("n_s1_scenes", "SAR"), ("n_landsat_scenes", "Landsat")]:
        is_missing = (df[missing_col] == 0).astype(int)
        table_label = pd.crosstab(is_missing, df["label"])
        odds_ratio, p_label = stats.fisher_exact(table_label.values) if table_label.shape == (2, 2) else (np.nan, np.nan)

        # Year effect via Fisher's exact isn't well-defined for >2x2; use chi-square
        # with a caveat if any expected cell count is small.
        table_year = pd.crosstab(is_missing, df["year"])
        try:
            chi2, p_year, dof, expected = stats.chi2_contingency(table_year)
            min_expected = float(expected.min())
        except ValueError:
            chi2, p_year, min_expected = np.nan, np.nan, np.nan

        results[feature_name] = {
            "n_missing": int(is_missing.sum()),
            "vs_label": {"odds_ratio": odds_ratio, "p_value": p_label, "test": "Fisher's exact"},
            "vs_year": {"chi2": chi2, "p_value": p_year, "min_expected_cell_count": min_expected,
                        "test": "chi-square (caveat: small expected counts likely given n=77)"},
        }
    return results


def benjamini_hochberg(p_values):
    """Returns FDR-corrected q-values, same order as input."""
    p = np.array(p_values)
    n = len(p)
    order = np.argsort(p)
    ranked_p = p[order]
    q = ranked_p * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    q_values = np.empty(n)
    q_values[order] = np.clip(q, 0, 1)
    return q_values.tolist()


def mann_whitney_by_outcome(df):
    """Per-feature Mann-Whitney U comparing outbreak (label=1) vs non-outbreak (label=0)
    events, with rank-biserial effect size — a classical-statistics complement to the
    SHAP ranking in classical_baseline_results.json, not a replacement for it."""
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    rows = []
    for col in FEATURE_COLUMNS:
        a = pos[col].dropna().values
        b = neg[col].dropna().values
        if len(a) < 2 or len(b) < 2:
            rows.append({"feature": col, "note": "insufficient non-missing data for this test"})
            continue
        u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial correlation: r = 1 - 2U / (n1 * n2)
        r_rb = 1 - (2 * u_stat) / (len(a) * len(b))
        rows.append({
            "feature": col, "n_outbreak": int(len(a)), "n_no_outbreak": int(len(b)),
            "median_outbreak": round(float(np.median(a)), 4), "median_no_outbreak": round(float(np.median(b)), 4),
            "U": float(u_stat), "p_value": float(p_val), "rank_biserial_r": round(float(r_rb), 4),
        })

    p_vals = [r["p_value"] for r in rows if "p_value" in r]
    q_vals = benjamini_hochberg(p_vals)
    q_iter = iter(q_vals)
    for r in rows:
        if "p_value" in r:
            r["fdr_q_value"] = round(next(q_iter), 6)
            effect = abs(r["rank_biserial_r"])
            r["effect_size_note"] = "small" if effect < 0.1 else ("medium" if effect < 0.3 else "large") if effect < 0.5 else "very large"
    return sorted(rows, key=lambda r: r.get("p_value", 1.0))


def severity_correlation(df):
    """Spearman correlation between each real feature and rice blast severity % (the
    24 rice-blast rows with a continuous severity value) — precedes, and motivates,
    the severity regression task rather than replacing it."""
    rice = df[df["disease"] == "rice_blast"]
    results = {}
    for sev_col in SEVERITY_COLUMNS:
        sev_results = []
        for col in FEATURE_COLUMNS:
            sub = rice[[col, sev_col]].dropna()
            if len(sub) < 4:
                continue
            rho, p_val = stats.spearmanr(sub[col], sub[sev_col])
            sev_results.append({"feature": col, "n": int(len(sub)), "spearman_rho": round(float(rho), 4),
                                 "p_value": float(p_val)})
        results[sev_col] = sorted(sev_results, key=lambda r: -abs(r["spearman_rho"]))
    return results


def confusion_matrix_report():
    with open(CLASSICAL_RESULTS_JSON) as f:
        classical = json.load(f)
    df = pd.read_csv(FEATURES_CSV)
    y_true = df["label"].astype(int).values
    y_prob = np.array(classical["RandomForest"]["leave_one_event_out"]["oof_probs"])
    valid = ~np.isnan(y_prob)
    y_true, y_prob = y_true[valid], y_prob[valid]
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    reporting_sentence = (
        f"At the standard 0.5 threshold on real leave-one-event-out predictions (n={len(y_true)}), "
        f"RandomForest achieves recall={recall:.2f} (catches {tp}/{tp+fn} real outbreaks) and "
        f"precision={precision:.2f} ({fp} false alarms among {tp+fp} flagged events). We emphasize "
        f"recall because a missed outbreak (false negative) is more costly than an unnecessary "
        f"fungicide/bactericide application prompted by a false alarm; precision is reported "
        f"alongside it so the cost of that trade-off is visible, not hidden behind a single metric."
    )

    return {
        "threshold": 0.5, "n_events": int(len(y_true)),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp),
                              "false_negative": int(fn), "true_positive": int(tp)},
        "precision": round(float(precision), 4), "recall": round(float(recall), 4), "f1": round(float(f1), 4),
        "reporting_sentence": reporting_sentence,
    }


def main():
    df = pd.read_csv(FEATURES_CSV)

    results = {
        "n_events": len(df),
        "descriptive_statistics": descriptive_statistics(df),
        "missingness_mechanism_check": missingness_mechanism_check(df),
        "mann_whitney_outbreak_vs_none": mann_whitney_by_outcome(df),
        "severity_correlations": severity_correlation(df),
        "confusion_matrix_report": confusion_matrix_report(),
    }

    logger.info("Top Mann-Whitney U results (by raw p-value):")
    for r in results["mann_whitney_outbreak_vs_none"][:5]:
        if "p_value" in r:
            logger.info(f"  {r['feature']}: U={r['U']:.1f}, p={r['p_value']:.4f}, q(FDR)={r['fdr_q_value']:.4f}, "
                        f"rank-biserial r={r['rank_biserial_r']:.3f} ({r['effect_size_note']})")

    logger.info(results["confusion_matrix_report"]["reporting_sentence"])

    out_path = f"{RESULTS_DIR}/statistical_tests.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved statistical tests to {out_path}")
    return results


if __name__ == "__main__":
    main()
