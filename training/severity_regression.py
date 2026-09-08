"""
Rice blast severity regression (proposal Section 9's severity/continuous-outcome
task): predicts leaf_blast_severity_pct and neck_blast_severity_pct — continuous,
not the binary outbreak label — restricted to the rice_blast rows (n=72, upazila-level
resolution, since the 2026-09-08 Phase B item 0 integration replaced the original
24 district-level rice-blast rows — see CLAUDE.md).

Per the CSE791 Week 6 lecture's "classical stats alongside ML metrics" guidance and
the plan approved 2026-09-08: features are restricted to those Phase A's Spearman
correlations (analysis/statistical_tests.py) already flagged as significant
(p<0.05, uncorrected — this is a deliberately narrowed follow-up regression on an
already-identified candidate set, not a fresh 14-test screen, so Benjamini-Hochberg
is not re-applied here) rather than all 14 features, to avoid overfitting a
regression on a small n. Because several of the significant candidates are
themselves highly collinear (precip_mean/sum/max/anomaly all correlate > 0.8 with
each other in this data — expected, they're different summaries of the same
rainfall signal), one representative per correlated cluster is kept for the multiple
regression, chosen as whichever cluster member has the strongest univariate
correlation with the target; this is disclosed explicitly rather than silently
picking one.

Reports, per target:
1. Univariate OLS (statsmodels) per significant feature: slope, 95% CI, R², p-value
   — the "classical regression stats" the lecture calls for.
2. A correlation-cluster reduction step (disclosed, not hidden).
3. Multiple OLS on the reduced feature set: coefficients + 95% CIs + p-values, R²,
   adjusted R².
4. Held-out leave-one-out CV (matches this project's CV convention elsewhere) RMSE
   and R² for both plain LinearRegression and Ridge(alpha=1.0) on the same reduced
   feature set — the "ML regression metrics" the lecture treats as complementary to
   the classical stats above, not a replacement for them.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SeverityRegression")

FEATURES_CSV = "data/processed/real_event_features.csv"
STATS_JSON = "results/statistical_tests.json"
RESULTS_DIR = "results"

SEVERITY_TARGETS = ["leaf_blast_severity_pct", "neck_blast_severity_pct"]
SIG_THRESHOLD = 0.05
COLLINEARITY_THRESHOLD = 0.8


def get_significant_features(target):
    with open(STATS_JSON) as f:
        stats_json = json.load(f)
    rows = stats_json["severity_correlations"][target]
    sig = [r for r in rows if r["p_value"] < SIG_THRESHOLD]
    return sig  # already sorted by |rho| descending, from analysis/statistical_tests.py


def reduce_collinear_features(df_rice, sig_rows):
    """Keeps one representative per correlated cluster (|Pearson r| > threshold among
    the candidate features themselves), preferring whichever cluster member has the
    strongest univariate correlation with the target (sig_rows is already sorted by
    |rho| descending, so the first unclustered feature encountered wins)."""
    candidate_features = [r["feature"] for r in sig_rows]
    corr = df_rice[candidate_features].corr(method="pearson").abs()

    kept, dropped = [], {}
    for feat in candidate_features:  # already strongest-first
        if any(corr.loc[feat, k] > COLLINEARITY_THRESHOLD for k in kept):
            redundant_with = next(k for k in kept if corr.loc[feat, k] > COLLINEARITY_THRESHOLD)
            dropped[feat] = redundant_with
            continue
        kept.append(feat)
    return kept, dropped


def univariate_ols(df_rice, feature, target):
    sub = df_rice[[feature, target]].dropna()
    X = sm.add_constant(sub[feature].values)
    y = sub[target].values
    model = sm.OLS(y, X).fit()
    ci = model.conf_int(alpha=0.05)
    return {
        "feature": feature, "n": int(len(sub)),
        "slope": float(model.params[1]), "slope_95pct_ci": [float(ci[1][0]), float(ci[1][1])],
        "intercept": float(model.params[0]), "r_squared": float(model.rsquared),
        "p_value": float(model.pvalues[1]),
    }


def multiple_ols(df_rice, features, target):
    sub = df_rice[features + [target]].dropna()
    X = sm.add_constant(sub[features].values)
    y = sub[target].values
    model = sm.OLS(y, X).fit()
    ci = model.conf_int(alpha=0.05)
    coefficients = {"const": {"coef": float(model.params[0]), "ci": [float(ci[0][0]), float(ci[0][1])],
                               "p_value": float(model.pvalues[0])}}
    for i, feat in enumerate(features, start=1):
        coefficients[feat] = {"coef": float(model.params[i]), "ci": [float(ci[i][0]), float(ci[i][1])],
                               "p_value": float(model.pvalues[i])}
    return {
        "n": int(len(sub)), "features": features,
        "r_squared": float(model.rsquared), "adj_r_squared": float(model.rsquared_adj),
        "f_pvalue": float(model.f_pvalue), "coefficients": coefficients,
    }


def loo_cv_metrics(df_rice, features, target):
    sub = df_rice[features + [target]].dropna().reset_index(drop=True)
    X_raw = sub[features].values
    y = sub[target].values
    n = len(sub)
    if n < len(features) + 3:
        return {"note": f"n={n} too small relative to {len(features)} features for a meaningful LOO CV"}

    loo = LeaveOneOut()
    results = {}
    for model_name, model_cls, kwargs in [("LinearRegression", LinearRegression, {}),
                                           ("Ridge_alpha1.0", Ridge, {"alpha": 1.0})]:
        oof_pred = np.full(n, np.nan)
        for train_idx, test_idx in loo.split(X_raw):
            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_raw[train_idx])
            X_test = imputer.transform(X_raw[test_idx])
            model = model_cls(**kwargs)
            model.fit(X_train, y[train_idx])
            oof_pred[test_idx] = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y, oof_pred)))
        r2 = float(r2_score(y, oof_pred))
        results[model_name] = {"loo_rmse": rmse, "loo_r_squared": r2}
    return results


def run_target(df_rice, target):
    sig_rows = get_significant_features(target)
    if not sig_rows:
        logger.warning(f"No features significantly correlated with {target} at p<{SIG_THRESHOLD} — skipping.")
        return {"note": "no significant univariate predictors at p<0.05"}

    univariate = [univariate_ols(df_rice, r["feature"], target) for r in sig_rows]
    kept, dropped = reduce_collinear_features(df_rice, sig_rows)
    logger.info(f"[{target}] {len(sig_rows)} significant features -> {len(kept)} after collinearity reduction "
                f"(threshold |r|>{COLLINEARITY_THRESHOLD}): kept={kept}, dropped(redundant_with)={dropped}")

    multiple = multiple_ols(df_rice, kept, target)
    loo = loo_cv_metrics(df_rice, kept, target)
    logger.info(f"[{target}] Multiple OLS R²={multiple['r_squared']:.3f}, adj R²={multiple['adj_r_squared']:.3f}, "
                f"n={multiple['n']}. LOO CV: {loo}")

    return {
        "n_significant_univariate_predictors": len(sig_rows),
        "univariate_ols": univariate,
        "collinearity_reduction": {"kept": kept, "dropped_redundant_with": dropped,
                                    "threshold_pearson_r": COLLINEARITY_THRESHOLD},
        "multiple_ols": multiple,
        "held_out_loo_cv": loo,
    }


def main():
    df = pd.read_csv(FEATURES_CSV)
    df_rice = df[df["disease"] == "rice_blast"].copy()
    logger.info(f"Rice blast subset: n={len(df_rice)} events (upazila-level, post-merge)")

    results = {"n_rice_blast_events": len(df_rice)}
    for target in SEVERITY_TARGETS:
        results[target] = run_target(df_rice, target)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "severity_regression_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved severity regression results to {out_path}")
    return results


if __name__ == "__main__":
    main()
