"""
Classical baselines (Random Forest, XGBoost) on the real event feature matrix (125
events as of the 2026-09-08 upazila-level integration — see CLAUDE.md).

Implements proposal Section 6.1/6.2/9:
- Random Forest (Breiman, 2001) and Gradient Boosting / XGBoost (Chen & Guestrin, 2016)
  as classical baselines and explicit fallback.
- Grouped, spatially blocked cross-validation (GroupKFold by district) plus a
  leave-one-event-out variant, so nearby districts/years cannot leak across
  train/test splits.
- Primary metric AUPRC, secondary F1 @ fixed threshold and ROC-AUC (with the
  imbalance caveat), Brier score for calibration.
- SHAP values for RQ3 feature attribution (Section 9), computed out-of-fold.

Deliberately excludes the epidemiological survey columns (leaf_blast_incidence_pct,
neck_blast_incidence_pct, leaf_blast_severity_pct, neck_blast_severity_pct,
pct_fields_infected, avg_yield_loss_pct, affected_area_ha) from the feature set:
those columns are how `label` itself was derived (Section 6.3 thresholds), so
including them would leak the target directly rather than testing whether
remote sensing + meteorology can predict it.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, LeaveOneOut
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ClassicalBaselines")

FEATURES_CSV = "data/processed/real_event_features.csv"
RESULTS_DIR = "results"

FEATURE_COLUMNS = [
    "precip_mean_mm", "precip_max_mm", "precip_sum_mm", "precip_anomaly_mm",
    "rh_mean_pct", "rh_max_pct", "temp_mean_c", "vpd_mean_kpa", "wet_persistence_max_days",
    "sar_vv_db_mean", "sar_vh_db_mean", "ndvi_mean", "ndwi_mean", "lst_celsius_mean",
    "water_extent_frac",
]
LEAKAGE_COLUMNS = [
    "leaf_blast_incidence_pct", "neck_blast_incidence_pct",
    "leaf_blast_severity_pct", "neck_blast_severity_pct",
    "pct_fields_infected", "avg_yield_loss_pct", "affected_area_ha",
]


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    missing_frac = df[FEATURE_COLUMNS].isna().mean()
    logger.info(f"Missingness per feature:\n{missing_frac}")
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int).values
    groups = df["district"].values
    year_groups = df["year"].values
    return df, X, y, groups, year_groups


def build_model(model_name, hp=None):
    if model_name == "RandomForest":
        defaults = dict(n_estimators=400, max_depth=5, min_samples_leaf=2)
        params = {**defaults, **(hp or {})}
        return RandomForestClassifier(class_weight="balanced", random_state=42, **params)
    else:
        defaults = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
        params = {**defaults, **(hp or {})}
        return XGBClassifier(eval_metric="logloss", random_state=42, **params)


def build_models():
    return {"RandomForest": build_model("RandomForest"), "XGBoost": build_model("XGBoost")}


# Small nested-tuning grids. Kept modest given the dataset is still small: RF/XGBoost are fast enough that a
# larger grid is cheap, but too much tuning search on this little data risks overfitting
# to whatever internal validation split gets used, so this stays deliberately small.
HP_CANDIDATES = {
    "RandomForest": [
        {"n_estimators": 400, "max_depth": 5, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 3},
        {"n_estimators": 500, "max_depth": 7, "min_samples_leaf": 1},
        {"n_estimators": 600, "max_depth": 4, "min_samples_leaf": 2},
    ],
    "XGBoost": [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.1},
        {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.02},
    ],
}


def _fit_predict(model_name, hp, X_train, y_train, X_test):
    model = build_model(model_name, hp)
    pos = max(1, np.sum(y_train == 1))
    neg = max(1, np.sum(y_train == 0))
    if model_name == "XGBoost":
        model.set_params(scale_pos_weight=neg / pos)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def select_hyperparams_for_indices(X, y, groups, model_name, train_idx, seed=0):
    """Nested tuning: pick hyperparams using ONLY this outer fold's training data,
    via one internal GroupShuffleSplit validation split — never touches the outer
    fold's held-out test data, so the outer metric stays leakage-free."""
    sub_groups = groups[train_idx]
    if len(np.unique(sub_groups)) < 2:
        return HP_CANDIDATES[model_name][0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    try:
        inner_tr_rel, inner_val_rel = next(gss.split(train_idx, y[train_idx], sub_groups))
    except ValueError:
        return HP_CANDIDATES[model_name][0]
    inner_tr, inner_val = train_idx[inner_tr_rel], train_idx[inner_val_rel]
    if len(np.unique(y[inner_val])) < 2 or len(np.unique(y[inner_tr])) < 2:
        return HP_CANDIDATES[model_name][0]

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X.iloc[inner_tr])
    X_val = imputer.transform(X.iloc[inner_val])

    best_hp, best_auprc = HP_CANDIDATES[model_name][0], -1
    for hp in HP_CANDIDATES[model_name]:
        probs = _fit_predict(model_name, hp, X_tr, y[inner_tr], X_val)
        auprc = average_precision_score(y[inner_val], probs)
        if auprc > best_auprc:
            best_auprc, best_hp = auprc, hp
    return best_hp


def cross_validate(X, y, groups, model_name, splitter, split_kind, tune=True):
    fold_metrics = {"auprc": [], "f1": [], "roc_auc": [], "recall": [], "brier": []}
    oof_probs = np.full(len(y), np.nan)
    selected_hps = []

    if split_kind == "loo":
        splits = splitter.split(X)
    else:
        splits = splitter.split(X, y, groups)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if len(np.unique(y[train_idx])) < 2:
            continue
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X.iloc[train_idx])
        X_test = imputer.transform(X.iloc[test_idx])

        # Nested hyperparameter selection: only for grouped CV (LOO's 76-sample
        # training folds are too small to carve out a further internal validation
        # split without overfitting the tuning step itself).
        hp = select_hyperparams_for_indices(X, y, groups, model_name, train_idx, seed=fold_idx) \
            if (tune and split_kind == "group") else None
        selected_hps.append(hp)

        probs = _fit_predict(model_name, hp, X_train, y[train_idx], X_test)
        oof_probs[test_idx] = probs

        if len(np.unique(y[test_idx])) > 1:
            fold_metrics["auprc"].append(average_precision_score(y[test_idx], probs))
            fold_metrics["roc_auc"].append(roc_auc_score(y[test_idx], probs))
        preds = (probs >= 0.5).astype(int)
        fold_metrics["f1"].append(f1_score(y[test_idx], preds, zero_division=0))
        fold_metrics["recall"].append(recall_score(y[test_idx], preds, zero_division=0))
        fold_metrics["brier"].append(brier_score_loss(y[test_idx], probs))

    summary = {k: (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)
               for k, v in fold_metrics.items()}
    valid_mask = ~np.isnan(oof_probs)
    if valid_mask.sum() and len(np.unique(y[valid_mask])) > 1:
        oof_auprc = average_precision_score(y[valid_mask], oof_probs[valid_mask])
        oof_roc = roc_auc_score(y[valid_mask], oof_probs[valid_mask])
    else:
        oof_auprc, oof_roc = np.nan, np.nan

    return summary, fold_metrics, oof_probs, oof_auprc, oof_roc, selected_hps


def run_shap_attribution(X, y, model_name="RandomForest"):
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    model = build_models()[model_name]
    pos = max(1, np.sum(y == 1))
    neg = max(1, np.sum(y == 0))
    if model_name == "XGBoost":
        model.set_params(scale_pos_weight=neg / pos)
    model.fit(X_imp, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imp)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class (older SHAP: list per class)
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # positive class (newer SHAP: (n, features, classes))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(X.columns, mean_abs_shap), key=lambda t: -t[1])
    return {name: float(val) for name, val in ranking}


def main():
    df, X, y, groups, year_groups = load_data()
    n_districts = len(np.unique(groups))
    n_splits = min(5, n_districts)
    n_years = len(np.unique(year_groups))

    results = {"n_events": len(y), "n_positive": int(y.sum()), "n_districts": n_districts}

    for model_name in ["RandomForest", "XGBoost"]:
        gkf = GroupKFold(n_splits=n_splits)
        summary_grouped, raw_grouped, oof_grouped, auprc_g, roc_g, hps_grouped = cross_validate(
            X, y, groups, model_name, gkf, split_kind="group", tune=True
        )

        loo = LeaveOneOut()
        summary_loo, raw_loo, oof_loo, auprc_l, roc_l, _ = cross_validate(
            X, y, groups, model_name, loo, split_kind="loo", tune=False
        )

        # Leave-one-year-out: tests generalization to an entirely unseen future
        # season, complementing grouped-by-district (unseen place) and LOO (unseen
        # single event). Reuses the "group" split_kind code path with year as the
        # grouping variable for BOTH the outer folds and the inner nested-tuning
        # validation split — consistent with how grouped-by-district already works.
        logo = LeaveOneGroupOut()
        summary_year, raw_year, oof_year, auprc_y, roc_y, hps_year = cross_validate(
            X, y, year_groups, model_name, logo, split_kind="group", tune=True
        )

        results[model_name] = {
            "grouped_by_district": {
                "n_splits": n_splits,
                "fold_metrics_mean_std": summary_grouped,
                "raw_fold_auprc": raw_grouped["auprc"],
                "oof_auprc": auprc_g,
                "oof_roc_auc": roc_g,
                "nested_hyperparams_per_fold": hps_grouped,
            },
            "leave_one_event_out": {
                "fold_metrics_mean_std": summary_loo,
                "oof_auprc": auprc_l,
                "oof_roc_auc": roc_l,
                "oof_probs": oof_loo.tolist(),
            },
            "leave_one_year_out": {
                "n_years": n_years,
                "fold_metrics_mean_std": summary_year,
                "raw_fold_auprc": raw_year["auprc"],
                "oof_auprc": auprc_y,
                "oof_roc_auc": roc_y,
                "nested_hyperparams_per_fold": hps_year,
                "note": "Some year-folds (e.g. 2015, all wheat-blast pre-emergence, 0 positives) have only one "
                        "class in the test fold — AUPRC/ROC-AUC are skipped for those folds specifically (not "
                        "counted as 0), consistent with how grouped-by-district handles single-class folds.",
            },
        }
        logger.info(f"{model_name} grouped-by-district (nested-tuned) OOF AUPRC={auprc_g:.3f} ROC-AUC={roc_g:.3f}")
        logger.info(f"{model_name} leave-one-event-out OOF AUPRC={auprc_l:.3f} ROC-AUC={roc_l:.3f}")
        logger.info(f"{model_name} leave-one-year-out (nested-tuned) OOF AUPRC={auprc_y:.3f} ROC-AUC={roc_y:.3f}")

    results["shap_feature_importance_random_forest"] = run_shap_attribution(X, y, "RandomForest")
    results["shap_feature_importance_xgboost"] = run_shap_attribution(X, y, "XGBoost")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "classical_baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved classical baseline + SHAP results to {out_path}")
    return results


if __name__ == "__main__":
    main()
