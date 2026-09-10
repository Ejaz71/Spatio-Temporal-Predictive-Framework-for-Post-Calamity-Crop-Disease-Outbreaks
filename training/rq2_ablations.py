"""
RQ2 ablations (proposal Section 9, "does fusing remote sensing + meteorology beat
either modality alone?"): remote-sensing-only and meteorology-only classical models,
evaluated with the exact same nested-tuned grouped-by-district + leave-one-event-out
CV as the full model in classical_baselines.py, so the three conditions
(remote-sensing-only, meteorology-only, full) are directly comparable — same models,
same tuning budget, same splits, same data.

Reuses classical_baselines.py's build_model / select_hyperparams_for_indices /
cross_validate unchanged — only the feature-column subset passed in differs.

Column ORDER matters: RandomForest samples `max_features` columns per split using an
index-based RNG seeded by `random_state`, so a different column order gives a different
(still valid) model and a different AUPRC. To guarantee the `full_model` condition here
is byte-identical to the primary result in classical_baselines.py, the three column
sets below are derived as ORDER-PRESERVING partitions of that module's FEATURE_COLUMNS
rather than being retyped. (Retyping them as METEOROLOGY + REMOTE_SENSING previously
made `full_model` score 0.741 vs the primary's 0.757 — same features, same data, only
the order differed.)
"""

import json
import logging
import os

import numpy as np
import pandas as pd

from training.classical_baselines import (
    FEATURES_CSV,
    FEATURE_COLUMNS,
    cross_validate,
)
from sklearn.model_selection import GroupKFold, LeaveOneOut

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RQ2Ablations")

RESULTS_DIR = "results"

_REMOTE_SENSING = {"sar_vv_db_mean", "sar_vh_db_mean", "ndvi_mean", "ndwi_mean",
                   "lst_celsius_mean", "water_extent_frac"}
# Order-preserving partition of the canonical FEATURE_COLUMNS.
REMOTE_SENSING_COLUMNS = [c for c in FEATURE_COLUMNS if c in _REMOTE_SENSING]
METEOROLOGY_COLUMNS = [c for c in FEATURE_COLUMNS if c not in _REMOTE_SENSING]
FULL_COLUMNS = list(FEATURE_COLUMNS)


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    y = df["label"].astype(int).values
    groups = df["district"].values
    return df, y, groups


def run_condition(name, columns, df, y, groups, model_name):
    X = df[columns].copy()
    n_districts = len(np.unique(groups))
    n_splits = min(5, n_districts)

    gkf = GroupKFold(n_splits=n_splits)
    summary_grouped, raw_grouped, oof_grouped, auprc_g, roc_g, hps_grouped = cross_validate(
        X, y, groups, model_name, gkf, split_kind="group", tune=True
    )
    loo = LeaveOneOut()
    summary_loo, raw_loo, oof_loo, auprc_l, roc_l, _ = cross_validate(
        X, y, groups, model_name, loo, split_kind="loo", tune=False
    )
    logger.info(f"[{name}] {model_name} grouped-by-district (nested-tuned) OOF AUPRC={auprc_g:.3f} ROC-AUC={roc_g:.3f} "
                f"(n_features={len(columns)})")
    logger.info(f"[{name}] {model_name} leave-one-event-out OOF AUPRC={auprc_l:.3f} ROC-AUC={roc_l:.3f}")
    return {
        "n_features": len(columns),
        "feature_columns": columns,
        "grouped_by_district": {
            "n_splits": n_splits,
            "fold_metrics_mean_std": summary_grouped,
            "oof_auprc": auprc_g,
            "oof_roc_auc": roc_g,
        },
        "leave_one_event_out": {
            "fold_metrics_mean_std": summary_loo,
            "oof_auprc": auprc_l,
            "oof_roc_auc": roc_l,
        },
    }


def main():
    df, y, groups = load_data()
    results = {"n_events": len(y), "n_positive": int(y.sum())}

    conditions = {
        "remote_sensing_only": REMOTE_SENSING_COLUMNS,
        "meteorology_only": METEOROLOGY_COLUMNS,
        "full_model": FULL_COLUMNS,
    }

    for model_name in ["RandomForest", "XGBoost"]:
        results[model_name] = {}
        for cond_name, columns in conditions.items():
            results[model_name][cond_name] = run_condition(cond_name, columns, df, y, groups, model_name)

    # Headline comparison for the report: does the full (fused) model beat both
    # single-modality ablations on the primary metric (grouped-by-district OOF AUPRC)?
    rf = results["RandomForest"]
    comparison = {
        "remote_sensing_only_auprc": rf["remote_sensing_only"]["grouped_by_district"]["oof_auprc"],
        "meteorology_only_auprc": rf["meteorology_only"]["grouped_by_district"]["oof_auprc"],
        "full_model_auprc": rf["full_model"]["grouped_by_district"]["oof_auprc"],
    }
    best_single = max(
        ("remote_sensing_only", comparison["remote_sensing_only_auprc"]),
        ("meteorology_only", comparison["meteorology_only_auprc"]),
        key=lambda t: t[1],
    )
    fusion_gain = comparison["full_model_auprc"] - best_single[1]
    comparison["best_single_modality"] = best_single[0]
    comparison["best_single_modality_auprc"] = best_single[1]
    comparison["full_model_gain_over_best_single_modality"] = fusion_gain
    comparison["interpretation"] = (
        f"RandomForest grouped-by-district OOF AUPRC: remote-sensing-only={comparison['remote_sensing_only_auprc']:.3f}, "
        f"meteorology-only={comparison['meteorology_only_auprc']:.3f}, full (both)={comparison['full_model_auprc']:.3f}. "
        f"The full model {'beats' if fusion_gain > 0 else 'does not beat'} the best single modality "
        f"({best_single[0]}) by {fusion_gain:+.3f} AUPRC. This is a point estimate on n={len(y)} events without a "
        f"paired significance test across ablations (unlike the RandomForest-vs-fusion comparison, which does have "
        f"one) — report the gap as descriptive evidence for which modality carries more signal, not as a "
        f"statistically confirmed difference."
    )
    results["rq2_headline_comparison"] = comparison
    logger.info(comparison["interpretation"])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "rq2_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved RQ2 ablation results to {out_path}")
    return results


if __name__ == "__main__":
    main()
