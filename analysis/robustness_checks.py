"""
Robustness / rigor layer for the primary classification result. None of this changes
the point estimate; it quantifies how much to trust it.

Five checks, all on the primary model (RandomForest, grouped-by-district nested-tuned
CV) unless noted:

1. Bootstrap 95% CI on the out-of-fold AUPRC and ROC-AUC, by CLUSTER bootstrap over
   districts (resampling whole districts, not individual events) so the interval
   respects the same grouped structure the CV uses. A point estimate of 0.72 with a
   CI of [0.55, 0.85] is a very different claim from [0.70, 0.74].

2. Label-permutation test: shuffle the outbreak labels, re-run the whole grouped CV,
   repeat. Gives an empirical p-value for "the model's AUPRC is above what this
   pipeline produces from noise". This is the honest answer to "is 0.72 actually
   better than chance, given only 125 events and 5 folds?". Nested tuning is disabled
   during permutation (tuning against shuffled labels is meaningless and just burns
   time); the observed run it is compared against uses the same fixed hyperparameters,
   so the comparison is like-for-like.

3. Paired bootstrap of meteorology-only vs. full-model AUPRC. The ablation already
   shows meteorology-only scoring higher; this tests whether that gap's CI excludes
   zero. If it does, the simpler 9-feature model is the defensible primary and the
   report should say so.

4. Label-threshold sensitivity (RICE events only -- the 72 rice labels come from a
   numeric rule, LBI>=15% OR NBI>=8%; the 53 wheat labels are literature
   presence/absence with no cutoff to vary). Re-derive rice labels across a grid of
   thresholds, re-run the full 125-event grouped CV each time, and check the headline
   AUPRC is not an artifact of the specific cutoff.

5. Decision-curve analysis: net benefit of the model across decision thresholds vs.
   the "treat all" and "treat none" defaults. For an outbreak early-warning system a
   plain AUPRC hides whether the model is actually useful at the operating points a
   decision-maker would pick.

Usage (from repo root, after training/classical_baselines.py has been run at least
once so the feature CSV and results exist):
    python3 -m analysis.robustness_checks
"""

import json
import logging
import os

import numpy as np
import pandas as pd

from training.classical_baselines import (
    FEATURE_COLUMNS,
    cross_validate,
)
from training.rq2_ablations import METEOROLOGY_COLUMNS

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # noqa: BLE001
    HAVE_MPL = False

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RobustnessChecks")

FEATURES_CSV = "data/processed/real_event_features.csv"
RESULTS_DIR = "results"
FIG_DIR = "results/figures"
RNG = np.random.default_rng(20260910)

N_BOOT = 2000
N_PERM = 300
N_SPLITS = 5


def _grouped_oof(X, y, groups, model_name="RandomForest", tune=True):
    """OOF probability vector from the same grouped nested CV the primary result uses."""
    splitter = GroupKFold(n_splits=N_SPLITS)
    _, _, oof_probs, oof_auprc, oof_roc, _ = cross_validate(
        X, y, groups, model_name, splitter, "group", tune=tune
    )
    return oof_probs, oof_auprc, oof_roc


def _cluster_bootstrap_indices(groups):
    """Resample whole districts with replacement; return the concatenated event indices."""
    uniq = np.unique(groups)
    drawn = RNG.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([np.where(groups == g)[0] for g in drawn])
    return idx


def _boot_ci(yy, p, index_fn, gg=None):
    auprcs, rocs = [], []
    for _ in range(N_BOOT):
        bi = index_fn(gg) if gg is not None else index_fn(len(yy))
        yb, pb = yy[bi], p[bi]
        if len(np.unique(yb)) < 2:
            continue
        auprcs.append(average_precision_score(yb, pb))
        rocs.append(roc_auc_score(yb, pb))
    return {
        "auprc_ci95": [float(np.percentile(auprcs, 2.5)), float(np.percentile(auprcs, 97.5))],
        "roc_auc_ci95": [float(np.percentile(rocs, 2.5)), float(np.percentile(rocs, 97.5))],
        "n_bootstrap": len(auprcs),
    }


def bootstrap_metric_ci(oof_probs, y, groups):
    """CI for AUPRC / ROC-AUC on a fixed OOF prediction vector, two ways:

    - cluster bootstrap over districts (resample whole groups) -- matches the grouped
      CV design, and is the number to report; it is wide here because most districts
      carry only 1-3 events, so resampling groups swings the positive count a lot.
    - plain event-level bootstrap -- narrower, ignores the grouping; reported only as
      context for how much of the width is the clustering.
    """
    mask = ~np.isnan(oof_probs)
    p, yy, gg = oof_probs[mask], y[mask], np.asarray(groups)[mask]
    clustered = _boot_ci(yy, p, _cluster_bootstrap_indices, gg=gg)
    plain = _boot_ci(yy, p, lambda n: RNG.integers(0, n, n))
    return {
        "auprc_point": float(average_precision_score(yy, p)),
        "roc_auc_point": float(roc_auc_score(yy, p)),
        "positive_rate": float(yy.mean()),
        "cluster_bootstrap_over_districts": clustered,
        "plain_event_bootstrap_context_only": plain,
        "auprc_ci95": clustered["auprc_ci95"],
        "roc_auc_ci95": clustered["roc_auc_ci95"],
        "method": "primary CI = cluster bootstrap over districts (whole groups resampled)",
    }


def permutation_test(X, y, groups):
    """Empirical p-value: P(AUPRC from shuffled labels >= observed). Fixed hyperparams
    (tune=False) on both sides so the comparison is like-for-like."""
    _, observed_auprc, _ = _grouped_oof(X, y, groups, tune=False)
    null = []
    for i in range(N_PERM):
        yp = RNG.permutation(y)
        try:
            _, a, _ = _grouped_oof(X, yp, groups, tune=False)
        except Exception:  # noqa: BLE001
            continue
        if not np.isnan(a):
            null.append(a)
        if (i + 1) % 50 == 0:
            logger.info(f"  permutation {i + 1}/{N_PERM}, running p "
                        f"~ {(1 + sum(v >= observed_auprc for v in null)) / (1 + len(null)):.4f}")
    null = np.array(null)
    p_value = (1 + int((null >= observed_auprc).sum())) / (1 + len(null))
    return {
        "observed_auprc_fixed_hp": float(observed_auprc),
        "n_permutations": int(len(null)),
        "null_auprc_mean": float(null.mean()),
        "null_auprc_p95": float(np.percentile(null, 95)),
        "null_auprc_max": float(null.max()),
        "p_value": float(p_value),
        "note": "tune=False both sides; p is one-sided P(null AUPRC >= observed)",
    }


def paired_bootstrap_met_vs_full(df, y, groups):
    """Cluster-bootstrap CI for (meteorology-only AUPRC) - (full-model AUPRC)."""
    X_full = df[FEATURE_COLUMNS].copy()
    X_met = df[METEOROLOGY_COLUMNS].copy()
    p_full, a_full, _ = _grouped_oof(X_full, y, groups)
    p_met, a_met, _ = _grouped_oof(X_met, y, groups)
    mask = ~np.isnan(p_full) & ~np.isnan(p_met)
    yy, gg = y[mask], np.asarray(groups)[mask]
    pf, pm = p_full[mask], p_met[mask]
    diffs = []
    for _ in range(N_BOOT):
        bi = _cluster_bootstrap_indices(gg)
        yb = yy[bi]
        if len(np.unique(yb)) < 2:
            continue
        diffs.append(average_precision_score(yb, pm[bi]) - average_precision_score(yb, pf[bi]))
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {
        "auprc_meteorology_only": float(a_met),
        "auprc_full_model": float(a_full),
        "difference_point": float(a_met - a_full),
        "difference_ci95": [lo, hi],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "n_bootstrap": len(diffs),
        "interpretation": (
            "meteorology-only is significantly better -> report it as primary"
            if lo > 0 else
            "difference not significant -> keep the full model as primary"
        ),
    }


def threshold_sensitivity(df, y_full):
    """Re-derive RICE labels over a grid of (LBI, NBI) thresholds; wheat labels fixed."""
    is_rice = (df["disease"] == "rice_blast").values
    groups = df["district"].values
    X = df[FEATURE_COLUMNS].copy()

    baseline = {"lbi": 15, "nbi": 8}
    grid = []
    for lbi in (10, 12, 15, 18, 20):
        for nbi in (5, 6, 8, 10):
            y = y_full.copy()
            rice_pos = ((df["leaf_blast_incidence_pct"] >= lbi) |
                        (df["neck_blast_incidence_pct"] >= nbi)).values
            y = np.where(is_rice, rice_pos.astype(int), y)
            if len(np.unique(y)) < 2:
                continue
            _, auprc, roc = _grouped_oof(X, y, groups, tune=False)
            grid.append({
                "lbi_threshold": lbi, "nbi_threshold": nbi,
                "n_positive": int(y.sum()), "n_rice_positive": int(y[is_rice].sum()),
                "auprc": float(auprc), "roc_auc": float(roc),
                "is_baseline": (lbi == baseline["lbi"] and nbi == baseline["nbi"]),
            })
    auprcs = [g["auprc"] for g in grid]
    # The "plausible band": thresholds close to the ones the source studies actually
    # used. Cutoffs like NBI>=10 discard ~a third of the rice positives and are not a
    # labelling choice anyone would defend -- reporting the full-grid range lets that
    # corner dominate a sensitivity claim it shouldn't. Report both, lead with the band.
    band = [g["auprc"] for g in grid
            if 12 <= g["lbi_threshold"] <= 20 and 5 <= g["nbi_threshold"] <= 8]
    baseline_auprc = next(g["auprc"] for g in grid if g["is_baseline"])
    return {
        "baseline_rule": "LBI>=15 OR NBI>=8",
        "baseline_auprc": float(baseline_auprc),
        "grid": grid,
        "full_grid_auprc_min": float(np.min(auprcs)), "full_grid_auprc_max": float(np.max(auprcs)),
        "plausible_band_auprc_min": float(np.min(band)), "plausible_band_auprc_max": float(np.max(band)),
        "plausible_band_auprc_range": float(np.max(band) - np.min(band)),
        "baseline_within_band_range": bool(np.min(band) <= baseline_auprc <= np.max(band)),
        "note": ("tune=False for speed/comparability across the grid; wheat labels held fixed. "
                 "'plausible band' = LBI in [12,20] AND NBI in [5,8]."),
    }


def decision_curve(oof_probs, y):
    """Net benefit across threshold probabilities, vs treat-all and treat-none."""
    mask = ~np.isnan(oof_probs)
    p, yy = oof_probs[mask], y[mask]
    n = len(yy)
    prevalence = yy.mean()
    pts = np.round(np.arange(0.05, 0.61, 0.05), 2)
    rows = []
    for pt in pts:
        pred = p >= pt
        tp = int(np.sum(pred & (yy == 1)))
        fp = int(np.sum(pred & (yy == 0)))
        nb_model = tp / n - fp / n * (pt / (1 - pt))
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append({
            "threshold": float(pt),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": 0.0,
            "model_beats_defaults": bool(nb_model > max(nb_all, 0.0)),
        })
    if HAVE_MPL:
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(pts, [r["net_benefit_model"] for r in rows], "o-", label="RandomForest")
        ax.plot(pts, [r["net_benefit_treat_all"] for r in rows], "--", label="Treat all")
        ax.axhline(0, color="k", lw=0.8, label="Treat none")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title("Decision curve — outbreak early warning (OOF predictions)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "decision_curve.png"), dpi=130)
        plt.close(fig)
    return {"prevalence": float(prevalence), "curve": rows,
            "figure": os.path.join(FIG_DIR, "decision_curve.png") if HAVE_MPL else None}


def main():
    df = pd.read_csv(FEATURES_CSV)
    y = df["label"].astype(int).values
    groups = df["district"].values
    X = df[FEATURE_COLUMNS].copy()

    logger.info("1/5 grouped OOF + cluster-bootstrap CI ...")
    oof_probs, oof_auprc, oof_roc = _grouped_oof(X, y, groups, tune=True)
    ci = bootstrap_metric_ci(oof_probs, y, groups)
    logger.info(f"    AUPRC {ci['auprc_point']:.3f}  CI95 {ci['auprc_ci95']}")

    logger.info(f"2/5 label-permutation test ({N_PERM} permutations) ...")
    perm = permutation_test(X, y, groups)
    logger.info(f"    p = {perm['p_value']:.4f}  (null mean {perm['null_auprc_mean']:.3f})")

    logger.info("3/5 paired bootstrap: meteorology-only vs full ...")
    paired = paired_bootstrap_met_vs_full(df, y, groups)
    logger.info(f"    diff {paired['difference_point']:+.3f}  CI95 {paired['difference_ci95']}  "
                f"{paired['interpretation']}")

    logger.info("4/5 label-threshold sensitivity (rice) ...")
    thr = threshold_sensitivity(df, y)
    logger.info(f"    plausible-band AUPRC {thr['plausible_band_auprc_min']:.3f}.."
                f"{thr['plausible_band_auprc_max']:.3f} (baseline {thr['baseline_auprc']:.3f}); "
                f"full grid {thr['full_grid_auprc_min']:.3f}..{thr['full_grid_auprc_max']:.3f}")

    logger.info("5/5 decision-curve analysis ...")
    dca = decision_curve(oof_probs, y)

    out = {
        "n_events": int(len(df)),
        "primary_oof_auprc_tuned": float(oof_auprc),
        "primary_oof_roc_auc_tuned": float(oof_roc),
        "bootstrap_ci": ci,
        "permutation_test": perm,
        "meteorology_vs_full_paired_bootstrap": paired,
        "label_threshold_sensitivity": thr,
        "decision_curve": dca,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "robustness_checks.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Saved {path}")
    return out


if __name__ == "__main__":
    main()
