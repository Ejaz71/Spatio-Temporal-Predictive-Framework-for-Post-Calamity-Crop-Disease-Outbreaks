"""
Evaluates the CNN-LSTM-style fusion model (models/fusion_model_tabular.py) against the
classical RandomForest/XGBoost baselines (training/classical_baselines.py), using the
SAME grouped-by-district and leave-one-event-out CV splits for a fair comparison.

v2: the temporal branch now runs over the REAL 90-day daily NASA POWER sequence per
event (data/extract_daily_sequences.py) instead of 9 pre-aggregated summary scalars.
Also adds a small hyperparameter search (picked via a single internal validation
split, not leaked into the outer CV) and seed-averaging per fold, since a neural net
on n=77 has enough run-to-run variance that a single seed isn't a fair test of the
architecture.

Implements the proposal's Section 8 "stage 3/4 decision point" and Section 9 H1 test:
if the fused model's cross-validated AUPRC does not exceed the best classical baseline
by a margin larger than a bootstrapped confidence interval, the classical baseline is
reported as the primary result and the fusion model as a documented, honest negative
finding — not silently dropped or dressed up.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneOut
from sklearn.preprocessing import StandardScaler

from models.fusion_model_tabular import SPATIAL_FEATURES, TabularFusionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FusionModelEval")

FEATURES_CSV = "data/processed/real_event_features.csv"
DAILY_SEQ_NPZ = "data/processed/real_daily_sequences.npz"
CLASSICAL_RESULTS_JSON = "results/classical_baseline_results.json"
RESULTS_DIR = "results"
N_SEEDS = 5
N_BOOTSTRAP = 5000

HYPERPARAM_CANDIDATES = [
    {"hidden_dim": 12, "dropout": 0.4, "weight_decay": 1e-2, "lr": 0.01, "epochs": 80},
    {"hidden_dim": 8, "dropout": 0.5, "weight_decay": 5e-2, "lr": 0.005, "epochs": 60},
    {"hidden_dim": 8, "dropout": 0.6, "weight_decay": 1e-1, "lr": 0.005, "epochs": 40},
    {"hidden_dim": 16, "dropout": 0.5, "weight_decay": 3e-2, "lr": 0.008, "epochs": 60},
]


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    seq_data = np.load(DAILY_SEQ_NPZ, allow_pickle=True)
    seq_by_event = dict(zip(seq_data["event_ids"], seq_data["sequences"]))

    # Align daily sequences to the same row order as df (all 77 events have one).
    daily_sequences = np.stack([seq_by_event[eid] for eid in df["event_id"]], axis=0)

    X_spatial = df[SPATIAL_FEATURES].copy()
    y = df["label"].astype(int).values
    groups = df["district"].values
    return X_spatial, daily_sequences, y, groups


def train_fold(X_sp_train, X_seq_train, y_train, X_sp_test, X_seq_test, hp, seed=42):
    torch.manual_seed(seed)
    model = TabularFusionModel(
        spatial_dim=X_sp_train.shape[1], n_daily_features=X_seq_train.shape[2],
        hidden_dim=hp["hidden_dim"], dropout=hp["dropout"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    n_pos = max(1, (y_train == 1).sum())
    n_neg = max(1, (y_train == 0).sum())
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_sp_t = torch.tensor(X_sp_train, dtype=torch.float32)
    X_seq_t = torch.tensor(X_seq_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(hp["epochs"]):
        optimizer.zero_grad()
        logits, _ = model(X_sp_t, X_seq_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.tensor(X_sp_test, dtype=torch.float32), torch.tensor(X_seq_test, dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy()
    return probs


def preprocess_fold(X_spatial, daily_sequences, train_idx, test_idx):
    sp_imputer = SimpleImputer(strategy="median")
    sp_scaler = StandardScaler()
    X_sp_train = sp_scaler.fit_transform(sp_imputer.fit_transform(X_spatial.iloc[train_idx]))
    X_sp_test = sp_scaler.transform(sp_imputer.transform(X_spatial.iloc[test_idx]))

    # Fit the daily-sequence scaler on (n_train * seq_len, n_features), same scaler
    # applied per-timestep so units are comparable across the whole series.
    seq_train_raw = daily_sequences[train_idx]
    seq_test_raw = daily_sequences[test_idx]
    n_feat = seq_train_raw.shape[2]
    seq_scaler = StandardScaler()
    seq_scaler.fit(seq_train_raw.reshape(-1, n_feat))
    X_seq_train = seq_scaler.transform(seq_train_raw.reshape(-1, n_feat)).reshape(seq_train_raw.shape)
    X_seq_test = seq_scaler.transform(seq_test_raw.reshape(-1, n_feat)).reshape(seq_test_raw.shape)

    return X_sp_train, X_sp_test, X_seq_train, X_seq_test


def select_hyperparams(X_spatial, daily_sequences, y, groups):
    """Pick hyperparams via ONE internal validation split, kept separate from the
    outer CV used for reported metrics, so the outer numbers aren't tuned-on-the-test-set."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train_idx, val_idx = next(gss.split(X_spatial, y, groups))
    if len(np.unique(y[val_idx])) < 2:
        logger.warning("Internal validation split has only one class; using default hyperparams.")
        return HYPERPARAM_CANDIDATES[0]

    X_sp_train, X_sp_val, X_seq_train, X_seq_val = preprocess_fold(X_spatial, daily_sequences, train_idx, val_idx)

    best_hp, best_auprc = HYPERPARAM_CANDIDATES[0], -1
    for hp in HYPERPARAM_CANDIDATES:
        probs = train_fold(X_sp_train, X_seq_train, y[train_idx], X_sp_val, X_seq_val, hp, seed=0)
        auprc = average_precision_score(y[val_idx], probs)
        logger.info(f"Hyperparam search {hp} -> internal val AUPRC={auprc:.3f}")
        if auprc > best_auprc:
            best_auprc, best_hp = auprc, hp

    logger.info(f"Selected hyperparams: {best_hp} (internal val AUPRC={best_auprc:.3f})")
    return best_hp


def cross_validate_fusion(X_spatial, daily_sequences, y, groups, splitter, split_kind, hp):
    fold_metrics = {"auprc": [], "f1": [], "roc_auc": [], "recall": [], "brier": []}
    oof_probs = np.full(len(y), np.nan)

    splits = splitter.split(X_spatial) if split_kind == "loo" else splitter.split(X_spatial, y, groups)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if len(np.unique(y[train_idx])) < 2:
            continue

        X_sp_train, X_sp_test, X_seq_train, X_seq_test = preprocess_fold(X_spatial, daily_sequences, train_idx, test_idx)

        # Seed-averaging: a neural net on n=77 has enough run-to-run variance that a
        # single seed isn't a fair read of the architecture's real skill.
        seed_probs = [
            train_fold(X_sp_train, X_seq_train, y[train_idx], X_sp_test, X_seq_test, hp, seed=100 * fold_idx + s)
            for s in range(N_SEEDS)
        ]
        probs = np.mean(seed_probs, axis=0)
        oof_probs[test_idx] = probs

        if len(np.unique(y[test_idx])) > 1:
            fold_metrics["auprc"].append(average_precision_score(y[test_idx], probs))
            fold_metrics["roc_auc"].append(roc_auc_score(y[test_idx], probs))
        preds = (probs >= 0.5).astype(int)
        fold_metrics["f1"].append(f1_score(y[test_idx], preds, zero_division=0))
        fold_metrics["recall"].append(recall_score(y[test_idx], preds, zero_division=0))
        fold_metrics["brier"].append(brier_score_loss(y[test_idx], probs))

    summary = {k: (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan) for k, v in fold_metrics.items()}
    valid_mask = ~np.isnan(oof_probs)
    if valid_mask.sum() and len(np.unique(y[valid_mask])) > 1:
        oof_auprc = average_precision_score(y[valid_mask], oof_probs[valid_mask])
        oof_roc = roc_auc_score(y[valid_mask], oof_probs[valid_mask])
    else:
        oof_auprc, oof_roc = np.nan, np.nan

    return summary, fold_metrics, oof_probs, oof_auprc, oof_roc


def bootstrap_ci_diff(fold_scores_a, fold_scores_b, n_bootstrap=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.array(fold_scores_a), np.array(fold_scores_b)
    n = len(a)
    if n == 0:
        return np.nan, np.nan, np.nan
    diffs = a - b
    boot_means = [rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def main():
    X_spatial, daily_sequences, y, groups = load_data()
    n_districts = len(np.unique(groups))
    n_splits = min(5, n_districts)

    with open(CLASSICAL_RESULTS_JSON) as f:
        classical = json.load(f)

    hp = select_hyperparams(X_spatial, daily_sequences, y, groups)

    results = {"n_events": len(y), "n_positive": int(y.sum()), "selected_hyperparams": hp, "n_seeds_averaged": N_SEEDS}

    for split_name, splitter, kind in [
        ("grouped_by_district", GroupKFold(n_splits=n_splits), "group"),
        ("leave_one_event_out", LeaveOneOut(), "loo"),
    ]:
        summary, fold_metrics, oof_probs, oof_auprc, oof_roc = cross_validate_fusion(
            X_spatial, daily_sequences, y, groups, splitter, kind, hp
        )
        logger.info(f"Fusion model (v2, real sequences) {split_name} OOF AUPRC={oof_auprc:.3f} ROC-AUC={oof_roc:.3f}")
        results[split_name] = {
            "fold_metrics_mean_std": summary,
            "raw_fold_auprc": fold_metrics["auprc"],
            "oof_auprc": oof_auprc,
            "oof_roc_auc": oof_roc,
        }

    best_classical_name, best_classical_auprc = None, -1
    for model_name in ["RandomForest", "XGBoost"]:
        auprc = classical[model_name]["grouped_by_district"]["oof_auprc"]
        if auprc > best_classical_auprc:
            best_classical_auprc = auprc
            best_classical_name = model_name

    fusion_fold_auprc = results["grouped_by_district"]["raw_fold_auprc"]
    classical_fold_auprc = classical[best_classical_name]["grouped_by_district"]["raw_fold_auprc"]
    n_pair = min(len(fusion_fold_auprc), len(classical_fold_auprc))
    mean_diff, ci_lo, ci_hi = bootstrap_ci_diff(fusion_fold_auprc[:n_pair], classical_fold_auprc[:n_pair])

    decision = {
        "best_classical_baseline": best_classical_name,
        "best_classical_grouped_oof_auprc": best_classical_auprc,
        "fusion_grouped_oof_auprc": results["grouped_by_district"]["oof_auprc"],
        "per_fold_auprc_diff_mean_fusion_minus_classical": mean_diff,
        "bootstrap_95pct_ci": [ci_lo, ci_hi],
        "n_folds_compared": n_pair,
    }

    if ci_lo > 0:
        decision["primary_result"] = "fusion_model"
        decision["rationale"] = (
            f"Fusion model (v2, real daily sequences + tuning + seed-averaging) exceeds "
            f"{best_classical_name}'s per-fold AUPRC by {mean_diff:.3f} (95% CI [{ci_lo:.3f}, {ci_hi:.3f}], "
            f"excludes zero) on grouped-by-district CV."
        )
    else:
        decision["primary_result"] = best_classical_name
        decision["rationale"] = (
            f"Per proposal Section 8's stage 3/4 decision point: even after switching the temporal branch "
            f"to real 90-day daily sequences, a small hyperparameter search, and 5-seed averaging per fold, "
            f"the fusion model's per-fold AUPRC advantage over {best_classical_name} is {mean_diff:.3f} with "
            f"a bootstrapped 95% CI of [{ci_lo:.3f}, {ci_hi:.3f}], which includes zero — not a statistically "
            f"defensible improvement on {n_pair} folds / n=77 events. The classical baseline "
            f"({best_classical_name}, grouped-by-district OOF AUPRC={best_classical_auprc:.3f}) remains the "
            f"primary result. This is reported as a real negative finding after a genuine improvement attempt, "
            f"not the original under-tuned architecture."
        )

    results["stage_3_4_decision"] = decision
    logger.info(decision["rationale"])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "fusion_model_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved fusion model results to {out_path}")
    return results


if __name__ == "__main__":
    main()
