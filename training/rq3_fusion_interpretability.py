"""
RQ3 interpretability for the fusion model (proposal Section 9): permutation
importance over its real input features, and inspection of its real temporal
attention weights — complementing the SHAP-based RandomForest attribution in
classical_baselines.py with an interpretability analysis of the fusion model itself.

Convention (matches the existing SHAP analysis, stated explicitly rather than
silently): an ensemble of N_SEEDS models is fit on ALL events (same
globally-selected hyperparameters as the LOO fusion evaluation), and both the
permutation importances and the attention-weight analysis below are computed
in-sample on that fit — standard practice for interpretability (not a claim about
held-out predictive skill, which is what fusion_model_eval.py's OOF metrics are for).

Two analyses:
1. Permutation importance: for each of the 5 spatial (remote-sensing) features and
   each of the 6 real daily-sequence (meteorological) channels, shuffle that
   feature/channel across events K times, re-run inference through the already-fit
   ensemble (no retraining — a forward pass only), and report the mean AUPRC drop.
2. Attention weights: extract the model's real softmax attention weights over the
   90-day sequence (right-aligned so day 90 = window_end, day 1 = 90 days before),
   averaged across the seed ensemble and across events, both overall and split by
   disease. Cross-checks the proposal's rice-blast/humidity vs. wheat-blast/temperature
   hypothesis by comparing each event's attention-weighted mean of RH/temperature
   against its plain (unweighted) mean, split by disease, via Mann-Whitney U (same
   test used throughout Phase A) — a positive gap means the model is disproportionately
   attending to higher-RH (or higher-temperature) days within that disease's events,
   not just averaging uniformly over the window.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from models.fusion_model_tabular import DAILY_SEQUENCE_FEATURES, SPATIAL_FEATURES, TabularFusionModel
from training.fusion_model_eval import DAILY_SEQ_NPZ, FEATURES_CSV, select_hyperparams

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RQ3FusionInterpretability")

RESULTS_DIR = "results"
N_SEEDS = 5
N_PERMUTATIONS = 30  # repeats per feature, for a stable importance estimate


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    seq_data = np.load(DAILY_SEQ_NPZ, allow_pickle=True)
    seq_by_event = dict(zip(seq_data["event_ids"], seq_data["sequences"]))
    daily_sequences = np.stack([seq_by_event[eid] for eid in df["event_id"]], axis=0)
    X_spatial = df[SPATIAL_FEATURES].copy()
    y = df["label"].astype(int).values
    groups = df["district"].values
    disease = df["disease"].values
    return df, X_spatial, daily_sequences, y, groups, disease


def fit_ensemble(X_sp, X_seq, y, hp):
    """Fits N_SEEDS models on ALL events (in-sample, interpretability convention —
    see module docstring), returns the fitted models for repeated forward passes."""
    import torch.nn as nn

    models = []
    for s in range(N_SEEDS):
        torch.manual_seed(s)
        model = TabularFusionModel(
            spatial_dim=X_sp.shape[1], n_daily_features=X_seq.shape[2],
            hidden_dim=hp["hidden_dim"], dropout=hp["dropout"],
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        n_pos, n_neg = max(1, (y == 1).sum()), max(1, (y == 0).sum())
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32))
        X_sp_t = torch.tensor(X_sp, dtype=torch.float32)
        X_seq_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        model.train()
        for _ in range(hp["epochs"]):
            optimizer.zero_grad()
            logits, _ = model(X_sp_t, X_seq_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
        model.eval()
        models.append(model)
    return models


def ensemble_predict(models, X_sp, X_seq):
    """Mean probability and mean attention weights across the seed ensemble, on
    whatever (possibly permuted) inputs are passed in — a forward pass only, no
    retraining, which is what makes permutation importance cheap here."""
    all_probs, all_attn = [], []
    X_sp_t = torch.tensor(X_sp, dtype=torch.float32)
    X_seq_t = torch.tensor(X_seq, dtype=torch.float32)
    with torch.no_grad():
        for model in models:
            logits, attn = model(X_sp_t, X_seq_t)
            all_probs.append(torch.sigmoid(logits).numpy())
            all_attn.append(attn.numpy())
    return np.mean(all_probs, axis=0), np.mean(all_attn, axis=0)


def permutation_importance(models, X_sp, X_seq, y, baseline_auprc, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    importances = {}

    for i, feat_name in enumerate(SPATIAL_FEATURES):
        drops = []
        for _ in range(N_PERMUTATIONS):
            X_sp_perm = X_sp.copy()
            perm_idx = rng.permutation(n)
            X_sp_perm[:, i] = X_sp_perm[perm_idx, i]
            probs, _ = ensemble_predict(models, X_sp_perm, X_seq)
            drops.append(baseline_auprc - average_precision_score(y, probs))
        importances[feat_name] = {"mean_auprc_drop": float(np.mean(drops)), "std_auprc_drop": float(np.std(drops)),
                                   "modality": "remote_sensing"}

    for i, feat_name in enumerate(DAILY_SEQUENCE_FEATURES):
        drops = []
        for _ in range(N_PERMUTATIONS):
            X_seq_perm = X_seq.copy()
            perm_idx = rng.permutation(n)
            X_seq_perm[:, :, i] = X_seq_perm[perm_idx, :, i]
            probs, _ = ensemble_predict(models, X_sp, X_seq_perm)
            drops.append(baseline_auprc - average_precision_score(y, probs))
        importances[feat_name] = {"mean_auprc_drop": float(np.mean(drops)), "std_auprc_drop": float(np.std(drops)),
                                   "modality": "meteorology_daily_sequence"}

    return importances


def attention_weight_analysis(attn_weights, daily_sequences_raw, disease):
    """attn_weights: (n_events, 90) real softmax weights, day 90 = window_end.
    daily_sequences_raw: (n_events, 90, 6) UNSCALED real daily values, so the
    attention-weighted means below are in real physical units (%RH, degC)."""
    rh_idx = DAILY_SEQUENCE_FEATURES.index("rh_pct")
    temp_idx = DAILY_SEQUENCE_FEATURES.index("temp_c")

    rh_series = daily_sequences_raw[:, :, rh_idx]
    temp_series = daily_sequences_raw[:, :, temp_idx]

    attn_weighted_rh = np.sum(attn_weights * rh_series, axis=1)
    plain_mean_rh = np.mean(rh_series, axis=1)
    rh_excess = attn_weighted_rh - plain_mean_rh  # >0: model attends more to higher-RH days than uniform

    attn_weighted_temp = np.sum(attn_weights * temp_series, axis=1)
    plain_mean_temp = np.mean(temp_series, axis=1)
    temp_excess = attn_weighted_temp - plain_mean_temp

    is_rice = disease == "rice_blast"
    is_wheat = disease == "wheat_blast"

    def mw_test(vals_a, vals_b):
        if len(vals_a) < 2 or len(vals_b) < 2:
            return {"note": "insufficient data"}
        u, p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        r_rb = 1 - (2 * u) / (len(vals_a) * len(vals_b))
        return {"n_a": int(len(vals_a)), "n_b": int(len(vals_b)), "median_a": float(np.median(vals_a)),
                "median_b": float(np.median(vals_b)), "U": float(u), "p_value": float(p),
                "rank_biserial_r": round(float(r_rb), 4)}

    hypothesis_check = {
        "rh_attention_excess_rice_vs_wheat": mw_test(rh_excess[is_rice], rh_excess[is_wheat]),
        "temp_attention_excess_rice_vs_wheat": mw_test(temp_excess[is_rice], temp_excess[is_wheat]),
        "interpretation": (
            "rh_attention_excess = (attention-weighted mean RH) - (plain mean RH) per event; positive means the "
            "model's attention disproportionately favors higher-humidity days within that event's window rather "
            "than weighting the whole window uniformly. If the proposal's rice-blast/humidity hypothesis is "
            "reflected in what the model actually learned, rice_blast events should show a larger rh_attention_excess "
            "than wheat_blast events (median_a vs median_b above, 'a'=rice_blast, 'b'=wheat_blast). Symmetric "
            "logic for temp_attention_excess and the wheat-blast/temperature hypothesis. This tests what the "
            "trained model's attention mechanism actually does, not the underlying epidemiology directly — a null "
            "result here means the model didn't learn that pattern (or attention isn't capturing it), not that the "
            "epidemiological hypothesis itself is false."
        ),
    }

    day_position = np.arange(1, attn_weights.shape[1] + 1)  # 1..90, 90 = window_end
    attention_curve = {
        "day_position_relative_to_window_end": day_position.tolist(),
        "mean_attention_all_events": attn_weights.mean(axis=0).tolist(),
        "mean_attention_rice_blast": attn_weights[is_rice].mean(axis=0).tolist() if is_rice.sum() else None,
        "mean_attention_wheat_blast": attn_weights[is_wheat].mean(axis=0).tolist() if is_wheat.sum() else None,
    }

    return {
        "attention_curve": attention_curve,
        "humidity_temperature_hypothesis_check": hypothesis_check,
    }


def main():
    df, X_spatial, daily_sequences, y, groups, disease = load_data()

    sp_imputer = SimpleImputer(strategy="median")
    sp_scaler = StandardScaler()
    X_sp_scaled = sp_scaler.fit_transform(sp_imputer.fit_transform(X_spatial))

    n_feat = daily_sequences.shape[2]
    seq_scaler = StandardScaler()
    seq_scaler.fit(daily_sequences.reshape(-1, n_feat))
    X_seq_scaled = seq_scaler.transform(daily_sequences.reshape(-1, n_feat)).reshape(daily_sequences.shape)

    hp = select_hyperparams(X_spatial, daily_sequences, y, groups, seed=0)
    logger.info(f"Fitting {N_SEEDS}-seed ensemble on all {len(y)} events (in-sample, interpretability "
                f"convention) with hyperparams: {hp}")
    models = fit_ensemble(X_sp_scaled, X_seq_scaled, y, hp)

    probs, attn_weights = ensemble_predict(models, X_sp_scaled, X_seq_scaled)
    baseline_auprc = average_precision_score(y, probs)
    logger.info(f"In-sample baseline AUPRC (ensemble fit on all data): {baseline_auprc:.3f}")

    logger.info(f"Running permutation importance ({N_PERMUTATIONS} shuffles/feature)...")
    importances = permutation_importance(models, X_sp_scaled, X_seq_scaled, y, baseline_auprc, seed=0)
    ranked = sorted(importances.items(), key=lambda kv: -kv[1]["mean_auprc_drop"])
    logger.info("Permutation importance ranking (mean AUPRC drop when shuffled):")
    for name, info in ranked:
        logger.info(f"  {name} ({info['modality']}): {info['mean_auprc_drop']:.4f} +/- {info['std_auprc_drop']:.4f}")

    attn_analysis = attention_weight_analysis(attn_weights, daily_sequences, disease)
    logger.info(attn_analysis["humidity_temperature_hypothesis_check"]["interpretation"])
    logger.info(f"rh_attention_excess: rice_blast median={attn_analysis['humidity_temperature_hypothesis_check']['rh_attention_excess_rice_vs_wheat'].get('median_a')}, "
                f"wheat_blast median={attn_analysis['humidity_temperature_hypothesis_check']['rh_attention_excess_rice_vs_wheat'].get('median_b')}, "
                f"p={attn_analysis['humidity_temperature_hypothesis_check']['rh_attention_excess_rice_vs_wheat'].get('p_value')}")
    logger.info(f"temp_attention_excess: rice_blast median={attn_analysis['humidity_temperature_hypothesis_check']['temp_attention_excess_rice_vs_wheat'].get('median_a')}, "
                f"wheat_blast median={attn_analysis['humidity_temperature_hypothesis_check']['temp_attention_excess_rice_vs_wheat'].get('median_b')}, "
                f"p={attn_analysis['humidity_temperature_hypothesis_check']['temp_attention_excess_rice_vs_wheat'].get('p_value')}")

    results = {
        "n_events": len(y), "n_positive": int(y.sum()), "n_seeds_ensemble": N_SEEDS,
        "hyperparameters": hp,
        "in_sample_baseline_auprc": float(baseline_auprc),
        "caveat": "Permutation importances and attention analysis below are computed in-sample on an "
                  "ensemble fit to ALL events — same interpretability convention as the RandomForest SHAP "
                  "analysis (see CLAUDE.md Known Issues). Not a claim about held-out predictive skill; see "
                  "fusion_model_results.json for that.",
        "permutation_importance": importances,
        "permutation_importance_ranked": [{"feature": n, **info} for n, info in ranked],
        "attention_analysis": attn_analysis,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "rq3_fusion_interpretability.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved RQ3 fusion interpretability results to {out_path}")
    return results


if __name__ == "__main__":
    main()
