"""
Calibration / reliability plot (proposal Section 9 calls for both Brier score AND a
calibration plot; only the Brier score existed until now). Uses the REAL
leave-one-event-out out-of-fold predicted probabilities already saved by
classical_baselines.py (RandomForest — the primary result) and fusion_model_eval.py
(the fusion model), binned via sklearn's calibration_curve. No new model fitting —
this is a visualization of predictions that already exist.

A well-calibrated model's curve tracks the diagonal (predicted probability == observed
outcome frequency in that bin). Given a dataset this size, bins are necessarily coarse
(5 bins here, not the textbook 10) — stated explicitly on the plot and in the log, not
hidden.
"""

import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CalibrationPlot")

FEATURES_CSV = "data/processed/real_event_features.csv"
CLASSICAL_RESULTS_JSON = "results/classical_baseline_results.json"
FUSION_RESULTS_JSON = "results/fusion_model_results.json"
OUT_PATH = "results/figures/calibration_plot.png"
N_BINS = 5  # coarse, appropriate for a dataset this size — not the textbook 10


def main():
    with open(CLASSICAL_RESULTS_JSON) as f:
        classical = json.load(f)
    with open(FUSION_RESULTS_JSON) as f:
        fusion = json.load(f)

    import pandas as pd
    y = pd.read_csv(FEATURES_CSV)["label"].astype(int).values

    rf_probs = np.array(classical["RandomForest"]["leave_one_event_out"]["oof_probs"])
    fusion_probs = np.array(fusion["leave_one_event_out"]["oof_probs"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    curves_summary = {}
    for name, probs, color in [("RandomForest (primary)", rf_probs, "#2c7fb8"),
                                ("Fusion model", fusion_probs, "#d95f02")]:
        valid = ~np.isnan(probs)
        y_valid, p_valid = y[valid], probs[valid]
        if len(np.unique(y_valid)) < 2:
            continue
        frac_pos, mean_pred = calibration_curve(y_valid, p_valid, n_bins=N_BINS, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", color=color, label=name)
        curves_summary[name] = {
            "n_events": int(valid.sum()),
            "mean_predicted_prob_per_bin": mean_pred.tolist(),
            "observed_frequency_per_bin": frac_pos.tolist(),
        }
        logger.info(f"{name}: mean predicted prob per bin={np.round(mean_pred, 3).tolist()}, "
                    f"observed frequency per bin={np.round(frac_pos, 3).tolist()}")

    ax.set_xlabel("Mean predicted probability (real LOO out-of-fold predictions)")
    ax.set_ylabel("Observed outbreak frequency")
    ax.set_title(f"Calibration / reliability plot (n={len(y)} real events, {N_BINS} quantile bins)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    logger.info(f"Saved calibration plot to {OUT_PATH}")

    summary_path = "results/calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"n_bins": N_BINS, "binning_strategy": "quantile",
                   "note": f"n={len(y)} is small for calibration analysis; {N_BINS} bins used "
                           "deliberately rather than the textbook 10, to keep per-bin counts meaningful.",
                   "curves": curves_summary}, f, indent=2, default=str)
    logger.info(f"Saved calibration summary to {summary_path}")


if __name__ == "__main__":
    main()
