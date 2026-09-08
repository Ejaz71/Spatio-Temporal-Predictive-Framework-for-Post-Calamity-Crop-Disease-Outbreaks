"""
Exploratory analysis of the SAR water-extent features, run BEFORE deciding whether to
add them to the model feature set.

Why this script exists as a separate, prior step
------------------------------------------------
The motivating hypothesis is specific and falsifiable: the remote-sensing modality has
so far underperformed meteorology (see results/rq2_ablation_results.json), and one
candidate explanation is representational rather than physical — `sar_vv_db_mean` is a
bounding-box MEAN, which cannot express partial surface water, so a real hydrological
signal could be present in the imagery and invisible to the model. `water_extent_frac`
(fraction of pixels below the open-water backscatter threshold) measures exactly what
the mean discards.

If we simply added the column to the feature set and re-ran everything, we would learn
only that some numbers changed. This script instead asks the question directly, and its
answer determines whether integration is warranted:

  1. Is the feature non-degenerate? (distribution, missingness)
  2. Is it actually *new* information, or a re-expression of sar_vv_db_mean? (rank
     correlation with the existing SAR/optical columns — a |rho| near 1 would mean the
     feature is redundant no matter how well it performs)
  3. Is it associated with outbreak occurrence? (Mann-Whitney U + rank-biserial)
  4. Is it associated with blast SEVERITY? (Spearman, and partial Spearman controlling
     for sar_vv_db_mean — the latter is the sharpest test of incremental value)

Confounding note: rice and wheat events differ in region, season and crop, so a pooled
outbreak test partly measures "rice vs wheat" rather than "outbreak vs not". Every
outcome test is therefore reported both pooled and within rice-blast events only.
Severity is rice-only by construction (the wheat rows have no severity percentages).

Multiplicity: p-values are FDR-corrected (Benjamini-Hochberg) within each family of
tests, and the three water variants are three thresholds of one underlying quantity, so
they are treated as a sensitivity check on each other, not as three independent findings.

Usage (run from repo root):
    python3 analysis/water_extent_exploration.py
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
logger = logging.getLogger("WaterExtentExploration")

FEATURES_CSV = "data/processed/real_event_features.csv"
RESULTS_DIR = "results"

WATER_COLUMNS = ["water_extent_frac", "water_extent_frac_strict", "water_extent_frac_max"]
COMPARISON_COLUMNS = ["sar_vv_db_mean", "sar_vh_db_mean", "ndwi_mean", "ndvi_mean",
                      "precip_sum_mm", "precip_anomaly_mm"]
SEVERITY_COLUMNS = ["leaf_blast_severity_pct", "neck_blast_severity_pct",
                    "leaf_blast_incidence_pct", "neck_blast_incidence_pct"]

# Below this many non-missing observations a correlation/test is not reported at all,
# rather than reported with a caveat — small-n rank statistics are too unstable to be
# worth putting in a report even hedged.
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


def distribution_summary(df):
    """Is the feature non-degenerate and physically plausible? A water fraction should
    sit in [0, 1], vary across events, and be low-but-nonzero in a dry-season window."""
    out = {}
    for col in WATER_COLUMNS:
        s = df[col].dropna()
        if s.empty:
            out[col] = {"note": "no values extracted"}
            continue
        out[col] = {
            "n_non_missing": int(s.size),
            "n_missing": int(df[col].isna().sum()),
            "mean": round(float(s.mean()), 5),
            "std": round(float(s.std()), 5),
            "min": round(float(s.min()), 5),
            "q25": round(float(s.quantile(0.25)), 5),
            "median": round(float(s.median()), 5),
            "q75": round(float(s.quantile(0.75)), 5),
            "max": round(float(s.max()), 5),
            "n_unique": int(s.nunique()),
            "within_unit_interval": bool(s.min() >= 0.0 and s.max() <= 1.0),
        }
        for disease, sub in df.groupby("disease"):
            ss = sub[col].dropna()
            if not ss.empty:
                out[col][f"median_{disease}"] = round(float(ss.median()), 5)
                out[col][f"n_{disease}"] = int(ss.size)
    return out


def redundancy_check(df):
    """Spearman rank correlation of each water column against the existing features.

    This is the gatekeeping question: if water_extent_frac is essentially a monotone
    re-expression of sar_vv_db_mean, it cannot rescue the remote-sensing modality no
    matter what its univariate association looks like — the model already had that
    information. A moderate correlation is expected and healthy (both are computed from
    the same backscatter), near-perfect is fatal to the hypothesis."""
    out = {}
    for col in WATER_COLUMNS:
        row = {}
        for other in COMPARISON_COLUMNS + [c for c in WATER_COLUMNS if c != col]:
            if other not in df.columns:
                continue
            pair = df[[col, other]].dropna()
            if len(pair) < MIN_N:
                continue
            rho, p = stats.spearmanr(pair[col], pair[other])
            row[other] = {"spearman_rho": round(float(rho), 4), "p_value": round(float(p), 6),
                          "n": int(len(pair))}
        out[col] = row
    return out


def outcome_association(df, subset_name):
    """Mann-Whitney U on outbreak vs non-outbreak, with rank-biserial effect size."""
    pos, neg = df[df["label"] == 1], df[df["label"] == 0]
    rows = []
    for col in WATER_COLUMNS:
        a, b = pos[col].dropna().values, neg[col].dropna().values
        if len(a) < MIN_N // 2 or len(b) < MIN_N // 2:
            rows.append({"feature": col, "subset": subset_name,
                         "note": f"insufficient data (n_pos={len(a)}, n_neg={len(b)})"})
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        r_rb = 1 - (2 * u) / (len(a) * len(b))
        rows.append({
            "feature": col, "subset": subset_name,
            "n_outbreak": int(len(a)), "n_no_outbreak": int(len(b)),
            "median_outbreak": round(float(np.median(a)), 5),
            "median_no_outbreak": round(float(np.median(b)), 5),
            "U": float(u), "p_value": float(p),
            "rank_biserial_r": round(float(r_rb), 4),
            "effect_size_note": effect_size_note(r_rb),
        })
    tested = [r for r in rows if "p_value" in r]
    if tested:
        q = benjamini_hochberg([r["p_value"] for r in tested])
        for r, qv in zip(tested, q):
            r["fdr_q_value"] = round(float(qv), 6)
    return rows


def _partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z, via rank residuals.

    Ranks all three, regresses rank(x) and rank(y) on rank(z) by ordinary least squares,
    and correlates the residuals. This is the standard rank-based partial correlation;
    it answers "does water extent track severity beyond what mean backscatter already
    explains?", which is precisely the incremental-value question. Returns (rho, p, n)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    zc = np.column_stack([np.ones_like(rz), rz])
    res_x = rx - zc @ np.linalg.lstsq(zc, rx, rcond=None)[0]
    res_y = ry - zc @ np.linalg.lstsq(zc, ry, rcond=None)[0]
    rho, p = stats.pearsonr(res_x, res_y)  # Pearson on rank residuals = partial Spearman
    return float(rho), float(p), int(len(rx))


def severity_association(df):
    """Spearman vs each severity/incidence outcome, plus the partial correlation
    controlling for sar_vv_db_mean (the incremental-value test)."""
    rice = df[df["disease"] == "rice_blast"]
    rows = []
    for col in WATER_COLUMNS:
        for sev in SEVERITY_COLUMNS:
            if sev not in rice.columns:
                continue
            pair = rice[[col, sev, "sar_vv_db_mean"]].dropna()
            if len(pair) < MIN_N:
                continue
            rho, p = stats.spearmanr(pair[col], pair[sev])
            entry = {"feature": col, "outcome": sev, "n": int(len(pair)),
                     "spearman_rho": round(float(rho), 4), "p_value": float(p)}
            prho, pp, pn = _partial_spearman(pair[col].values, pair[sev].values,
                                             pair["sar_vv_db_mean"].values)
            entry["partial_rho_controlling_sar_vv"] = round(prho, 4)
            entry["partial_p_value"] = round(pp, 6)
            entry["retains_signal_after_control"] = bool(abs(prho) >= 0.15 and pp < 0.05)
            # Distinguish the two very different reasons a correlation can fail to
            # survive the control, because they license opposite conclusions: an
            # ATTENUATED correlation means sar_vv_db_mean already carried the signal
            # (bad for this feature's case), whereas a correlation that was never
            # significant to begin with says nothing about redundancy at all.
            if entry["retains_signal_after_control"]:
                entry["interpretation"] = "retains signal beyond sar_vv_db_mean"
            elif p >= 0.05:
                entry["interpretation"] = "no significant raw association to explain"
            elif abs(prho) < abs(rho):
                entry["interpretation"] = "attenuated by sar_vv_db_mean (largely redundant)"
            else:
                entry["interpretation"] = "raw association significant, partial not"
            rows.append(entry)
    if rows:
        q = benjamini_hochberg([r["p_value"] for r in rows])
        for r, qv in zip(rows, q):
            r["fdr_q_value"] = round(float(qv), 6)
    return sorted(rows, key=lambda r: r["p_value"])


SATURATION_FLAG = 0.90


def implausible_value_check(df):
    """Flags events whose wettest scene classifies (nearly) the ENTIRE ~13 km box as
    open water.

    In a Dec-Mar dry-season window over inland Bangladesh this is not physically
    credible: it indicates a degraded scene rather than real inundation — a swath-edge
    tile that is mostly fill, or a radar-shadow/layover artifact, both of which push
    gamma0 uniformly low and mimic specular water. It matters most for
    `water_extent_frac_max`, which by taking a maximum over scenes is the variant most
    exposed to a single bad observation; the two mean-based variants average such a
    scene against its neighbours. Flagged, not silently dropped — the decision about
    what to do with these belongs in the write-up, not hidden in a filter."""
    sat = df[df["water_extent_frac_max"] >= SATURATION_FLAG]
    return {
        "saturation_threshold": SATURATION_FLAG,
        "n_saturated_scenes": int(len(sat)),
        "saturated_events": [
            {"event_id": r["event_id"],
             "water_extent_frac_max": round(float(r["water_extent_frac_max"]), 4),
             "water_extent_frac": round(float(r["water_extent_frac"]), 4),
             "n_s1_scenes": int(r["n_s1_scenes"]) if pd.notna(r["n_s1_scenes"]) else None}
            for _, r in sat.iterrows()
        ],
    }


def missingness_note(df):
    """Water extent is missing exactly where Sentinel-1 had no usable scene. Confirm
    that, so the report can state the missingness mechanism rather than guess it."""
    n_missing = int(df["water_extent_frac"].isna().sum())
    no_scene = df["n_s1_scenes"].fillna(0) == 0
    return {
        "n_missing_water_extent": n_missing,
        "n_events_with_zero_s1_scenes": int(no_scene.sum()),
        "missing_all_explained_by_zero_scenes":
            bool(df.loc[df["water_extent_frac"].isna(), "n_s1_scenes"].fillna(0).eq(0).all()),
        "missing_rate_pct": round(100.0 * n_missing / len(df), 2),
    }


def main():
    df = pd.read_csv(FEATURES_CSV)
    missing_cols = [c for c in WATER_COLUMNS if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"{missing_cols} not in {FEATURES_CSV} — run data/add_water_extent_feature.py first")

    n_pop = int(df["water_extent_frac"].notna().sum())
    logger.info(f"{n_pop}/{len(df)} events have a water-extent value")
    if n_pop < 60:
        logger.warning(f"Only {n_pop} events populated — backfill is incomplete. Results below are "
                       f"PRELIMINARY and must not be reported as final.")

    results = {
        "n_events": int(len(df)),
        "n_populated": n_pop,
        "backfill_complete": bool(n_pop >= int((df["n_s1_scenes"].fillna(0) > 0).sum())),
        "distribution": distribution_summary(df),
        "implausible_values": implausible_value_check(df),
        "missingness": missingness_note(df),
        "redundancy_vs_existing_features": redundancy_check(df),
        "outcome_association_pooled": outcome_association(df, "all_events"),
        "outcome_association_rice_only": outcome_association(df[df["disease"] == "rice_blast"], "rice_blast_only"),
        "severity_association_rice": severity_association(df),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/water_extent_exploration.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ---- readable summary -------------------------------------------------
    logger.info("=" * 72)
    logger.info("DISTRIBUTION")
    for col, d in results["distribution"].items():
        if "note" in d:
            logger.info(f"  {col}: {d['note']}")
            continue
        logger.info(f"  {col}: median={d['median']:.4f} IQR=[{d['q25']:.4f},{d['q75']:.4f}] "
                    f"range=[{d['min']:.4f},{d['max']:.4f}] n={d['n_non_missing']}")

    logger.info("REDUNDANCY (Spearman vs existing features; |rho|>0.9 would mean no new information)")
    for col, row in results["redundancy_vs_existing_features"].items():
        top = sorted(row.items(), key=lambda kv: -abs(kv[1]["spearman_rho"]))[:3]
        logger.info(f"  {col}: " + ", ".join(f"{k} rho={v['spearman_rho']:+.3f}" for k, v in top))

    for key in ("outcome_association_pooled", "outcome_association_rice_only"):
        logger.info(f"OUTBREAK ASSOCIATION — {key.replace('outcome_association_', '')}")
        for r in results[key]:
            if "note" in r:
                logger.info(f"  {r['feature']}: {r['note']}")
                continue
            logger.info(f"  {r['feature']}: median {r['median_outbreak']:.4f} (outbreak) vs "
                        f"{r['median_no_outbreak']:.4f} (no outbreak), p={r['p_value']:.4f}, "
                        f"q={r['fdr_q_value']:.4f}, r={r['rank_biserial_r']:+.3f} ({r['effect_size_note']})")

    logger.info("SEVERITY ASSOCIATION (rice only; partial rho controls for sar_vv_db_mean)")
    for r in results["severity_association_rice"][:8]:
        logger.info(f"  {r['feature']} vs {r['outcome']}: rho={r['spearman_rho']:+.3f} "
                    f"(p={r['p_value']:.4f}, q={r['fdr_q_value']:.4f}, n={r['n']}), "
                    f"partial rho={r['partial_rho_controlling_sar_vv']:+.3f} — {r['interpretation']}")

    if results["implausible_values"]["n_saturated_scenes"]:
        logger.warning("IMPLAUSIBLE VALUES — inspect before trusting water_extent_frac_max:")
        for e in results["implausible_values"]["saturated_events"]:
            logger.warning(f"  {e['event_id']}: max={e['water_extent_frac_max']:.4f}, "
                           f"mean={e['water_extent_frac']:.4f}, n_s1_scenes={e['n_s1_scenes']}")

    logger.info("=" * 72)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
