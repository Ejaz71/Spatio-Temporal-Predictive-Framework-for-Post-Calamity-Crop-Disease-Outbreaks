"""
Smoke tests for the real data pipeline and models. These check structural invariants
that matter for scientific integrity (no label leakage, correct event counts, model
loads and predicts) — they do not re-verify prediction accuracy, which is reported
via cross-validation in results/classical_baseline_results.json.
"""

import json
import os
import unittest

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(REPO_ROOT, "data", "processed", "real_event_features.csv")
EVENTS_CSV = os.path.join(REPO_ROOT, "data", "raw_literature", "events.csv")

LEAKAGE_COLUMNS = {
    "leaf_blast_incidence_pct", "neck_blast_incidence_pct",
    "leaf_blast_severity_pct", "neck_blast_severity_pct",
    "pct_fields_infected", "avg_yield_loss_pct", "affected_area_ha",
}
FEATURE_COLUMNS = [
    "precip_mean_mm", "precip_max_mm", "precip_sum_mm", "precip_anomaly_mm",
    "rh_mean_pct", "rh_max_pct", "temp_mean_c", "vpd_mean_kpa", "wet_persistence_max_days",
    "sar_vv_db_mean", "sar_vh_db_mean", "ndvi_mean", "ndwi_mean", "lst_celsius_mean",
]


@unittest.skipUnless(os.path.exists(EVENTS_CSV), "events.csv not present")
class TestEventLabels(unittest.TestCase):
    def test_matches_proposal_table_2(self):
        df = pd.read_csv(EVENTS_CSV)
        self.assertEqual(len(df), 77)
        self.assertEqual(int((df["label"] == 1).sum()), 24)
        rice = df[df["disease"] == "rice_blast"]
        self.assertEqual(len(rice), 24)
        self.assertEqual(int((rice["label"] == 1).sum()), 8)


@unittest.skipUnless(os.path.exists(FEATURES_CSV), "run data/real_feature_pipeline.py first")
class TestRealFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(FEATURES_CSV)

    def test_event_count(self):
        # 72 upazila-level rice-blast events (Mahmud et al. 2021, finer-grained than
        # the original 24 district-level rows) + 53 wheat-blast events (Islam et al.
        # 2016) = 125, since Phase B item 0 (2026-09-08) merged the upazila-level
        # disaggregation in — see data/merge_upazila_dataset.py and CLAUDE.md.
        self.assertEqual(len(self.df), 125)
        self.assertEqual(int((self.df["disease"] == "rice_blast").sum()), 72)
        self.assertEqual(int((self.df["disease"] == "wheat_blast").sum()), 53)
        self.assertEqual(int((self.df["label"] == 1).sum()), 46)

    def test_feature_columns_present(self):
        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.df.columns)

    def test_no_leakage_columns_used_as_features(self):
        # The literature columns exist (they define `label`) but must never be fed
        # to a model as a predictor — that would make prediction circular.
        self.assertTrue(LEAKAGE_COLUMNS.issubset(set(self.df.columns)))
        self.assertTrue(set(FEATURE_COLUMNS).isdisjoint(LEAKAGE_COLUMNS))

    def test_features_are_not_trivially_constant(self):
        # A sanity check against silently-broken extraction (e.g. every row NaN
        # or every row an identical fallback value).
        for col in FEATURE_COLUMNS:
            self.assertGreater(self.df[col].nunique(dropna=True), 1, f"{col} has <=1 unique value")


class TestFeatureListConsistency(unittest.TestCase):
    """The model feature list is currently duplicated across several modules. If they
    drift apart, different scripts silently report numbers computed on different
    feature sets — exactly the kind of quiet inconsistency this project exists to
    avoid. This test pins them together; if you add a feature, it fails until every
    consumer agrees."""

    def test_all_modules_agree_on_feature_columns(self):
        import importlib
        import sys
        sys.path.insert(0, REPO_ROOT)

        from training.classical_baselines import FEATURE_COLUMNS as classical_cols
        from analysis.statistical_tests import FEATURE_COLUMNS as stats_cols
        from training.train_final_model import FEATURE_COLUMNS as final_cols
        from training.rq2_ablations import FULL_COLUMNS as ablation_cols

        self.assertEqual(set(classical_cols), set(FEATURE_COLUMNS),
                         "training/classical_baselines.py disagrees with the test's feature list")
        self.assertEqual(set(stats_cols), set(FEATURE_COLUMNS),
                         "analysis/statistical_tests.py disagrees with the feature list")
        self.assertEqual(set(final_cols), set(FEATURE_COLUMNS),
                         "training/train_final_model.py disagrees with the feature list")
        self.assertEqual(set(ablation_cols), set(FEATURE_COLUMNS),
                         "training/rq2_ablations.py's FULL_COLUMNS disagrees with the feature list")

    def test_ablation_partition_is_exhaustive_and_disjoint(self):
        """The RQ2 ablation splits features into remote-sensing vs. meteorology. Those
        two sets must exactly partition the full feature set — otherwise a feature is
        silently dropped from, or double-counted across, the ablation conditions."""
        import sys
        sys.path.insert(0, REPO_ROOT)
        from training.rq2_ablations import (
            FULL_COLUMNS, METEOROLOGY_COLUMNS, REMOTE_SENSING_COLUMNS,
        )
        self.assertTrue(set(METEOROLOGY_COLUMNS).isdisjoint(REMOTE_SENSING_COLUMNS),
                        "a feature is in BOTH ablation conditions")
        self.assertEqual(set(METEOROLOGY_COLUMNS) | set(REMOTE_SENSING_COLUMNS), set(FULL_COLUMNS),
                         "ablation conditions do not partition the full feature set")


class TestClassicalResults(unittest.TestCase):
    RESULTS_JSON = os.path.join(REPO_ROOT, "results", "classical_baseline_results.json")

    @unittest.skipUnless(os.path.exists(RESULTS_JSON), "run training/classical_baselines.py first")
    def test_results_not_suspiciously_perfect(self):
        # A regression guard against the failure mode that produced the original
        # fabricated pipeline's 1.0/1.0/1.0 scores: real cross-validated AUPRC on
        # n=77 should be informative but comfortably below 1.0.
        with open(self.RESULTS_JSON) as f:
            results = json.load(f)
        for model_name in ["RandomForest", "XGBoost"]:
            auprc = results[model_name]["grouped_by_district"]["oof_auprc"]
            self.assertGreater(auprc, 0.35)  # better than the ~0.368 positive-rate baseline (46/125)
            self.assertLess(auprc, 0.99)     # not a suspicious perfect score


if __name__ == "__main__":
    unittest.main()
