"""
Merges the 72-row upazila-level rice blast dataset (real_upazila_features.csv) with
the 53 real wheat-blast events from the original 77-event dataset
(real_event_features.csv), replacing the 24 district-level rice-blast aggregate rows
with their finer-grained upazila-level disaggregation — not supplementing them (the
upazila table is a strict, higher-resolution breakdown of the same underlying Mahmud
et al. 2021 survey, so keeping both would double-count the same real observations).

Also merges the daily NASA POWER sequences (real_daily_sequences.npz) the same way,
so every event_id in the merged feature CSV has a matching entry in the merged
sequences file — required by fusion_model_eval.py's load_data().

Backs up the pre-merge files (with a timestamp-free "_77_original" / "_pre_merge"
suffix) before overwriting, so the original 77-event snapshot is never lost.
"""

import logging
import os
import shutil

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MergeUpazilaDataset")

ORIGINAL_FEATURES_CSV = "data/processed/real_event_features.csv"
UPAZILA_FEATURES_CSV = "data/processed/real_upazila_features.csv"
ORIGINAL_SEQ_NPZ = "data/processed/real_daily_sequences.npz"
UPAZILA_SEQ_NPZ = "data/processed/real_upazila_daily_sequences.npz"

BACKUP_FEATURES_CSV = "data/processed/real_event_features_77_original.csv"
BACKUP_SEQ_NPZ = "data/processed/real_daily_sequences_77_original.npz"


def merge_features():
    original = pd.read_csv(ORIGINAL_FEATURES_CSV)
    upazila = pd.read_csv(UPAZILA_FEATURES_CSV)

    n_rice_district_level = (original["disease"] == "rice_blast").sum()
    wheat = original[original["disease"] == "wheat_blast"].copy()

    logger.info(f"Original dataset: {len(original)} events ({n_rice_district_level} rice-blast "
                f"district-level, {len(wheat)} wheat-blast). Replacing rice-blast rows with "
                f"{len(upazila)} upazila-level rice-blast rows.")

    merged = pd.concat([upazila, wheat], ignore_index=True, sort=False)
    logger.info(f"Merged dataset: {len(merged)} events ({len(upazila)} rice-blast upazila-level + "
                f"{len(wheat)} wheat-blast) = {len(merged)}, {int(merged['label'].sum())} positive")

    if not os.path.exists(BACKUP_FEATURES_CSV):
        shutil.copy(ORIGINAL_FEATURES_CSV, BACKUP_FEATURES_CSV)
        logger.info(f"Backed up original 77-event dataset to {BACKUP_FEATURES_CSV}")

    merged.to_csv(ORIGINAL_FEATURES_CSV, index=False)
    logger.info(f"Saved merged dataset to {ORIGINAL_FEATURES_CSV} (overwriting the 77-event snapshot)")
    return merged


def merge_daily_sequences(merged_event_ids):
    original_seq = np.load(ORIGINAL_SEQ_NPZ, allow_pickle=True)
    upazila_seq = np.load(UPAZILA_SEQ_NPZ, allow_pickle=True)

    orig_event_ids = original_seq["event_ids"]
    orig_sequences = original_seq["sequences"]
    wheat_mask = np.array([eid.startswith("WB_") for eid in orig_event_ids])
    wheat_event_ids = orig_event_ids[wheat_mask]
    wheat_sequences = orig_sequences[wheat_mask]

    upazila_event_ids = upazila_seq["event_ids"]
    upazila_sequences = upazila_seq["sequences"]

    combined_event_ids = np.concatenate([upazila_event_ids, wheat_event_ids])
    combined_sequences = np.concatenate([upazila_sequences, wheat_sequences], axis=0)

    logger.info(f"Merged daily sequences: {combined_sequences.shape} for {len(combined_event_ids)} events "
                f"({len(upazila_event_ids)} upazila rice-blast + {len(wheat_event_ids)} wheat-blast)")

    # Sanity check: every event in the merged feature CSV must have a sequence, and vice versa.
    missing_from_seq = set(merged_event_ids) - set(combined_event_ids.tolist())
    extra_in_seq = set(combined_event_ids.tolist()) - set(merged_event_ids)
    if missing_from_seq:
        raise ValueError(f"{len(missing_from_seq)} merged events have NO daily sequence: {missing_from_seq}")
    if extra_in_seq:
        logger.warning(f"{len(extra_in_seq)} daily sequences have no matching merged event (will be unused): "
                        f"{extra_in_seq}")

    if not os.path.exists(BACKUP_SEQ_NPZ):
        shutil.copy(ORIGINAL_SEQ_NPZ, BACKUP_SEQ_NPZ)
        logger.info(f"Backed up original 77-event daily sequences to {BACKUP_SEQ_NPZ}")

    np.savez_compressed(ORIGINAL_SEQ_NPZ, sequences=combined_sequences, event_ids=combined_event_ids)
    logger.info(f"Saved merged daily sequences to {ORIGINAL_SEQ_NPZ}")


def main():
    merged = merge_features()
    merge_daily_sequences(merged["event_id"].tolist())
    logger.info("Merge complete. Downstream scripts (classical_baselines.py, fusion_model_eval.py, etc.) "
                "now operate on the merged dataset automatically — no path changes needed.")


if __name__ == "__main__":
    main()
