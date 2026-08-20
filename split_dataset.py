"""
Patient-level stratified train/val/test split for the selected DR dataset.

Reads the manifest produced by select_best_images_from_manifest.py
(selected_manifest.csv) and splits it into train/val/test sets such that:

  1. No patient's images appear in more than one split (prevents leakage
     from same-patient left/right eye or multi-visit images).
  2. Each DR grade (0-4) is distributed across splits in approximately the
     configured TRAIN_FRAC / VAL_FRAC / TEST_FRAC proportions, computed by
     image count (not just patient count), so the already-quota-balanced
     class ratio from selection is preserved in every split.

A patient may have images spanning more than one grade (e.g. different
grade per eye). Each patient is assigned to a single split based on their
*dominant* grade (the most frequent grade among their own images), and all
of that patient's images move together.

Output: three manifest CSVs (train_manifest.csv, val_manifest.csv,
test_manifest.csv) with the same columns as the input manifest, written to
OUTPUT_DIR.
"""

import os
import random
import pandas as pd

# ---------------- CONFIG ----------------
MANIFEST_CSV_PATH = r"D:\Thesis Dataset\final dataset\selected_manifest.csv"
OUTPUT_DIR = r"D:\Thesis Dataset\final dataset\splits"

IMAGE_COL = "image"
LABEL_COL = "level"
PATIENT_COL = "patient_id"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

RANDOM_SEED = 42
# -----------------------------------------


def get_patient_dominant_grade(df):
    """Returns a Series mapping patient_id -> most frequent grade among their images."""
    return df.groupby(PATIENT_COL)[LABEL_COL].agg(lambda x: x.value_counts().idxmax())


def get_patient_image_counts(df):
    """Returns a Series mapping patient_id -> number of images for that patient."""
    return df.groupby(PATIENT_COL).size()


def assign_patients_to_splits(patient_ids, image_counts, rng):
    """
    Greedily assigns a shuffled list of patient_ids to train/val/test so that
    the running image-count totals approach TRAIN_FRAC/VAL_FRAC/TEST_FRAC of
    the group's total image count. Whole patients only move as a unit.
    """
    patient_ids = list(patient_ids)
    rng.shuffle(patient_ids)

    total_images = sum(image_counts[pid] for pid in patient_ids)
    target_train = total_images * TRAIN_FRAC
    target_val = total_images * VAL_FRAC
    # test gets whatever remains

    train_ids, val_ids, test_ids = [], [], []
    train_count = val_count = 0

    for pid in patient_ids:
        n = image_counts[pid]
        # Fill train until it reaches its target, then val, then everything else to test.
        if train_count < target_train:
            train_ids.append(pid)
            train_count += n
        elif val_count < target_val:
            val_ids.append(pid)
            val_count += n
        else:
            test_ids.append(pid)

    return train_ids, val_ids, test_ids


def build_split(df):
    assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-6, \
        "TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.0"

    rng = random.Random(RANDOM_SEED)

    patient_grade = get_patient_dominant_grade(df)
    patient_image_count = get_patient_image_counts(df)

    train_patients, val_patients, test_patients = [], [], []

    # Split independently within each grade group so class ratios are
    # preserved in every split, then combine.
    for grade in sorted(patient_grade.unique()):
        grade_patients = patient_grade[patient_grade == grade].index.tolist()
        t_ids, v_ids, te_ids = assign_patients_to_splits(
            grade_patients, patient_image_count, rng
        )
        train_patients.extend(t_ids)
        val_patients.extend(v_ids)
        test_patients.extend(te_ids)

    # Sanity check: no patient appears in more than one split.
    train_set, val_set, test_set = set(train_patients), set(val_patients), set(test_patients)
    assert not (train_set & val_set), "Patient leakage between train and val!"
    assert not (train_set & test_set), "Patient leakage between train and test!"
    assert not (val_set & test_set), "Patient leakage between val and test!"

    train_df = df[df[PATIENT_COL].isin(train_set)].reset_index(drop=True)
    val_df = df[df[PATIENT_COL].isin(val_set)].reset_index(drop=True)
    test_df = df[df[PATIENT_COL].isin(test_set)].reset_index(drop=True)

    return train_df, val_df, test_df


def print_summary(name, split_df, full_df):
    n_images = len(split_df)
    n_patients = split_df[PATIENT_COL].nunique()
    pct = 100 * n_images / len(full_df)
    print(f"\n{name}: {n_images} images ({pct:.1f}%), {n_patients} unique patients")
    print(split_df[LABEL_COL].value_counts().sort_index().rename("count").to_frame())


def main():
    df = pd.read_csv(MANIFEST_CSV_PATH)
    df = df[[IMAGE_COL, LABEL_COL, PATIENT_COL]].dropna().copy()

    train_df, val_df, test_df = build_split(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train_manifest.csv")
    val_path = os.path.join(OUTPUT_DIR, "val_manifest.csv")
    test_path = os.path.join(OUTPUT_DIR, "test_manifest.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print_summary("Train", train_df, df)
    print_summary("Val", val_df, df)
    print_summary("Test", test_df, df)

    print(f"\nManifests written to:\n  {train_path}\n  {val_path}\n  {test_path}")


if __name__ == "__main__":
    main()