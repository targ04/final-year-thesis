"""
Selects a balanced subset of IMAGES images from the full cropped EyePACS
dataset, using a quota per DR grade plus a quality score (sharpness) to
break ties within oversupplied classes.

Assumes:
    - Images are already contour-cropped and resized to 512x512.
    - The labels CSV has two columns: 'image' (filename without extension)
      and 'level' (DR grade 0-4). This matches the standard EyePACS
      trainLabels.csv format. Adjust COLUMN NAMES below if different.
    - Filenames follow the EyePACS convention "<patient_id>_<left/right>.jpeg"
      so a patient_id can be extracted for later patient-level splitting.

Usage:
    1. Set INPUT_IMAGE_FOLDER, INPUT_CSV_PATH, OUTPUT_FOLDER below.
    2. Adjust IMAGES if you want a different subset size.
    3. Run. Selected images are copied to OUTPUT_FOLDER, and a manifest
       CSV (selected_manifest.csv) is written there with filename, level,
       patient_id, and quality_score columns.
"""

import os
import shutil
import cv2
import pandas as pd

# ---------------- CONFIG ----------------
IMAGES = 15000                                       # total subset size, adjust freely
INPUT_IMAGE_FOLDER = r"D:\Thesis Dataset\cropped and resized"    # <-- set your input image folder
INPUT_CSV_PATH = r"D:\Thesis Dataset\labels\all_labels.csv"           # <-- set your labels CSV path
OUTPUT_FOLDER = r"D:\Thesis Dataset\best quality images"        # <-- set your output folder

IMAGE_COL = "image"          # column name for filename (without extension) in CSV
LABEL_COL = "level"          # column name for DR grade (0-4) in CSV
IMAGE_EXT = ".png"          # extension of the actual image files on disk

# Fraction of the "remaining budget" (after reserving all of grades 3 & 4)
# allocated to grades 0, 1, 2. Must sum to 1.0.
FRACTIONS_FOR_MAJORITY_CLASSES = {0: 0.35, 1: 0.30, 2: 0.35}
# -----------------------------------------


def extract_patient_id(filename):
    """EyePACS filenames look like '10_left' or '10_right' -> patient_id '10'."""
    base = filename.split("_")[0]
    return base


def compute_sharpness(image_path):
    """Laplacian variance as a simple sharpness/quality proxy. Higher = sharper."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return -1.0
    return cv2.Laplacian(image, cv2.CV_64F).var()


def build_selection(df):
    df = df.copy()
    df["patient_id"] = df[IMAGE_COL].apply(extract_patient_id)

    reserved = df[df[LABEL_COL].isin([3, 4])]
    remaining_budget = IMAGES - len(reserved)

    if remaining_budget < 0:
        raise ValueError(
            f"IMAGES={IMAGES} is smaller than the number of grade 3+4 images "
            f"({len(reserved)}). Increase IMAGES or handle this case explicitly."
        )

    selected_rows = [reserved]

    for grade, fraction in FRACTIONS_FOR_MAJORITY_CLASSES.items():
        pool = df[df[LABEL_COL] == grade].copy()
        quota = int(remaining_budget * fraction)
        quota = min(quota, len(pool))

        print(f"Scoring {len(pool)} grade-{grade} images for sharpness...")
        pool["quality_score"] = pool[IMAGE_COL].apply(
            lambda fname: compute_sharpness(os.path.join(INPUT_IMAGE_FOLDER, fname + IMAGE_EXT))
        )

        pool_sorted = pool.sort_values("quality_score", ascending=False)
        selected_rows.append(pool_sorted.head(quota))

    reserved = reserved.copy()
    reserved["quality_score"] = reserved[IMAGE_COL].apply(
        lambda fname: compute_sharpness(os.path.join(INPUT_IMAGE_FOLDER, fname + IMAGE_EXT))
    )
    selected_rows[0] = reserved

    final_selection = pd.concat(selected_rows, ignore_index=True)
    return final_selection


def copy_selected_images(selection_df):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    total = len(selection_df)

    for i, row in enumerate(selection_df.itertuples(index=False), start=1):
        fname = getattr(row, IMAGE_COL) + IMAGE_EXT
        src = os.path.join(INPUT_IMAGE_FOLDER, fname)
        dst = os.path.join(OUTPUT_FOLDER, fname)

        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found, skipping.")

        if i % 500 == 0 or i == total:
            print(f"Copied {i}/{total} images")


def main():
    df = pd.read_csv(INPUT_CSV_PATH)
    df = df[[IMAGE_COL, LABEL_COL]].dropna()

    print("Available images per grade:")
    print(df[LABEL_COL].value_counts().sort_index())

    selection = build_selection(df)

    print("\nFinal selected images per grade:")
    print(selection[LABEL_COL].value_counts().sort_index())
    print(f"Total selected: {len(selection)}")

    copy_selected_images(selection)

    manifest_path = os.path.join(OUTPUT_FOLDER, "selected_manifest.csv")
    selection.to_csv(manifest_path, index=False)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()