"""
Reads the sharpness manifest CSV and copies the top-quality images by DR grade
from the input folder to the output folder.

This version preserves most of the grade 3 and grade 4 images, allocates the
remaining budget across grades 0/1/2 using the same quota logic as the
original selection script, and ignores missing files so that as close to
IMAGES images as possible are copied when enough valid files exist.

Any rounding shortfall from the initial quota pass is backfilled
proportionally from grades 0/1/2 (not from the unrestricted pool), so the
class ratio designed by FRACTIONS_FOR_MAJORITY_CLASSES is preserved. Only if
grades 0/1/2 are fully exhausted does it fall back to the unrestricted pool,
with a printed warning.
"""

import os
import shutil
import pandas as pd

# ---------------- CONFIG ----------------
IMAGES = 15000
INPUT_IMAGE_FOLDER = r"D:\Thesis Dataset\cropped and resized"
MANIFEST_CSV_PATH = r"D:\Thesis Dataset\labels\sharpness_manifest.csv"
OUTPUT_FOLDER = r"D:\Thesis Dataset\best images"

IMAGE_COL = "image"
LABEL_COL = "level"
IMAGE_EXT = ".png"

# Fraction of the remaining budget (after reserving all grade 3 & 4 images)
# allocated to grades 0, 1, 2. Must sum to 1.0.
FRACTIONS_FOR_MAJORITY_CLASSES = {0: 0.35, 1: 0.30, 2: 0.35}
# -----------------------------------------


def extract_patient_id(filename):
    return filename.split("_")[0]


def build_selection(df):
    df = df.copy()
    df = df[df["image_exists"]].copy()
    df["patient_id"] = df[IMAGE_COL].apply(extract_patient_id)

    reserved_pool = df[df[LABEL_COL].isin([3, 4])].copy()
    reserved_pool = reserved_pool.sort_values("quality_score", ascending=False)

    if len(reserved_pool) > IMAGES:
        raise ValueError(
            f"IMAGES={IMAGES} is smaller than the number of available grade 3+4 images "
            f"({len(reserved_pool)}). Increase IMAGES or handle this case explicitly."
        )

    selected_frames = [reserved_pool]
    remaining_budget = IMAGES - len(reserved_pool)

    if remaining_budget > 0:
        for grade, fraction in FRACTIONS_FOR_MAJORITY_CLASSES.items():
            pool = df[df[LABEL_COL] == grade].copy()
            pool = pool.sort_values("quality_score", ascending=False)
            quota = int(remaining_budget * fraction)
            quota = min(quota, len(pool))
            if quota > 0:
                selected_frames.append(pool.head(quota))

    selection = pd.concat(selected_frames, ignore_index=True)
    selection = selection.drop_duplicates(subset=[IMAGE_COL]).reset_index(drop=True)

    # --- Proportional backfill for rounding shortfall ---
    # int() truncation in the quota pass above almost always leaves a small
    # gap versus IMAGES. Fill that gap from grades 0/1/2 in the same
    # proportions as FRACTIONS_FOR_MAJORITY_CLASSES, using each grade's next
    # best-remaining images, instead of pulling from the unrestricted pool
    # (which would silently skew the selection back toward grade 0).
    if len(selection) < IMAGES:
        shortfall = IMAGES - len(selection)
        print(f"Backfilling {shortfall} image(s) proportionally across grades 0/1/2...")

        for grade, fraction in FRACTIONS_FOR_MAJORITY_CLASSES.items():
            if shortfall <= 0:
                break

            pool = df[df[LABEL_COL] == grade].copy()
            pool = pool[~pool[IMAGE_COL].isin(selection[IMAGE_COL])]
            pool = pool.sort_values("quality_score", ascending=False)

            take = min(int(round(shortfall * fraction)), len(pool))
            if take > 0:
                selection = pd.concat([selection, pool.head(take)], ignore_index=True)
                selection = selection.drop_duplicates(subset=[IMAGE_COL]).reset_index(drop=True)
                shortfall = IMAGES - len(selection)

    # --- Unrestricted fallback, only if grades 0/1/2 are fully exhausted ---
    if len(selection) < IMAGES:
        remaining = IMAGES - len(selection)
        available_pool = df.loc[~df[IMAGE_COL].isin(selection[IMAGE_COL])].copy()

        if len(available_pool) < remaining:
            print(
                f"Warning: only {len(available_pool)} additional valid images are available, "
                f"but {remaining} more were needed. Proceeding with {len(selection) + len(available_pool)} "
                f"images instead of the requested IMAGES={IMAGES}."
            )
            remaining = len(available_pool)
        else:
            print(
                f"Warning: grades 0/1/2 pools exhausted before reaching IMAGES={IMAGES}. "
                f"Falling back to the unrestricted pool for the remaining {remaining} image(s), "
                "which may skew the intended class ratio."
            )

        available_pool = available_pool.sort_values("quality_score", ascending=False)
        selection = pd.concat([selection, available_pool.head(remaining)], ignore_index=True)
        selection = selection.drop_duplicates(subset=[IMAGE_COL]).reset_index(drop=True)

    if len(selection) != IMAGES:
        print(
            f"Note: final selection size is {len(selection)}, requested IMAGES={IMAGES}. "
            "This is expected if the dataset does not contain enough valid images to fill the quota."
        )

    return selection


def copy_selected_images(selection_df):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    total = len(selection_df)
    copied = 0
    missing = 0

    for row in selection_df.itertuples(index=False):
        fname = getattr(row, IMAGE_COL) + IMAGE_EXT
        src = os.path.join(INPUT_IMAGE_FOLDER, fname)
        dst = os.path.join(OUTPUT_FOLDER, fname)

        if not os.path.exists(src):
            missing += 1
            print(f"Warning: {src} not found, skipping.")
            continue

        shutil.copy2(src, dst)
        copied += 1

        if copied % 500 == 0 or copied == total:
            print(f"Copied {copied}/{total} images")

    if missing > 0:
        print(
            f"\nWarning: {missing} file(s) listed in the manifest were missing at copy time "
            f"and were skipped. Copied {copied}/{total} images total."
        )


def main():
    df = pd.read_csv(MANIFEST_CSV_PATH)
    df = df[[IMAGE_COL, LABEL_COL, "patient_id", "quality_score", "image_exists"]].dropna()

    selection = build_selection(df)

    print("\nSelected images per grade:")
    print(selection[LABEL_COL].value_counts().sort_index())

    copy_selected_images(selection)

    manifest_path = os.path.join(OUTPUT_FOLDER, "selected_manifest.csv")
    selection.to_csv(manifest_path, index=False)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()