"""
Computes sharpness scores for EyePACS images and writes a manifest CSV.

The script reads the labels CSV, computes Laplacian variance for each image,
and writes an output CSV containing image name, level, patient_id,
quality_score, and whether the image file exists on disk.
"""

import os
import cv2
import pandas as pd

# ---------------- CONFIG ----------------
INPUT_IMAGE_FOLDER = r"D:\Thesis Dataset\cropped and resized"
INPUT_CSV_PATH = r"D:\Thesis Dataset\labels\all_labels.csv"
OUTPUT_CSV_PATH = r"D:\Thesis Dataset\labels\sharpness_manifest.csv"

IMAGE_COL = "image"
LABEL_COL = "level"
IMAGE_EXT = ".png"
# -----------------------------------------


def extract_patient_id(filename):
    return filename.split("_")[0]


def compute_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return -1.0
    return cv2.Laplacian(image, cv2.CV_64F).var()


def main():
    df = pd.read_csv(INPUT_CSV_PATH)
    df = df[[IMAGE_COL, LABEL_COL]].dropna().copy()
    df["patient_id"] = df[IMAGE_COL].apply(extract_patient_id)

    def score_image(fname):
        full_path = os.path.join(INPUT_IMAGE_FOLDER, fname + IMAGE_EXT)
        return compute_sharpness(full_path)

    df["quality_score"] = df[IMAGE_COL].apply(score_image)
    df["image_exists"] = df[IMAGE_COL].apply(
        lambda fname: os.path.exists(os.path.join(INPUT_IMAGE_FOLDER, fname + IMAGE_EXT))
    )

    missing_count = int((~df["image_exists"]).sum())
    if missing_count > 0:
        print(f"Warning: {missing_count} rows refer to missing image files and will be marked as missing.")

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Sharpness manifest written to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
