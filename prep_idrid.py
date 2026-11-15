"""
Prepares IDRiD Part A (Lesion Segmentation, 81 images) for training the
lesion segmentation model.

What it does
  1. Downloads the dataset via kagglehub (or uses a local extracted folder).
  2. Finds the 81 segmentation images (IDRiD_01..IDRiD_81, two-digit names;
     the 3-digit IDRiD_001.. images belong to the grading set and are ignored)
     and their masks (IDRiD_XX_MA / HE / EX / SE / OD).
  3. Applies the SAME crop + 512x512 resize as crop_and_resize.py to each image,
     and the identical crop box to every mask, so training data matches what
     the pipeline feeds the models.
  4. Writes images, binary masks, split.csv, and a feasibility report: how big
     each lesion type is at 512x512 (microaneurysms are the risk).

Run from the same folder as crop_and_resize.py:
    pip install kagglehub opencv-python numpy
    python prep_idrid.py
"""

import os
import re
import json
import math
from pathlib import Path

os.environ["KAGGLEHUB_CACHE"] = "E:/kagglehub"   # same cache as download_dataset.py

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

basepath = Path(__file__).parent

from dataset_prep.crop_and_resize import analyze_fundus_mask   # same fundus-circle detection as EyePACS

# ---------------- CONFIG ----------------
KAGGLE_SLUG = "aaryapatel98/indian-diabetic-retinopathy-image-dataset"
# Set to an already-extracted folder (e.g. the official "A. Segmentation" folder
# from IEEE DataPort) to skip the download. Searched recursively.
IDRID_ROOT = None

OUT_DIR = Path(r"D:\Thesis Dataset\idrid_lesions")
TARGET_SIZE = 512
LESIONS = ["MA", "HE", "EX", "SE", "OD"]
N_TRAIN = 54                     # IDRiD_01..54 = official train, 55..81 = official test

# Area-downsampling a mask can erase tiny lesions, so MA keeps any pixel with
# partial coverage; larger lesions keep pixels that are >=25% covered.
DOWNSAMPLE_THRESHOLD = {"MA": 0, "HE": 64, "EX": 64, "SE": 64, "OD": 64}

IMG_RE = re.compile(r"^IDRiD_(\d{2})\.jpg$", re.IGNORECASE)
MASK_RE = re.compile(r"^IDRiD_(\d{2})_(MA|HE|EX|SE|OD)\.(tif|tiff|png)$", re.IGNORECASE)
# -----------------------------------------


def get_root():
    if IDRID_ROOT:
        return Path(IDRID_ROOT)
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_SLUG)
    print("Dataset at:", path)
    return Path(path)


def discover(root):
    images, masks = {}, {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        m = IMG_RE.match(p.name)
        if m:
            images.setdefault(int(m.group(1)), p)
            continue
        m = MASK_RE.match(p.name)
        if m:
            masks.setdefault(int(m.group(1)), {}).setdefault(m.group(2).upper(), p)
    return images, masks


def process_one(num, img_path, mask_paths, diam_log):
    img = cv2.imread(str(img_path))
    x, y, w, h, ok, reason = analyze_fundus_mask(img)
    if not ok:
        print(f"  note: IDRiD_{num:02d} flagged by crop check ({reason}); using its box anyway.")
    if w < 10 or h < 10:
        x, y, w, h = 0, 0, img.shape[1], img.shape[0]

    out_img = cv2.resize(img[y:y + h, x:x + w], (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    scale_area = (w / TARGET_SIZE) * (h / TARGET_SIZE)   # native px per 512-px pixel

    out_masks = {}
    for les in LESIONS:
        p = mask_paths.get(les)
        if p is None:
            out_masks[les] = np.zeros((TARGET_SIZE, TARGET_SIZE), np.uint8)   # no file = none annotated
            continue
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape[:2] != img.shape[:2]:
            print(f"  WARNING: mask/image size mismatch for IDRiD_{num:02d}_{les}, using empty mask.")
            out_masks[les] = np.zeros((TARGET_SIZE, TARGET_SIZE), np.uint8)
            continue
        m = (m[y:y + h, x:x + w] > 0).astype(np.uint8)

        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            diam_log[les].append(math.sqrt(4 * area / (math.pi * scale_area)))   # equiv. diameter at 512 px

        small = cv2.resize(m * 255, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        out_masks[les] = ((small > DOWNSAMPLE_THRESHOLD[les]) * 255).astype(np.uint8)
    return out_img, out_masks


def main():
    root = get_root()
    images, masks = discover(root)
    print(f"Found {len(images)} segmentation images (expected 81).")
    for les in LESIONS:
        print(f"  {les}: {sum(1 for n in images if les in masks.get(n, {}))} images with a mask file")

    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "masks").mkdir(parents=True, exist_ok=True)

    diam_log = {les: [] for les in LESIONS}
    rows = []
    for num in sorted(images):
        name = f"IDRiD_{num:02d}"
        out_img, out_masks = process_one(num, images[num], masks.get(num, {}), diam_log)
        cv2.imwrite(str(OUT_DIR / "images" / f"{name}.png"), out_img)
        for les in LESIONS:
            cv2.imwrite(str(OUT_DIR / "masks" / f"{name}_{les}.png"), out_masks[les])
        rows.append({"image": name, "split": "train" if num <= N_TRAIN else "test"})
    pd.DataFrame(rows).to_csv(OUT_DIR / "split.csv", index=False)

    print("\n--- Feasibility: lesion size at 512x512 (equivalent diameter, pixels) ---")
    report = {}
    for les in LESIONS:
        d = np.array(diam_log[les])
        if d.size == 0:
            print(f"{les}: no components found")
            continue
        report[les] = {
            "n_components": int(d.size),
            "diam_p10": float(np.percentile(d, 10)),
            "diam_median": float(np.median(d)),
            "diam_p90": float(np.percentile(d, 90)),
            "frac_under_3px": float((d < 3).mean()),
        }
        r = report[les]
        print(f"{les}: {r['n_components']:6d} components | diameter p10/median/p90 = "
              f"{r['diam_p10']:.1f}/{r['diam_median']:.1f}/{r['diam_p90']:.1f} px | "
              f"{r['frac_under_3px'] * 100:.0f}% under 3 px")
    with open(OUT_DIR / "feasibility_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()