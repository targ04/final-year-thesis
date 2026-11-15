"""
Lesion analysis on EyePACS images using the lesion segmentation model trained
on IDRiD (train_lesion_unet.py).

For every image it computes, per lesion type (MA, HE, EX, SE):
  - count          : connected lesion regions (after minimum-size filtering)
  - area_frac      : lesion area as % of the retina
  - soft           : mean predicted probability over the retina (threshold-free
                     burden, immune to threshold-tuning problems)
  - per-quadrant counts (image-relative: upper/lower x left/right)

Predictions are restricted to the retina (eroded fundus mask, so the bright rim
does not create false lesions), and the optic disc (largest predicted OD blob,
dilated) is excluded from lesion maps.

Then it checks whether lesion burden tracks DR grade, which is the real test of
whether an IDRiD-trained model transfers to EyePACS (no lesion masks exist here):
  - mean burden per grade, grade-0 false-positive level
  - Spearman correlation with grade, AUC for grade>=2 and grade>=3
  - boxplot figure and a few overlay images per grade for visual sanity checks

--- Methodology: use val for decisions, test once for reporting ---
SPLIT below selects which EyePACS manifest to run. If you want to change any
threshold (THRESHOLD_OVERRIDE) or filter size after looking at results, look at
the "val" run, then run "test" once with the final settings. Tuning on the
test run would leak.

Place in the same folder as train_lesion_unet.py. Usage:
    python eyepacs_lesion_analysis.py
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from train_lesion_unet import (
    preprocess, IMAGENET_MEAN, IMAGENET_STD, LESIONS, COUNT_CLASSES, MIN_AREA, ENCODER, CKPT_DIR,
)

# ---------------- CONFIG ----------------
SPLIT = "test"      # "val" for decisions, "test" for the final reported run

SPLIT_DIR = Path(r"E:\Thesis Dataset\final dataset\splits")
MANIFEST_PATH = SPLIT_DIR / f"{SPLIT}_manifest.csv"
IMAGE_FOLDER = Path(r"E:\Thesis Dataset\final dataset\images")
IMAGE_EXT = ".png"
IMAGE_COL, LABEL_COL, PATIENT_COL = "image", "level", "patient_id"

MODEL_PATH = CKPT_DIR / "best_lesion_unet.pt"
THRESHOLDS_PATH = CKPT_DIR / "lesion_thresholds.json"
# Optional per-lesion threshold overrides, e.g. {"HE": 0.3}. Decide on the val run only.
THRESHOLD_OVERRIDE = {}

OUT_DIR = Path(__file__).resolve().parent / f"lesion_analysis_{SPLIT}"
BATCH_SIZE = 8
OVERLAYS_PER_GRADE = 3

RETINA_ERODE_PX = 8          # shrink the fundus mask to drop false positives on the bright rim
OD_DILATE_PX = 12            # margin around the detected optic disc
EXCLUDE_OD_FOR = {"MA", "HE", "EX", "SE"}

OVERLAY_COLORS = {"MA": (0, 255, 255), "HE": (0, 0, 255), "EX": (0, 255, 0), "SE": (255, 0, 0)}  # BGR
# -----------------------------------------

IDX = {l: i for i, l in enumerate(LESIONS)}
QUADS = ["UL", "UR", "LL", "LR"]


def ellipse(px):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def retina_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(5))
    return cv2.erode(mask, ellipse(RETINA_ERODE_PX)).astype(bool)


def optic_disc_exclusion(od_prob, retina, thr):
    """Largest predicted OD blob, dilated. Empty mask if none found."""
    od_bin = ((od_prob >= thr) & retina).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(od_bin, connectivity=8)
    if n <= 1:
        return np.zeros_like(retina)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < MIN_AREA["OD"]:
        return np.zeros_like(retina)
    return cv2.dilate((lab == k).astype(np.uint8), ellipse(OD_DILATE_PX)).astype(bool)


def analyse_image(probs, retina, thresholds):
    """probs: (C, H, W). Returns (row dict of features, {lesion: boolean mask} for overlays)."""
    H, W = retina.shape
    retina_px = max(int(retina.sum()), 1)
    od_excl = optic_disc_exclusion(probs[IDX["OD"]], retina, thresholds["OD"])
    valid = retina & ~od_excl

    row = {"retina_px": retina_px, "od_found": bool(od_excl.any())}
    masks = {}
    for l in COUNT_CLASSES:
        region = valid if l in EXCLUDE_OD_FOR else retina
        p = probs[IDX[l]]
        row[f"{l}_soft"] = float(p[region].sum() / retina_px)

        binm = ((p >= thresholds[l]) & region).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(binm, connectivity=8)
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA[l]]

        row[f"{l}_count"] = len(keep)
        row[f"{l}_area_frac"] = 100.0 * float(sum(stats[i, cv2.CC_STAT_AREA] for i in keep)) / retina_px
        quad = {q: 0 for q in QUADS}
        for i in keep:
            cx, cy = cents[i]
            quad[("U" if cy < H / 2 else "L") + ("L" if cx < W / 2 else "R")] += 1
        for q in QUADS:
            row[f"{l}_{q}"] = quad[q]
        masks[l] = np.isin(lab, keep)
    return row, masks


def make_overlay(img_bgr, masks):
    out = img_bgr.copy()
    for l, m in masks.items():
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, OVERLAY_COLORS[l], 1)
    return np.concatenate([img_bgr, out], axis=1)


@torch.no_grad()
def run_model(model, batch_rgb, device):
    x = np.stack([(im.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD for im in batch_rgb])
    t = torch.from_numpy(x.transpose(0, 3, 1, 2)).to(device)
    return torch.sigmoid(model(t)).cpu().numpy()


def summarise(df):
    feats = [c for c in df.columns if re.search(r"_(count|area_frac|soft)$", c)]
    labels = df["true_label"].values

    print("\n--- Mean per DR grade ---")
    print(df.groupby("true_label")[feats].mean().round(3).T.to_string())

    rows = []
    for c in feats:
        v = df[c].values
        rho = spearmanr(v, labels)[0]
        auc2 = roc_auc_score(labels >= 2, v) if len(set(labels >= 2)) > 1 else float("nan")
        auc3 = roc_auc_score(labels >= 3, v) if len(set(labels >= 3)) > 1 else float("nan")
        g0, g4 = df[df.true_label == 0][c], df[df.true_label == 4][c]
        rows.append({"feature": c, "spearman_vs_grade": rho, "AUC_grade>=2": auc2, "AUC_grade>=3": auc3,
                     "grade0_median": g0.median(), "grade0_p90": g0.quantile(0.9),
                     "grade4_median": g4.median() if len(g4) else float("nan")})
    summary = pd.DataFrame(rows).set_index("feature")
    print("\n--- Does lesion burden track DR grade? ---")
    print(summary.round(3).to_string())
    return summary


def make_boxplots(df, out_path):
    fig, axes = plt.subplots(2, len(COUNT_CLASSES), figsize=(4 * len(COUNT_CLASSES), 7))
    for j, l in enumerate(COUNT_CLASSES):
        for i, (suffix, title) in enumerate([("count", "count"), ("area_frac", "area (% of retina)")]):
            data = [df[df.true_label == g][f"{l}_{suffix}"].values for g in range(5)]
            axes[i][j].boxplot(data, showfliers=False)
            axes[i][j].set_xticklabels(range(5))
            axes[i][j].set_title(f"{l} {title}")
            axes[i][j].set_xlabel("DR grade")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | split: {SPLIT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "overlays").mkdir(exist_ok=True)

    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)
    thresholds.update(THRESHOLD_OVERRIDE)
    print("Thresholds:", thresholds)

    model = smp.Unet(ENCODER, encoder_weights=None, in_channels=3, classes=len(LESIONS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    df = pd.read_csv(MANIFEST_PATH)
    overlay_names = set(df.groupby(LABEL_COL).head(OVERLAYS_PER_GRADE)[IMAGE_COL])

    rows = []
    for start in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[start:start + BATCH_SIZE]
        imgs = [cv2.imread(str(IMAGE_FOLDER / (n + IMAGE_EXT))) for n in chunk[IMAGE_COL]]
        valid = [i for i, im in enumerate(imgs) if im is not None]
        if len(valid) < len(imgs):
            print(f"  WARNING: {len(imgs) - len(valid)} unreadable image(s) in batch starting at {start}, skipped.")
        if not valid:
            continue
        probs = run_model(model, [preprocess(imgs[i]) for i in valid], device)

        for k, i in enumerate(valid):
            r = chunk.iloc[i]
            feat, masks = analyse_image(probs[k], retina_mask(imgs[i]), thresholds)
            rows.append({"image": r[IMAGE_COL], "patient_id": r.get(PATIENT_COL),
                         "true_label": int(r[LABEL_COL]), **feat})
            if r[IMAGE_COL] in overlay_names:
                name = f"grade{int(r[LABEL_COL])}_{r[IMAGE_COL]}.png"
                cv2.imwrite(str(OUT_DIR / "overlays" / name), make_overlay(imgs[i], masks))

        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  processed {min(start + BATCH_SIZE, len(df))}/{len(df)}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "lesion_features.csv", index=False)

    summary = summarise(out_df)
    summary.to_csv(OUT_DIR / "burden_vs_grade_summary.csv")
    make_boxplots(out_df, OUT_DIR / "burden_vs_grade_boxplots.png")
    print(f"\nSaved features, summary, boxplots and overlays to {OUT_DIR}")


if __name__ == "__main__":
    main()
