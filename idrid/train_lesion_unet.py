"""
Trains a multi-label U-Net (ResNet-34 encoder) to segment diabetic retinopathy
lesions on IDRiD (MA, HE, EX, SE + optic disc), then evaluates it on the 27
official test images.

Protocol (no test leakage)
  - 54 official train images -> 44 train / 10 validation (fixed seed).
  - Best epoch chosen by mean average precision (AP) on validation.
  - Per-lesion probability thresholds tuned on validation only.
  - The 27 test images are used once, at the end.

Reported on test, per lesion type
  - AP (the IDRiD challenge metric), pixel Dice
  - lesion-level precision/recall (connected components)
  - per-image COUNT agreement with ground truth (MAE, Spearman) - this is what
    the lesion-counting module in the thesis actually relies on.

Preprocessing (Ben Graham local-contrast normalisation) is applied identically
at train and inference time to reduce camera/colour differences between IDRiD
and EyePACS; reuse preprocess() when running on EyePACS images later.

Setup:
    pip install segmentation-models-pytorch scikit-learn scipy opencv-python pandas
Usage (after prep_idrid.py):
    python train_lesion_unet.py
"""

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

# ---------------- CONFIG ----------------
DATA_DIR = Path(r"E:\Thesis Dataset\idrid_lesions")
CKPT_DIR = DATA_DIR / "checkpoints"

LESIONS = ["MA", "HE", "EX", "SE", "OD"]
COUNT_CLASSES = ["MA", "HE", "EX", "SE"]      # lesions we count/select on; OD is auxiliary
MIN_AREA = {"MA": 2, "HE": 4, "EX": 4, "SE": 8, "OD": 50}   # px at 512x512; predicted blobs smaller are ignored

ENCODER = "resnet34"
N_VAL = 10
EPOCHS = 150
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# -----------------------------------------


def preprocess(img_bgr):
    """Ben Graham normalisation -> RGB uint8. Use this on EyePACS images too."""
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 10)
    out = cv2.addWeighted(img_bgr, 4, blur, -4, 128)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def load_split(names):
    X, Y = [], []
    for name in names:
        img = cv2.imread(str(DATA_DIR / "images" / f"{name}.png"))
        X.append(preprocess(img))
        Y.append(np.stack([
            cv2.imread(str(DATA_DIR / "masks" / f"{name}_{l}.png"), cv2.IMREAD_GRAYSCALE) > 0
            for l in LESIONS
        ]).astype(np.uint8))
    return X, Y


class LesionDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, train):
        self.X, self.Y, self.train = X, Y, train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.Y[i]            # x: HWC uint8, y: CHW uint8
        if self.train:
            if random.random() < 0.5:
                x, y = x[:, ::-1], y[:, :, ::-1]
            if random.random() < 0.5:
                x, y = x[::-1], y[:, ::-1]
            k = random.randint(0, 3)
            x, y = np.rot90(x, k, (0, 1)), np.rot90(y, k, (1, 2))
        x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        return x, y


def loss_fn(logits, target, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + target.sum((0, 2, 3))
    dice = 1 - (2 * inter + 1) / (denom + 1)
    return bce + dice.mean()


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    out = []
    for i in range(0, len(X), BATCH_SIZE):
        batch = np.stack([(x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD for x in X[i:i + BATCH_SIZE]])
        t = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device)
        out.append(torch.sigmoid(model(t)).cpu().numpy())
    return np.concatenate(out)        # (N, C, H, W)


def mean_ap(probs, Y):
    aps = {}
    for c, l in enumerate(LESIONS):
        gt = np.stack(Y)[:, c].ravel()
        aps[l] = average_precision_score(gt, probs[:, c].ravel()) if gt.any() else float("nan")
    return aps


def tune_thresholds(probs, Y):
    thresholds = {}
    gt_all = np.stack(Y)
    for c, l in enumerate(LESIONS):
        gt = gt_all[:, c].ravel().astype(bool)
        pr = probs[:, c].ravel()
        best_t, best_f1 = 0.5, -1
        for t in np.arange(0.1, 0.91, 0.05):
            pred = pr >= t
            tp = (pred & gt).sum()
            f1 = 2 * tp / (pred.sum() + gt.sum() + 1e-9)
            if f1 > best_f1:
                best_t, best_f1 = float(t), float(f1)
        thresholds[l] = best_t
    return thresholds


def filtered_components(mask, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return lab, keep


def evaluate_test(probs, Y, names, thresholds):
    gt_all = np.stack(Y)
    results, count_rows = {}, []
    for c, l in enumerate(LESIONS):
        gt = gt_all[:, c]
        ap = average_precision_score(gt.ravel(), probs[:, c].ravel()) if gt.any() else float("nan")
        pred_bin = probs[:, c] >= thresholds[l]
        dice = 2 * (pred_bin & gt.astype(bool)).sum() / (pred_bin.sum() + gt.sum() + 1e-9)

        gt_hit = gt_tot = pr_hit = pr_tot = 0
        pred_counts, gt_counts = [], []
        for i in range(len(names)):
            g_lab, g_keep = filtered_components(gt[i], 1)
            p_lab, p_keep = filtered_components(pred_bin[i], MIN_AREA[l])
            p_mask = np.isin(p_lab, p_keep)
            g_mask = gt[i].astype(bool)
            gt_tot += len(g_keep)
            gt_hit += sum(1 for k in g_keep if p_mask[g_lab == k].any())
            pr_tot += len(p_keep)
            pr_hit += sum(1 for k in p_keep if g_mask[p_lab == k].any())
            pred_counts.append(len(p_keep))
            gt_counts.append(len(g_keep))
            count_rows.append({"image": names[i], "lesion": l, "gt_count": len(g_keep), "pred_count": len(p_keep)})

        rho = spearmanr(pred_counts, gt_counts)[0] if len(set(gt_counts)) > 1 and len(set(pred_counts)) > 1 else float("nan")
        results[l] = {
            "threshold": thresholds[l], "AP": float(ap), "pixel_dice": float(dice),
            "lesion_recall": gt_hit / max(gt_tot, 1), "lesion_precision": pr_hit / max(pr_tot, 1),
            "count_MAE": float(np.mean(np.abs(np.array(pred_counts) - np.array(gt_counts)))),
            "count_spearman": float(rho),
            "gt_components": gt_tot, "pred_components": pr_tot,
        }
    return results, pd.DataFrame(count_rows)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    split = pd.read_csv(DATA_DIR / "split.csv")
    train_all = split[split["split"] == "train"]["image"].tolist()
    test_names = split[split["split"] == "test"]["image"].tolist()
    rng = random.Random(SEED)
    rng.shuffle(train_all)
    val_names, train_names = train_all[:N_VAL], train_all[N_VAL:]
    print(f"train={len(train_names)} val={len(val_names)} test={len(test_names)}")

    Xtr, Ytr = load_split(train_names)
    Xva, Yva = load_split(val_names)
    Xte, Yte = load_split(test_names)

    pos = np.stack(Ytr).sum((0, 2, 3)).astype(np.float64)
    neg = np.stack(Ytr)[:, 0].size - pos
    pw = np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1, 50)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    print("pos_weight:", dict(zip(LESIONS, np.round(pw, 1))))

    model = smp.Unet(ENCODER, encoder_weights="imagenet", in_channels=3, classes=len(LESIONS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    loader = torch.utils.data.DataLoader(LesionDataset(Xtr, Ytr, True), batch_size=BATCH_SIZE, shuffle=True)

    best_score, best_path, log = -1.0, CKPT_DIR / "best_lesion_unet.pt", []
    for epoch in range(EPOCHS):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                loss = loss_fn(model(x), y, pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * x.size(0)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            aps = mean_ap(predict(model, Xva, device), Yva)
            score = float(np.nanmean([aps[l] for l in COUNT_CLASSES]))
            log.append({"epoch": epoch + 1, "train_loss": total / len(Xtr), "val_mean_AP": score, **{f"val_AP_{k}": v for k, v in aps.items()}})
            print(f"Epoch {epoch + 1}/{EPOCHS} ({time.time() - t0:.1f}s) loss={total / len(Xtr):.4f} "
                  f"val_mAP={score:.4f} | " + " ".join(f"{k}={v:.3f}" for k, v in aps.items()))
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), best_path)
                print(f"  -> new best val mAP {score:.4f}")
    pd.DataFrame(log).to_csv(CKPT_DIR / "training_log.csv", index=False)

    model.load_state_dict(torch.load(best_path, map_location=device))
    thresholds = tune_thresholds(predict(model, Xva, device), Yva)
    with open(CKPT_DIR / "lesion_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print("Val-tuned thresholds:", thresholds)

    results, counts_df = evaluate_test(predict(model, Xte, device), Yte, test_names, thresholds)
    with open(CKPT_DIR / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    counts_df.to_csv(CKPT_DIR / "test_counts.csv", index=False)

    print("\n--- TEST RESULTS (27 official test images) ---")
    print(pd.DataFrame(results).T.round(3).to_string())
    print(f"\nSaved to {CKPT_DIR}")


if __name__ == "__main__":
    main()