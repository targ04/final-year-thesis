"""
Ensemble evaluation: averages softmax outputs from a trained ResNet-50 and a
trained ViT-B/16 (both fine-tuned for 5-class DR severity grading) over the
same manifest, and reports quadratic weighted kappa + accuracy for:
  - ResNet-50 alone
  - ViT-B/16 alone
  - Ensemble (softmax-averaged)

Also saves per-image ensemble probabilities and predictions to a CSV, since
the Referral Agent's temperature scaling step will need the raw averaged
probabilities/logits, not just the final metric.

Usage:
    python ensemble_eval.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import cohen_kappa_score, accuracy_score

# ---------------- CONFIG ----------------
# EDIT THESE PATHS
BASE_DIR = Path(__file__).resolve().parent

MANIFEST_PATH = r"D:\Thesis Dataset\final dataset\splits\val_manifest.csv"  # or test_manifest.csv
IMAGE_FOLDER = r"D:\Thesis Dataset\final dataset\images"
IMAGE_EXT = ".png"

RESNET_WEIGHTS_PATH = BASE_DIR / "resnet50" / "best_model.pt"
VIT_WEIGHTS_PATH = BASE_DIR / "vitb" / "run2" / "best_model.pt"

OUTPUT_CSV_PATH = r"ensemble_val_predictions.csv"

IMAGE_COL = "image"
LABEL_COL = "level"
NUM_CLASSES = 5

RESNET_INPUT_SIZE = 224
VIT_INPUT_SIZE = 224

BATCH_SIZE = 32
NUM_WORKERS = 4

# Weight given to each model's softmax output when averaging.
# 0.5/0.5 = simple average. Adjust later if one model consistently
# outperforms the other and you want a weighted ensemble instead.
RESNET_WEIGHT = 0.5
VIT_WEIGHT = 0.5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# -----------------------------------------


class DRDataset(Dataset):
    def __init__(self, manifest_df, image_folder, transform):
        self.df = manifest_df.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_folder, row[IMAGE_COL] + IMAGE_EXT)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(row[LABEL_COL])
        return image, label


def get_eval_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_vit_b(num_classes, dropout_p=0.3):
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes)
    )
    return model


def load_model(model, weights_path, device):
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def get_softmax_outputs(model, loader, device):
    """Runs the model over the loader and returns (probs, labels) as numpy arrays.
    Order is preserved so outputs from different loaders over the same
    manifest can be aligned/averaged directly."""
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def report_metrics(name, labels, preds):
    kappa = cohen_kappa_score(labels, preds, weights="quadratic")
    acc = accuracy_score(labels, preds)
    print(f"{name:20s} | quadratic kappa = {kappa:.4f} | accuracy = {acc:.4f}")
    return kappa, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(MANIFEST_PATH)

    # --- ResNet-50 pass ---
    resnet_dataset = DRDataset(df, IMAGE_FOLDER, get_eval_transform(RESNET_INPUT_SIZE))
    resnet_loader = DataLoader(resnet_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
    resnet_model = build_resnet50(NUM_CLASSES)
    resnet_model = load_model(resnet_model, RESNET_WEIGHTS_PATH, device)
    print("Running ResNet-50 inference...")
    resnet_probs, labels = get_softmax_outputs(resnet_model, resnet_loader, device)

    # --- ViT-B/16 pass ---
    vit_dataset = DRDataset(df, IMAGE_FOLDER, get_eval_transform(VIT_INPUT_SIZE))
    vit_loader = DataLoader(vit_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    vit_model = build_vit_b(NUM_CLASSES)
    vit_model = load_model(vit_model, VIT_WEIGHTS_PATH, device)
    print("Running ViT-B/16 inference...")
    vit_probs, vit_labels = get_softmax_outputs(vit_model, vit_loader, device)

    # Sanity check: both passes should have iterated the manifest in the
    # same order (shuffle=False on both loaders), so labels must match.
    assert np.array_equal(labels, vit_labels), (
        "Label order mismatch between ResNet-50 and ViT-B passes — "
        "check that both loaders use shuffle=False and the same manifest."
    )

    # --- Ensemble ---
    ensemble_probs = RESNET_WEIGHT * resnet_probs + VIT_WEIGHT * vit_probs
    ensemble_preds = ensemble_probs.argmax(axis=1)

    resnet_preds = resnet_probs.argmax(axis=1)
    vit_preds = vit_probs.argmax(axis=1)

    print("\n--- Results ---")
    report_metrics("ResNet-50 (solo)", labels, resnet_preds)
    report_metrics("ViT-B/16 (solo)", labels, vit_preds)
    report_metrics("Ensemble (avg)", labels, ensemble_preds)

    # --- Save per-image predictions + probabilities for downstream use
    # (e.g. temperature scaling calibration for the Referral Agent) ---
    out_df = df.copy()
    out_df["true_label"] = labels
    out_df["resnet_pred"] = resnet_preds
    out_df["vit_pred"] = vit_preds
    out_df["ensemble_pred"] = ensemble_preds
    for c in range(NUM_CLASSES):
        out_df[f"ensemble_prob_class{c}"] = ensemble_probs[:, c]
    for c in range(NUM_CLASSES):
        out_df[f"resnet_prob_class{c}"] = resnet_probs[:, c]
    for c in range(NUM_CLASSES):
        out_df[f"vit_prob_class{c}"] = vit_probs[:, c]

    out_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nPer-image predictions and probabilities saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
