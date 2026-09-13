"""
Trains a ViT-B/16 classifier for 5-class DR severity grading (0-4).

Reads train_manifest.csv / val_manifest.csv (from split_dataset.py), loads
images from the "best images" folder, fine-tunes an ImageNet-pretrained
ViT-B/16 with class-weighted cross-entropy loss, and evaluates each epoch
using quadratic weighted kappa (the primary metric) alongside accuracy.

Supports resuming from the last checkpoint if interrupted (important on a
shared GPU where runs may be preempted).

Usage:
    python train_vit_b.py

Outputs (written to CHECKPOINT_DIR):
    best_model.pt        - weights-only checkpoint of the best val-kappa epoch
    last_checkpoint.pt    - full state (model + optimizer + epoch) for resuming
    training_log.csv       - per-epoch train/val loss, val kappa, val accuracy
"""

import os
import time
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import cohen_kappa_score, accuracy_score

# ---------------- CONFIG ----------------
TRAIN_MANIFEST_PATH = r"D:\Thesis Dataset\splits\train_manifest.csv"
VAL_MANIFEST_PATH = r"D:\Thesis Dataset\splits\val_manifest.csv"
IMAGE_FOLDER = r"D:\Thesis Dataset\best images"
IMAGE_EXT = ".png"

CHECKPOINT_DIR = r"D:\Thesis Dataset\checkpoints\vit_b"

IMAGE_COL = "image"
LABEL_COL = "level"
NUM_CLASSES = 5

INPUT_SIZE = 224          # ViT-B/16 standard input resolution (torchvision IMAGENET1K_V1 weights)
BATCH_SIZE = 32            # Reduce if you hit OOM on the shared GPU - ViT-B is heavier than ResNet-50
NUM_EPOCHS = 30
LEARNING_RATE = 3e-5        # Lower than ResNet-50 - ViT fine-tuning is more sensitive to LR
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

RESUME = True              # If True and last_checkpoint.pt exists, resume from it
RANDOM_SEED = 42

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


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        # Conservative color jitter only - retinal color carries diagnostic
        # signal (microaneurysms, hemorrhages, exudates), so hue/saturation
        # are left untouched.
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def compute_class_weights(train_df, num_classes):
    """Inverse-frequency class weights, normalized to mean 1.0."""
    counts = train_df[LABEL_COL].value_counts().reindex(range(num_classes), fill_value=0)
    total = counts.sum()
    weights = total / (num_classes * counts.replace(0, 1))
    weights = weights / weights.mean()
    print("Class counts (train):")
    print(counts)
    print("Class weights:")
    print(weights)
    return torch.tensor(weights.values, dtype=torch.float32)


def build_model(num_classes):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, scaler, device, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                             enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, kappa, acc


def main():
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(TRAIN_MANIFEST_PATH)
    val_df = pd.read_csv(VAL_MANIFEST_PATH)

    train_dataset = DRDataset(train_df, IMAGE_FOLDER, get_train_transform())
    val_dataset = DRDataset(val_df, IMAGE_FOLDER, get_eval_transform())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    class_weights = compute_class_weights(train_df, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    best_kappa = -1.0
    log_rows = []

    last_ckpt_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    log_path = os.path.join(CHECKPOINT_DIR, "training_log.csv")

    if RESUME and os.path.exists(last_ckpt_path):
        print(f"Resuming from {last_ckpt_path}")
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_kappa = checkpoint["best_kappa"]
        if os.path.exists(log_path):
            log_rows = pd.read_csv(log_path).to_dict("records")
        print(f"Resumed at epoch {start_epoch}, best_kappa so far = {best_kappa:.4f}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()

        train_loss, train_kappa, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, train=True
        )
        val_loss, val_kappa, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, train=False
        )

        elapsed = time.time() - t0
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} ({elapsed:.1f}s) | "
              f"train_loss={train_loss:.4f} train_kappa={train_kappa:.4f} | "
              f"val_loss={val_loss:.4f} val_kappa={val_kappa:.4f} val_acc={val_acc:.4f}")

        log_rows.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_kappa": train_kappa,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_kappa": val_kappa,
            "val_acc": val_acc,
        })
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        # Always save the latest state so training can resume after interruption.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_kappa": best_kappa,
        }, last_ckpt_path)

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best val_kappa={best_kappa:.4f}. Saved to {best_model_path}")

    print(f"\nTraining complete. Best val_kappa = {best_kappa:.4f}")
    print(f"Best model weights: {best_model_path}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
