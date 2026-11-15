"""
Visual Evidence Generation: Grad-CAM (ResNet-50) + Attention Rollout (ViT-B/16).

Computes WHERE each backbone's attention concentrated on each fundus image
for its predicted grade, and whether the two independently-trained
backbones agree on that region. This is the only honest way to let the
Explanation Agent say anything about "why" a classification was assigned -
the pipeline has no lesion-level detector, so this produces LOCATION and
CROSS-MODEL AGREEMENT evidence only. It never identifies lesion types
(microaneurysm, hemorrhage, exudate, etc.) - that would be fabricated, since
nothing in this pipeline detects them. The Explanation Agent's prompt is
written to respect this boundary explicitly.

--- Methodological note: Grad-CAM vs attention rollout are not equivalent ---
Grad-CAM (ResNet-50) is class-discriminative: it uses the gradient of the
TARGET CLASS logit, so it shows evidence specifically for the predicted
grade. Attention rollout (ViT-B/16) is gradient-free and class-agnostic -
it shows where the model attended overall, not evidence specific to the
predicted class. This is a known asymmetry between the two methods, not a
bug. Treat "both backbones concentrated attention in the same region" as
corroborating evidence, not as two class-discriminative signals agreeing.

--- Target class ---
Both saliency methods are computed for the ENSEMBLE's final calibrated
prediction (read from referral_triage_output.csv's calibrated_pred column),
not each backbone's own argmax, so the saliency explains the diagnosis that
was actually output.

--- Output ---
A manifest CSV with one row per image: quadrant + concentration ratio for
each backbone, and whether they agree. Optionally, overlay PNGs for a small
sample (for thesis figures) - not the full set, to keep this fast and not
fill the disk, since what the Explanation Agent actually consumes is the
CSV, not the images.

Usage:
    python generate_saliency.py
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path

# ---------------- CONFIG ----------------
basepath = Path(__file__).parent

RESNET_WEIGHTS_PATH = basepath.parent / "diagnosis_agent" / r"resnet50\best_model.pt"
VIT_WEIGHTS_PATH =  basepath.parent / "diagnosis_agent" / r"vitb\run2\best_model.pt"

MANIFEST_PATH = basepath.parent / "referral_agent" / "referral_triage_output.csv"
IMAGE_FOLDER = r"E:\Thesis Dataset\final dataset\images"
IMAGE_EXT = ".png"

OUTPUT_CSV_PATH = basepath / "saliency_manifest.csv"
OUTPUT_OVERLAY_DIR = basepath / "saliency_overlays"
SAVE_OVERLAY_SAMPLE_N = 50   # only the first N processed images get overlay PNGs saved

IMAGE_COL = "image"
PATIENT_COL = "patient_id"
PRED_COL = "calibrated_pred"   # target class for saliency = the ensemble's final prediction

NUM_CLASSES = 5
INPUT_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# -----------------------------------------


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_vit_b(num_classes, dropout_p=0.3):
    # Matches ensemble_eval.py's build_vit_b exactly - must match for the
    # saved state_dict's keys to load correctly.
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


# ---------------- Grad-CAM (ResNet-50) ----------------

class ResNetGradCAM:
    """Standard Grad-CAM hooked on layer4 (last conv block before the
    global average pool) - class-discriminative via backprop to the
    target class logit."""

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        model.layer4.register_forward_hook(self._save_activations)
        model.layer4.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dims
        cam = (weights * self.activations).sum(dim=1).squeeze(0)  # (H, W)
        cam = torch.relu(cam)
        return cam.cpu().numpy()


# ---------------- Attention rollout (ViT-B/16) ----------------

def patch_vit_to_capture_attention(vit_model, storage):
    """Monkey-patches every EncoderBlock's forward to request attention
    weights (torchvision discards them by default via need_weights=False
    for speed). Appends each layer's (1, seq_len, seq_len) attention
    weight tensor to `storage` on every forward() call - clear storage
    before each new forward pass.

    NOTE: this mirrors torchvision's EncoderBlock.forward structure
    (ln_1 -> self_attention -> dropout -> residual -> ln_2 -> mlp ->
    residual) as of the torchvision version this was written against. If
    this breaks after a torchvision upgrade, diff this against the
    installed torchvision/models/vision_transformer.py EncoderBlock.forward.
    """
    for block in vit_model.encoder.layers:
        def make_patched_forward(blk):
            def patched_forward(input):
                x = blk.ln_1(input)
                attn_output, attn_weights = blk.self_attention(
                    x, x, x, need_weights=True, average_attn_weights=True
                )
                storage.append(attn_weights.detach())
                x = blk.dropout(attn_output)
                x = x + input
                y = blk.ln_2(x)
                y = blk.mlp(y)
                return x + y
            return patched_forward

        block.forward = make_patched_forward(block)


def compute_attention_rollout(attn_weights_list):
    """Abnar & Zuidema (2020) attention rollout: adds the residual
    connection's identity to each layer's attention, renormalizes, and
    composes across layers via matrix multiplication. Returns the CLS
    token's rolled-out attention to each patch token (gradient-free,
    class-agnostic - see module docstring)."""
    seq_len = attn_weights_list[0].shape[-1]
    device = attn_weights_list[0].device
    rollout = torch.eye(seq_len, device=device).unsqueeze(0)

    for attn in attn_weights_list:
        attn_res = attn + torch.eye(seq_len, device=device).unsqueeze(0)
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(attn_res, rollout)

    cls_to_patches = rollout[0, 0, 1:]  # exclude CLS-to-CLS self-attention
    return cls_to_patches.cpu().numpy()


# ---------------- Shared: heatmap -> quadrant descriptor ----------------

def heatmap_to_quadrant(heatmap_2d):
    """Returns (quadrant_label, concentration_ratio). quadrant_label is
    image-relative (upper-left/upper-right/lower-left/lower-right) - NOT
    mapped to anatomical terms (nasal/temporal/superior/inferior), since
    that mapping depends on eye laterality/capture conventions this
    pipeline doesn't track, and an incorrect anatomical claim is worse
    than a correctly-hedged image-relative one. concentration_ratio is the
    fraction of total activation held by the top 10% most-active cells -
    higher means focal/localized attention, lower means diffuse."""
    h, w = heatmap_2d.shape
    total = heatmap_2d.sum()
    if total <= 1e-8:
        return "undetermined", 0.0

    yy, xx = np.mgrid[0:h, 0:w]
    centroid_row = (yy * heatmap_2d).sum() / total / h
    centroid_col = (xx * heatmap_2d).sum() / total / w

    vertical = "upper" if centroid_row < 0.5 else "lower"
    horizontal = "left" if centroid_col < 0.5 else "right"
    quadrant = f"{vertical}-{horizontal}"

    flat_sorted = np.sort(heatmap_2d.flatten())[::-1]
    top_k = max(1, int(0.1 * flat_sorted.size))
    concentration_ratio = float(flat_sorted[:top_k].sum() / total)

    return quadrant, concentration_ratio


def save_overlay(original_image, heatmap_2d, out_path):
    heatmap_resized = np.array(
        Image.fromarray((heatmap_2d / (heatmap_2d.max() + 1e-8) * 255).astype(np.uint8))
        .resize(original_image.size, resample=Image.BILINEAR)
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(original_image)
    ax.imshow(heatmap_resized, cmap="jet", alpha=0.4)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CPU works but is noticeably slower for this script - "
          f"a full backward pass per image for Grad-CAM plus a forward pass for attention rollout).")

    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded {len(df)} rows from {MANIFEST_PATH}")

    resnet_model = load_model(build_resnet50(NUM_CLASSES), RESNET_WEIGHTS_PATH, device)
    vit_model = load_model(build_vit_b(NUM_CLASSES), VIT_WEIGHTS_PATH, device)

    gradcam = ResNetGradCAM(resnet_model)

    attn_storage = []
    patch_vit_to_capture_attention(vit_model, attn_storage)

    transform = get_eval_transform()

    os.makedirs(OUTPUT_OVERLAY_DIR, exist_ok=True)

    results = []
    for i, row in df.iterrows():
        image_name = row[IMAGE_COL]
        target_class = int(row[PRED_COL])
        img_path = os.path.join(IMAGE_FOLDER, image_name + IMAGE_EXT)

        try:
            pil_image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"  WARNING: image not found for {image_name}, skipping.")
            continue

        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        # --- ResNet-50 Grad-CAM (class-discriminative, needs gradients) ---
        resnet_cam = gradcam.compute(input_tensor, target_class)
        resnet_quadrant, resnet_concentration = heatmap_to_quadrant(resnet_cam)

        # --- ViT-B/16 attention rollout (gradient-free) ---
        attn_storage.clear()
        with torch.no_grad():
            vit_model(input_tensor)
        cls_to_patches = compute_attention_rollout(attn_storage)
        grid_size = int(np.sqrt(cls_to_patches.shape[0]))  # 14x14 for 224/16 patches
        vit_map = cls_to_patches.reshape(grid_size, grid_size)
        vit_quadrant, vit_concentration = heatmap_to_quadrant(vit_map)

        agree = resnet_quadrant == vit_quadrant

        results.append({
            IMAGE_COL: image_name,
            PATIENT_COL: row.get(PATIENT_COL),
            "target_grade": target_class,
            "resnet_quadrant": resnet_quadrant,
            "resnet_concentration": round(resnet_concentration, 4),
            "vit_quadrant": vit_quadrant,
            "vit_concentration": round(vit_concentration, 4),
            "attention_agreement": agree,
        })

        if i < SAVE_OVERLAY_SAMPLE_N:
            save_overlay(pil_image, resnet_cam, os.path.join(OUTPUT_OVERLAY_DIR, f"{image_name}_resnet_cam.png"))
            save_overlay(pil_image, vit_map, os.path.join(OUTPUT_OVERLAY_DIR, f"{image_name}_vit_attn.png"))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(df)}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV_PATH, index=False)

    agreement_rate = out_df["attention_agreement"].mean()
    print(f"\nDone. {len(out_df)} images processed.")
    print(f"Cross-model attention agreement rate (same quadrant): {agreement_rate * 100:.1f}%")
    print(f"Saliency manifest saved to {OUTPUT_CSV_PATH}")
    print(f"Sample overlays ({min(SAVE_OVERLAY_SAMPLE_N, len(out_df))} images) saved to {OUTPUT_OVERLAY_DIR}")


if __name__ == "__main__":
    main()