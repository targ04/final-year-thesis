"""
Lesion measurement tool - wraps the IDRiD-trained U-Net (idrid/train_lesion_unet.py)
so the agent can call it on a single preprocessed (cropped, 512x512) fundus image.

Reuses the exact feature extraction of idrid/eyepacs_lesion_analysis.py
(retina mask, optic-disc exclusion, min-area filtering, quadrant counts), so a
feature computed here matches lesion_features.csv for the same image.

Returns per lesion type (MA, HE, EX, SE): count, area_frac (% of retina),
soft burden, per-quadrant counts, plus binary masks for the GUI overlay.

These are MODEL ESTIMATES transferred from IDRiD (different camera/population,
44 training images). Known behaviour on EyePACS (val/test burden analysis):
MA tracks grade best, HE over-fires on grade 0, SE carries ~no signal.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "idrid"))   # eyepacs_lesion_analysis does `from train_lesion_unet import ...`

import segmentation_models_pytorch as smp                      # noqa: E402
from train_lesion_unet import preprocess, LESIONS, ENCODER, COUNT_CLASSES   # noqa: E402
from eyepacs_lesion_analysis import (retina_mask, analyse_image, run_model,  # noqa: E402
                                     make_overlay, QUADS)

CKPT_DIR = ROOT / "idrid" / "idrid_lesions" / "checkpoints"
MODEL_PATH = CKPT_DIR / "best_lesion_unet.pt"
THRESHOLDS_PATH = CKPT_DIR / "lesion_thresholds.json"
RULE_PATH = ROOT / "agentic" / "results" / "selected_lesion_rule.json"   # chosen on VAL by offline_ablation.py

LESION_NAMES = {"MA": "microaneurysm", "HE": "haemorrhage", "EX": "hard exudate", "SE": "soft exudate"}
QUAD_NAMES = {"UL": "upper-left", "UR": "upper-right", "LL": "lower-left", "LR": "lower-right"}


class LesionTool:
    def __init__(self, device):
        self.device = device
        self.model = smp.Unet(ENCODER, encoder_weights=None, in_channels=3, classes=len(LESIONS))
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        self.model.to(device).eval()
        with open(THRESHOLDS_PATH) as f:
            self.thresholds = json.load(f)
        with open(RULE_PATH) as f:
            self.rule = json.load(f)      # {"feature": "MA_area_frac", "threshold": ...}

    def measure(self, img_bgr):
        """img_bgr: cropped 512x512 BGR uint8. Returns (features dict, masks dict)."""
        probs = run_model(self.model, [preprocess(img_bgr)], self.device)[0]
        return analyse_image(probs, retina_mask(img_bgr), self.thresholds)

    def measure_path(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        feats, masks = self.measure(img)
        return feats, masks, img

    def discordant(self, feats):
        """Deterministic verify-before-clear check. True = lesion burden is too high
        for an auto-clear (rule and threshold fixed on VAL, see offline_ablation.py)."""
        return feats[self.rule["feature"]] > self.rule["threshold"]


def summarise(feats):
    """Compact, LLM-friendly summary: only lesion types with at least one region."""
    out = {}
    for l in COUNT_CLASSES:
        n = int(feats[f"{l}_count"])
        if n == 0:
            continue
        quads = {QUAD_NAMES[q]: int(feats[f"{l}_{q}"]) for q in QUADS if feats[f"{l}_{q}"] > 0}
        top_quad = max(quads, key=quads.get) if quads else None
        out[l] = {"name": LESION_NAMES[l], "count": n,
                  "area_pct_of_retina": round(float(feats[f"{l}_area_frac"]), 3),
                  "most_regions_in": top_quad, "by_quadrant": quads}
    return out


def overlay(img_bgr, masks):
    """Lesion contours drawn on the image (right half of make_overlay's side-by-side)."""
    both = make_overlay(img_bgr, masks)
    return both[:, img_bgr.shape[1]:]
