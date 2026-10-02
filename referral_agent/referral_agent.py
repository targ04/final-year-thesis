"""
Referral Agent: calibrates the Diagnosis Agent's ensemble output via
temperature scaling, then applies a triage policy that decides, per image,
whether the case can be auto-cleared, must be referred due to severity, or
must be referred because the model isn't confident enough.

--- Why temperature scaling on softmax outputs works here ---
Classic temperature scaling (Guo et al., 2017) divides pre-softmax logits by
a learned scalar T before the softmax. ensemble_eval.py only saved softmax
PROBABILITIES (not raw logits), because the ensemble itself is an average of
two models' softmax outputs, which doesn't have a single underlying logit
vector anyway. We work around this with pseudo-logits: z = log(p + eps).
Since softmax(log(p)) == p exactly, and softmax is shift-invariant, log(p)
is a valid stand-in for "logits" for the purpose of fitting T. This is a
standard, mathematically sound trick for calibrating an already-softmaxed
(e.g. ensembled) distribution.

--- Methodology note worth flagging in your thesis / to double check ---
The ensemble weight sweep (resnet_weight=0.60) was already selected using
the val set. Fitting temperature ALSO on val set is common practice but
means val is now doing double duty (model selection + calibration fitting).
If you have a separate held-out test_manifest.csv that was NOT used for the
weight sweep, prefer fitting T on val and reporting final ECE/triage
behavior on test — that's the methodologically cleaner story. If test data
isn't available, fitting and reporting both on val is defensible but should
be stated as a limitation.

--- Triage policy (clinically motivated, tune thresholds as needed) ---
1. MANDATORY REFERRAL: if the predicted grade is in MANDATORY_REFERRAL_GRADES
   (moderate NPDR and worse), refer regardless of confidence. Severity-based
   safety-first gating: a low-confidence "grade 3" should never be quietly
   auto-cleared just because the model wasn't sure.
2. CONFIDENT AUTO-CLEAR: if predicted grade is mild/no-DR AND calibrated
   confidence >= CONFIDENCE_THRESHOLD, auto-clear (no human review needed).
3. UNCERTAIN -> REFER: if predicted grade is mild/no-DR but calibrated
   confidence is below threshold, refer for human review anyway (the model
   isn't sure enough to auto-clear).

Usage:
    python referral_agent.py
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ---------------- CONFIG ----------------

BASE_DIR = Path(__file__).resolve().parent

# Input: the per-image ensemble predictions CSV saved by ensemble_eval.py
ENSEMBLE_PREDICTIONS_CSV = BASE_DIR / "new_ensemble_val_predictions.csv"

# If you have a separate held-out split (recommended - see methodology note
# above), point this at its own ensemble_eval.py output. Leave as None to
# fit and evaluate on the same file above.
HELD_OUT_PREDICTIONS_CSV = BASE_DIR / "new_ensemble_test_predictions.csv"

NUM_CLASSES = 5
LABEL_COL = "true_label"
PROB_COL_PREFIX = "ensemble_prob_class"   # matches ensemble_eval.py's output columns

OUTPUT_TRIAGE_CSV = BASE_DIR / "referral_triage_output.csv"
OUTPUT_TEMPERATURE_JSON = BASE_DIR / "temperature_scaling.json"

# Grades 2 (moderate NPDR), 3 (severe NPDR), 4 (PDR) are always referred
# regardless of model confidence. Adjust if your clinical framing differs.
MANDATORY_REFERRAL_GRADES = {2, 3, 4}

# Calibrated confidence required to auto-clear a mild/no-DR prediction.
CONFIDENCE_THRESHOLD = 0.85

ECE_NUM_BINS = 15
EPS = 1e-12
# -----------------------------------------


def load_predictions(csv_path, num_classes):
    df = pd.read_csv(csv_path)
    prob_cols = [f"{PROB_COL_PREFIX}{c}" for c in range(num_classes)]
    probs = df[prob_cols].values.astype(np.float64)
    labels = df[LABEL_COL].values.astype(np.int64)
    return df, probs, labels


def probs_to_pseudo_logits(probs, eps=EPS):
    return np.log(probs + eps)


class TemperatureScaler(nn.Module):
    """Learns a single scalar T such that softmax(pseudo_logits / T) is
    better calibrated on the fitting set, per Guo et al. 2017."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # T starts at 1.0

    def forward(self, pseudo_logits):
        temperature = torch.exp(self.log_temperature)
        return pseudo_logits / temperature

    @property
    def temperature(self):
        return torch.exp(self.log_temperature).item()


def fit_temperature(pseudo_logits_np, labels_np, max_iter=200, lr=0.01):
    pseudo_logits = torch.tensor(pseudo_logits_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)

    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)
    nll_criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled = scaler(pseudo_logits)
        loss = nll_criterion(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    final_loss = nll_criterion(scaler(pseudo_logits), labels).item()
    print(f"Fitted temperature T = {scaler.temperature:.4f} (final NLL = {final_loss:.4f})")
    return scaler.temperature


def apply_temperature(probs_np, temperature, eps=EPS):
    pseudo_logits = probs_to_pseudo_logits(probs_np, eps)
    scaled_logits = torch.tensor(pseudo_logits, dtype=torch.float32) / temperature
    calibrated_probs = F.softmax(scaled_logits, dim=1).numpy()
    return calibrated_probs


def compute_ece(probs, labels, num_bins=ECE_NUM_BINS):
    """Expected Calibration Error: bins predictions by confidence (max prob),
    and measures the gap between confidence and actual accuracy in each bin,
    weighted by bin size."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_report = []

    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        bin_size = in_bin.sum()
        if bin_size == 0:
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (bin_size / len(confidences)) * abs(bin_acc - bin_conf)
        bin_report.append({
            "bin_range": f"({lo:.2f}, {hi:.2f}]",
            "count": int(bin_size),
            "avg_confidence": round(float(bin_conf), 4),
            "accuracy": round(float(bin_acc), 4),
        })

    return ece, pd.DataFrame(bin_report)


def triage_case(predicted_grade, calibrated_confidence):
    if predicted_grade in MANDATORY_REFERRAL_GRADES:
        return "REFER_SEVERITY"
    if calibrated_confidence >= CONFIDENCE_THRESHOLD:
        return "AUTO_CLEAR"
    return "REFER_LOW_CONFIDENCE"


def main():
    print("Loading ensemble predictions...")
    fit_df, fit_probs, fit_labels = load_predictions(ENSEMBLE_PREDICTIONS_CSV, NUM_CLASSES)

    if HELD_OUT_PREDICTIONS_CSV:
        eval_df, eval_probs, eval_labels = load_predictions(HELD_OUT_PREDICTIONS_CSV, NUM_CLASSES)
        print("Fitting temperature on the fit set, evaluating calibration on the held-out set.")
    else:
        eval_df, eval_probs, eval_labels = fit_df, fit_probs, fit_labels
        print("No separate held-out set provided - fitting and evaluating on the same file. "
              "See the methodology note in this script's docstring.")

    # --- Uncalibrated ECE (baseline) ---
    uncalibrated_ece, uncalibrated_bins = compute_ece(eval_probs, eval_labels)
    print(f"\nUncalibrated ECE: {uncalibrated_ece:.4f}")

    # --- Fit temperature ---
    pseudo_logits = probs_to_pseudo_logits(fit_probs)
    temperature = fit_temperature(pseudo_logits, fit_labels)

    with open(OUTPUT_TEMPERATURE_JSON, "w") as f:
        json.dump({"temperature": temperature}, f, indent=2)
    print(f"Temperature saved to {OUTPUT_TEMPERATURE_JSON}")

    # --- Apply calibration to the evaluation set ---
    calibrated_probs = apply_temperature(eval_probs, temperature)
    calibrated_ece, calibrated_bins = compute_ece(calibrated_probs, eval_labels)
    print(f"Calibrated ECE:   {calibrated_ece:.4f}  (T={temperature:.4f})")

    if calibrated_ece < uncalibrated_ece:
        print(f"Calibration improved ECE by {uncalibrated_ece - calibrated_ece:.4f}.")
    else:
        print("Warning: calibration did NOT improve ECE on this set - "
              "worth checking whether the fit set is representative of the eval set.")

    print("\n--- Reliability bins (calibrated) ---")
    print(calibrated_bins.to_string(index=False))

    # --- Triage decisions ---
    predicted_grades = calibrated_probs.argmax(axis=1)
    calibrated_confidences = calibrated_probs.max(axis=1)

    triage_decisions = [
        triage_case(grade, conf)
        for grade, conf in zip(predicted_grades, calibrated_confidences)
    ]

    out_df = eval_df.copy()
    out_df["calibrated_pred"] = predicted_grades
    out_df["calibrated_confidence"] = calibrated_confidences
    for c in range(NUM_CLASSES):
        out_df[f"calibrated_prob_class{c}"] = calibrated_probs[:, c]
    out_df["triage_decision"] = triage_decisions

    out_df.to_csv(OUTPUT_TRIAGE_CSV, index=False)
    print(f"\nPer-image triage decisions saved to {OUTPUT_TRIAGE_CSV}")

    print("\n--- Triage decision counts ---")
    print(pd.Series(triage_decisions).value_counts())

    # --- Sanity check: how many mandatory-referral cases would have been
    # auto-cleared under a naive "just threshold confidence" policy with no
    # severity override? Useful for your thesis discussion of why the
    # severity gate matters. ---
    naive_auto_clear = calibrated_confidences >= CONFIDENCE_THRESHOLD
    severity_overridden = naive_auto_clear & np.isin(predicted_grades, list(MANDATORY_REFERRAL_GRADES))
    print(f"\nCases where the severity gate overrode a high-confidence prediction "
          f"that a naive confidence-only policy would have auto-cleared: {severity_overridden.sum()}")


if __name__ == "__main__":
    main()
