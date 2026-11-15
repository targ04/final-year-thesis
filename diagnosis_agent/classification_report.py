"""
Computes classical classification metrics (precision, recall, F1 - per-class
and macro/weighted averages) plus a confusion matrix from an ensemble
predictions CSV (the file saved by ensemble_eval.py or referral_agent.py).

Usage:
    python classification_report.py
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score
from pathlib import Path

# ---------------- CONFIG ----------------
base_dir = Path(__file__).parent
PREDICTIONS_CSV = base_dir / "new_ensemble_test_predictions.csv"

# Column names - adjust if reading referral_triage_output.csv instead
# (that file uses "calibrated_pred" instead of "ensemble_pred")
LABEL_COL = "true_label"
PRED_COL = "ensemble_pred"     # use "calibrated_pred" for referral_triage_output.csv

CLASS_NAMES = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "PDR (4)"]
# -----------------------------------------


def main():
    df = pd.read_csv(PREDICTIONS_CSV)
    labels = df[LABEL_COL].values
    preds = df[PRED_COL].values

    print(f"Loaded {len(df)} predictions from {PREDICTIONS_CSV}\n")

    # --- Per-class + aggregate precision/recall/F1 ---
    print("--- Classification report ---")
    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    print(report)

    # --- Confusion matrix ---
    print("--- Confusion matrix (rows = true, cols = predicted) ---")
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print(cm_df.to_string())

    # --- Headline metrics for quick reference ---
    print("\n--- Headline metrics ---")
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds, weights="quadratic")
    print(f"Accuracy:              {acc:.4f}")
    print(f"Quadratic weighted kappa: {kappa:.4f}")


if __name__ == "__main__":
    main()
