"""
Safety-net miss rate for the Referral Agent.

The severity gate guarantees every PREDICTED grade 2/3/4 case gets referred
(REFER_SEVERITY). But that guarantee is only as good as the prediction
itself: if the model predicts grade 0/1 (and is confident enough to
AUTO_CLEAR) while the TRUE label was actually grade 2, 3, or 4, that's a
real missed referral - a patient with genuine moderate-to-proliferative DR
who would have been sent home.

This is arguably a more clinically meaningful "agent effectiveness" number
than kappa or accuracy alone, since it directly measures the one failure
mode that matters most for patient safety: a confident wrong call that
bypasses the severity gate entirely because the wrong call was 0/1 to begin
with.

Usage:
    python safety_net_miss_rate.py
"""

import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
basepath = Path(__file__).parent
TRIAGE_CSV_PATH = basepath / "referral_triage_output.csv"

LABEL_COL = "true_label"
TRIAGE_COL = "triage_decision"
CONFIDENCE_COL = "calibrated_confidence"
PRED_COL = "calibrated_pred"

MANDATORY_REFERRAL_GRADES = {2, 3, 4}
# -----------------------------------------


def main():
    df = pd.read_csv(TRIAGE_CSV_PATH)

    auto_cleared = df[df[TRIAGE_COL] == "AUTO_CLEAR"].copy()
    total_auto_cleared = len(auto_cleared)

    if total_auto_cleared == 0:
        print("No AUTO_CLEAR cases found - nothing to compute.")
        return

    # --- The core safety-net miss rate ---
    missed = auto_cleared[auto_cleared[LABEL_COL].isin(MANDATORY_REFERRAL_GRADES)].copy()
    miss_count = len(missed)
    miss_rate = miss_count / total_auto_cleared

    print("--- Safety-net miss rate ---")
    print(f"Total AUTO_CLEAR cases:                    {total_auto_cleared}")
    print(f"AUTO_CLEAR cases with true grade >= 2:      {miss_count}")
    print(f"Safety-net miss rate (of AUTO_CLEAR cases): {miss_rate:.4f} ({miss_rate * 100:.2f}%)")

    # --- Breakdown by true grade, since a miss to grade 2 is very
    # different clinically from a miss to grade 4 ---
    if miss_count > 0:
        print("\n--- Missed cases by true grade ---")
        print(missed[LABEL_COL].value_counts().sort_index())

        print("\n--- Missed cases detail (predicted grade, true grade, confidence) ---")
        detail_cols = [c for c in [PRED_COL, LABEL_COL, CONFIDENCE_COL] if c in missed.columns]
        print(missed[detail_cols].sort_values(LABEL_COL, ascending=False).to_string(index=False))
    else:
        print("\nNo missed cases - every AUTO_CLEAR case had a true grade of 0 or 1.")

    # --- Context: overall dataset prevalence of grade 2+, for framing the
    # miss rate against the base rate rather than in isolation ---
    overall_severe_rate = df[LABEL_COL].isin(MANDATORY_REFERRAL_GRADES).mean()
    print(f"\nFor context: {overall_severe_rate * 100:.2f}% of the FULL evaluated set "
          f"has true grade >= 2 (base rate).")

    # --- Also report the inverse: of all TRUE grade 2+ cases, what fraction
    # were correctly caught by the referral pipeline (via REFER_SEVERITY or
    # REFER_LOW_CONFIDENCE), i.e. the pipeline's overall referral recall for
    # cases that clinically needed it ---
    true_severe = df[df[LABEL_COL].isin(MANDATORY_REFERRAL_GRADES)]
    caught = true_severe[true_severe[TRIAGE_COL] != "AUTO_CLEAR"]
    referral_recall = len(caught) / len(true_severe) if len(true_severe) > 0 else float("nan")
    print(f"\nOf all true grade >= 2 cases, {referral_recall * 100:.2f}% were referred "
          f"(caught by REFER_SEVERITY or REFER_LOW_CONFIDENCE) rather than auto-cleared.")


if __name__ == "__main__":
    main()
