"""
Structuring Agent.

Takes the Referral Agent's per-image output (predicted grade, calibrated
confidence, per-class probabilities, triage decision) and converts it into
a clean, schema'd structured record per image - plus a templated natural-
language summary of that record.

This is deliberately NOT an LLM step. It's a formatting/schema layer: every
fact in the structured record and summary is a direct, deterministic
readout of upstream agent outputs (Diagnosis Agent's probabilities,
Referral Agent's triage decision). The Explanation Agent (next agent in the
pipeline) is what will take this structured, already-verified fact-set and
ground it against clinical guideline text via RAG. Keeping this step
deterministic and LLM-free means every fact the Explanation Agent later
references is traceable back to a specific model output - important for
your thesis's grounding/faithfulness story, since none of these clinical
facts are hallucinated or paraphrased.

Usage:
    python structuring_agent.py
"""

import json
import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
# Input: the Referral Agent's per-image triage output
basepath = Path(__file__).parent
TRIAGE_CSV_PATH = basepath.parent / "referral_agent" / "referral_triage_output.csv"

OUTPUT_JSON_PATH = basepath / "structured_records.json"
OUTPUT_CSV_PATH = basepath / "structured_records.csv"

IMAGE_COL = "image"
PATIENT_COL = "patient_id"
LABEL_COL = "true_label"          # kept for evaluation/thesis use only - see note below
PRED_COL = "calibrated_pred"
CONFIDENCE_COL = "calibrated_confidence"
PROB_COL_PREFIX = "calibrated_prob_class"
TRIAGE_COL = "triage_decision"

NUM_CLASSES = 5

# Must match CONFIDENCE_THRESHOLD in referral_agent.py, since the summary
# text below references this value directly.
CONFIDENCE_THRESHOLD = 0.85
MANDATORY_REFERRAL_GRADES = {2, 3, 4}

GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR (PDR)",
}
# -----------------------------------------


def build_referral_reason(triage_decision, grade_label, confidence, threshold):
    """Deterministic, templated explanation of WHY the triage decision was
    made - grounded directly in the triage rule that fired, not generated
    text. This is the fact-set the Explanation Agent will elaborate on."""
    conf_pct = f"{confidence * 100:.1f}%"
    threshold_pct = f"{threshold * 100:.0f}%"

    if triage_decision == "REFER_SEVERITY":
        return (
            f"Referred for specialist review because the predicted grade "
            f"({grade_label}) meets the mandatory referral threshold "
            f"(moderate NPDR or worse). This referral applies regardless of "
            f"model confidence, per the severity-first triage policy."
        )
    elif triage_decision == "AUTO_CLEAR":
        return (
            f"No referral required. Predicted grade ({grade_label}) with "
            f"calibrated confidence {conf_pct}, which meets the auto-clear "
            f"confidence threshold of {threshold_pct}."
        )
    elif triage_decision == "REFER_LOW_CONFIDENCE":
        return (
            f"Referred for specialist review because calibrated confidence "
            f"({conf_pct}) fell below the auto-clear threshold "
            f"({threshold_pct}), despite a {grade_label.lower()} prediction. "
            f"The model is not confident enough in this case to clear it "
            f"without human review."
        )
    else:
        return f"Unrecognized triage decision: {triage_decision}"


def build_structured_record(row, include_ground_truth):
    predicted_grade = int(row[PRED_COL])
    grade_label = GRADE_LABELS[predicted_grade]
    confidence = float(row[CONFIDENCE_COL])
    triage_decision = row[TRIAGE_COL]

    class_probabilities = {
        GRADE_LABELS[c]: round(float(row[f"{PROB_COL_PREFIX}{c}"]), 4)
        for c in range(NUM_CLASSES)
    }

    record = {
        "image_id": row[IMAGE_COL],
        "patient_id": row[PATIENT_COL] if PATIENT_COL in row.index else None,
        "diagnosis": {
            "predicted_grade": predicted_grade,
            "grade_label": grade_label,
            "calibrated_confidence": round(confidence, 4),
            "class_probabilities": class_probabilities,
        },
        "referral": {
            "decision": triage_decision,
            "reason": build_referral_reason(
                triage_decision, grade_label, confidence, CONFIDENCE_THRESHOLD
            ),
        },
        "structured_summary": (
            f"{grade_label} (grade {predicted_grade}) detected with "
            f"{confidence * 100:.1f}% calibrated confidence. "
            f"{build_referral_reason(triage_decision, grade_label, confidence, CONFIDENCE_THRESHOLD)}"
        ),
    }

    # True label is included ONLY for thesis evaluation/error-analysis use
    # (e.g. checking structured records against known misses like the
    # safety-net miss rate cases). A real deployment-facing schema would
    # omit this entirely, since ground truth isn't available at inference
    # time - keep this flag in mind if you reuse this schema downstream.
    if include_ground_truth and LABEL_COL in row.index:
        record["_evaluation_only"] = {
            "true_label": int(row[LABEL_COL]),
            "true_grade_label": GRADE_LABELS[int(row[LABEL_COL])],
            "correct_prediction": int(row[LABEL_COL]) == predicted_grade,
        }

    return record


def main():
    df = pd.read_csv(TRIAGE_CSV_PATH)
    has_ground_truth = LABEL_COL in df.columns

    records = [
        build_structured_record(row, include_ground_truth=has_ground_truth)
        for _, row in df.iterrows()
    ]

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Structured records (JSON) saved to {OUTPUT_JSON_PATH}")

    # Flat CSV version for quick inspection / spreadsheet use
    flat_rows = []
    for r in records:
        flat_row = {
            "image_id": r["image_id"],
            "patient_id": r["patient_id"],
            "predicted_grade": r["diagnosis"]["predicted_grade"],
            "grade_label": r["diagnosis"]["grade_label"],
            "calibrated_confidence": r["diagnosis"]["calibrated_confidence"],
            "referral_decision": r["referral"]["decision"],
            "referral_reason": r["referral"]["reason"],
            "structured_summary": r["structured_summary"],
        }
        if "_evaluation_only" in r:
            flat_row["true_label"] = r["_evaluation_only"]["true_label"]
            flat_row["correct_prediction"] = r["_evaluation_only"]["correct_prediction"]
        flat_rows.append(flat_row)

    pd.DataFrame(flat_rows).to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Structured records (flat CSV) saved to {OUTPUT_CSV_PATH}")

    # Print a couple of example records so you can sanity check the output
    print("\n--- Example structured records ---")
    for r in records[:3]:
        print(json.dumps(r, indent=2))
        print()


if __name__ == "__main__":
    main()
