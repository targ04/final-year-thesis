"""
What does 5-class grading add over a binary referable / non-referable output (as in LumineticsCore)?
A binary system says only "refer". ICO 2017 Table 2a assigns a DIFFERENT follow-up interval to every
grade (0: 1-2 y, 1: 6-12 mo, 2: 3-6 mo, 3: < 3 mo, 4: < 1 mo), so the grade carries urgency.
This script measures how well the 5-class ensemble supports urgency triage. CPU only, saved predictions.

Output: agentic/results/urgency_analysis.csv
Usage:  python agentic/urgency_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
T = json.load(open(ROOT / "referral_agent" / "temperature_scaling.json"))["temperature"]


def load(split):
    p = pd.read_csv(ROOT / "referral_agent" / f"new_ensemble_{split}_predictions.csv")
    P = p[[f"ensemble_prob_class{i}" for i in range(5)]].values
    z = np.log(P + 1e-12) / T
    c = np.exp(z - z.max(1, keepdims=True))
    c /= c.sum(1, keepdims=True)
    return p["true_label"].values, c


def main():
    rows = []
    for split in ["val", "test"]:
        y, c = load(split)
        g = c.argmax(1)
        vt, pv = y >= 3, g >= 3
        ref = y >= 2
        queue = ref | (g >= 2)                      # cases that would reach a specialist queue
        q = y[queue][np.argsort(-c[queue, 3:].sum(1))]
        half = len(q) // 2
        rows.append({
            "split": split,
            "urgent_prevalence_%": round(100 * vt.mean(), 1),
            "urgent(grade>=3)_sensitivity_%": round(100 * (pv & vt).sum() / vt.sum(), 1),
            "urgent_specificity_%": round(100 * (~pv & ~vt).sum() / (~vt).sum(), 1),
            "urgent_AUC": round(roc_auc_score(vt, c[:, 3:].sum(1)), 3),
            "AUC_urgent_vs_moderate_within_referable": round(roc_auc_score(y[ref] >= 3, c[ref, 3:].sum(1)), 3),
            "ICO_interval_exact_%": round(100 * (g == y).mean(), 1),
            "ICO_interval_within_one_step_%": round(100 * (abs(g - y) <= 1).mean(), 1),
            "interval_later_than_correct_%": round(100 * (g < y).mean(), 1),
            "interval_two+_steps_later_%": round(100 * (g <= y - 2).mean(), 2),
            "true_severe_given_urgent_interval_%": round(100 * (g[y == 3] >= 3).mean(), 1),
            "true_PDR_given_urgent_interval_%": round(100 * (g[y == 4] >= 3).mean(), 1),
            "true_PDR_given_<1_month_%": round(100 * (g[y == 4] == 4).mean(), 1),
            "urgent_cases_in_first_half_of_queue_%": round(100 * (q[:half] >= 3).sum() / (q >= 3).sum(), 1),
        })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "agentic" / "results" / "urgency_analysis.csv", index=False)
    print(out.T.to_string(header=False))


if __name__ == "__main__":
    main()
