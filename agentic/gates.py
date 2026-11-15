"""
Auto-clear gate shared by offline_ablation.py, orchestrator_agentic.py and gradio_app.py.

Two rules (the decision is always: severity gate first, then this gate for grade 0/1):
  "max_class"  (v1 / midsem): clear if the single most likely grade's calibrated
               probability >= 0.85.
  "nonref_sum" (v2, adopted): clear if P(non-referable) = P(grade 0) + P(grade 1) >= t.
               The clinical decision is binary (refer at moderate NPDR or worse, the
               same more-than-mild threshold as FDA-cleared autonomous systems), so the
               confidence should be in that binary outcome, not in one grade. t is
               selected on VAL by binary_gate_analysis.py (written to
               results/binary_gate_selected.json).
"""

import json
from pathlib import Path

import numpy as np

SELECTED_PATH = Path(__file__).resolve().parent / "results" / "binary_gate_selected.json"
V1_THRESHOLD = 0.85
MANDATORY = 2


def load_selected():
    if SELECTED_PATH.exists():
        with open(SELECTED_PATH) as f:
            return json.load(f)
    return {"rule": "max_class", "threshold": V1_THRESHOLD}


def p_nonref(cal):
    cal = np.asarray(cal)
    return cal[..., :MANDATORY].sum(-1)


def clear_score(cal, rule):
    """The confidence the gate thresholds: top-class prob (v1) or P(grade 0 + 1) (v2)."""
    cal = np.asarray(cal)
    return cal.max(-1) if rule == "max_class" else p_nonref(cal)


def decide(cal, rule, threshold):
    """Single image: returns REFER_SEVERITY / AUTO_CLEAR / REFER_LOW_CONFIDENCE."""
    g = int(np.argmax(cal))
    if g >= MANDATORY:
        return "REFER_SEVERITY"
    return "AUTO_CLEAR" if float(clear_score(cal, rule)) >= threshold else "REFER_LOW_CONFIDENCE"


def decide_batch(cal, rule, threshold):
    cal = np.asarray(cal)
    g = cal.argmax(1)
    s = clear_score(cal, rule)
    return np.where(g >= MANDATORY, "REFER_SEVERITY",
                    np.where(s >= threshold, "AUTO_CLEAR", "REFER_LOW_CONFIDENCE"))
