"""
Binary (referable vs non-referable) view of the 5-class ensemble, and selection of the
auto-clear gate. CPU only - uses the saved calibrated ensemble predictions.

Answers two questions for the report:
 1. How does the 5-class model perform as a binary more-than-mild DR detector
    (grade >= 2 = referable, the threshold used by FDA-cleared autonomous systems
    such as LumineticsCore)? -> sensitivity / specificity / AUC, ROC figure.
 2. Should the auto-clear gate threshold the top single-grade probability (v1) or the
    summed non-referable probability P(0)+P(1)? -> v2 threshold chosen on VAL as the
    LOWEST threshold whose val misses do not exceed v1's (same safety, more clears),
    then both gates reported once on TEST.

Outputs (agentic/results/):
  binary_operating_point.csv     sens/spec/AUC per split (argmax binary decision)
  binary_gate_comparison.csv     v1 gate vs summed-probability gates, val + test
  binary_gate_selected.json      the selected gate (read by gates.py)
  binary_roc.png                 ROC of P(referable) with operating points

Usage:  python agentic/binary_gate_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "agentic" / "results"
T = json.load(open(ROOT / "referral_agent" / "temperature_scaling.json"))["temperature"]
V1_THR = 0.85
GRID = np.round(np.arange(0.80, 0.995, 0.005), 3)

# Published reference point, NOT computed here and NOT on this data (pivotal trial,
# Abramoff et al., npj Digital Medicine 2018). Plotted only for orientation.
LUMINETICS_PIVOTAL = {"sensitivity": 0.872, "specificity": 0.907}


def load(split):
    p = pd.read_csv(ROOT / "referral_agent" / f"new_ensemble_{split}_predictions.csv")
    P = p[[f"ensemble_prob_class{i}" for i in range(5)]].values
    z = np.log(P + 1e-12) / T
    c = np.exp(z - z.max(1, keepdims=True))
    c /= c.sum(1, keepdims=True)
    return p["true_label"].values, c


def gate_metrics(y, c, rule, thr):
    g = c.argmax(1)
    score = c.max(1) if rule == "max_class" else c[:, :2].sum(1)
    clear = (g < 2) & (score >= thr)
    ref = y >= 2
    n, miss = int(clear.sum()), int((clear & ref).sum())
    return {"auto_clear": n, "missed_referable": miss,
            "miss_rate_%": round(100 * miss / max(n, 1), 2),
            "referral_recall_%": round(100 * (ref & ~clear).sum() / ref.sum(), 2),
            "specificity_of_clearing_%": round(100 * (~ref & clear).sum() / (~ref).sum(), 2),
            "referral_workload_%": round(100 * (1 - clear.mean()), 2)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {s: load(s) for s in ["val", "test"]}

    # 1. binary operating point of the argmax prediction
    rows = []
    for s, (y, c) in data.items():
        ref, pred_ref = y >= 2, c.argmax(1) >= 2
        rows.append({"split": s, "n": len(y), "referable_prevalence_%": round(100 * ref.mean(), 2),
                     "sensitivity_%": round(100 * (pred_ref & ref).sum() / ref.sum(), 2),
                     "specificity_%": round(100 * (~pred_ref & ~ref).sum() / (~ref).sum(), 2),
                     "AUC_P(referable)": round(roc_auc_score(ref, c[:, 2:].sum(1)), 4)})
    op = pd.DataFrame(rows)
    op.to_csv(OUT / "binary_operating_point.csv", index=False)
    print("=== Binary referable-DR performance (argmax grade >= 2) ===\n" + op.to_string(index=False))

    # 2. gate selection on VAL
    yv, cv = data["val"]
    v1_val_miss = gate_metrics(yv, cv, "max_class", V1_THR)["missed_referable"]
    ok = [t for t in GRID if gate_metrics(yv, cv, "nonref_sum", t)["missed_referable"] <= v1_val_miss]
    t_sel = float(min(ok))
    sel = {"rule": "nonref_sum", "threshold": t_sel, "selected_on": "val",
           "criterion": f"lowest P(grade0+grade1) threshold with val misses <= v1 gate's ({v1_val_miss})"}
    json.dump(sel, open(OUT / "binary_gate_selected.json", "w"), indent=2)
    print(f"\nSelected on VAL: {sel}")

    rows = []
    for s, (y, c) in data.items():
        for name, rule, thr in [("v1: top-grade prob >= 0.85", "max_class", V1_THR),
                                ("P(0+1) >= 0.85 (same threshold, unsafe)", "nonref_sum", V1_THR),
                                (f"v2: P(0+1) >= {t_sel} (selected on val)", "nonref_sum", t_sel)]:
            rows.append({"split": s, "gate": name, **gate_metrics(y, c, rule, thr)})
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "binary_gate_comparison.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== Auto-clear gate comparison ===\n" + comp.to_string(index=False))

    # cases the v2 gate clears that v1 referred for low confidence (test)
    yt, ct = data["test"]
    g = ct.argmax(1)
    moved = (g < 2) & (ct.max(1) < V1_THR) & (ct[:, :2].sum(1) >= t_sel)
    print(f"\nTEST: {moved.sum()} cases v1 referred for low confidence are cleared by v2; "
          f"true grades {np.bincount(yt[moved], minlength=5).tolist()}")

    # 3. ROC figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for s, (y, c) in data.items():
        fpr, tpr, _ = roc_curve(y >= 2, c[:, 2:].sum(1))
        ax.plot(fpr, tpr, label=f"{s} (AUC {roc_auc_score(y >= 2, c[:, 2:].sum(1)):.3f})")
    t_op = op.set_index("split").loc["test"]
    ax.scatter(1 - t_op["specificity_%"] / 100, t_op["sensitivity_%"] / 100, zorder=5, color="black",
               label="this model, test: argmax grade >= 2")
    ax.scatter(1 - LUMINETICS_PIVOTAL["specificity"], LUMINETICS_PIVOTAL["sensitivity"], zorder=5,
               facecolors="none", edgecolors="red", s=60,
               label="LumineticsCore pivotal trial\n(published; different data)")
    ax.plot([0, 1], [0, 1], ls=":", color="grey")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity (referable DR, grade >= 2)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "binary_roc.png", dpi=200)
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
