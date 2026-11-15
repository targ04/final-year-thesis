"""
All thesis statistics in one place. CPU only, reads saved outputs - no models, no LLM.

Confidence intervals: 95% percentile bootstrap, resampling PATIENTS (both eyes of a patient
are correlated, so resampling images would understate uncertainty), 2000 resamples, seed 42.
Proportions with few events also get a Wilson interval.

Sections (each writes agentic/results/stats/<name>.csv; STATS_SUMMARY.md collects everything):
  A  grading: ResNet-50 / ViT-B/16 / ensemble - accuracy, QWK, linear kappa, MAE, off-by-k,
     per-class P/R/F1, confusion matrices (counts and row-normalised), McNemar tests
  B  calibration: ECE / MCE / Brier / NLL before and after temperature scaling (5-class), and
     calibration of the binary P(non-referable) that the v2 gate thresholds
  C  binary endpoints: referable (grade>=2) and vision-threatening (grade>=3) - sens, spec,
     PPV, NPV, F1, AUC with CIs
  D  triage policies (v1 gate, v2 gate, + lesion, + all tools / LLM planner): decision counts,
     auto-clear rate, misses, miss rate, referral recall, workload, decision x true-grade tables,
     paired comparisons of missed cases
  E  screening-prevalence projection: per-grade policy rates re-weighted to the grade mix of the
     full 88k EyePACS pool (the test set is enriched for disease)
  F  agentic verification: overrides and catches per tool, tool paths, planner vs fixed rule
  G  lesion module: IDRiD test metrics + EyePACS burden-vs-grade (val, test)
  H  explanations: pass rate (overall, per grade, per decision), attempts, length, citations
  I  saliency: backbone attention agreement and its relation to correctness
  J  bilateral: fellow-eye availability, inter-eye grade agreement in the ground truth
  K  urgency: ICO interval agreement and referral-queue prioritisation, with CIs
  L  dataset: class distributions of the 15k splits and the 88k pool

Usage:  python agentic/thesis_statistics.py
"""

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "agentic" / "results"
OUT = RES / "stats"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_rules import check_explanation, SENT_SPLIT   # noqa: E402

T = json.load(open(ROOT / "referral_agent" / "temperature_scaling.json"))["temperature"]
GATE = json.load(open(RES / "binary_gate_selected.json"))
LESION_RULE = json.load(open(RES / "selected_lesion_rule.json"))
LABELS = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
B = 2000
RNG = np.random.default_rng(42)
SUMMARY = []


# ---------------------------------------------------------------- helpers
def calib(P, t=T):
    z = np.log(P + 1e-12) / t
    c = np.exp(z - z.max(1, keepdims=True))
    return c / c.sum(1, keepdims=True)


def load_preds(split):
    p = pd.read_csv(ROOT / "referral_agent" / f"new_ensemble_{split}_predictions.csv")
    get = lambda m: p[[f"{m}_prob_class{i}" for i in range(5)]].values
    return p, {"resnet": get("resnet"), "vit": get("vit"), "ensemble": get("ensemble")}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def patient_boot(patients, stat_fn, b=B):
    """stat_fn(idx) -> float or dict; returns (point, lo, hi) per key."""
    pts = np.asarray(patients)
    uniq = np.unique(pts)
    groups = {u: np.where(pts == u)[0] for u in uniq}
    point = stat_fn(np.arange(len(pts)))
    keys = point.keys() if isinstance(point, dict) else [None]
    draws = {k: [] for k in keys}
    for _ in range(b):
        samp = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[u] for u in samp])
        v = stat_fn(idx)
        for k in keys:
            draws[k].append(v[k] if k is not None else v)
    out = {}
    for k in keys:
        arr = np.array(draws[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        pv = point[k] if k is not None else point
        out[k] = (pv, np.percentile(arr, 2.5), np.percentile(arr, 97.5))
    return out if isinstance(point, dict) else out[None]


def fmt(v, lo, hi, pct=True, d=1):
    if pct:
        return f"{100 * v:.{d}f}% [{100 * lo:.{d}f}, {100 * hi:.{d}f}]"
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def mcnemar(a_correct, b_correct):
    """Exact McNemar on paired correctness. Returns (b01, b10, p)."""
    b01 = int((~a_correct & b_correct).sum())
    b10 = int((a_correct & ~b_correct).sum())
    n = b01 + b10
    p = binomtest(b01, n, 0.5).pvalue if n else 1.0
    return b01, b10, p


def save(df, name, title=None, note=None):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    if title:
        SUMMARY.append(f"\n### {title}\n`stats/{name}.csv`" + (f" - {note}" if note else "") + "\n")
        SUMMARY.append(df.to_markdown(index=False))


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e, m = 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            gap = abs(conf[s].mean() - correct[s].mean())
            e += s.mean() * gap
            m = max(m, gap)
    return e, m


# ---------------------------------------------------------------- A grading
def section_a():
    rows, cls_rows = [], []
    for split in ["val", "test"]:
        p, probs = load_preds(split)
        y, pid = p.true_label.values, p.patient_id.values
        preds = {m: probs[m].argmax(1) for m in probs}
        for m, g in preds.items():
            def stat(idx, g=g):
                yy, gg = y[idx], g[idx]
                return {"acc": (yy == gg).mean(), "qwk": cohen_kappa_score(yy, gg, weights="quadratic")}
            bs = patient_boot(pid, stat)
            err = np.abs(g - y)
            rows.append({"split": split, "model": m, "n": len(y),
                         "accuracy": fmt(*bs["acc"]), "QWK": fmt(*bs["qwk"], pct=False, d=4),
                         "linear_kappa": round(cohen_kappa_score(y, g, weights="linear"), 4),
                         "MAE_grades": round(err.mean(), 3),
                         "exact_%": round(100 * (err == 0).mean(), 2),
                         "off_by_1_%": round(100 * (err == 1).mean(), 2),
                         "off_by_2+_%": round(100 * (err >= 2).mean(), 2)})
            P, R, F, S = precision_recall_fscore_support(y, g, labels=range(5), zero_division=0)
            for k in range(5):
                cls_rows.append({"split": split, "model": m, "grade": f"{k} {LABELS[k]}",
                                 "precision": round(P[k], 4), "recall": round(R[k], 4),
                                 "F1": round(F[k], 4), "support": int(S[k])})
            for avg in ["macro", "weighted"]:
                P_, R_, F_, _ = precision_recall_fscore_support(y, g, average=avg, zero_division=0)
                cls_rows.append({"split": split, "model": m, "grade": f"{avg} avg", "precision": round(P_, 4),
                                 "recall": round(R_, 4), "F1": round(F_, 4), "support": len(y)})
            cm = confusion_matrix(y, g, labels=range(5))
            cmdf = pd.DataFrame(cm, columns=[f"pred {k}" for k in range(5)])
            cmdf.insert(0, "true", [f"{k} {LABELS[k]}" for k in range(5)])
            save(cmdf, f"A_confusion_{m}_{split}",
                 f"A. Confusion matrix - {m}, {split} (rows true, columns predicted)" if m == "ensemble" else None)
            cmn = (cm / cm.sum(1, keepdims=True)).round(3)
            cmndf = pd.DataFrame(cmn, columns=[f"pred {k}" for k in range(5)])
            cmndf.insert(0, "true", [f"{k} {LABELS[k]}" for k in range(5)])
            save(cmndf, f"A_confusion_{m}_{split}_rownorm",
                 f"A. Confusion matrix (row-normalised = per-grade recall) - {m}, {split}" if m == "ensemble" else None)
        # McNemar: ensemble vs each backbone on exact-grade correctness
        for other in ["resnet", "vit"]:
            b01, b10, pv = mcnemar(preds[other] == y, preds["ensemble"] == y)
            rows.append({"split": split, "model": f"McNemar ensemble vs {other}",
                         "n": len(y), "accuracy": f"ensemble-only correct {b01}, {other}-only correct {b10}",
                         "QWK": f"exact p = {pv:.4g}"})
    save(pd.DataFrame(rows), "A_grading_summary", "A. Grading performance (95% patient-bootstrap CI)",
         "McNemar rows compare exact-grade correctness, paired by image")
    save(pd.DataFrame(cls_rows), "A_per_class", "A. Per-class precision / recall / F1")


# ---------------------------------------------------------------- B calibration
def section_b():
    rows = []
    for split in ["val", "test"]:
        p, probs = load_preds(split)
        y = p.true_label.values
        for name, P in [("uncalibrated", probs["ensemble"]), (f"temperature T={T:.4f}", calib(probs["ensemble"]))]:
            conf, g = P.max(1), P.argmax(1)
            e, m = ece(conf, (g == y).astype(float))
            onehot = np.eye(5)[y]
            pnr = P[:, :2].sum(1)
            nonref = (y < 2).astype(float)
            eb, mb = ece(pnr, nonref)
            rows.append({"split": split, "probabilities": name,
                         "ECE_5class": round(e, 4), "MCE_5class": round(m, 4),
                         "Brier_5class": round(((P - onehot) ** 2).sum(1).mean(), 4),
                         "NLL": round(-np.log(P[np.arange(len(y)), y] + 1e-12).mean(), 4),
                         "ECE_binary_P(nonref)": round(eb, 4), "MCE_binary": round(mb, 4),
                         "Brier_binary": round(((pnr - nonref) ** 2).mean(), 4)})
    save(pd.DataFrame(rows), "B_calibration", "B. Calibration (15 equal-width bins)",
         "binary columns = calibration of P(grade 0+1), the quantity the v2 gate thresholds")
    # reliability table of the binary probability on test (for a figure)
    p, probs = load_preds("test")
    P = calib(probs["ensemble"])
    pnr, nonref = P[:, :2].sum(1), (p.true_label.values < 2)
    edges = np.linspace(0, 1, 11)
    rel = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (pnr > lo) & (pnr <= hi)
        if s.any():
            rel.append({"P(nonref) bin": f"({lo:.1f}, {hi:.1f}]", "n": int(s.sum()),
                        "mean P(nonref)": round(pnr[s].mean(), 4),
                        "observed non-referable": round(nonref[s].mean(), 4)})
    save(pd.DataFrame(rel), "B_reliability_binary_test", "B. Reliability of calibrated P(non-referable), test")


# ---------------------------------------------------------------- C binary endpoints
def binary_metrics(y_true_pos, pred_pos, score, pid):
    def stat(idx):
        t, pr = y_true_pos[idx], pred_pos[idx]
        tp, tn = (t & pr).sum(), (~t & ~pr).sum()
        fp, fn = (~t & pr).sum(), (t & ~pr).sum()
        return {"sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1), "ppv": tp / max(tp + fp, 1),
                "npv": tn / max(tn + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "auc": roc_auc_score(t, score[idx]) if 0 < t.sum() < len(t) else np.nan}
    return patient_boot(pid, stat)


def section_c():
    rows = []
    for split in ["val", "test"]:
        p, probs = load_preds(split)
        y, pid = p.true_label.values, p.patient_id.values
        for m in ["resnet", "vit", "ensemble"]:
            P = calib(probs[m]) if m == "ensemble" else probs[m]
            g = P.argmax(1)
            for name, thr in [("referable (grade>=2)", 2), ("vision-threatening (grade>=3)", 3)]:
                bs = binary_metrics(y >= thr, g >= thr, P[:, thr:].sum(1), pid)
                rows.append({"split": split, "model": m, "endpoint": name,
                             "prevalence_%": round(100 * (y >= thr).mean(), 2),
                             **{k.upper(): fmt(*bs[k], pct=k != "auc", d=1 if k != "auc" else 4)
                                for k in ["sens", "spec", "ppv", "npv", "f1", "auc"]}})
    save(pd.DataFrame(rows), "C_binary_endpoints", "C. Binary endpoints from argmax grade (95% CI)",
         "AUC uses P(grade>=k); ensemble probabilities are temperature-calibrated")


# ---------------------------------------------------------------- D triage policies
def load_policy_runs():
    runs = {}
    for name, f in [("v2 gate", "v2_none_triage.csv"), ("v2 gate + lesion", "v2_lesion_only_triage.csv"),
                    ("v2 gate + lesion + all tools", "v2_all_tools_triage.csv"),
                    ("v2 gate + lesion + LLM planner (final)", "v2_llm.csv")]:
        d = pd.read_csv(RES / f)[["image_id", "patient_id", "true_label", "triage_decision", "initial_decision"]]
        runs[name] = d.set_index("image_id")
    # v1 gate (midsem) from saved predictions
    p, probs = load_preds("test")
    P = calib(probs["ensemble"])
    g, c = P.argmax(1), P.max(1)
    dec = np.where(g >= 2, "REFER_SEVERITY", np.where(c >= 0.85, "AUTO_CLEAR", "REFER_LOW_CONFIDENCE"))
    v1 = pd.DataFrame({"image_id": p.image, "patient_id": p.patient_id, "true_label": p.true_label,
                       "triage_decision": dec, "initial_decision": dec}).set_index("image_id")
    ordered = {"v1 gate (midsem)": v1}
    ordered.update(runs)
    idx = v1.index
    return {k: v.loc[idx] for k, v in ordered.items()}


def section_d(runs):
    rows, xtabs = [], []
    for name, d in runs.items():
        y, pid = d.true_label.values, d.patient_id.values
        ac = (d.triage_decision == "AUTO_CLEAR").values

        def stat(idx):
            a, yy = ac[idx], y[idx]
            ref = yy >= 2
            return {"clear_rate": a.mean(), "miss_rate": (a & ref).sum() / max(a.sum(), 1),
                    "recall": (ref & ~a).sum() / max(ref.sum(), 1),
                    "clear_spec": (~ref & a).sum() / max((~ref).sum(), 1)}
        bs = patient_boot(pid, stat)
        miss = int((ac & (y >= 2)).sum())
        wl = wilson(miss, int(ac.sum()))
        counts = d.triage_decision.value_counts()
        rows.append({"policy": name, "AUTO_CLEAR": int(ac.sum()),
                     "auto_clear_rate": fmt(*bs["clear_rate"]),
                     "REFER_SEVERITY": int(counts.get("REFER_SEVERITY", 0)),
                     "REFER_LOW_CONFIDENCE": int(counts.get("REFER_LOW_CONFIDENCE", 0)),
                     "REFER_VERIFIED": int(counts.get("REFER_VERIFIED", 0)),
                     "missed_referable": miss,
                     "miss_rate_among_cleared": fmt(*bs["miss_rate"], d=2),
                     "miss_rate_Wilson": f"[{100 * wl[0]:.2f}, {100 * wl[1]:.2f}]",
                     "referral_recall": fmt(*bs["recall"], d=2),
                     "non-referable_correctly_cleared": fmt(*bs["clear_spec"])})
        xt = pd.crosstab(d.triage_decision, d.true_label).reindex(
            ["AUTO_CLEAR", "REFER_LOW_CONFIDENCE", "REFER_VERIFIED", "REFER_SEVERITY"]).dropna(how="all").fillna(0).astype(int)
        xt.columns = [f"true {c}" for c in xt.columns]
        xt.insert(0, "decision", xt.index)
        xt.insert(0, "policy", name)
        xtabs.append(xt)
    save(pd.DataFrame(rows), "D_triage_policies", "D. Triage policies on test (95% patient-bootstrap CI)",
         "miss rate = referable among auto-cleared; 'non-referable correctly cleared' = auto-clears / all true grade 0-1")
    save(pd.concat(xtabs).reset_index(drop=True), "D_decision_by_true_grade", "D. Decision x true grade (test)")

    # paired comparison of missed referable cases between policies (same referable patients)
    pairs = [("v1 gate (midsem)", "v2 gate"), ("v2 gate", "v2 gate + lesion + LLM planner (final)"),
             ("v1 gate (midsem)", "v2 gate + lesion + LLM planner (final)")]
    prow = []
    for a, b in pairs:
        A, Bp = runs[a], runs[b]
        ref = A.true_label.values >= 2
        caught_a = (A.triage_decision != "AUTO_CLEAR").values[ref]
        caught_b = (Bp.triage_decision != "AUTO_CLEAR").values[ref]
        b01, b10, pv = mcnemar(caught_a, caught_b)
        ma = int((~caught_a).sum())
        mb = int((~caught_b).sum())

        def dstat(idx, A=A, Bp=Bp):
            r = A.true_label.values[idx] >= 2
            return (int(((A.triage_decision.values[idx] == "AUTO_CLEAR") & r).sum())
                    - int(((Bp.triage_decision.values[idx] == "AUTO_CLEAR") & r).sum()))
        diff = patient_boot(A.patient_id.values, dstat)
        prow.append({"policy A": a, "policy B": b, "misses A": ma, "misses B": mb,
                     "caught by B only": b01, "caught by A only": b10,
                     "misses avoided (A-B) [95% CI]": f"{diff[0]} [{diff[1]:.0f}, {diff[2]:.0f}]",
                     "exact McNemar p": round(pv, 4)})
    save(pd.DataFrame(prow), "D_paired_miss_comparison", "D. Paired comparison of missed referable cases (test)",
         "with 10-16 events the tests are underpowered: report the direction and CI, not just p")


# ---------------------------------------------------------------- E screening prevalence
def section_e(runs):
    pool = pd.read_csv(ROOT / "all_88k_labels.csv")
    pi = pool.level.value_counts(normalize=True).sort_index().reindex(range(5)).values
    rows = [{"policy": "grade mix of the 88k EyePACS pool",
             **{f"grade {g} share": f"{100 * pi[g]:.1f}%" for g in range(5)}}]
    save(pd.DataFrame(rows), "E_pool_grade_mix")
    out = []
    for name, d in runs.items():
        y = d.true_label.values
        ac = (d.triage_decision == "AUTO_CLEAR").values
        r = np.array([ac[y == g].mean() for g in range(5)])          # per-grade auto-clear rate on test
        clear = (pi * r).sum()
        miss = (pi[2:] * r[2:]).sum()
        out.append({"policy": name,
                    **{f"clear rate | true {g}": f"{100 * r[g]:.1f}%" for g in range(5)},
                    "projected auto-clear rate": f"{100 * clear:.1f}%",
                    "projected specialist workload": f"{100 * (1 - clear):.1f}%",
                    "projected miss rate among cleared": f"{100 * miss / clear:.2f}%",
                    "missed referable per 1,000 screened": round(1000 * miss, 1),
                    "referable per 1,000 screened": round(1000 * pi[2:].sum(), 1)})
    save(pd.DataFrame(out), "E_screening_projection",
         "E. Projection to screening prevalence (grade mix of the full 88k EyePACS pool)",
         f"pool: {100 * pi[2:].sum():.1f}% referable vs 51.8% in the enriched test set; assumes per-grade behaviour transfers")


# ---------------------------------------------------------------- F verification
def section_f():
    d = pd.read_csv(RES / "v2_llm.csv")
    d["ver"] = d.verification.apply(ast.literal_eval)
    cand = d[d.initial_decision == "AUTO_CLEAR"]
    rows = []
    for tool in ["lesion_check", "fellow_eye", "tta"]:
        ran = cand.ver.apply(lambda vv: any(v["tool"] == tool for v in vv))
        disc = cand.ver.apply(lambda vv: any(v["tool"] == tool and v["discordant"] for v in vv))
        rows.append({"check": tool, "auto-clear candidates": len(cand), "times run": int(ran.sum()),
                     "discordant (override to refer)": int(disc.sum()),
                     "of which truly referable": int((disc & (cand.true_label >= 2)).sum()),
                     "of which true grade 0-1": int((disc & (cand.true_label < 2)).sum())})
    ref_c = cand[cand.true_label >= 2]
    rows.append({"check": "ALL (any discordant)", "auto-clear candidates": len(cand),
                 "times run": "", "discordant (override to refer)": int((cand.triage_decision == "REFER_VERIFIED").sum()),
                 "of which truly referable": int(((cand.triage_decision == "REFER_VERIFIED") & (cand.true_label >= 2)).sum()),
                 "of which true grade 0-1": int(((cand.triage_decision == "REFER_VERIFIED") & (cand.true_label < 2)).sum())})
    save(pd.DataFrame(rows), "F_verification_tools", "F. What each verification check contributed (test, final run)",
         f"{len(ref_c)} referable cases among the {len(cand)} candidates")

    def path(r):
        if r.initial_decision != "AUTO_CLEAR":
            return f"no verification ({r.initial_decision})"
        return " -> ".join(v["tool"] + ("(X)" if v["discordant"] else "(ok)") for v in r.ver) + f" => {r.triage_decision}"
    pc = d.apply(path, axis=1).value_counts().rename_axis("path").reset_index(name="images")
    save(pc, "F_agent_paths", "F. Paths taken by the agent (test, final run)", "(X) = discordant")
    pa = pd.read_csv(RES / "v2_planner_analysis.csv")
    save(pa, "F_planner_vs_fixed_rule", "F. LLM planner vs fixed rule")


# ---------------------------------------------------------------- G lesion module
def section_g():
    tr = json.load(open(ROOT / "idrid" / "idrid_lesions" / "checkpoints" / "test_results.json"))
    rows = []
    for k, v in tr.items():
        if isinstance(v, dict) and "AP" in v:
            rows.append({"lesion": k, **{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}})
    save(pd.DataFrame(rows), "G_idrid_test_metrics", "G. Lesion U-Net on the IDRiD test set (27 images)")
    for split in ["val", "test"]:
        f = ROOT / "idrid" / f"lesion_analysis_{split}" / "burden_vs_grade_summary.csv"
        if f.exists():
            save(pd.read_csv(f).round(4), f"G_eyepacs_burden_vs_grade_{split}",
                 f"G. Lesion burden vs DR grade on EyePACS {split} (transfer, no EyePACS lesion labels)")


# ---------------------------------------------------------------- H explanations
def section_h():
    d = pd.read_csv(RES / "v2_llm.csv")
    counts = d.lesion_counts.apply(ast.literal_eval)
    d["fails"] = [check_explanation(t, c) for t, c in zip(d.why_explanation.fillna(""), counts)]
    d["pass"] = d.fails.apply(len) == 0
    d["n_sent"] = d.why_explanation.fillna("").apply(lambda t: len([s for s in SENT_SPLIT.split(t) if s.strip()]))
    d["n_words"] = d.why_explanation.fillna("").str.split().apply(len)
    d["cites_ICO_T1"] = d.why_explanation.fillna("").str.contains(r"Table 1", regex=False)
    rows = [{"group": "all", "n": len(d), "pass_%": round(100 * d["pass"].mean(), 2),
             "mean_attempts": round(d.explanation_attempts.mean(), 2),
             "first_attempt_pass_%": round(100 * ((d.explanation_attempts == 1) & d.citation_ok.astype(bool)).mean(), 2),
             "mean_words": round(d.n_words.mean(), 1), "mean_sentences": round(d.n_sent.mean(), 2),
             "cites_ICO_Table_1_%": round(100 * d.cites_ICO_T1.mean(), 1)}]
    for col, title in [("predicted_grade", "predicted grade"), ("triage_decision", "decision")]:
        for k, g in d.groupby(col):
            rows.append({"group": f"{title}: {k}", "n": len(g), "pass_%": round(100 * g["pass"].mean(), 2),
                         "mean_attempts": round(g.explanation_attempts.mean(), 2),
                         "first_attempt_pass_%": round(100 * ((g.explanation_attempts == 1) & g.citation_ok.astype(bool)).mean(), 2),
                         "mean_words": round(g.n_words.mean(), 1), "mean_sentences": round(g.n_sent.mean(), 2),
                         "cites_ICO_Table_1_%": round(100 * g.cites_ICO_T1.mean(), 1)})
    lo, hi = wilson(int(d["pass"].sum()), len(d))
    save(pd.DataFrame(rows), "H_explanations", "H. Explanation quality (verifier rules v2.1)",
         f"overall pass Wilson 95% CI [{100 * lo:.2f}, {100 * hi:.2f}]")
    src = d.why_explanation.fillna("").str.findall(r"\b(ICO|AAO)\b[^.]{0,100}?p\.\s?(\d+)").explode().dropna()
    sc = src.apply(lambda t: f"{t[0]} p.{t[1]}").value_counts().head(12).rename_axis("citation").reset_index(name="count")
    save(sc, "H_citations", "H. Most cited guideline pages (PDF page numbers)")


# ---------------------------------------------------------------- I saliency
def section_i():
    d = pd.read_csv(RES / "v2_llm.csv")
    s = d.saliency.apply(ast.literal_eval)
    d["agree"] = s.apply(lambda x: bool(x["attention_agreement"]))
    d["r_conc"] = s.apply(lambda x: x["resnet_concentration"])
    d["v_conc"] = s.apply(lambda x: x["vit_concentration"])
    d["correct"] = d.predicted_grade == d.true_label
    rows = [{"group": "all", "n": len(d), "backbones agree on quadrant_%": round(100 * d.agree.mean(), 1),
             "accuracy when agree_%": round(100 * d[d.agree].correct.mean(), 1),
             "accuracy when disagree_%": round(100 * d[~d.agree].correct.mean(), 1),
             "median ResNet concentration": round(d.r_conc.median(), 3),
             "median ViT concentration": round(d.v_conc.median(), 3)}]
    for g, x in d.groupby("true_label"):
        rows.append({"group": f"true grade {g}", "n": len(x),
                     "backbones agree on quadrant_%": round(100 * x.agree.mean(), 1),
                     "accuracy when agree_%": round(100 * x[x.agree].correct.mean(), 1) if x.agree.any() else None,
                     "accuracy when disagree_%": round(100 * x[~x.agree].correct.mean(), 1) if (~x.agree).any() else None,
                     "median ResNet concentration": round(x.r_conc.median(), 3),
                     "median ViT concentration": round(x.v_conc.median(), 3)})
    p = chi2_contingency(pd.crosstab(d.agree, d.correct).values)[1]
    save(pd.DataFrame(rows), "I_saliency", "I. Backbone attention agreement (test)",
         f"agreement vs correctness chi-square p = {p:.3g}; concentration = share of saliency in the top region")


# ---------------------------------------------------------------- J bilateral
def section_j():
    rows = []
    for split in ["val", "test"]:
        p, _ = load_preds(split)
        p["side"] = p.image.str.extract(r"_(left|right)$")[0]
        piv = p.pivot_table(index="patient_id", columns="side", values="true_label", aggfunc="first").dropna()
        diff = (piv["left"] - piv["right"]).abs()
        both_ref = ((piv["left"] >= 2) != (piv["right"] >= 2))
        rows.append({"split": split, "images": len(p), "patients": p.patient_id.nunique(),
                     "patients with both eyes": len(piv),
                     "images with fellow eye available_%": round(100 * 2 * len(piv) / len(p), 1),
                     "both eyes same grade_%": round(100 * (diff == 0).mean(), 1),
                     "eyes differ by 1_%": round(100 * (diff == 1).mean(), 1),
                     "eyes differ by 2+_%": round(100 * (diff >= 2).mean(), 1),
                     "referability differs between eyes_%": round(100 * both_ref.mean(), 1)})
    save(pd.DataFrame(rows), "J_bilateral", "J. Bilateral (two-eye) statistics of the ground truth",
         "motivates using the fellow eye as evidence rather than averaging the two eyes")


# ---------------------------------------------------------------- K urgency
def section_k():
    rows = []
    for split in ["val", "test"]:
        p, probs = load_preds(split)
        y, pid = p.true_label.values, p.patient_id.values
        P = calib(probs["ensemble"])
        g = P.argmax(1)

        def stat(idx):
            yy, gg, pp = y[idx], g[idx], P[idx]
            refq = (yy >= 2) | (gg >= 2)
            q = yy[refq][np.argsort(-pp[refq, 3:].sum(1))]
            h = len(q) // 2
            return {"exact": (gg == yy).mean(), "within1": (np.abs(gg - yy) <= 1).mean(),
                    "later": (gg < yy).mean(), "later2": (gg <= yy - 2).mean(),
                    "queue": (q[:h] >= 3).sum() / max((q >= 3).sum(), 1),
                    "pdr_urgent": (gg[yy == 4] >= 3).mean() if (yy == 4).any() else np.nan}
        bs = patient_boot(pid, stat)
        rows.append({"split": split,
                     "ICO interval exact": fmt(*bs["exact"]), "within one step": fmt(*bs["within1"]),
                     "later than correct (under-triage)": fmt(*bs["later"]),
                     "2+ steps later": fmt(*bs["later2"], d=2),
                     "true PDR flagged urgent (<3 mo)": fmt(*bs["pdr_urgent"]),
                     "urgent cases in first half of referral queue": fmt(*bs["queue"])})
    save(pd.DataFrame(rows), "K_urgency", "K. Urgency / ICO interval agreement (95% patient-bootstrap CI)",
         "random queue ordering would put 50% of urgent cases in the first half")


# ---------------------------------------------------------------- L dataset
def section_l():
    rows = []
    for split in ["train", "val", "test"]:
        f = ROOT / "referral_agent" / f"new_ensemble_{split}_predictions.csv"
        if f.exists():
            y = pd.read_csv(f).true_label
            rows.append({"set": f"15k {split}", "images": len(y),
                         **{f"grade {g}": f"{(y == g).sum()} ({100 * (y == g).mean():.1f}%)" for g in range(5)},
                         "referable_%": round(100 * (y >= 2).mean(), 1)})
    pool = pd.read_csv(ROOT / "all_88k_labels.csv").level
    rows.append({"set": "full EyePACS pool (88k)", "images": len(pool),
                 **{f"grade {g}": f"{(pool == g).sum()} ({100 * (pool == g).mean():.1f}%)" for g in range(5)},
                 "referable_%": round(100 * (pool >= 2).mean(), 1)})
    save(pd.DataFrame(rows), "L_dataset", "L. Class distributions",
         "train split counts are in the midsem report (Table 2); only val/test predictions are saved here")


# ---------------------------------------------------------------- M ensemble weight sweep
def section_m():
    f = ROOT / "diagnosis_agent" / "ensemble_weight_sweep.csv"
    if f.exists():
        save(pd.read_csv(f).round(4), "M_ensemble_weight_sweep", "M. Ensemble weight sweep (validation)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.append("# Thesis statistics (generated by agentic/thesis_statistics.py)\n\n95% CIs: patient-level "
                   "bootstrap, 2000 resamples, seed 42. Test set n = 2,240 unless stated.")
    section_l()
    section_a()
    section_b()
    section_c()
    runs = load_policy_runs()
    section_d(runs)
    section_e(runs)
    section_f()
    section_g()
    section_h()
    section_i()
    section_j()
    section_k()
    section_m()
    (RES / "STATS_SUMMARY.md").write_text("\n".join(SUMMARY), encoding="utf-8")
    print(f"Wrote {len(list(OUT.glob('*.csv')))} tables to {OUT} and {RES / 'STATS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
