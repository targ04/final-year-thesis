"""
How much of the test-split QWK (0.872) is explained by image sharpness?  (CPU only, ~1 minute)

The 15k subset was selected sharpest-first within each grade, so sharpness falls with grade
(median 245 for grade 0, ~42-49 for grades 3-4). Three analyses:

  1. Sharpness-only baseline: grade predicted from sharpness alone (cut-points fitted on the
     TRAIN split of the 15k, applied once to TEST). How much QWK does sharpness give by itself?
  2. Ensemble with the sharpness cue removed: re-weight TEST so that, within each sharpness bin,
     every grade has the same weight as in the test split overall (inverse-probability weights).
     Sharpness then carries no information about grade, so the weighted QWK is what the ensemble
     achieves from disease alone. Also: QWK / AUC within sharpness terciles.
  3. Does sharpness add to the ensemble? Rank correlation between sharpness and the ensemble's
     expected grade WITHIN each true grade (should be ~0 for a model that ignores sharpness).

Usage (repo root):  python agentic/sharpness_ablation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score
import json

ROOT = Path(__file__).resolve().parent.parent
SHARPNESS_CSV = Path(r"E:\EyePACS\sharpness_manifest.csv")
USED_CSV = Path(r"E:\EyePACS\selected_15k_manifest.csv")
N_BINS = 20
N_BOOT = 2000
OUT = ROOT / "agentic" / "results" / "sharpness_ablation.md"

T = json.load(open(ROOT / "referral_agent" / "temperature_scaling.json"))["temperature"]


def strip(s):
    return s.astype(str).str.replace(r"\.(png|jpe?g|tiff?)$", "", regex=True, case=False)


def calibrate(P):
    z = np.log(P + 1e-12) / T
    c = np.exp(z - z.max(1, keepdims=True))
    return c / c.sum(1, keepdims=True)


def qwk(y, p, w=None):
    return cohen_kappa_score(y, p, weights="quadratic", sample_weight=w)


def fit_cutpoints(s, y):
    """Monotone mapping: lower sharpness -> higher grade. 4 cut-points on log-sharpness,
    initialised at the class-proportion quantiles, refined by coordinate ascent on QWK."""
    x = -np.log(s)                       # higher x = blurrier
    props = np.cumsum(np.bincount(y, minlength=5) / len(y))[:4]
    cuts = np.quantile(x, props)
    grid = np.quantile(x, np.linspace(0.005, 0.995, 200))
    best = qwk(y, np.searchsorted(cuts, x))
    for _ in range(5):
        for k in range(4):
            lo = cuts[k - 1] if k > 0 else -np.inf
            hi = cuts[k + 1] if k < 3 else np.inf
            for c in grid[(grid > lo) & (grid < hi)]:
                trial = cuts.copy(); trial[k] = c
                v = qwk(y, np.searchsorted(trial, x))
                if v > best:
                    best, cuts = v, trial
    return cuts, best


def boot(n, fn, rng):
    vals = [fn(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return np.nanpercentile(vals, [2.5, 97.5])


def main():
    rng = np.random.default_rng(0)
    sh = pd.read_csv(SHARPNESS_CSV)
    sh["image"] = strip(sh["image"])
    q = dict(zip(sh.image, sh.quality_score))

    preds = {}
    for split in ["val", "test"]:
        p = pd.read_csv(ROOT / "referral_agent" / f"new_ensemble_{split}_predictions.csv")
        p["image"] = strip(p["image"])
        p["sharp"] = p.image.map(q)
        P = calibrate(p[[f"ensemble_prob_class{k}" for k in range(5)]].values)
        p["pred"], p["p_ref"], p["exp_grade"] = P.argmax(1), P[:, 2:].sum(1), (P * np.arange(5)).sum(1)
        preds[split] = p.dropna(subset=["sharp"]).reset_index(drop=True)
    test = preds["test"]
    used = pd.read_csv(USED_CSV)
    used["image"] = strip(used["image"])
    lab = "level" if "level" in used.columns else used.columns[1]
    train = used[~used.image.isin(set(preds["val"].image) | set(test.image))].copy()
    train["sharp"] = train.image.map(q)
    train = train.dropna(subset=["sharp"])
    y, s = test.true_label.values, test.sharp.values

    lines = ["# How much of the test QWK comes from sharpness?", ""]
    base = qwk(y, test.pred)
    lo, hi = boot(len(y), lambda i: qwk(y[i], test.pred.values[i]), rng)
    lines += [f"Ensemble on the test split: QWK {base:.3f} [{lo:.3f}, {hi:.3f}] (n = {len(y)})", ""]

    # ---- 1. sharpness-only baseline
    cuts, train_qwk = fit_cutpoints(train.sharp.values, train[lab].astype(int).values)
    sp = np.searchsorted(cuts, -np.log(s))
    sq = qwk(y, sp)
    slo, shi = boot(len(y), lambda i: qwk(y[i], sp[i]), rng)
    auc_s = roc_auc_score(y >= 2, -s)
    lines += ["## 1. Sharpness alone (cut-points fitted on the train split)",
              f"QWK on test: {sq:.3f} [{slo:.3f}, {shi:.3f}]   (train fit: {train_qwk:.3f})",
              f"AUC for referable DR using sharpness alone: {auc_s:.3f}   (ensemble: {roc_auc_score(y >= 2, test.p_ref):.3f})",
              f"Sharpness cut-points (grade 0|1|2|3|4): {np.round(np.exp(-cuts[::-1]), 1).tolist()[::-1]}", ""]

    # ---- 2. remove the cue: inverse-probability weights so grade is independent of sharpness bin
    test["bin"] = pd.qcut(test.sharp, N_BINS, labels=False, duplicates="drop")
    p_grade = test.true_label.value_counts(normalize=True)
    p_grade_bin = test.groupby("bin").true_label.value_counts(normalize=True)
    w = np.array([p_grade[g] / p_grade_bin[(b, g)] for b, g in zip(test.bin, test.true_label)])
    w = w / w.mean()
    wq = qwk(y, test.pred, w)
    wlo, whi = boot(len(y), lambda i: qwk(y[i], test.pred.values[i], w[i]), rng)
    ess = w.sum() ** 2 / (w ** 2).sum()
    wsq = qwk(y, sp, w)
    lines += ["## 2. Ensemble with the sharpness cue removed (re-weighted so grade is independent of sharpness)",
              f"Weighted QWK, ensemble: {wq:.3f} [{wlo:.3f}, {whi:.3f}]   (effective n = {ess:.0f})",
              f"Weighted QWK, sharpness alone: {wsq:.3f}   (residual cue left after re-weighting; 0 = fully removed."
              " Grades that barely overlap in sharpness, e.g. grade 0 vs 3/4, cannot be fully balanced)",
              f"Weighted AUC referable, ensemble: {roc_auc_score(y >= 2, test.p_ref, sample_weight=w):.3f}", "",
              "Within sharpness terciles (unweighted):"]
    test["tercile"] = pd.qcut(test.sharp, 3, labels=["blurriest third", "middle third", "sharpest third"])
    rows = []
    for t_, g in test.groupby("tercile", observed=True):
        yy = g.true_label.values
        rows.append({"tercile": t_, "n": len(g), "sharpness_range": f"{g.sharp.min():.0f}-{g.sharp.max():.0f}",
                     "grades": g.true_label.value_counts().sort_index().to_dict(),
                     "QWK_ensemble": round(qwk(yy, g.pred), 3),
                     "AUC_referable": round(roc_auc_score(yy >= 2, g.p_ref), 3) if 0 < (yy >= 2).sum() < len(yy) else None})
    lines += [pd.DataFrame(rows).to_string(index=False), ""]

    # ---- 3. does the ensemble use sharpness within a true grade?
    rows = []
    for g, d in test.groupby("true_label"):
        r, p = spearmanr(d.sharp, d.exp_grade)
        rows.append({"true_grade": g, "n": len(d), "median_sharpness": round(d.sharp.median(), 1),
                     "spearman(sharpness, predicted grade)": round(r, 3), "p": f"{p:.1g}"})
    lines += ["## 3. Within each true grade: does a blurrier image get a higher predicted grade?",
              pd.DataFrame(rows).to_string(index=False), "",
              "Negative correlations = the ensemble predicts higher grades for blurrier images of the SAME grade.", ""]

    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()