"""
Why do held-out EyePACS grade 0 images get P(grade 0) = 0.17 when test-split grade 0 images get 0.83?

Two checks:
  1. Selection confound (CPU): did selecting the sharpest images per grade make image sharpness
     track DR grade in the 15k subset, so the models could learn "blurrier = worse"?
       - median sharpness per grade in the 15k, and in the held-out sample
       - on the original test split (stored predictions): does predicted severity rise as
         sharpness falls within true grade 0?
  2. Pipeline check (GPU, N_CHECK images): re-run the hold-out preprocessing + ensemble on
     test-split images and compare with the stored test probabilities. If they match, the
     hold-out pipeline is identical to the one used for the thesis results.

Usage (repo root):  python agentic/check_shift.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agentic"))
import eval_eyepacs_holdout as ev  # noqa: E402  (reuses its paths and functions)

N_CHECK = 200
CHECK_DIR = ev.EYEPACS_DIR / "shift_check_512"


def strip(s):
    return ev.strip_ext(s)


def main():
    sh = pd.read_csv(ev.SHARPNESS_CSV)
    sh["image"] = strip(sh["image"])
    q = dict(zip(sh.image, sh.quality_score))
    used = pd.read_csv(ev.USED_CSV)
    used["image"] = strip(used["image"])
    used["quality_score"] = used.image.map(q)
    lab = "level" if "level" in used.columns else used.columns[1]

    print("=== 1a. Sharpness (quality_score) by grade")
    t = pd.DataFrame({"15k subset (median)": used.groupby(lab).quality_score.median(),
                      "15k subset (n)": used.groupby(lab).size(),
                      "all EyePACS (median)": sh.groupby("level").quality_score.median()})
    ho = ev.OUT_DIR / "holdout_per_image.csv"
    if ho.exists():
        h = pd.read_csv(ho)
        t["hold-out (median)"] = h.groupby("true_label").quality_score.median()
    print(t.round(1).to_string())

    test = pd.read_csv(ROOT / "referral_agent" / "new_ensemble_test_predictions.csv")
    test["image"] = strip(test["image"])
    test["quality_score"] = test.image.map(q)
    P = ev.calibrate(test[[f"ensemble_prob_class{k}" for k in range(5)]].values)
    test["p_not0"] = 1 - P[:, 0]
    g0 = test[test.true_label == 0].dropna(subset=["quality_score"])
    rho, p = spearmanr(g0.quality_score, g0.p_not0)
    print("\n=== 1b. Original test split, true grade 0 only")
    print(f"Spearman(sharpness, P(grade > 0)) = {rho:.3f} (p = {p:.2g}, n = {len(g0)})")
    g0 = g0.assign(sharpness_quintile=pd.qcut(g0.quality_score, 5, labels=False))
    print(g0.groupby("sharpness_quintile").agg(median_sharpness=("quality_score", "median"),
                                               mean_P_grade_gt0=("p_not0", "mean")).round(3).to_string())

    print(f"\n=== 2. Pipeline check on {N_CHECK} test-split images")
    sample = test.sample(N_CHECK, random_state=0).reset_index(drop=True)
    sel = sample[["image"]].assign(patient_id=sample.image.str.split("_").str[0],
                                   true_label=sample.true_label, quality_score=sample.quality_score)
    ev.PREP_DIR = CHECK_DIR
    ev.PRED_CACHE = CHECK_DIR / "predictions.csv"
    ev.CROP_LOG = CHECK_DIR / "crop_log.csv"
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    ev.preprocess(sel)
    ev.predict(sel)
    new = pd.read_csv(ev.PRED_CACHE).drop_duplicates("image").set_index("image")
    cols = [f"ensemble_prob_class{k}" for k in range(5)]
    old = sample.set_index("image")[cols]
    d = (new.loc[old.index, cols] - old).abs()
    print(f"max |difference| in ensemble probabilities: {d.values.max():.4f}; mean: {d.values.mean():.5f}")
    print("argmax agreement:", f"{(new.loc[old.index, cols].values.argmax(1) == old.values.argmax(1)).mean():.3f}")
    print("-> near 0 means the hold-out pipeline reproduces the thesis pipeline exactly.")


if __name__ == "__main__":
    main()