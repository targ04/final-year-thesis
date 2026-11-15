"""
Offline ablation of the verify-before-clear policy (CPU only, no GPU / LLM).

Replays triage from saved ensemble predictions + lesion features and compares:
  P0  static           : severity gate + confidence gate (midsem policy)
  P1  tighter-conf     : static, but auto-clear threshold raised until it removes
                         the SAME number of clears as P3 (the honest baseline: does
                         the lesion tool beat simply being more conservative?)
  P2  +fellow-eye      : auto-clear candidate referred if fellow eye predicted >= 2
  P3  +lesion          : auto-clear candidate referred if lesion burden is discordant
  P4  +lesion+fellow   : both checks (fixed rule: "always run every tool")

Overrides are ONE-WAY: a check can turn AUTO_CLEAR into a referral, never the
reverse. So referral recall can only go up and workload can only go up.

Protocol: lesion feature + threshold are chosen on VAL only (criterion below),
then applied unchanged to TEST once.

Auto-clear gate: the one selected by binary_gate_analysis.py (P(grade0+grade1) >= t,
see gates.py) - run that script first. P0-v1 (midsem gate) is reported for reference.

Usage:  python agentic/binary_gate_analysis.py && python agentic/offline_ablation.py
"""

import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates import load_selected, V1_THRESHOLD   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRED = {"val": ROOT / "referral_agent" / "new_ensemble_val_predictions.csv",
        "test": ROOT / "referral_agent" / "new_ensemble_test_predictions.csv"}
LES = {"val": ROOT / "idrid" / "lesion_analysis_val" / "lesion_features.csv",
       "test": ROOT / "idrid" / "lesion_analysis_test" / "lesion_features.csv"}
TEMPERATURE = json.load(open(ROOT / "referral_agent" / "temperature_scaling.json"))["temperature"]
OUT_DIR = ROOT / "agentic" / "results"

GATE = load_selected()                 # {"rule": "nonref_sum", "threshold": 0.94, ...}
CONF_THRESHOLD = GATE["threshold"]
MANDATORY = 2                       # grade >= 2 -> REFER_SEVERITY
CANDIDATE_FEATURES = ["MA_area_frac", "MA_count", "EX_area_frac", "HE_area_frac"]
PERCENTILES = [80, 85, 90, 95]      # of the feature among val auto-clears with true grade < 2
MAX_EXTRA_REFERRAL_FRAC = 0.10      # selection constraint: at most 10% of val clears overridden


def load(split):
    p = pd.read_csv(PRED[split])
    probs = p[[f"ensemble_prob_class{i}" for i in range(5)]].values
    z = np.log(probs + 1e-12) / TEMPERATURE
    cal = np.exp(z - z.max(1, keepdims=True))
    cal /= cal.sum(1, keepdims=True)
    p["pred"] = cal.argmax(1)
    p["conf_v1"] = cal.max(1)                                        # top single-grade probability
    p["conf"] = cal.max(1) if GATE["rule"] == "max_class" else cal[:, :2].sum(1)   # gate score
    les = pd.read_csv(LES[split]).drop(columns=["patient_id", "true_label"])
    df = p.merge(les, on="image", how="left", validate="one_to_one")
    assert df["MA_count"].notna().all(), f"missing lesion features in {split}"

    # fellow eye = the patient's other image in the same split (patient-level split)
    side = df["image"].str.extract(r"_(left|right)$")[0]
    other = df["image"].str.replace(r"_(left|right)$", "", regex=True) + "_" + side.map({"left": "right", "right": "left"})
    pred_by_img = dict(zip(df["image"], df["pred"]))
    df["fellow_pred"] = other.map(pred_by_img)
    return df


def triage(df, conf_thr=CONF_THRESHOLD, lesion_rule=None, fellow=False, score="conf"):
    sev = df["pred"] >= MANDATORY
    clear = ~sev & (df[score] >= conf_thr)
    override = pd.Series(False, index=df.index)
    if lesion_rule is not None:
        feat, thr = lesion_rule
        override |= clear & (df[feat] > thr)
    if fellow:
        override |= clear & (df["fellow_pred"] >= MANDATORY)
    return np.where(sev, "REFER_SEVERITY", np.where(clear & ~override, "AUTO_CLEAR",
                    np.where(override, "REFER_VERIFIED", "REFER_LOW_CONFIDENCE")))


def metrics(df, decision, name):
    d = pd.Series(decision, index=df.index)
    y = df["true_label"]
    ac = d == "AUTO_CLEAR"
    referable = y >= MANDATORY
    miss = int((ac & referable).sum())
    return {"policy": name, "auto_clear": int(ac.sum()), "missed_referable": miss,
            "safety_net_miss_rate_%": round(100 * miss / max(ac.sum(), 1), 2),
            "referral_recall_%": round(100 * (referable & ~ac).sum() / referable.sum(), 2),
            "referral_workload_%": round(100 * (~ac).mean(), 2),
            "verifier_overrides": int((d == "REFER_VERIFIED").sum())}


def select_lesion_rule(val):
    """On VAL: candidate rules = (feature, percentile of feature among correctly
    clearable auto-clears). Keep rules overriding <= MAX_EXTRA_REFERRAL_FRAC of
    clears; pick the one catching the most misses, ties -> fewest overrides."""
    base = pd.Series(triage(val), index=val.index)
    ac = val[base == "AUTO_CLEAR"]
    clean = ac[ac["true_label"] < MANDATORY]
    rows = []
    for f in CANDIDATE_FEATURES:
        for q in PERCENTILES:
            thr = float(clean[f].quantile(q / 100))
            flagged = ac[f] > thr
            rows.append({"feature": f, "percentile": q, "threshold": thr,
                         "misses_caught": int((flagged & (ac["true_label"] >= MANDATORY)).sum()),
                         "misses_total": int((ac["true_label"] >= MANDATORY).sum()),
                         "overrides": int(flagged.sum()),
                         "override_frac": round(flagged.mean(), 4)})
    sweep = pd.DataFrame(rows)
    ok = sweep[sweep["override_frac"] <= MAX_EXTRA_REFERRAL_FRAC]
    best = ok.sort_values(["misses_caught", "overrides"], ascending=[False, True]).iloc[0]
    return (best["feature"], best["threshold"]), sweep, best


def matched_conf_threshold(df, n_clears_target):
    """Smallest confidence threshold giving <= n_clears_target auto-clears."""
    cand = df[df["pred"] < MANDATORY]["conf"].sort_values(ascending=False).values
    if n_clears_target <= 0:
        return 1.0
    return float(np.nextafter(cand[n_clears_target - 1], 1.0)) if n_clears_target < len(cand) else CONF_THRESHOLD


def run_split(df, rule, label):
    rows = [metrics(df, triage(df, conf_thr=V1_THRESHOLD, score="conf_v1"), "P0-v1 static, midsem gate (top-grade>=0.85)"),
            metrics(df, triage(df), f"P0 static, v2 gate (P(0+1)>={CONF_THRESHOLD})")]
    p3 = triage(df, lesion_rule=rule)
    n_p3 = int((p3 == "AUTO_CLEAR").sum())
    thr = matched_conf_threshold(df, n_p3)
    rows.append(metrics(df, triage(df, conf_thr=thr), f"P1 tighter-conf (thr={thr:.3f})"))
    rows.append(metrics(df, triage(df, fellow=True), "P2 +fellow-eye"))
    rows.append(metrics(df, p3, f"P3 +lesion ({rule[0]}>{rule[1]:.4g})"))
    p4 = triage(df, lesion_rule=rule, fellow=True)
    thr4 = matched_conf_threshold(df, int((p4 == "AUTO_CLEAR").sum()))
    rows.append(metrics(df, triage(df, conf_thr=thr4), f"P1b tighter-conf matched to P4 (thr={thr4:.3f})"))
    rows.append(metrics(df, p4, "P4 +lesion+fellow"))
    out = pd.DataFrame(rows)
    out.insert(0, "split", label)
    return out


def tradeoff_curve(val, test, out_png):
    """Safety vs workload: missed referable cases as a function of how many cases are
    auto-cleared. Confidence-only curve = sweep the auto-clear threshold. Verification
    curve = keep the gate threshold and sweep the lesion percentile (thresholds taken from VAL),
    with the fellow-eye check on. A curve lower-left is strictly better."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = pd.Series(triage(val), index=val.index)
    ac = val[base == "AUTO_CLEAR"]
    clean = ac[ac["true_label"] < MANDATORY]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    rows = []
    for ax, (name, df) in zip(axes, [("val", val), ("test", test)]):
        conf_pts = []
        for thr in np.linspace(CONF_THRESHOLD, 0.9995, 60):
            m = metrics(df, triage(df, conf_thr=thr), "")
            conf_pts.append((m["auto_clear"], m["missed_referable"]))
        ver_pts = []
        for q in range(50, 100, 2):
            lthr = float(clean["MA_area_frac"].quantile(q / 100))
            m = metrics(df, triage(df, lesion_rule=("MA_area_frac", lthr), fellow=True), "")
            ver_pts.append((m["auto_clear"], m["missed_referable"]))
            rows.append({"split": name, "lesion_percentile": q, **m})
        for pts, lab in [(conf_pts, "confidence threshold only"), (ver_pts, "verify-before-clear (lesion + fellow eye)")]:
            x, y = zip(*sorted(pts))
            ax.step(x, y, where="post", label=lab)
        ax.set_title(f"{name.upper()} split")
        ax.set_xlabel("Auto-cleared images (fewer = more specialist workload)")
        ax.set_ylabel("Missed referable (true grade >= 2) among auto-cleared")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_png.with_suffix(".csv"), index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val, test = load("val"), load("test")

    rule, sweep, best = select_lesion_rule(val)
    sweep.to_csv(OUT_DIR / "lesion_rule_sweep_val.csv", index=False)
    print("Selected on VAL:", best.to_dict())
    with open(OUT_DIR / "selected_lesion_rule.json", "w") as f:
        json.dump({"feature": rule[0], "threshold": rule[1], "percentile": int(best["percentile"]),
                   "selected_on": "val"}, f, indent=2)

    res = pd.concat([run_split(val, rule, "val"), run_split(test, rule, "test")], ignore_index=True)
    res.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))

    tradeoff_curve(val, test, OUT_DIR / "safety_workload_tradeoff.png")

    # missed cases on test under the full fixed-rule policy, for error analysis
    d = triage(test, lesion_rule=rule, fellow=True)
    miss = test[(d == "AUTO_CLEAR") & (test["true_label"] >= MANDATORY)]
    miss[["image", "true_label", "pred", "conf", rule[0], "fellow_pred"]].to_csv(
        OUT_DIR / "remaining_misses_test.csv", index=False)


if __name__ == "__main__":
    main()
