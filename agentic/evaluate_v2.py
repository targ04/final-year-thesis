"""
Builds the thesis tables from the v2 orchestrator runs in agentic/results/:
  v2_none_triage.csv, v2_lesion_only_triage.csv, v2_all_tools_triage.csv,
  v2_llm_triage.csv   (triage-only ablation runs)
  v2_llm.csv          (full run with explanations, optional)

Outputs agentic/results/v2_ablation_table.csv, v2_planner_analysis.csv and,
if the full run exists, v2_explanation_quality.csv. Prints all three.

Usage:  python agentic/evaluate_v2.py
"""

import ast
from pathlib import Path

import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_rules import check_explanation   # noqa: E402

RES = Path(__file__).resolve().parent / "results"
MODES = [("none", "Static triage (v2 gate, no verification)"), ("lesion_only", "+ lesion gate"),
         ("all_tools", "+ lesion gate + all tools (fixed rule)"), ("llm", "+ lesion gate + LLM planner")]


def load(name):
    p = RES / f"v2_{name}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["tools_called", "planner_log", "verifier_failures", "verification"]:
        if c in df:
            df[c] = df[c].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x[:1] in "[{" else [])
    return df


def triage_row(df, label):
    ac = df.triage_decision == "AUTO_CLEAR"
    ref = df.true_label >= 2
    cand = df.initial_decision == "AUTO_CLEAR"
    return {"policy": label, "auto_clear": int(ac.sum()), "missed_referable": int((ac & ref).sum()),
            "miss_rate_%": round(100 * (ac & ref).sum() / max(ac.sum(), 1), 2),
            "referral_recall_%": round(100 * (ref & ~ac).sum() / ref.sum(), 2),
            "workload_%": round(100 * (~ac).mean(), 2),
            "verified_referrals": int((df.triage_decision == "REFER_VERIFIED").sum()),
            "tool_calls": int(df.n_tool_calls.sum()),
            "tool_calls_per_candidate": round(df.n_tool_calls.sum() / max(cand.sum(), 1), 2)}


def main():
    runs = {m: load(f"{m}_triage") for m, _ in MODES}
    rows = [triage_row(runs[m], lab) for m, lab in MODES if runs[m] is not None]
    table = pd.DataFrame(rows)
    table.to_csv(RES / "v2_ablation_table.csv", index=False)
    pd.set_option("display.width", 220)
    print("=== Ablation (test set) ===\n" + table.to_string(index=False))

    fixed, llm = runs.get("all_tools"), runs.get("llm")
    if fixed is not None and llm is not None:
        m = fixed.merge(llm, on="image_id", suffixes=("_fixed", "_llm"))
        m = m[m.initial_decision_fixed == "AUTO_CLEAR"]
        # cases where the fixed rule still had a tool left to run when the LLM stopped
        stopped_early = m.apply(lambda r: len(r.tools_called_llm) < len(r.tools_called_fixed), axis=1)
        same_tools = m.apply(lambda r: r.tools_called_llm == r.tools_called_fixed, axis=1)
        missed_by_stopping = (m.triage_decision_fixed == "REFER_VERIFIED") & (m.triage_decision_llm == "AUTO_CLEAR")
        fallbacks = sum(p.get("fallback", False) for log in llm.planner_log for p in log)
        n_llm_calls = sum(len(log) for log in llm.planner_log)
        actions = pd.Series([p["action"] for log in llm.planner_log for p in log]).value_counts().to_dict()
        pa = pd.DataFrame([{
            "auto_clear_candidates": len(m),
            "planner_llm_calls": n_llm_calls,
            "unusable_outputs_fallback": fallbacks,
            "planner_action_counts": actions,
            "identical_tool_sequence_to_fixed_rule_%": round(100 * same_tools.mean(), 1),
            "stopped_before_fixed_rule_%": round(100 * stopped_early.mean(), 1),
            "tool_calls_saved_vs_fixed": int(m.n_tool_calls_fixed.sum() - m.n_tool_calls_llm.sum()),
            "referrals_lost_by_stopping_early": int(missed_by_stopping.sum()),
            "of_which_truly_referable": int((missed_by_stopping & (m.true_label_fixed >= 2)).sum()),
        }])
        pa.to_csv(RES / "v2_planner_analysis.csv", index=False)
        print("\n=== LLM planner vs fixed rule ===\n" + pa.T.to_string(header=False))

    full = load("llm")
    if full is not None and "explanation_attempts" in full:
        # Re-score every saved explanation offline. v2.0 = rules used during the first full run
        # (two false-positive patterns, see verifier_rules.py); v2.1 = corrected rules.
        def counts(row):
            if isinstance(row.get("lesion_counts"), str):
                return ast.literal_eval(row["lesion_counts"])
            summ = row.get("lesion_summary")
            summ = ast.literal_eval(summ) if isinstance(summ, str) else (summ or {})
            return {k: v["count"] for k, v in summ.items()}   # types absent from the summary had 0 regions

        def reason(f):
            if f.startswith("states "):
                return "lesion count does not match measurement"
            if "sentences (limit" in f:
                return "too many sentences"
            return f

        # Score with the rules the run itself used (recorded per row); older runs without the
        # column were generated under v2.0, so they are also re-scored with v2.1.
        used = set(full["verifier_rules"].dropna()) if "verifier_rules" in full else {"v2.0"}
        versions = ["v2.1"] if used == {"v2.1"} else ["v2.0", "v2.1"]
        rows, reasons = [], {}
        for ver in versions:
            fails = [check_explanation(t, c, version=ver)
                     for t, c in zip(full.why_explanation.fillna(""), [counts(r) for _, r in full.iterrows()])]
            ok = pd.Series([not f for f in fails])
            rows.append({"rules": ver, "images": len(full), "final_explanation_passes_%": round(100 * ok.mean(), 2),
                         "failing": int((~ok).sum())})
            reasons[ver] = pd.Series([reason(x) for f in fails for x in f], dtype=object).value_counts()
        eq = pd.DataFrame(rows)
        eq["mean_generation_attempts_during_run"] = round(full.explanation_attempts.mean(), 2)
        eq["passed_first_attempt_during_run"] = int(((full.explanation_attempts == 1) & full.citation_ok.astype(bool)).sum())
        eq.to_csv(RES / "v2_explanation_quality.csv", index=False)
        print("\n=== Explanation verifier (final saved explanations, re-scored offline) ===\n" + eq.to_string(index=False))
        for ver, r in reasons.items():
            print(f"\nFailure reasons under rules {ver}:\n" + (r.to_string() if len(r) else "none"))


if __name__ == "__main__":
    main()
