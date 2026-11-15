"""
Rewrites the deterministic `timeframe_fact` (next steps) column of saved v2 results with the
decision-aware wording of orchestrator_agentic.build_next_steps. No model or LLM is involved.

Why: the first full v2 run built the sentence from the predicted grade alone, so grade 0/1 cases
that were referred (REFER_LOW_CONFIDENCE, REFER_VERIFIED) said "referral is not required".
No reported metric uses this column; this only removes the contradiction from the saved outputs.

Usage:  python agentic/fix_next_steps.py      (edits agentic/results/v2_llm.json and .csv in place)
"""

import ast
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "agentic" / "results"
V1_SRC = (ROOT / "langgraph" / "orchestrator_graph.py").read_text(encoding="utf-8")

# read the timeframe table and setting from the v1 module source without importing torch etc.
_ns = {}
exec(re.search(r"GRADE_TIMEFRAMES = \{.*?\n\}", V1_SRC, re.S).group(0), _ns)
GRADE_TIMEFRAMES = _ns["GRADE_TIMEFRAMES"]
RESOURCE_SETTING = ast.literal_eval(re.search(r'^RESOURCE_SETTING\s*=\s*(.+?)(\s*#.*)?$', V1_SRC, re.M).group(1))
TIMEFRAME_TABLE = {"high": "Table 2a", "low": "Table 2b"}
GRADE_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR (PDR)"}


def build_next_steps(grade, decision):   # identical to orchestrator_agentic.build_next_steps
    interval, required = GRADE_TIMEFRAMES[RESOURCE_SETTING][grade]
    cite = f"(ICO Guidelines for Diabetic Eye Care, 2017, {TIMEFRAME_TABLE[RESOURCE_SETTING]}, p.8)"
    setting = f"{RESOURCE_SETTING}-resource setting"
    if decision in ("REFER_LOW_CONFIDENCE", "REFER_VERIFIED") and not required:
        why = ("the model is not confident enough that the eye is non-referable"
               if decision == "REFER_LOW_CONFIDENCE"
               else "pre-clearance verification found evidence that contradicts the clear decision")
        return (f"Refer for specialist review, because {why}. For reference, if {GRADE_LABELS[grade]} were "
                f"confirmed, ICO guidelines ({setting}) would recommend re-examination in {interval} without "
                f"referral {cite}; the referral decision takes precedence.")
    return (f"Per ICO guidelines ({setting}): recommended re-examination/referral interval is {interval}; "
            f"referral is {'required' if required else 'not required'} {cite}.")


def main():
    jpath, cpath = RES / "v2_llm.json", RES / "v2_llm.csv"
    rows = json.load(open(jpath))
    changed = 0
    for r in rows:
        new = build_next_steps(int(r["predicted_grade"]), r["triage_decision"])
        changed += new != r.get("timeframe_fact")
        r["timeframe_fact"] = new
    json.dump(rows, open(jpath, "w"), indent=2, default=str)
    df = pd.read_csv(cpath)
    df["timeframe_fact"] = [build_next_steps(int(g), d) for g, d in zip(df.predicted_grade, df.triage_decision)]
    df.to_csv(cpath, index=False)
    still = sum(("referral is not required" in t) and d != "AUTO_CLEAR"
                for t, d in zip(df.timeframe_fact, df.triage_decision))
    print(f"Updated {changed} of {len(rows)} rows. Referred cases still saying 'not required': {still}")


if __name__ == "__main__":
    main()
