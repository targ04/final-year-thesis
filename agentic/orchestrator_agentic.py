"""
Agentic orchestrator (v2) - goal-directed verification of auto-clear decisions.

--- Why v2 exists (finding from the v1 run, langgraph/orchestrator_graph.py) ---
v1 escalated only close-margin cases. A top-2 margin < 0.15 caps the top
probability near 0.575, far below the 0.85 auto-clear threshold, so escalation
could never change refer-vs-clear: all 52 triage changes it made on test were
REFER_SEVERITY <-> REFER_LOW_CONFIDENCE (both referrals), and the LLM approved
escalation in 146/146 consultations. v2 points agency at the one decision with
clinical consequence: sending a patient home without human review (AUTO_CLEAR).

--- Agent goal and safety constraint ---
Goal: auto-clear as many patients as is safe. Constraint (enforced in code, not
by the LLM): verification can only turn AUTO_CLEAR into a referral, never the
reverse, so the agent can never make the system less safe than v1's policy.

--- Flow (LangGraph) ---
  perceive  : ensemble grade probabilities (with per-patient memory cache)
              + lesion measurement (IDRiD U-Net, agentic/lesion_tool.py)
  triage    : temperature scaling + severity gate + auto-clear gate. v2 gate: clear grade 0/1
              predictions only if P(grade 0 + grade 1) >= t (t selected on VAL, see gates.py /
              binary_gate_analysis.py) - the decision is binary (refer at >= moderate NPDR, as in
              FDA-cleared systems), so confidence is measured in that binary outcome, not in one grade
  if AUTO_CLEAR candidate:
    lesion_check : deterministic gate (rule chosen on VAL) -> discordant => REFER_VERIFIED
    planner      : LLM chooses the next verification tool (fellow eye / TTA) or stops,
                   seeing the evidence gathered so far and each tool's cost
    tool         : runs the chosen tool; discordance is judged IN CODE => REFER_VERIFIED
    (loop planner <-> tool, at most one call per tool)
  structuring -> saliency -> explanation (RAG + measured lesion evidence)
  -> verifier (rule checks; failures fed back to the explainer, bounded retries)

--- Modes (for the ablation; set PLANNER_MODE) ---
  "none"        static v1 policy, no verification            (= midsem triage)
  "lesion_only" deterministic lesion gate only
  "all_tools"   fixed rule: lesion gate + every available tool
  "llm"         lesion gate + LLM planner choosing tools
SKIP_EXPLANATION=True runs triage only (minutes, no explanation LLM calls) -
use it for the four ablation runs, then one full "llm" run with explanations.

Run from the repo root:   python -m agentic.orchestrator_agentic
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ollama                                            # noqa: E402
from langgraph.graph import StateGraph, END              # noqa: E402
import langgraph.orchestrator_graph as v1                # noqa: E402  (reused models, saliency, retrieval)
from agentic.lesion_tool import LesionTool, summarise as summarise_lesions, overlay as lesion_overlay  # noqa: E402
from agentic.gates import load_selected, decide, clear_score, p_nonref  # noqa: E402
from agentic.verifier_rules import check_explanation as check_rules, MAX_SENTENCES, RULES_VERSION  # noqa: E402

# ---------------- CONFIG ----------------
PLANNER_MODE = "llm"          # "none" | "lesion_only" | "all_tools" | "llm"
SKIP_EXPLANATION = False      # True for fast triage-only ablation runs

OUT_DIR = ROOT / "agentic" / "results"
SAVE_EVERY = 25

GRADE_LABELS = v1.GRADE_LABELS
MANDATORY = 2
GATE = load_selected()                      # auto-clear gate chosen on VAL (binary_gate_analysis.py)
CONF_THRESHOLD = GATE["threshold"]          # P(grade 0 + grade 1) >= this (v1 used top-grade prob >= 0.85)

TOOLS = {   # verification tools the planner can choose from (lesion gate always runs first)
    "fellow_eye": {"cost": "1 forward pass of the ensemble on the patient's other eye",
                   "checks": "whether the other eye shows referable (moderate or worse) DR"},
    "tta": {"cost": "4 forward passes (flips + rotation) of the ensemble on this image",
            "checks": "whether the clear decision is stable under the augmentations the models were trained with"},
}

MAX_EXPLANATION_ATTEMPTS = 3

# Official grade definitions - International Classification of DR, the SAME scale as the EyePACS
# labels. Verbatim from ICO Guidelines for Diabetic Eye Care (2017), Table 1 (PDF page 7, printed
# page 2). Given to the LLM as a fixed fact (like the referral timeframes) because retrieval alone
# tended to return AAO Appendix 5 (PDF pp. 60-65), which is the ETDRS scale - a different scale.
ICO_TABLE1_CITATION = "(ICO Guidelines for Diabetic Eye Care, 2017, Table 1, p.7)"
ICO_TABLE1 = {
    0: "No apparent DR: no abnormalities.",
    1: "Mild nonproliferative DR: microaneurysms only.",
    2: ("Moderate nonproliferative DR: microaneurysms and other signs (e.g., dot and blot hemorrhages, hard "
        "exudates, cotton wool spots), but less than severe nonproliferative DR."),
    3: ("Severe nonproliferative DR: moderate nonproliferative DR with any of the following: intraretinal "
        "hemorrhages (>=20 in each quadrant); definite venous beading (in 2 quadrants); intraretinal "
        "microvascular abnormalities (in 1 quadrant); and no signs of proliferative retinopathy."),
    4: ("Proliferative DR: severe nonproliferative DR and 1 or more of the following: neovascularization; "
        "vitreous/preretinal hemorrhage."),
}
# Referral timeframes: ICO 2017 Table 2a (high resource) / 2b (low-intermediate), PDF page 8 - verified
# identical to v1.GRADE_TIMEFRAMES; re-stated here only to name the table in the citation.
TIMEFRAME_TABLE = {"high": "Table 2a", "low": "Table 2b"}
EXPLAIN_LESIONS = ["MA", "HE", "EX"]   # SE carries no grade signal on EyePACS (AUC ~0.59) - not shown to the LLM
# -----------------------------------------


class State(TypedDict, total=False):
    image_id: str
    image_path: str
    patient_id: Any
    true_label: Optional[int]
    ensemble_probs: np.ndarray
    memory_hit: bool
    lesion_features: Dict[str, Any]
    lesion_summary: Dict[str, Any]
    calibrated_probs: np.ndarray
    predicted_grade: int
    calibrated_confidence: float
    p_nonreferable: float
    class_probabilities: Dict[str, float]
    initial_decision: str
    triage_decision: str
    verification: List[Dict[str, Any]]     # ordered record of every check/tool and its outcome
    next_action: str
    planner_log: List[Dict[str, Any]]
    structured_record: Dict[str, Any]
    saliency: Dict[str, Any]
    why_explanation: str
    timeframe_fact: str
    verifier_failures: List[str]
    citation_ok: bool
    explanation_attempts: int


# ---------------- helpers ----------------

PRED_CACHE: Dict[str, np.ndarray] = {}   # memory: ensemble probs per image, shared across the patient's eyes


def calibrate(probs):
    z = np.log(probs + 1e-12) / v1.TEMPERATURE
    e = np.exp(z - z.max())
    return e / e.sum()


def ensemble_probs_for(image_id, path):
    """Returns (probs, memory_hit)."""
    if image_id in PRED_CACHE:
        return PRED_CACHE[image_id], True
    probs = v1.ensemble_predict(Image.open(path).convert("RGB"))
    PRED_CACHE[image_id] = probs
    return probs, False


def fellow_of(state):
    iid = state["image_id"]
    side = "left" if iid.endswith("_left") else ("right" if iid.endswith("_right") else None)
    other = {"left": "right", "right": "left"}.get(side)
    if other is None:
        return None, None
    name = v1.FELLOW_LOOKUP.get(state["patient_id"], {}).get(other)
    if not name:
        return None, None
    path = os.path.join(v1.IMAGE_FOLDER, name + v1.IMAGE_EXT)
    return (name, path) if os.path.exists(path) else (None, None)


def available_tools(state):
    used = {v["tool"] for v in state.get("verification", [])}
    out = []
    for t in TOOLS:
        if t in used:
            continue
        if t == "fellow_eye" and fellow_of(state)[0] is None:
            continue
        out.append(t)
    return out


# ---------------- nodes ----------------

def perceive_node(state):
    probs, hit = ensemble_probs_for(state["image_id"], state["image_path"])
    feats, _masks, _img = LESION_TOOL.measure_path(state["image_path"])
    return {"ensemble_probs": probs, "memory_hit": hit, "lesion_features": feats,
            "lesion_summary": summarise_lesions(feats), "verification": [], "planner_log": [],
            "explanation_attempts": 0}


def triage_node(state):
    cal = calibrate(state["ensemble_probs"])
    g, c = int(np.argmax(cal)), float(cal.max())
    d = decide(cal, GATE["rule"], CONF_THRESHOLD)
    return {"calibrated_probs": cal, "predicted_grade": g, "calibrated_confidence": c,
            "p_nonreferable": float(p_nonref(cal)),
            "class_probabilities": {GRADE_LABELS[i]: round(float(cal[i]), 4) for i in range(5)},
            "initial_decision": d, "triage_decision": d}


def route_after_triage(state):
    if state["triage_decision"] != "AUTO_CLEAR" or PLANNER_MODE == "none":
        return "done"
    return "verify"


def lesion_check_node(state):
    f = state["lesion_features"]
    feat = LESION_TOOL.rule["feature"]
    disc = bool(LESION_TOOL.discordant(f))
    rec = {"tool": "lesion_check", "discordant": disc,
           "detail": f"{feat}={f[feat]:.4f} (threshold {LESION_TOOL.rule['threshold']:.4f}, set on validation)"}
    upd = {"verification": state["verification"] + [rec]}
    if disc:
        upd["triage_decision"] = "REFER_VERIFIED"
    return upd


def route_after_check(state):
    if state["triage_decision"] == "REFER_VERIFIED":
        return "done"
    if PLANNER_MODE == "lesion_only" or not available_tools(state):
        return "done"
    return "plan"


def planner_node(state):
    tools = available_tools(state)
    if PLANNER_MODE == "all_tools":
        return {"next_action": tools[0],
                "planner_log": state["planner_log"] + [{"action": tools[0], "reasoning": "fixed rule: run every tool"}]}

    evidence = "\n".join(f"- {v['tool']}: {'DISCORDANT' if v['discordant'] else 'concordant'} ({v['detail']})"
                         for v in state["verification"])
    tool_lines = "\n".join(f"- {t}: checks {TOOLS[t]['checks']}; cost: {TOOLS[t]['cost']}" for t in tools)
    les = state["lesion_summary"]
    les_txt = ", ".join(f"{v['count']} {v['name']} region(s)" for k, v in les.items() if k in EXPLAIN_LESIONS) or "none"
    prompt = (
        "You are the verification planner of a diabetic retinopathy screening system. The classifier wants to "
        "AUTO-CLEAR this patient (no specialist review). A missed referable case is far more costly than an "
        "unnecessary referral, but every tool call costs compute and time.\n\n"
        f"Classifier: {GRADE_LABELS[state['predicted_grade']]} (grade {state['predicted_grade']}); calibrated "
        f"probability of NON-referable DR (grade 0 or 1) {state['p_nonreferable']:.3f} "
        f"(auto-clear threshold {CONF_THRESHOLD}).\n"
        f"Lesion model estimates: {les_txt}.\n"
        f"Evidence gathered so far:\n{evidence}\n\n"
        f"Tools still available:\n{tool_lines}\n\n"
        "Decide the single next step: call one tool, or 'stop' if the evidence already makes the clear decision "
        "trustworthy enough. Any discordant tool result automatically refers the patient.\n"
        f'Respond with ONLY JSON: {{"action": {" | ".join(json.dumps(t) for t in tools)} | "stop", '
        '"reasoning": "one short sentence"}}'
    )
    try:
        r = ollama.chat(model=v1.OLLAMA_MODEL_NAME, messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0.1, "num_predict": 120})
        raw = r["message"]["content"]
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        dec = json.loads(m.group(0) if m else raw)
        action = dec.get("action", "")
        reasoning = dec.get("reasoning", "")
        if action not in tools + ["stop"]:
            raise ValueError(f"invalid action {action!r}")
        fallback = False
    except Exception as e:
        # Safe default when the LLM output is unusable: gather MORE evidence, not less.
        action, reasoning, fallback = tools[0], f"planner output unusable ({e}); defaulting to next tool", True
    return {"next_action": action,
            "planner_log": state["planner_log"] + [{"action": action, "reasoning": reasoning, "fallback": fallback}]}


def route_after_planner(state):
    return "stop" if state["next_action"] == "stop" else "call"


def tool_node(state):
    t = state["next_action"]
    if t == "fellow_eye":
        name, path = fellow_of(state)
        probs, hit = ensemble_probs_for(name, path)
        cal = calibrate(probs)
        g = int(np.argmax(cal))
        rec = {"tool": "fellow_eye", "discordant": g >= MANDATORY, "memory_hit": hit,
               "detail": f"fellow eye {name}: {GRADE_LABELS[g]} at {cal.max():.3f}"
                         + (" (from patient memory)" if hit else "")}
    elif t == "tta":
        cal = calibrate(v1.tta_predict(Image.open(state["image_path"]).convert("RGB")))
        g, sc = int(np.argmax(cal)), float(clear_score(cal, GATE["rule"]))
        rec = {"tool": "tta", "discordant": g >= MANDATORY or sc < CONF_THRESHOLD,
               "detail": f"TTA: {GRADE_LABELS[g]}, P(non-referable) {float(p_nonref(cal)):.3f}"}
    else:
        raise ValueError(t)
    upd = {"verification": state["verification"] + [rec]}
    if rec["discordant"]:
        upd["triage_decision"] = "REFER_VERIFIED"
    return upd


def structuring_node(state):
    label = GRADE_LABELS[state["predicted_grade"]]
    d = state["triage_decision"]
    pn = f"{state['p_nonreferable'] * 100:.1f}%"
    thr = f"{CONF_THRESHOLD * 100:.0f}%"
    if d == "REFER_VERIFIED":
        bad = [v["detail"] for v in state["verification"] if v["discordant"]]
        reason = (f"The classifier predicted {label} with a {pn} calibrated probability of non-referable DR, "
                  f"which would normally be auto-cleared, but pre-clearance verification found discordant "
                  f"evidence ({'; '.join(bad)}). Referred for specialist review.")
    elif d == "REFER_SEVERITY":
        reason = v1.build_referral_reason(d, label, state["calibrated_confidence"], CONF_THRESHOLD)
    elif d == "AUTO_CLEAR":
        reason = (f"No referral required. Predicted {label}; calibrated probability of non-referable DR "
                  f"(grade 0 or 1) is {pn}, meeting the auto-clear threshold of {thr}, and pre-clearance "
                  f"verification found no discordant evidence.")
    else:
        reason = (f"Referred for specialist review: predicted {label}, but the calibrated probability of "
                  f"non-referable DR (grade 0 or 1) is {pn}, below the auto-clear threshold of {thr}.")
    rec = {"diagnosis": {"predicted_grade": state["predicted_grade"], "grade_label": label,
                         "calibrated_confidence": round(state["calibrated_confidence"], 4),
                         "p_nonreferable": round(state["p_nonreferable"], 4),
                         "class_probabilities": state["class_probabilities"]},
           "lesion_estimates": state["lesion_summary"],
           "referral": {"initial_decision": state["initial_decision"], "decision": d, "reason": reason,
                        "verification": state["verification"]}}
    return {"structured_record": rec}


def route_after_structuring(state):
    return "end" if SKIP_EXPLANATION else "explain"


def saliency_node(state):
    return v1.saliency_node(state)


# ---------------- explanation + verifier ----------------

def lesion_evidence_text(summary):
    parts = []
    for l in EXPLAIN_LESIONS:
        if l in summary:
            s = summary[l]
            parts.append(f"{s['count']} {s['name']} region(s), most in the {s['most_regions_in']} quadrant")
    if not parts:
        return "The lesion segmentation model estimated no microaneurysm, haemorrhage or hard exudate regions."
    return "Lesion segmentation model estimates: " + "; ".join(parts) + "."


def build_prompts(state, context_block, feedback):
    label = GRADE_LABELS[state["predicted_grade"]]
    sal = state.get("saliency")
    if sal:
        agree = ("Both backbones concentrated attention in the " + sal["resnet_quadrant"] + " region."
                 if sal["attention_agreement"] else
                 f"ResNet-50 attended mostly to the {sal['resnet_quadrant']} region and ViT-B/16 to the "
                 f"{sal['vit_quadrant']} region (they do not agree).")
    else:
        agree = "No attention data."
    alt = v1.build_alt_grade_note(state["class_probabilities"], state["predicted_grade"])
    alt_grade = next((k for k, v in GRADE_LABELS.items() if alt and alt.startswith(v + " (")), None)
    alt_txt = (f"The next most probable grade was {alt}. Its official definition: {ICO_TABLE1[alt_grade]}"
               if alt and alt_grade is not None else "No other grade had a meaningfully high probability.")
    definition = f"{ICO_TABLE1[state['predicted_grade']]} {ICO_TABLE1_CITATION}"

    system = (
        "You explain to an ophthalmologist why an automated diabetic retinopathy grade is plausible. Use ONLY "
        "this evidence:\n"
        "1. LESION ESTIMATES from a separate segmentation model (trained on another dataset). Quote the counts "
        "exactly as given and call them estimates; never invent a number or a lesion type not listed.\n"
        "2. ATTENTION LOCATION of the two classifier backbones. Report location and agreement only; never say a "
        "backbone detected, identified, saw or found anything, and never interpret what the pattern means.\n"
        "3. OFFICIAL GRADE DEFINITION (International Classification of DR - the grading scale used): paraphrase "
        f"it briefly and cite it exactly as {ICO_TABLE1_CITATION}. This is the ONLY grade definition you may "
        "use. The retrieved excerpts may describe a DIFFERENT scale (ETDRS: 'standard photograph', 'fields 4 "
        "through 7') - never use or cite ETDRS criteria. Use retrieved excerpts only for brief extra context "
        "that agrees with the official definition, cited as (SOURCE, p.N).\n"
        "4. ALTERNATE GRADE, if given, and what its official definition would imply.\n\n"
        "Rules: start directly with the evidence (do NOT begin with or repeat the predicted grade, confidence or "
        "referral timeframe); do not use the words 'indicating', 'indicates' or 'suggests'; "
        f"at most {MAX_SENTENCES} sentences, one paragraph, no lists."
    )
    user = (f"Predicted grade (known, do not restate): {label}\n\n"
            f"Official definition of this grade: {definition}\n\n"
            f"{lesion_evidence_text(state['lesion_summary'])}\n\nAttention: {agree}\n\nAlternate grade: {alt_txt}\n\n"
            f"Retrieved guideline excerpts:\n{context_block}\n\nWrite the explanation now.")
    if feedback:
        user += "\n\nYour previous attempt broke these rules - fix them:\n- " + "\n- ".join(feedback)
    return system, user


def build_next_steps(grade, decision):
    """Next-steps sentence from the FINAL decision, citing ICO Table 2a/2b. The ICO interval is a
    function of grade only; when a grade 0/1 case is nevertheless referred (low confidence or
    discordant verification), the referral takes precedence and the sentence must say so."""
    interval, required = v1.GRADE_TIMEFRAMES[v1.RESOURCE_SETTING][grade]
    cite = f"(ICO Guidelines for Diabetic Eye Care, 2017, {TIMEFRAME_TABLE[v1.RESOURCE_SETTING]}, p.8)"
    setting = f"{v1.RESOURCE_SETTING}-resource setting"
    if decision in ("REFER_LOW_CONFIDENCE", "REFER_VERIFIED") and not required:
        why = ("the model is not confident enough that the eye is non-referable"
               if decision == "REFER_LOW_CONFIDENCE"
               else "pre-clearance verification found evidence that contradicts the clear decision")
        return (f"Refer for specialist review, because {why}. For reference, if {GRADE_LABELS[grade]} were "
                f"confirmed, ICO guidelines ({setting}) would recommend re-examination in {interval} without "
                f"referral {cite}; the referral decision takes precedence.")
    return (f"Per ICO guidelines ({setting}): recommended re-examination/referral interval is {interval}; "
            f"referral is {'required' if required else 'not required'} {cite}.")


def explanation_node(state):
    label = GRADE_LABELS[state["predicted_grade"]]
    query = f"{label} diabetic retinopathy clinical features and diagnostic criteria"
    docs = v1.VECTOR_STORE.similarity_search(v1.QUERY_INSTRUCTION + query, k=v1.TOP_K)
    system, user = build_prompts(state, v1.format_context(docs), state.get("verifier_failures"))
    r = ollama.chat(model=v1.OLLAMA_MODEL_NAME,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    options={"temperature": 0.2, "num_predict": 260})
    return {"why_explanation": r["message"]["content"].strip(),
            "timeframe_fact": build_next_steps(state["predicted_grade"], state["triage_decision"]),
            "explanation_attempts": state.get("explanation_attempts", 0) + 1}


def check_explanation(text, state):
    """Rule checks live in agentic/verifier_rules.py (shared with evaluate_v2.py's offline re-scoring)."""
    feats = state["lesion_features"]
    return check_rules(text, {l: int(feats[f"{l}_count"]) for l in ["MA", "HE", "EX", "SE"]})


def verifier_node(state):
    fails = check_explanation(state["why_explanation"], state)
    return {"verifier_failures": fails, "citation_ok": not fails}


def route_after_verifier(state):
    if state["citation_ok"] or state["explanation_attempts"] >= MAX_EXPLANATION_ATTEMPTS:
        return "accept"
    return "retry"


# ---------------- graph ----------------

def build_graph():
    g = StateGraph(State)
    for name, fn in [("perceive", perceive_node), ("triage", triage_node), ("lesion_check", lesion_check_node),
                     ("planner", planner_node), ("tool", tool_node), ("structuring", structuring_node),
                     ("saliency", saliency_node), ("explanation", explanation_node), ("verifier", verifier_node)]:
        g.add_node(name, fn)
    g.set_entry_point("perceive")
    g.add_edge("perceive", "triage")
    g.add_conditional_edges("triage", route_after_triage, {"verify": "lesion_check", "done": "structuring"})
    g.add_conditional_edges("lesion_check", route_after_check, {"plan": "planner", "done": "structuring"})
    g.add_conditional_edges("planner", route_after_planner, {"call": "tool", "stop": "structuring"})
    g.add_conditional_edges("tool", route_after_check, {"plan": "planner", "done": "structuring"})
    g.add_conditional_edges("structuring", route_after_structuring, {"explain": "saliency", "end": END})
    g.add_edge("saliency", "explanation")
    g.add_edge("explanation", "verifier")
    g.add_conditional_edges("verifier", route_after_verifier, {"retry": "explanation", "accept": END})
    return g.compile()


def setup():
    """Loads v1's models / temperature / vector store / fellow-eye lookup, plus the lesion tool."""
    global LESION_TOOL
    manifest = v1.setup()
    LESION_TOOL = LesionTool(v1.DEVICE)
    PRED_CACHE.clear()
    print(f"Auto-clear gate (from VAL): {GATE} | Lesion rule (from VAL): {LESION_TOOL.rule} | mode={PLANNER_MODE} | skip_explanation={SKIP_EXPLANATION}")
    return manifest


def serialize(state):
    ver = state.get("verification", [])
    return {
        "image_id": state["image_id"], "patient_id": state.get("patient_id"), "true_label": state.get("true_label"),
        "predicted_grade": state["predicted_grade"], "calibrated_confidence": state["calibrated_confidence"],
        "p_nonreferable": state.get("p_nonreferable"),
        "initial_decision": state["initial_decision"], "triage_decision": state["triage_decision"],
        "tools_called": [v["tool"] for v in ver if v["tool"] != "lesion_check"],
        "n_tool_calls": sum(v["tool"] != "lesion_check" for v in ver),
        "verification": ver, "planner_log": state.get("planner_log", []),
        "planner_llm_calls": sum(not p.get("reasoning", "").startswith("fixed rule") for p in state.get("planner_log", [])),
        "memory_hit": state.get("memory_hit"),
        "lesion_summary": state.get("lesion_summary"),
        "saliency": state.get("saliency"),
        "why_explanation": state.get("why_explanation"), "timeframe_fact": state.get("timeframe_fact"),
        "citation_ok": state.get("citation_ok"), "verifier_failures": state.get("verifier_failures"),
        "explanation_attempts": state.get("explanation_attempts", 0),
        "verifier_rules": RULES_VERSION,
        "lesion_counts": {l: int(state["lesion_features"][f"{l}_count"]) for l in ["MA", "HE", "EX", "SE"]},
    }


def main():
    manifest = setup()
    app = build_graph()
    tag = f"{PLANNER_MODE}{'_triage' if SKIP_EXPLANATION else ''}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json, out_csv = OUT_DIR / f"v2_{tag}.json", OUT_DIR / f"v2_{tag}.csv"

    results = json.load(open(out_json)) if out_json.exists() else []
    done = {r["image_id"] for r in results}
    if done:
        print(f"Resuming: {len(done)} already done.")

    def save():
        json.dump(results, open(out_json, "w"), indent=2, default=str)
        pd.DataFrame(results).to_csv(out_csv, index=False)

    for i, row in manifest.iterrows():
        iid = row[v1.IMAGE_COL]
        if iid in done:
            continue
        path = os.path.join(v1.IMAGE_FOLDER, iid + v1.IMAGE_EXT)
        if not os.path.exists(path):
            print(f"  missing image {iid}, skipped")
            continue
        init = {"image_id": iid, "image_path": path, "patient_id": row.get(v1.PATIENT_COL),
                "true_label": int(row[v1.LABEL_COL])}
        try:
            final = app.invoke(init)
        except Exception as e:
            print(f"  {iid} failed ({e}); will retry on next run")
            continue
        results.append(serialize(final))
        if len(results) % SAVE_EVERY == 0:
            save()
            print(f"  {len(results)}/{len(manifest)}")
    save()

    df = pd.DataFrame(results)
    ac = df[df.triage_decision == "AUTO_CLEAR"]
    print(f"\nmode={PLANNER_MODE}: auto_clear={len(ac)}, missed referable={int((ac.true_label >= 2).sum())}, "
          f"REFER_VERIFIED={int((df.triage_decision == 'REFER_VERIFIED').sum())}, "
          f"tool calls={int(df.n_tool_calls.sum())}. Saved {out_csv}")
    print(f"gate: {GATE['rule']} >= {CONF_THRESHOLD}")
    fallbacks = sum(p.get("fallback", False) for log in df.planner_log for p in log)
    if PLANNER_MODE == "llm":
        print(f"planner LLM calls={int(df.planner_llm_calls.sum())}, unusable outputs (fell back to next tool)={fallbacks}"
              + ("  <-- CHECK: see planner_log for the error" if fallbacks else ""))


if __name__ == "__main__":
    main()
