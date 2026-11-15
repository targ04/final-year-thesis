"""
LangGraph Orchestrator - ties Diagnosis, Referral, Structuring, Saliency,
and the Explanation Agent into one graph with REAL conditional control
flow, tool-use, and reflection - three of Grzybowski et al.'s four
agentic capabilities [2] that a flat batch pipeline doesn't have, even
when every individual stage is well-built.

--- What's agentic here, and what deliberately isn't ---
Diagnosis (ensemble forward pass), Referral (temperature scaling + triage
policy), and Structuring (templated fact readout) remain exactly as
deterministic as they were as standalone scripts - wrapping a dict lookup
in a graph node doesn't make it an agent (same mistake this project
already caught once, see explanation_agent_generate.py's docstring).
What's genuinely agentic:

  1. THE ORCHESTRATOR (this script): decides, from state, whether a
     diagnosis is confident enough to triage directly, or whether it's
     worth the extra compute to re-score via test-time augmentation
     and/or bilateral (fellow-eye) evidence first. A cheap deterministic
     pre-filter (MARGIN_THRESHOLD) decides WHETHER to even consult the
     orchestrator LLM - most images are confidently diagnosed and don't
     need it. When consulted, the LLM reasons about whether escalation
     could plausibly change the TRIAGE outcome (do the top-2 candidate
     grades fall in different triage buckets?), not just the raw grade.
  2. THE EXPLANATION AGENT (reused from explanation_agent_generate.py):
     retrieval-grounded synthesis, prompt unchanged from the validated
     version.
  3. THE VERIFIER LOOP: the same citation/grounding check used as a QA
     metric in the batch script now actually triggers regeneration on
     failure instead of just being logged - genuine reflection, bounded
     by MAX_EXPLANATION_RETRIES so an unfixable case still terminates.

--- Escalation is bounded, not iterative ---
TTA and bilateral fusion each run AT MOST ONCE per image. There is no
repeated "re-check confidence, escalate again" loop - that would be both
an infinite-loop risk and hard to justify methodologically (at what point
does more augmentation stop being informative?).

--- Bilateral fusion is evidence, not blending ---
The fellow eye's own prediction is ATTACHED to state as supporting
context, never mathematically averaged into this eye's probability
vector. DR severity can genuinely differ between a patient's two eyes, so
blending two different images' predictions as if they were repeated
measurements of the same thing would be a methodological overclaim -
concordance is informative, fusion isn't.

--- Nothing here is refit ---
The temperature scaling value and the vector store are loaded from what
referral_agent.py / explanation_agent_ingest.py already produced. This
script orchestrates already-validated pieces; it does not retrain or
refit anything.

Setup:
    pip install langgraph
Everything else (Ollama + llama3.1, BAAI/bge-base-en-v1.5, trained
checkpoints, temperature_scaling.json, vector_store_bge) must already
exist on disk from the scripts already built.

Usage:
    python orchestrator_graph.py
"""

import os
import re
import json
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END

# ---------------- CONFIG ----------------
basepath = Path(__file__).parent

RESNET_WEIGHTS_PATH = basepath.parent / "diagnosis_agent" / r"resnet50\best_model.pt"
VIT_WEIGHTS_PATH = basepath.parent / "diagnosis_agent" / r"vitb\run2\best_model.pt"
TEMPERATURE_JSON_PATH = basepath.parent / "referral_agent" / "temperature_scaling.json"
VECTOR_STORE_DIR = basepath.parent / "explanation_agent" / "vector_store_bge"

IMAGE_FOLDER = r"E:\Thesis Dataset\final dataset\images"
IMAGE_EXT = ".png"
MANIFEST_PATH = r"E:\Thesis Dataset\final dataset\splits\test_manifest.csv"   # columns: image, level, patient_id

OUTPUT_JSON_PATH = basepath / "orchestrated_results_updated.json"
OUTPUT_CSV_PATH = basepath / "orchestrated_results_updated.csv"
SAVE_EVERY = 25

IMAGE_COL = "image"
LABEL_COL = "level"
PATIENT_COL = "patient_id"
NUM_CLASSES = 5
INPUT_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RESNET_WEIGHT = 0.6   # matches the ensemble weight sweep's chosen value
VIT_WEIGHT = 0.4

MARGIN_THRESHOLD = 0.15
MANDATORY_REFERRAL_GRADES = {2, 3, 4}
CONFIDENCE_THRESHOLD = 0.85   # must match referral_agent.py's own threshold

RESOURCE_SETTING = "high"
GRADE_TIMEFRAMES = {
    "high": {0: ("1-2 years", False), 1: ("6-12 months", False), 2: ("3-6 months", True),
             3: ("<3 months", True), 4: ("<1 month", True)},
    "low": {0: ("1-2 years", False), 1: ("1-2 years", False), 2: ("6-12 months", True),
            3: ("<3 months", True), 4: ("<1 month", True)},
}
TIMEFRAME_CITATION = "(ICO Guidelines for Diabetic Eye Care, 2017, p.8)"

GRADE_LABELS = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR (PDR)"}
ALT_GRADE_THRESHOLD = 0.15

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this question for retrieving supporting documents: "
TOP_K = 4

OLLAMA_MODEL_NAME = "llama3.1"
CITATION_PATTERN = re.compile(r"(?:ICO|AAO)[^.]{0,100}?p\.\s?\d+", re.IGNORECASE)
NO_CRITERIA_FOUND_TEXT = "No additional guideline-specific clinical criteria found in the retrieved context."
MAX_EXPLANATION_RETRIES = 3   # allows 2 real retries (3 total attempts) - the retry
                              # counter increments starting from the first failed
                              # attempt, so this value is "total attempts allowed",
                              # not "number of retries on top of the first"
# -----------------------------------------


class PipelineState(TypedDict, total=False):
    image_id: str
    image_path: str
    patient_id: Any
    true_label: Optional[int]
    fellow_eye_image: Optional[str]
    fellow_eye_prediction: Optional[Dict[str, Any]]
    ensemble_probs: np.ndarray
    escalate: bool
    escalation_method: str
    escalation_reasoning: str
    calibrated_probs: np.ndarray
    predicted_grade: int
    calibrated_confidence: float
    class_probabilities: Dict[str, float]
    triage_decision: str
    structured_record: Dict[str, Any]
    saliency: Dict[str, Any]
    retrieval_query: str
    why_explanation: str
    timeframe_fact: str
    has_citation: bool
    no_criteria_found: bool
    citation_ok: bool
    explanation_retries: int


# ---------------- Model building (matches ensemble_eval.py exactly) ----------------

def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_vit_b(num_classes, dropout_p=0.3):
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))
    return model


def load_model(model, weights_path, device):
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def ensemble_predict(pil_image):
    input_tensor = EVAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        resnet_probs = F.softmax(RESNET_MODEL(input_tensor), dim=1).cpu().numpy()[0]
        vit_probs = F.softmax(VIT_MODEL(input_tensor), dim=1).cpu().numpy()[0]
    return RESNET_WEIGHT * resnet_probs + VIT_WEIGHT * vit_probs


def tta_predict(pil_image):
    """Averages ensemble softmax across the same augmentation family used
    at TRAINING time (horizontal/vertical flip, rotation) - re-sampling
    invariances the models were trained to have, not introducing new ones."""
    views = [pil_image, pil_image.transpose(Image.FLIP_LEFT_RIGHT),
             pil_image.transpose(Image.FLIP_TOP_BOTTOM), pil_image.rotate(180)]
    return np.mean([ensemble_predict(v) for v in views], axis=0)


# ---------------- Grad-CAM / attention rollout (matches generate_saliency.py) ----------------

class ResNetGradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        model.layer4.register_forward_hook(self._save_activations)
        model.layer4.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1).squeeze(0))
        return cam.cpu().numpy()


def patch_vit_to_capture_attention(vit_model, storage):
    for block in vit_model.encoder.layers:
        def make_patched_forward(blk):
            def patched_forward(input):
                x = blk.ln_1(input)
                attn_output, attn_weights = blk.self_attention(x, x, x, need_weights=True, average_attn_weights=True)
                storage.append(attn_weights.detach())
                x = blk.dropout(attn_output)
                x = x + input
                y = blk.ln_2(x)
                y = blk.mlp(y)
                return x + y
            return patched_forward
        block.forward = make_patched_forward(block)


def compute_attention_rollout(attn_weights_list):
    seq_len = attn_weights_list[0].shape[-1]
    device = attn_weights_list[0].device
    rollout = torch.eye(seq_len, device=device).unsqueeze(0)
    for attn in attn_weights_list:
        attn_res = attn + torch.eye(seq_len, device=device).unsqueeze(0)
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(attn_res, rollout)
    return rollout[0, 0, 1:].cpu().numpy()


def heatmap_to_quadrant(heatmap_2d):
    h, w = heatmap_2d.shape
    total = heatmap_2d.sum()
    if total <= 1e-8:
        return "undetermined", 0.0
    yy, xx = np.mgrid[0:h, 0:w]
    centroid_row = (yy * heatmap_2d).sum() / total / h
    centroid_col = (xx * heatmap_2d).sum() / total / w
    quadrant = f"{'upper' if centroid_row < 0.5 else 'lower'}-{'left' if centroid_col < 0.5 else 'right'}"
    flat_sorted = np.sort(heatmap_2d.flatten())[::-1]
    top_k = max(1, int(0.1 * flat_sorted.size))
    return quadrant, float(flat_sorted[:top_k].sum() / total)


# ---------------- Deterministic facts (matches structuring/explanation_agent_generate.py) ----------------

def build_timeframe_sentence(predicted_grade):
    interval, referral_required = GRADE_TIMEFRAMES[RESOURCE_SETTING][predicted_grade]
    req_text = "referral is required" if referral_required else "referral is not required"
    return (f"Per ICO guidelines ({RESOURCE_SETTING}-resource setting): recommended "
            f"re-examination/referral interval is {interval}; {req_text} {TIMEFRAME_CITATION}.")


def build_alt_grade_note(class_probabilities, predicted_grade):
    sorted_grades = sorted(class_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    for label, prob in sorted_grades:
        grade_num = [k for k, v in GRADE_LABELS.items() if v == label][0]
        if grade_num != predicted_grade and prob >= ALT_GRADE_THRESHOLD:
            return f"{label} (grade {grade_num}) at {prob * 100:.1f}% probability"
    return None


def build_referral_reason(triage_decision, grade_label, confidence, threshold):
    conf_pct = f"{confidence * 100:.1f}%"
    threshold_pct = f"{threshold * 100:.0f}%"
    if triage_decision == "REFER_SEVERITY":
        return (f"Referred for specialist review because the predicted grade ({grade_label}) meets the "
                f"mandatory referral threshold (moderate NPDR or worse). This referral applies regardless "
                f"of model confidence, per the severity-first triage policy.")
    elif triage_decision == "AUTO_CLEAR":
        return (f"No referral required. Predicted grade ({grade_label}) with calibrated confidence "
                f"{conf_pct}, which meets the auto-clear confidence threshold of {threshold_pct}.")
    elif triage_decision == "REFER_LOW_CONFIDENCE":
        return (f"Referred for specialist review because calibrated confidence ({conf_pct}) fell below "
                f"the auto-clear threshold ({threshold_pct}), despite a {grade_label.lower()} prediction.")
    return f"Unrecognized triage decision: {triage_decision}"


def format_context(docs):
    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "?")
        lines.append(f"[{i}] ({source}, p.{page})\n{doc.page_content.strip()}")
    return "\n\n".join(lines)


def build_why_prompt(grade_label, saliency_row, alt_grade_note, context_block):
    """Same validated prompt as explanation_agent_generate.py - do not
    redesign this, it's already been through two rounds of fixing a real
    hallucination and a boundary violation."""
    if saliency_row is not None:
        r_focal = "focal" if saliency_row["resnet_concentration"] >= 0.3 else "diffuse"
        v_focal = "focal" if saliency_row["vit_concentration"] >= 0.3 else "diffuse"
        if saliency_row["resnet_quadrant"] == saliency_row["vit_quadrant"]:
            attention_desc = (f"Both backbones concentrated attention in the {saliency_row['resnet_quadrant']} "
                               f"region (ResNet-50: {r_focal}, {saliency_row['resnet_concentration']:.2f}; "
                               f"ViT-B/16: {v_focal}, {saliency_row['vit_concentration']:.2f}).")
        else:
            attention_desc = (f"ResNet-50's attention concentrated in the {saliency_row['resnet_quadrant']} region "
                               f"({r_focal}, {saliency_row['resnet_concentration']:.2f}); ViT-B/16's in the "
                               f"{saliency_row['vit_quadrant']} region ({v_focal}, {saliency_row['vit_concentration']:.2f}) "
                               f"- the two backbones do not agree on a single region.")
    else:
        attention_desc = "No saliency/attention data available for this image."

    alt_grade_text = f"The next most probable grade was {alt_grade_note}." if alt_grade_note else \
        "No other grade had a meaningfully high probability."

    system_prompt = (
        "You are explaining WHY an automated diabetic retinopathy (DR) grade is plausible, for an "
        "ophthalmologist reviewing the case. You have exactly three kinds of evidence, and must not "
        "exceed them:\n\n"
        "1. ATTENTION LOCATION: where two independently-trained model backbones concentrated their "
        "attention, whether they agree, and how focal vs. diffuse each was. FORBIDDEN: any sentence "
        "implying the models 'detected', 'identified', 'saw', or 'found' a feature, lesion, or finding. "
        "ALSO FORBIDDEN: any interpretation of what the attention PATTERN means (e.g. 'indicating a "
        "complex pattern of abnormalities', 'suggesting multiple findings') - disagreement between the "
        "two backbones is not evidence of complexity, it may just mean one or both backbones attended "
        "to something unrelated to the grade. Report ONLY location, agreement, and focal/diffuse - never "
        "what it implies.\n"
        "2. RETRIEVED GUIDELINE TEXT: use ONLY the sentence(s) that explicitly name THIS grade - retrieved "
        "text is a glossary with multiple DIFFERENT terms defined back-to-back; never use a definition "
        "unless it explicitly names this exact grade. PARAPHRASE rather than quoting at length - a short "
        "paraphrase with a citation is preferred over a long verbatim quote, which risks running out of "
        "space mid-sentence. Cite in the exact form (SOURCE, p.N). If no excerpt explicitly defines this "
        f"grade, output exactly: '{NO_CRITERIA_FOUND_TEXT}'\n"
        "3. ALTERNATE GRADE: the next most probable grade, if meaningfully likely, and its clinical "
        "implication. Do not conclude by reasserting the predicted grade is correct.\n\n"
        "Do not restate the predicted grade name, confidence, or referral timeframe. HARD LIMIT: at most "
        "3 sentences, one paragraph, no headers or bullet points."
    )
    user_prompt = (
        f"Predicted grade (already known, do not restate): {grade_label}\n\n"
        f"Attention evidence: {attention_desc}\n\n"
        f"Alternate grade: {alt_grade_text}\n\n"
        f"Retrieved guideline excerpts:\n{context_block}\n\n"
        f"Write the WHY explanation now."
    )
    return system_prompt, user_prompt


def generate_text(system_prompt, user_prompt):
    response = ollama.chat(
        model=OLLAMA_MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        options={"temperature": 0.2, "num_predict": 220},   # raised from 180 - verbatim guideline
                                                              # quotes were running out of budget
                                                              # mid-sentence on the 3rd sentence
    )
    return response["message"]["content"].strip()


def build_fellow_eye_lookup(manifest_df):
    """Maps patient_id -> {"left": image_name or None, "right": image_name or None}."""
    lookup = {}
    for _, row in manifest_df.iterrows():
        image_name = row[IMAGE_COL]
        side = "left" if image_name.endswith("_left") else ("right" if image_name.endswith("_right") else None)
        if side is None:
            continue
        lookup.setdefault(row[PATIENT_COL], {"left": None, "right": None})
        lookup[row[PATIENT_COL]][side] = image_name
    return lookup


# ---------------- Orchestrator decision (the genuinely agentic control-flow step) ----------------

def orchestrator_decide(state):
    """Only reached when route_after_diagnosis has already confirmed (in
    code, not by the LLM) that the margin is close AND the two candidate
    grades fall in genuinely different triage buckets - so the premise
    "resolving this could change the clinical outcome" is guaranteed true
    by the time the LLM sees it, not something the LLM has to verify."""
    probs = state["ensemble_probs"]
    sorted_grades = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    top1_idx, top1_prob = sorted_grades[0]
    top2_idx, top2_prob = sorted_grades[1]
    fellow_available = state.get("fellow_eye_image") is not None

    prompt = (
        f"An automated diabetic retinopathy classifier is uncertain between two grades for this fundus "
        f"image. These two grades fall in DIFFERENT triage buckets (one would change the clinical action "
        f"taken), so resolving the uncertainty matters:\n"
        f"- Top prediction: {GRADE_LABELS[top1_idx]} (grade {top1_idx}), probability {top1_prob:.3f}\n"
        f"- Runner-up: {GRADE_LABELS[top2_idx]} (grade {top2_idx}), probability {top2_prob:.3f}\n"
        f"Fellow eye image available: {fellow_available}\n\n"
        f"Decide whether to gather more evidence before triaging, and which: 'tta' (test-time "
        f"augmentation re-scoring, always available), 'bilateral' (fellow-eye evidence, only if "
        f"available), 'both', or 'none' if the margin is close enough that more evidence is unlikely "
        f"to help.\n\n"
        f'Respond with ONLY a JSON object: {{"escalate": true or false, "method": "tta"|"bilateral"|'
        f'"both"|"none", "reasoning": "one short sentence"}}'
    )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 150},
        )
        raw = response["message"]["content"].strip()
        # Extract just the {...} substring rather than only stripping markdown
        # fences - the model sometimes adds prose around the JSON despite the
        # "ONLY a JSON object" instruction, and a bare fence-strip fails on
        # that extra text, silently defaulting to no escalation.
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        decision = json.loads(json_match.group(0) if json_match else raw)
        method = decision.get("method", "none")
        if method in ("bilateral", "both") and not fellow_available:
            method = "tta" if method == "both" else "none"
        return {
            "escalate": bool(decision.get("escalate", False)) and method != "none",
            "escalation_method": method,
            "escalation_reasoning": decision.get("reasoning", ""),
        }
    except Exception as e:
        # Safe default: don't escalate. The case still gets triaged normally,
        # just without the extra evidence - never block the pipeline on a
        # malformed LLM response.
        return {"escalate": False, "escalation_method": "none",
                "escalation_reasoning": f"orchestrator response unparseable, defaulting to no escalation ({e})"}


# ---------------- Graph nodes ----------------

def diagnosis_node(state):
    pil_image = Image.open(state["image_path"]).convert("RGB")
    probs = ensemble_predict(pil_image)
    side = "left" if state["image_id"].endswith("_left") else ("right" if state["image_id"].endswith("_right") else None)
    other_side = {"left": "right", "right": "left"}.get(side)
    fellow = FELLOW_LOOKUP.get(state["patient_id"], {}).get(other_side) if other_side else None
    return {"ensemble_probs": probs, "fellow_eye_image": fellow, "explanation_retries": 0}


def route_after_diagnosis(state):
    """Two deterministic gates before the orchestrator LLM is even
    consulted - both are facts the model shouldn't be trusted to reason
    about correctly when the answer is just a lookup:
      1. margin - if the top prediction is already clearly ahead, there's
         nothing to resolve.
      2. bucket equality - even with a close margin, if BOTH candidate
         grades map to the same triage decision (e.g. grade 2 and grade 3
         are both MANDATORY_REFERRAL), escalating cannot change the
         clinical outcome. An earlier version let the LLM judge this and
         it got a real case wrong (claimed "different triage buckets" for
         two grades that were both MANDATORY_REFERRAL) - moved to code."""
    sorted_grades = sorted(enumerate(state["ensemble_probs"]), key=lambda x: x[1], reverse=True)
    top1_idx, top1_prob = sorted_grades[0]
    top2_idx, top2_prob = sorted_grades[1]

    if top1_prob - top2_prob >= MARGIN_THRESHOLD:
        return "skip_to_referral"
    if (top1_idx in MANDATORY_REFERRAL_GRADES) == (top2_idx in MANDATORY_REFERRAL_GRADES):
        return "skip_to_referral"
    return "consult_orchestrator"


def orchestrator_node(state):
    return orchestrator_decide(state)


def route_after_orchestrator(state):
    return "escalate" if state.get("escalate") else "skip_to_referral"


def escalate_node(state):
    method = state["escalation_method"]
    updates = {}
    pil_image = Image.open(state["image_path"]).convert("RGB")

    if method in ("tta", "both"):
        updates["ensemble_probs"] = tta_predict(pil_image)

    if method in ("bilateral", "both") and state.get("fellow_eye_image"):
        fellow_path = os.path.join(IMAGE_FOLDER, state["fellow_eye_image"] + IMAGE_EXT)
        if os.path.exists(fellow_path):
            fellow_image = Image.open(fellow_path).convert("RGB")
            fellow_probs = ensemble_predict(fellow_image)
            fellow_pred = int(np.argmax(fellow_probs))
            updates["fellow_eye_prediction"] = {
                "image": state["fellow_eye_image"], "predicted_grade": fellow_pred,
                "grade_label": GRADE_LABELS[fellow_pred], "confidence": float(fellow_probs[fellow_pred]),
            }
    return updates


def referral_node(state):
    probs = state["ensemble_probs"]
    pseudo_logits = np.log(probs + 1e-12) / TEMPERATURE
    exp = np.exp(pseudo_logits - pseudo_logits.max())
    calibrated_probs = exp / exp.sum()

    predicted_grade = int(np.argmax(calibrated_probs))
    confidence = float(calibrated_probs[predicted_grade])

    if predicted_grade in MANDATORY_REFERRAL_GRADES:
        decision = "REFER_SEVERITY"
    elif confidence >= CONFIDENCE_THRESHOLD:
        decision = "AUTO_CLEAR"
    else:
        decision = "REFER_LOW_CONFIDENCE"

    class_probabilities = {GRADE_LABELS[c]: round(float(calibrated_probs[c]), 4) for c in range(NUM_CLASSES)}
    return {"calibrated_probs": calibrated_probs, "predicted_grade": predicted_grade,
            "calibrated_confidence": confidence, "triage_decision": decision,
            "class_probabilities": class_probabilities}


def structuring_node(state):
    grade_label = GRADE_LABELS[state["predicted_grade"]]
    reason = build_referral_reason(state["triage_decision"], grade_label, state["calibrated_confidence"], CONFIDENCE_THRESHOLD)
    record = {
        "diagnosis": {"predicted_grade": state["predicted_grade"], "grade_label": grade_label,
                      "calibrated_confidence": round(state["calibrated_confidence"], 4),
                      "class_probabilities": state["class_probabilities"]},
        "referral": {"decision": state["triage_decision"], "reason": reason},
    }
    return {"structured_record": record}


def saliency_node(state):
    pil_image = Image.open(state["image_path"]).convert("RGB")
    input_tensor = EVAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    target_class = state["predicted_grade"]

    resnet_cam = GRADCAM.compute(input_tensor, target_class)
    resnet_quadrant, resnet_conc = heatmap_to_quadrant(resnet_cam)

    ATTN_STORAGE.clear()
    with torch.no_grad():
        VIT_MODEL(input_tensor)
    cls_to_patches = compute_attention_rollout(ATTN_STORAGE)
    grid_size = int(np.sqrt(cls_to_patches.shape[0]))
    vit_quadrant, vit_conc = heatmap_to_quadrant(cls_to_patches.reshape(grid_size, grid_size))

    return {"saliency": {"resnet_quadrant": resnet_quadrant, "resnet_concentration": resnet_conc,
                          "vit_quadrant": vit_quadrant, "vit_concentration": vit_conc,
                          "attention_agreement": resnet_quadrant == vit_quadrant}}


def explanation_node(state):
    grade_label = GRADE_LABELS[state["predicted_grade"]]
    timeframe_fact = build_timeframe_sentence(state["predicted_grade"])
    alt_grade_note = build_alt_grade_note(state["class_probabilities"], state["predicted_grade"])

    query = f"{grade_label} diabetic retinopathy clinical features and diagnostic criteria"
    docs = VECTOR_STORE.similarity_search(QUERY_INSTRUCTION + query, k=TOP_K)
    context_block = format_context(docs)

    system_prompt, user_prompt = build_why_prompt(grade_label, state.get("saliency"), alt_grade_note, context_block)
    if state.get("explanation_retries", 0) > 0:
        user_prompt += ("\n\nIMPORTANT: your previous attempt either made an uncited claim or was cut off "
                         "before finishing a sentence. Keep claims cited and make sure your response ends "
                         "with a complete sentence this time.")

    why_explanation = generate_text(system_prompt, user_prompt)
    no_criteria_found = why_explanation.strip().startswith("No additional guideline-specific clinical criteria")
    has_citation = bool(CITATION_PATTERN.search(why_explanation))

    return {"why_explanation": why_explanation, "timeframe_fact": timeframe_fact,
            "has_citation": has_citation, "no_criteria_found": no_criteria_found, "retrieval_query": query}


def verifier_node(state):
    """Checks citation presence AND that generation actually finished - a
    response can satisfy the citation regex via an earlier sentence while
    its LAST sentence got cut off mid-quote by the token budget (observed
    in practice: a Moderate NPDR quote truncated to "...standard
    photograph" with no closing punctuation or citation). A truncated
    response is malformed regardless of what an earlier sentence cited."""
    explanation = state["why_explanation"].strip()
    ends_cleanly = bool(re.search(r'[.!?]"?$', explanation))
    ok = (state["has_citation"] or state["no_criteria_found"]) and ends_cleanly
    retries = state.get("explanation_retries", 0) + (0 if ok else 1)
    return {"citation_ok": ok, "explanation_retries": retries}


def route_after_verifier(state):
    if state["citation_ok"] or state["explanation_retries"] >= MAX_EXPLANATION_RETRIES:
        return "accept"
    return "retry"


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("escalate_evidence", escalate_node)
    graph.add_node("referral", referral_node)
    graph.add_node("structuring", structuring_node)
    graph.add_node("saliency", saliency_node)
    graph.add_node("explanation", explanation_node)
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("diagnosis")
    graph.add_conditional_edges("diagnosis", route_after_diagnosis,
                                 {"consult_orchestrator": "orchestrator", "skip_to_referral": "referral"})
    graph.add_conditional_edges("orchestrator", route_after_orchestrator,
                                 {"escalate": "escalate_evidence", "skip_to_referral": "referral"})
    graph.add_edge("escalate_evidence", "referral")
    graph.add_edge("referral", "structuring")
    graph.add_edge("structuring", "saliency")
    graph.add_edge("saliency", "explanation")
    graph.add_edge("explanation", "verifier")
    graph.add_conditional_edges("verifier", route_after_verifier, {"retry": "explanation", "accept": END})

    return graph.compile()


def setup():
    """Loads every model/store/config ONCE - called before the graph runs
    any image. All of this is REUSED from already-validated artifacts;
    nothing is refit here."""
    global DEVICE, EVAL_TRANSFORM, RESNET_MODEL, VIT_MODEL, GRADCAM, ATTN_STORAGE
    global TEMPERATURE, VECTOR_STORE, FELLOW_LOOKUP

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    EVAL_TRANSFORM = get_eval_transform()
    RESNET_MODEL = load_model(build_resnet50(NUM_CLASSES), RESNET_WEIGHTS_PATH, DEVICE)
    VIT_MODEL = load_model(build_vit_b(NUM_CLASSES), VIT_WEIGHTS_PATH, DEVICE)

    GRADCAM = ResNetGradCAM(RESNET_MODEL)
    ATTN_STORAGE = []
    patch_vit_to_capture_attention(VIT_MODEL, ATTN_STORAGE)

    with open(TEMPERATURE_JSON_PATH, "r") as f:
        TEMPERATURE = json.load(f)["temperature"]
    print(f"Loaded fitted temperature: {TEMPERATURE:.4f}")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})
    VECTOR_STORE = Chroma(persist_directory=str(VECTOR_STORE_DIR), embedding_function=embeddings)

    manifest_df = pd.read_csv(MANIFEST_PATH)
    FELLOW_LOOKUP = build_fellow_eye_lookup(manifest_df)
    return manifest_df


def load_existing_results(path):
    if not os.path.exists(path):
        return [], set()
    with open(path, "r") as f:
        results = json.load(f)
    done_ids = {r["image_id"] for r in results}
    print(f"Resuming: {len(done_ids)} images already processed, will be skipped.")
    return results, done_ids


def serialize_result(image_id, state):
    return {
        "image_id": image_id,
        "patient_id": state.get("patient_id"),
        "true_label": state.get("true_label"),
        "escalated": state.get("escalate", False),
        "escalation_method": state.get("escalation_method", "none"),
        "escalation_reasoning": state.get("escalation_reasoning", ""),
        "fellow_eye_prediction": state.get("fellow_eye_prediction"),
        "predicted_grade": state.get("predicted_grade"),
        "grade_label": GRADE_LABELS.get(state.get("predicted_grade")),
        "calibrated_confidence": state.get("calibrated_confidence"),
        "triage_decision": state.get("triage_decision"),
        "saliency": state.get("saliency"),
        "why_explanation": state.get("why_explanation"),
        "timeframe_fact": state.get("timeframe_fact"),
        "citation_ok": state.get("citation_ok"),
        "explanation_retries": state.get("explanation_retries", 0),
    }


def main():
    manifest_df = setup()
    app = build_graph()

    results, done_ids = load_existing_results(OUTPUT_JSON_PATH)
    newly_processed = 0

    for _, row in manifest_df.iterrows():
        image_id = row[IMAGE_COL]
        if image_id in done_ids:
            continue

        img_path = os.path.join(IMAGE_FOLDER, image_id + IMAGE_EXT)
        if not os.path.exists(img_path):
            print(f"  WARNING: image not found for {image_id}, skipping.")
            continue

        initial_state = {
            "image_id": image_id,
            "image_path": img_path,
            "patient_id": row.get(PATIENT_COL),
            "true_label": int(row[LABEL_COL]) if LABEL_COL in row.index else None,
        }

        try:
            final_state = app.invoke(initial_state)
        except Exception as e:
            print(f"  WARNING: pipeline failed for {image_id} ({e}) - skipping, will retry next run.")
            continue

        results.append(serialize_result(image_id, final_state))
        newly_processed += 1

        if newly_processed % SAVE_EVERY == 0:
            with open(OUTPUT_JSON_PATH, "w") as f:
                json.dump(results, f, indent=2, default=str)
            pd.DataFrame(results).to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"  ...{newly_processed} new images processed this run (saved checkpoint).")

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    pd.DataFrame(results).to_csv(OUTPUT_CSV_PATH, index=False)

    escalated = sum(1 for r in results if r["escalated"])
    print(f"\nDone. {newly_processed} new images processed this run. Total: {len(results)}")
    print(f"Escalated (orchestrator chose TTA/bilateral/both): {escalated} "
          f"({escalated / max(len(results), 1) * 100:.1f}%)")
    print(f"Saved to {OUTPUT_JSON_PATH} and {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()