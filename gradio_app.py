"""
Gradio GUI for the LangGraph orchestrator.

Upload a fundus photograph (optionally the fellow eye too) and watch the
pipeline run node by node: diagnosis -> (orchestrator -> evidence
gathering, only when the case is genuinely uncertain) -> referral ->
structuring -> saliency -> explanation -> verifier. The final view shows the
grade, triage decision, calibrated probabilities, the grounded
explanation, and the Grad-CAM / attention-rollout heatmaps.

This file IMPORTS orchestrator_graph.py - it does not modify or duplicate
it, so the graph logic stays in exactly one place. Keep it in the same
folder as orchestrator_graph.py, crop_and_resize.py and vector_store_bge.

--- Preprocessing ---
Training images went through crop_and_resize.py (crop to the fundus circle,
resize to 512x512). Uploaded photos get the same treatment before entering
the graph, otherwise raw photos with black borders / different sizes would
be out-of-distribution for the models.

--- Fellow eye ---
If a second photo is uploaded, it is registered as the patient's fellow eye
so the orchestrator can use bilateral evidence when it chooses to. Without
one, bilateral evidence is simply unavailable and only TTA can be chosen.

--- Notes ---
- Requires the same things orchestrator_graph.py does (trained checkpoints,
  temperature_scaling.json, vector_store_bge, Ollama running with llama3.1),
  plus: pip install gradio
- orchestrator_graph.setup() still reads MANIFEST_PATH once at startup (to
  build its fellow-eye lookup), so that CSV must exist.
- One request is processed at a time (shared models/hooks are not
  thread-safe). Do not run this at the same time as the batch run on a
  shared GPU - both use the GPU and Ollama.
- Research prototype from an academic thesis. Not a medical device and not
  validated for clinical use.

Usage:
    python gradio_app.py
"""

import os
import re
import tempfile
from pathlib import Path
import threading
import traceback

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import gradio as gr

import langgraph.orchestrator_graph as og          # v1 module: models, saliency, retrieval (shared)
import agentic.orchestrator_agentic as ag          # v2 agentic graph (verify-before-clear + lesion tool)
from agentic.lesion_tool import overlay as lesion_overlay, summarise as summarise_lesions, LESION_NAMES
from dataset_prep.crop_and_resize import crop_and_resize

APP = None
RUN_LOCK = threading.Lock()

DISCLAIMER = (
    "> **Research prototype** from an academic thesis. Not a medical device and not "
    "validated for clinical use - do not use it to make care decisions."
)


def pil_to_bgr_array(pil_image):
    """PIL -> contiguous uint8 BGR array via raw bytes. Deliberately avoids
    np.array(pil_image) / Image.fromarray: those go through PIL's array
    interface, which breaks (or yields non-ndarray objects) when the
    installed Pillow and NumPy versions don't match - cv2 then rejects the
    result with "src is not a numerical tuple"."""
    rgb = pil_image.convert("RGB")
    w, h = rgb.size
    arr = np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(h, w, 3)
    return np.ascontiguousarray(arr[:, :, ::-1])


def bgr_array_to_pil(bgr):
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return Image.frombytes("RGB", (w, h), rgb.tobytes())


def preprocess_fundus(pil_image):
    """Same crop + 512x512 resize the training images went through.

    The current crop_and_resize.py returns (image, is_valid, reason) - it
    flags composite/non-circular images instead of silently resizing them.
    Older versions returned just the image, so both shapes are handled."""
    result = crop_and_resize(pil_to_bgr_array(pil_image))
    if isinstance(result, tuple):
        processed, is_valid, reason = result
        if not is_valid:
            raise ValueError(f"this does not look like a single valid fundus photograph ({reason})")
    else:
        processed = result
    return bgr_array_to_pil(processed)


def make_overlay(pil_image, heatmap_2d, alpha=0.4):
    heat = heatmap_2d / (heatmap_2d.max() + 1e-8)
    heat_u8 = np.ascontiguousarray((heat * 255).astype(np.uint8))
    hh, hw = heat_u8.shape
    heat_img = Image.frombytes("L", (hw, hh), heat_u8.tobytes()).resize(pil_image.size, Image.BILINEAR)
    w, h = heat_img.size
    heat_arr = np.frombuffer(heat_img.tobytes(), dtype=np.uint8).reshape(h, w)
    colored = plt.get_cmap("jet")(heat_arr / 255.0)[..., :3] * 255
    base = np.frombuffer(pil_image.convert("RGB").tobytes(), dtype=np.uint8).reshape(h, w, 3)
    blended = np.ascontiguousarray(((1 - alpha) * base + alpha * colored).astype(np.uint8))
    return Image.frombytes("RGB", (w, h), blended.tobytes())


def compute_saliency_maps(pil_image, grade):
    """Re-computes the two heatmaps for display (the graph's saliency node
    only keeps quadrant/concentration descriptors, not the maps). Must run
    WITHOUT torch.no_grad for the Grad-CAM backward pass."""
    input_tensor = og.EVAL_TRANSFORM(pil_image).unsqueeze(0).to(og.DEVICE)
    resnet_cam = og.GRADCAM.compute(input_tensor, grade)

    og.ATTN_STORAGE.clear()
    with torch.no_grad():
        og.VIT_MODEL(input_tensor)
    cls_to_patches = og.compute_attention_rollout(og.ATTN_STORAGE)
    grid = int(np.sqrt(cls_to_patches.shape[0]))
    vit_map = cls_to_patches.reshape(grid, grid)

    return make_overlay(pil_image, resnet_cam), make_overlay(pil_image, vit_map)


def describe_update(node, upd, state):
    if node == "perceive":
        probs = upd["ensemble_probs"]
        order = np.argsort(probs)[::-1]
        top, second = int(order[0]), int(order[1])
        les = upd["lesion_summary"]
        les_txt = ", ".join(f"{v['count']} {v['name']}" for v in les.values()) or "no lesion regions"
        return (f"**Perception** - ensemble top: {og.GRADE_LABELS[top]} ({probs[top] * 100:.1f}%), runner-up: "
                f"{og.GRADE_LABELS[second]} ({probs[second] * 100:.1f}%); lesion model: {les_txt}")
    if node == "triage":
        return (f"**Triage** - calibrated {og.GRADE_LABELS[upd['predicted_grade']]} at "
                f"{upd['calibrated_confidence'] * 100:.1f}%; P(non-referable, grade 0+1) "
                f"{upd['p_nonreferable'] * 100:.1f}% vs auto-clear threshold {ag.CONF_THRESHOLD * 100:.0f}% "
                f"-> `{upd['triage_decision']}`"
                + (" - candidate for auto-clear, starting verification" if upd["triage_decision"] == "AUTO_CLEAR" else ""))
    if node in ("lesion_check", "tool"):
        v = upd["verification"][-1]
        verdict = "DISCORDANT -> refer (`REFER_VERIFIED`)" if v["discordant"] else "concordant"
        name = {"lesion_check": "Lesion check", "fellow_eye": "Fellow-eye tool", "tta": "TTA tool"}[v["tool"]]
        return f"**{name}** - {v['detail']}: {verdict}"
    if node == "planner":
        p = upd["planner_log"][-1]
        act = "stop verifying" if p["action"] == "stop" else f"call `{p['action']}`"
        return f"**Planner (LLM)** - {act}: {p['reasoning']}"
    if node == "structuring":
        return f"**Structuring** - record built; final decision `{state['triage_decision']}`"
    if node == "saliency":
        sal = upd["saliency"]
        agree = "agree" if sal["attention_agreement"] else "do not agree"
        return (f"**Saliency** - ResNet-50: {sal['resnet_quadrant']}, ViT-B/16: {sal['vit_quadrant']} "
                f"(backbones {agree})")
    if node == "explanation":
        return f"**Explanation Agent** - draft {upd['explanation_attempts']} generated from guidelines + lesion estimates"
    if node == "verifier":
        if upd["citation_ok"]:
            return "**Verifier** - passed all rule checks"
        fails = "; ".join(upd["verifier_failures"])
        if state.get("explanation_attempts", 0) < ag.MAX_EXPLANATION_ATTEMPTS:
            return f"**Verifier** - failed ({fails}); sending feedback and regenerating"
        return f"**Verifier** - failed ({fails}); attempt limit reached, keeping last draft"
    return f"**{node}**"


def _test_binary_op():
    f = Path(__file__).resolve().parent / "agentic" / "results" / "binary_operating_point.csv"
    try:
        r = pd.read_csv(f).set_index("split").loc["test"]
        return f"sensitivity {r['sensitivity_%']:.2f}% / specificity {r['specificity_%']:.2f}%"
    except Exception:
        return "see agentic/results/binary_operating_point.csv"


TEST_BINARY_OP = _test_binary_op()


def binary_readout(state):
    """Same model, read the way binary autonomous systems (e.g. LumineticsCore) report:
    more-than-mild DR (grade >= 2) present or not. Not LumineticsCore itself."""
    p_ref = 1.0 - state["p_nonreferable"]
    detected = p_ref > 0.5
    return (f"\n\n---\n**Binary readout (LumineticsCore-style, same model):** P(more-than-mild DR, grade >= 2) = "
            f"{p_ref * 100:.1f}% -> **more-than-mild DR {'detected' if detected else 'not detected'}** "
            f"(at 50%).\n\nOn the test set this binary decision has {TEST_BINARY_OP} "
            f"(published LumineticsCore pivotal trial: 87.2% / 90.7%, on different data). "
            f"The triage above is stricter: it clears only when P(non-referable) >= "
            f"{ag.CONF_THRESHOLD * 100:.0f}%.")


def lesion_table(summary):
    if not summary:
        return "**Lesion estimates:** no lesion regions above threshold."
    rows = ["| Lesion | Regions | Area (% retina) | Upper-left | Upper-right | Lower-left | Lower-right |",
            "|---|---|---|---|---|---|---|"]
    for l, v in summary.items():
        q = v["by_quadrant"]
        rows.append(f"| {v['name']} ({l}) | {v['count']} | {v['area_pct_of_retina']:.3f} | "
                    + " | ".join(str(q.get(k, 0)) for k in ["upper-left", "upper-right", "lower-left", "lower-right"])
                    + " |")
    return ("**Lesion estimates** (U-Net trained on IDRiD; model estimates, not ground truth - "
            "haemorrhage counts tend to be over-estimated, soft-exudate counts are unreliable on this data)\n\n"
            + "\n".join(rows))


def render_trace(lines):
    return "\n".join(f"- {line}" for line in lines)


GROUND_TRUTH = {}   # image name (e.g. "4877_left") -> (grade, split); loaded once in main()
FULL_LABELS_PATH = Path(__file__).resolve().parent / "all_88k_labels.csv"   # full EyePACS pool, no splits


def load_ground_truth():
    """Used ONLY for the 'Reveal true grade' box - labels never enter the graph.
    1. train/val/test manifests of the 15k subset (folder of og.MANIFEST_PATH): split known.
    2. all_88k_labels.csv for everything else. Such an image was never used by the models,
       UNLESS the same patient's other eye is in the 15k (then it is not independent)."""
    split_dir = Path(og.MANIFEST_PATH).parent
    gt, patient_split = {}, {}
    for split in ["train", "val", "test"]:
        f = split_dir / f"{split}_manifest.csv"
        if f.exists():
            df = pd.read_csv(f)
            for n, l in zip(df[og.IMAGE_COL].astype(str), df[og.LABEL_COL]):
                gt[n] = (int(l), split)
                patient_split[n.rsplit("_", 1)[0]] = split
    n_subset = len(gt)
    if FULL_LABELS_PATH.exists():
        full = pd.read_csv(FULL_LABELS_PATH)
        for n, l in zip(full["image"].astype(str), full["level"]):
            if n in gt:
                continue
            sib = patient_split.get(n.rsplit("_", 1)[0])
            gt[n] = (int(l), f"leak:{sib}" if sib else "unseen")
    print(f"Ground truth: {n_subset} images in the 15k subset, {len(gt) - n_subset} more from {FULL_LABELS_PATH.name}")
    return gt


def image_name(path):
    """'.../4877_left.png' -> '4877_left'. Gradio keeps the uploaded file's original name."""
    return Path(path).stem if path else None


def side_from_name(name):
    m = re.search(r"_(left|right)$", name or "")
    return m.group(1) if m else None


SOURCE_TEXT = {
    "train": "15k subset, TRAIN split - the models were trained on this image, so the result does not count",
    "val": "15k subset, VAL split - used to tune ensemble weights, temperature and lesion threshold (slightly optimistic)",
    "test": "15k subset, TEST split - held out",
    "unseen": "full EyePACS pool, outside the 15k - never seen by the models (not quality-selected, may be blurrier)",
}


def truth_line(name, predicted=None, role="Main photograph"):
    if name not in GROUND_TRUTH:
        return (f"- **{role}** `{name}`: no label found in the 15k manifests or {FULL_LABELS_PATH.name} "
                f"(upload dataset images with their original filename, e.g. `4877_left.png`).")
    g, src = GROUND_TRUTH[name]
    if src.startswith("leak:"):
        src_txt = (f"full EyePACS pool, outside the 15k - but this patient's other eye is in the "
                   f"{src.split(':')[1].upper()} split, so it is not fully independent")
    else:
        src_txt = SOURCE_TEXT[src]
    line = f"- **{role}** `{name}`: **{og.GRADE_LABELS[g]} (grade {g})**"
    if predicted is not None:
        diff = predicted - g
        verdict = "correct" if diff == 0 else f"model was {abs(diff)} grade{'s' if abs(diff) > 1 else ''} {'too high' if diff > 0 else 'too low'}"
        line += f" - model predicted {og.GRADE_LABELS[predicted]} (grade {predicted}): {verdict}"
        if g >= 2 and predicted < 2:
            line += " - a referable case graded as non-referable"
    return line + f"\n  - Source: {src_txt}."


def run_pipeline(primary_path_in, fellow_path_in):

    def out(trace="", summary="", probs=None, expl="", pre=None, rn=None, vt=None, les=None, les_md="", truth=""):
        return (trace, summary, probs, expl, pre, rn, vt, les, les_md, truth)

    if primary_path_in is None:
        yield out("Please upload the main fundus photograph first.")
        return

    with RUN_LOCK, tempfile.TemporaryDirectory() as tmpdir:
        try:
            main_name, fellow_name_in = image_name(primary_path_in), image_name(fellow_path_in)
            notes = []
            # The top box is always the eye being graded. Left/right is read from the filename
            # (4877_left.png) for the trace only - it has no effect on any prediction. Without it,
            # the two uploads are paired internally as left/right placeholders.
            detected = side_from_name(main_name)
            eye_side = detected or "left"
            other = "right" if eye_side == "left" else "left"
            if fellow_path_in is not None:
                f_side = side_from_name(fellow_name_in)
                if detected and f_side and f_side == eye_side:
                    notes.append(f"Warning: both photographs look like **{eye_side}** eyes (from filenames) - "
                                 f"the fellow eye should be the patient's other eye.")
                pid_m, pid_f = main_name.rsplit("_", 1)[0], (fellow_name_in or "").rsplit("_", 1)[0]
                if detected and f_side and pid_m != pid_f:
                    notes.append(f"Warning: the two filenames belong to different patients ({pid_m} vs {pid_f}).")

            primary = preprocess_fundus(Image.open(primary_path_in))
            image_id = f"upload_{eye_side}" if fellow_path_in is not None else "upload"
            primary_path = os.path.join(tmpdir, image_id + ".png")
            primary.save(primary_path)

            # Point the graph's fellow-eye machinery at this session's temp folder.
            patient_id = "gui_session"
            og.IMAGE_FOLDER = tmpdir
            og.IMAGE_EXT = ".png"
            og.FELLOW_LOOKUP = {patient_id: {"left": None, "right": None}}
            ag.PRED_CACHE.clear()   # upload names repeat across runs - never reuse another session's memory
            if fellow_path_in is not None:
                fellow_id = f"upload_{other}"
                preprocess_fundus(Image.open(fellow_path_in)).save(os.path.join(tmpdir, fellow_id + ".png"))
                og.FELLOW_LOOKUP[patient_id][eye_side] = image_id
                og.FELLOW_LOOKUP[patient_id][other] = fellow_id

            initial_state = {"image_id": image_id, "image_path": primary_path,
                             "patient_id": patient_id, "true_label": None}

            state = dict(initial_state)
            side_txt = f" ({eye_side} eye, from filename)" if detected else ""
            trace = [f"**Input** - main photograph `{main_name}`{side_txt}"
                     + (f"; fellow eye `{fellow_name_in}`" if fellow_path_in is not None else "; no fellow eye")]
            trace += notes
            yield out(render_trace(trace) + "\n\nRunning...", pre=primary)

            for step in APP.stream(initial_state, stream_mode="updates"):
                for node, upd in step.items():
                    state.update(upd)
                    trace.append(describe_update(node, upd, state))
                yield out(render_trace(trace), pre=primary)

            grade = state["predicted_grade"]
            label = og.GRADE_LABELS[grade]
            reason = state["structured_record"]["referral"]["reason"]
            summary = (
                f"### Result\n"
                f"**{label}** (grade {grade}) - calibrated confidence "
                f"{state['calibrated_confidence'] * 100:.1f}%\n\n"
                f"**Triage decision:** `{state['triage_decision']}`"
                + (f" (initially `{state['initial_decision']}`)" if state["initial_decision"] != state["triage_decision"] else "")
                + f"\n\n{reason}"
                + binary_readout(state)
            )

            explanation = f"### Explanation\n{state['why_explanation']}\n\n**Next steps:** {state['timeframe_fact']}"
            if not state.get("citation_ok"):
                explanation += ("\n\n> Warning: this explanation failed the automated checks ("
                                + "; ".join(state.get("verifier_failures") or []) + ") - treat it with extra caution.")

            feats, masks, img_bgr = ag.LESION_TOOL.measure_path(primary_path)
            lesion_img = bgr_array_to_pil(np.ascontiguousarray(lesion_overlay(img_bgr, masks)))
            resnet_overlay, vit_overlay = compute_saliency_maps(primary, grade)

            truth = [truth_line(main_name, grade)]
            if fellow_path_in is not None:
                fellow_pred = next((v["detail"] for v in state.get("verification", []) if v["tool"] == "fellow_eye"), None)
                truth.append(truth_line(fellow_name_in, role="Fellow eye")
                             + (f" (fellow-eye tool saw: {fellow_pred})" if fellow_pred else ""))

            yield out(render_trace(trace), summary, state["class_probabilities"], explanation,
                      primary, resnet_overlay, vit_overlay, lesion_img, lesion_table(state["lesion_summary"]),
                      "\n".join(truth))

        except ValueError as e:
            yield out(f"Could not process the upload: {e}")
        except Exception as e:
            traceback.print_exc()   # full traceback in the terminal running the app
            yield out(f"Pipeline failed: `{e}`\n\nFull traceback is in the terminal.")


def build_demo():
    with gr.Blocks(title="Agentic DR Screening - Demo") as demo:
        gr.Markdown("# Agentic AI for Diabetic Retinopathy - Demo")
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column(scale=1):
                # type="filepath" keeps the original filename, used to detect the eye side
                # (4877_left.png) and to look up the true grade.
                primary = gr.Image(type="filepath", label="Main photograph - the eye being graded")
                fellow = gr.Image(type="filepath", label="Fellow eye - the same patient's OTHER eye (optional)")
                gr.Markdown("The top photograph is always the eye being graded. Upload dataset images with their "
                            "original filenames (e.g. `4877_left.png`) so the eye side and true grade can be read.")
                run_btn = gr.Button("Run pipeline", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### Pipeline trace")
                trace = gr.Markdown()
                summary = gr.Markdown()
                probs = gr.Label(label="Calibrated grade probabilities", num_top_classes=5)

        explanation = gr.Markdown()
        with gr.Accordion("Reveal true grade (from the dataset labels - for cross-checking only)", open=False):
            truth_md = gr.Markdown()

        with gr.Row():
            pre_img = gr.Image(label="Preprocessed input", type="pil", interactive=False)
            resnet_img = gr.Image(label="ResNet-50 Grad-CAM (class-specific)", type="pil", interactive=False)
            vit_img = gr.Image(label="ViT-B/16 attention rollout (not class-specific)", type="pil",
                               interactive=False)
            lesion_img = gr.Image(label="Lesion estimates (yellow MA, red HE, green EX, blue SE)",
                                  type="pil", interactive=False)
        lesion_md = gr.Markdown()

        run_btn.click(
            run_pipeline,
            inputs=[primary, fellow],
            outputs=[trace, summary, probs, explanation, pre_img, resnet_img, vit_img, lesion_img, lesion_md, truth_md],
        )
    return demo


def main():
    global APP
    print("Loading models, vector store and graph (one-time)...")
    global GROUND_TRUTH
    # The GUI always runs the full agentic system, whatever the batch-run settings at the top of
    # orchestrator_agentic.py were last left at (e.g. SKIP_EXPLANATION=True from a triage-only run).
    ag.PLANNER_MODE = "llm"
    ag.SKIP_EXPLANATION = False
    # Portable: off the lab PC the E: drive is absent, so fall back to manifests copied into the repo
    # (data/splits/{train,val,test}_manifest.csv - small CSVs, used for the fellow-eye lookup and the
    # "Reveal true grade" box only).
    if not os.path.exists(og.MANIFEST_PATH):
        local = Path(__file__).resolve().parent / "data" / "splits" / "test_manifest.csv"
        if local.exists():
            og.MANIFEST_PATH = str(local)
            print(f"Using repo copy of the split manifests: {local.parent}")
        else:
            raise FileNotFoundError(f"{og.MANIFEST_PATH} not found - copy the split manifests to {local.parent}")
    ag.setup()
    APP = ag.build_graph()
    GROUND_TRUTH = load_ground_truth()

    demo = build_demo()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()