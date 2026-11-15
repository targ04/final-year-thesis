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
import tempfile
import threading
import traceback

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import gradio as gr

import langgraph.orchestrator_graph as og
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


def describe_update(node, upd):
    if node == "diagnosis":
        probs = upd["ensemble_probs"]
        order = np.argsort(probs)[::-1]
        top, second = int(order[0]), int(order[1])
        return (f"**Diagnosis ensemble** - top: {og.GRADE_LABELS[top]} ({probs[top] * 100:.1f}%), "
                f"runner-up: {og.GRADE_LABELS[second]} ({probs[second] * 100:.1f}%)")
    if node == "orchestrator":
        if upd.get("escalate"):
            return (f"**Orchestrator** - escalating via `{upd['escalation_method']}`: "
                    f"{upd.get('escalation_reasoning', '')}")
        return f"**Orchestrator** - no escalation: {upd.get('escalation_reasoning', '')}"
    if node == "escalate_evidence":
        parts = []
        if "ensemble_probs" in upd:
            parts.append("test-time augmentation re-scored the image")
        fe = upd.get("fellow_eye_prediction")
        if fe:
            parts.append(f"fellow eye predicted {fe['grade_label']} ({fe['confidence'] * 100:.0f}%)")
        return "**Evidence gathering** - " + ("; ".join(parts) if parts else "no new evidence available")
    if node == "referral":
        return (f"**Referral** - calibrated grade {og.GRADE_LABELS[upd['predicted_grade']]} at "
                f"{upd['calibrated_confidence'] * 100:.1f}% -> `{upd['triage_decision']}`")
    if node == "structuring":
        return "**Structuring** - structured record built"
    if node == "saliency":
        s = upd["saliency"]
        agree = "agree" if s["attention_agreement"] else "do not agree"
        return (f"**Saliency** - ResNet-50: {s['resnet_quadrant']}, ViT-B/16: {s['vit_quadrant']} "
                f"(backbones {agree})")
    if node == "explanation":
        return "**Explanation Agent** - retrieved guideline text and generated the explanation"
    if node == "verifier":
        if upd["citation_ok"]:
            return "**Verifier** - passed (cited, ends cleanly)"
        if upd["explanation_retries"] < og.MAX_EXPLANATION_RETRIES:
            return "**Verifier** - failed, regenerating"
        return "**Verifier** - failed, retry limit reached; keeping last output"
    return f"**{node}**"


def render_trace(lines):
    return "\n".join(f"- {line}" for line in lines)


def run_pipeline(primary_img, fellow_img, eye_side):
    if primary_img is None:
        yield ("Please upload a fundus photograph first.", "", None, "", None, None, None)
        return

    with RUN_LOCK, tempfile.TemporaryDirectory() as tmpdir:
        try:
            primary = preprocess_fundus(primary_img)
            other = "right" if eye_side == "left" else "left"

            image_id = f"upload_{eye_side}" if fellow_img is not None else "upload"
            primary_path = os.path.join(tmpdir, image_id + ".png")
            primary.save(primary_path)

            # Point the graph's fellow-eye machinery at this session's temp folder.
            patient_id = "gui_session"
            og.IMAGE_FOLDER = tmpdir
            og.IMAGE_EXT = ".png"
            og.FELLOW_LOOKUP = {patient_id: {"left": None, "right": None}}
            if fellow_img is not None:
                fellow_name = f"upload_{other}"
                preprocess_fundus(fellow_img).save(os.path.join(tmpdir, fellow_name + ".png"))
                og.FELLOW_LOOKUP[patient_id][eye_side] = image_id
                og.FELLOW_LOOKUP[patient_id][other] = fellow_name

            initial_state = {
                "image_id": image_id,
                "image_path": primary_path,
                "patient_id": patient_id,
                "true_label": None,
            }

            state = dict(initial_state)
            trace, seen = [], set()
            yield ("Running...", "", None, "", primary, None, None)

            for step in APP.stream(initial_state, stream_mode="updates"):
                for node, upd in step.items():
                    if node == "referral" and "orchestrator" not in seen:
                        trace.append("**Orchestrator** - not consulted (wide margin, or both candidate "
                                     "grades lead to the same triage outcome)")
                    seen.add(node)
                    state.update(upd)
                    trace.append(describe_update(node, upd))
                yield (render_trace(trace), "", None, "", primary, None, None)

            grade = state["predicted_grade"]
            label = og.GRADE_LABELS[grade]
            reason = state["structured_record"]["referral"]["reason"]
            summary = (
                f"### Result\n"
                f"**{label}** (grade {grade}) - calibrated confidence "
                f"{state['calibrated_confidence'] * 100:.1f}%\n\n"
                f"**Triage decision:** `{state['triage_decision']}`\n\n{reason}"
            )

            explanation = f"### Explanation\n{state['why_explanation']}\n\n**Next steps:** {state['timeframe_fact']}"
            fe = state.get("fellow_eye_prediction")
            if fe:
                explanation += (f"\n\n**Fellow eye:** predicted {fe['grade_label']} "
                                f"({fe['confidence'] * 100:.0f}% confidence).")
            if not state.get("citation_ok"):
                explanation += ("\n\n> Warning: this explanation did not pass the automated "
                                "citation/completeness check - treat it with extra caution.")

            resnet_overlay, vit_overlay = compute_saliency_maps(primary, grade)

            yield (render_trace(trace), summary, state["class_probabilities"], explanation,
                   primary, resnet_overlay, vit_overlay)

        except ValueError as e:
            yield (f"Could not process the upload: {e}", "", None, "", None, None, None)
        except Exception as e:
            traceback.print_exc()   # full traceback in the terminal running the app
            yield (f"Pipeline failed: `{e}`\n\nInput types: main={type(primary_img).__name__}, "
                   f"fellow={type(fellow_img).__name__}. Full traceback is in the terminal.",
                   "", None, "", None, None, None)


def build_demo():
    with gr.Blocks(title="Agentic DR Screening - Demo") as demo:
        gr.Markdown("# Agentic AI for Diabetic Retinopathy - Demo")
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column(scale=1):
                primary = gr.Image(type="pil", label="Fundus photograph")
                fellow = gr.Image(type="pil", label="Fellow-eye photograph (optional)")
                eye_side = gr.Radio(
                    ["left", "right"], value="left",
                    label="Which eye is the main photograph? (only used if a fellow-eye photo is provided)",
                )
                run_btn = gr.Button("Run pipeline", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### Pipeline trace")
                trace = gr.Markdown()
                summary = gr.Markdown()
                probs = gr.Label(label="Calibrated grade probabilities", num_top_classes=5)

        explanation = gr.Markdown()

        with gr.Row():
            pre_img = gr.Image(label="Preprocessed input", type="pil", interactive=False)
            resnet_img = gr.Image(label="ResNet-50 Grad-CAM (class-specific)", type="pil", interactive=False)
            vit_img = gr.Image(label="ViT-B/16 attention rollout (not class-specific)", type="pil",
                               interactive=False)

        run_btn.click(
            run_pipeline,
            inputs=[primary, fellow, eye_side],
            outputs=[trace, summary, probs, explanation, pre_img, resnet_img, vit_img],
        )
    return demo


def main():
    global APP
    print("Loading models, vector store and graph (one-time)...")
    og.setup()
    APP = og.build_graph()

    demo = build_demo()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()