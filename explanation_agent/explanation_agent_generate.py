"""
Explanation Agent - Generation.

Reads the Structuring Agent's per-image structured records and the
saliency manifest (generate_saliency.py), retrieves relevant clinical
guideline passages from the vector store built by
explanation_agent_ingest.py, and generates a grounded, physician-oriented
WHY + NEXT-STEPS explanation per image via a local, free LLM (Ollama).

--- Architecture: deterministic facts vs. genuine LLM synthesis ---
Earlier versions of this script asked the LLM to either (a) write a free
paragraph grounded in retrieved context, or (b) extract a single
referral-timeframe fact from that context. Both turned out to be the wrong
job for an LLM: (a) rewarded restating facts already in the structured
record plus padding with irrelevant retrieved chunks; (b) is a pure
function of (grade, resource_setting) - a deterministic table lookup, not
a generation task, and running it through an LLM added hallucination risk
for zero benefit over a Python dict.

This version splits the explanation into two parts, computed differently:
  1. DETERMINISTIC FACTS (code, not LLM): the referral/re-examination
     timeframe, via GRADE_TIMEFRAMES below - always correct, zero latency,
     zero hallucination risk, exactly in the spirit of the Structuring
     Agent's own "deterministic, traceable facts" design.
  2. LLM SYNTHESIS (genuinely generative, varies per image): WHY this
     grade is plausible, using (a) WHERE each backbone's attention
     concentrated on this specific image and whether the two
     independently-trained backbones agree (from the saliency manifest -
     this is the one thing that is genuinely different per image, not
     just per grade), (b) retrieved guideline text describing what
     clinically characterizes this grade, and (c) if a second grade has
     meaningfully high probability, what that would mean.

--- Hard boundary: location evidence only, never lesion claims ---
Neither Grad-CAM nor attention rollout detects specific lesions
(microaneurysm, hemorrhage, exudate, etc.) - nothing in this pipeline
does. The prompt below explicitly forbids the LLM from claiming any
specific lesion was detected; it may only describe WHERE attention
concentrated and WHETHER the two backbones agree. This boundary is
non-negotiable - crossing it would fabricate evidence the pipeline never
produced, which is a worse failure than the padding problem this version
fixes.

--- GRADE_TIMEFRAMES provenance ---
Transcribed from the ICO Guidelines for Diabetic Eye Care (2017), p.8,
Tables 2a (high resource) and 2b (low-/intermediate resource), as
extracted via explanation_agent_ingest.py's retrieval in this project's
development. Spot-check these against the source PDF directly before
relying on them for thesis results - they were transcribed from retrieved
chunk text, not independently re-verified against the primary document.

--- Vector store / embedding config MUST match explanation_agent_ingest.py exactly ---
Same embedding model, same QUERY_INSTRUCTION prefix, same persisted
directory - see that script's docstring for why this matters.

--- LLM: local via Ollama (free) ---
Requires Ollama installed and running, plus the model pulled once:
    ollama pull llama3.1

Install dependencies first:
    pip install langchain langchain-community langchain-huggingface langchain-chroma chromadb sentence-transformers ollama pandas

Usage:
    python explanation_agent_generate.py
"""

import os
import re
import json
from pathlib import Path

import pandas as pd
import ollama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------------- CONFIG ----------------
basepath = Path(__file__).parent

# Must match explanation_agent_ingest.py exactly.
VECTOR_STORE_DIR = basepath / "vector_store_bge"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this question for retrieving supporting documents: "
TOP_K = 4

# Inputs
STRUCTURED_RECORDS_JSON_PATH = basepath.parent / "structuring_agent" / "structured_records.json"
SALIENCY_MANIFEST_PATH = basepath.parent / "saliency_script" / "saliency_manifest.csv"

OUTPUT_JSON_PATH = basepath / "explanations_updated.json"
OUTPUT_CSV_PATH = basepath / "explanations_updated.csv"

OLLAMA_MODEL_NAME = "llama3.1"
OLLAMA_TEMPERATURE = 0.2
OLLAMA_MAX_TOKENS = 220   # raised from 180 - verbatim guideline quotes were running out of
                          # budget mid-sentence on the 3rd sentence (see ends_cleanly check below)

SAVE_EVERY = 25

# ICO Table 2a (high resource) / 2b (low-/intermediate resource) - see
# provenance note in the module docstring above.
RESOURCE_SETTING = "high"   # "high" or "low"

GRADE_TIMEFRAMES = {
    "high": {
        0: ("1-2 years", False),
        1: ("6-12 months", False),
        2: ("3-6 months", True),
        3: ("<3 months", True),
        4: ("<1 month", True),
    },
    "low": {
        0: ("1-2 years", False),
        1: ("1-2 years", False),
        2: ("6-12 months", True),
        3: ("<3 months", True),
        4: ("<1 month", True),
    },
}
TIMEFRAME_CITATION = "(ICO Guidelines for Diabetic Eye Care, 2017, p.8)"

# A second grade's probability must clear this to be worth mentioning -
# below this it's noise, not a meaningful alternate possibility.
ALT_GRADE_THRESHOLD = 0.15

GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR (PDR)",
}

# Matches both the requested "(SOURCE, p.N)" shape AND the variant the
# model actually tends to produce - "SOURCE ... (YEAR, p.N)", source name
# outside the parens, year+page inside. Two rounds of explicit formatting
# instructions didn't change the model's habit, and the latter form is
# just as traceable (source name + page number both present, close
# together) - so detect what's actually traceable rather than keep fighting
# the exact punctuation shape.
CITATION_PATTERN = re.compile(r"(?:ICO|AAO)[^.]{0,100}?p\.\s?\d+", re.IGNORECASE)
NO_CRITERIA_FOUND_TEXT = "No additional guideline-specific clinical criteria found in the retrieved context."
# -----------------------------------------


def build_timeframe_sentence(predicted_grade):
    """Deterministic - computed in code, never by the LLM. See module
    docstring for why."""
    interval, referral_required = GRADE_TIMEFRAMES[RESOURCE_SETTING][predicted_grade]
    req_text = "referral is required" if referral_required else "referral is not required"
    return (
        f"Per ICO guidelines ({RESOURCE_SETTING}-resource setting): recommended "
        f"re-examination/referral interval is {interval}; {req_text} "
        f"{TIMEFRAME_CITATION}."
    )


def build_alt_grade_note(class_probabilities, predicted_grade):
    """Returns a short description of the next-most-probable grade if it
    clears ALT_GRADE_THRESHOLD, else None. This is the one piece of
    genuinely per-image-varying information beyond attention location -
    two images with the same predicted grade can have very different
    runner-up probabilities, and that's clinically meaningful."""
    sorted_grades = sorted(class_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    for label, prob in sorted_grades:
        grade_num = [k for k, v in GRADE_LABELS.items() if v == label][0]
        if grade_num != predicted_grade and prob >= ALT_GRADE_THRESHOLD:
            return f"{label} (grade {grade_num}) at {prob * 100:.1f}% probability"
    return None


def build_retrieval_query(record):
    """Retrieves clinical DEFINING FEATURES for this grade, not referral
    timeframes (those are now deterministic, see build_timeframe_sentence)."""
    grade_label = record["diagnosis"]["grade_label"]
    return f"{grade_label} diabetic retinopathy clinical features and diagnostic criteria"


def format_context(docs):
    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "?")
        lines.append(f"[{i}] ({source}, p.{page})\n{doc.page_content.strip()}")
    return "\n\n".join(lines)


def build_prompt(record, context_block, saliency_row, alt_grade_note):
    diagnosis = record["diagnosis"]

    if saliency_row is not None:
        resnet_focal = "focal" if saliency_row["resnet_concentration"] >= 0.3 else "diffuse"
        vit_focal = "focal" if saliency_row["vit_concentration"] >= 0.3 else "diffuse"
        if saliency_row["resnet_quadrant"] == saliency_row["vit_quadrant"]:
            attention_desc = (
                f"Both backbones concentrated attention in the {saliency_row['resnet_quadrant']} "
                f"region of the image (ResNet-50: {resnet_focal}, concentration {saliency_row['resnet_concentration']:.2f}; "
                f"ViT-B/16: {vit_focal}, concentration {saliency_row['vit_concentration']:.2f})."
            )
        else:
            attention_desc = (
                f"ResNet-50's attention concentrated in the {saliency_row['resnet_quadrant']} region "
                f"({resnet_focal}, concentration {saliency_row['resnet_concentration']:.2f}); "
                f"ViT-B/16's attention concentrated in the {saliency_row['vit_quadrant']} region "
                f"({vit_focal}, concentration {saliency_row['vit_concentration']:.2f}) - the two backbones "
                f"do not agree on a single region."
            )
    else:
        attention_desc = "No saliency/attention data available for this image."

    alt_grade_text = (
        f"The next most probable grade was {alt_grade_note}."
        if alt_grade_note else
        "No other grade had a meaningfully high probability."
    )

    system_prompt = (
        "You are explaining WHY an automated diabetic retinopathy (DR) grade is "
        "plausible, for an ophthalmologist reviewing the case. You have exactly "
        "three kinds of evidence, and you must not exceed them:\n\n"
        "1. ATTENTION LOCATION: where two independently-trained model backbones "
        "concentrated their attention on this image, whether they agree, and how "
        "focal vs. diffuse each one's attention was. This tells you WHERE the "
        "models focused and HOW CONCENTRATED that focus was - NOTHING ELSE. "
        "FORBIDDEN: any sentence implying the models 'detected', 'identified', "
        "'saw', or 'found' a feature, lesion, or finding. 'Detecting features "
        "characteristic of X' is FORBIDDEN even if X is the correct grade - "
        "that claims feature detection, which does not exist in this pipeline. "
        "You may ONLY describe location, agreement, and focal/diffuse pattern - "
        "never infer or imply WHY the model attended there.\n"
        "2. RETRIEVED GUIDELINE TEXT: excerpts describing what clinically "
        "characterizes this grade in general. Use ONLY the sentence(s) that "
        "explicitly name THIS grade - retrieved text is a glossary with multiple "
        "DIFFERENT terms defined back-to-back (e.g. 'Moderate NPDR' and 'Moderate "
        "visual loss' are unrelated terms that may appear adjacent); never use a "
        "definition unless it explicitly names this exact grade. Cite any claim "
        "in the exact form (SOURCE, p.N) - e.g. (AAO Preferred Practice Pattern, "
        "p.64). PARAPHRASE rather than quoting at length - a short paraphrase with a "
        "citation is preferred over a long verbatim quote, which risks running out of "
        "space mid-sentence. If no excerpt explicitly defines this grade, output exactly: "
        f"'{NO_CRITERIA_FOUND_TEXT}'\n"
        "3. ALTERNATE GRADE: the next most probable grade, if any was meaningfully "
        "likely, and what it would mean clinically if that were the true grade "
        "instead. Do not conclude by reasserting the predicted grade is correct - "
        "state the alternate grade's implication and stop.\n\n"
        "Do not restate the predicted grade name, the confidence percentage, or "
        "any referral timeframe - those are already known and handled elsewhere. "
        "HARD LIMIT: output at most 3 sentences total, one paragraph, no line "
        "breaks, no headers, no bullet points. If you cannot say something "
        "meaningful in 3 sentences, say less, not more."
    )

    user_prompt = (
        f"Predicted grade (already known, do not restate): {diagnosis['grade_label']}\n\n"
        f"Attention evidence: {attention_desc}\n\n"
        f"Alternate grade: {alt_grade_text}\n\n"
        f"Retrieved guideline excerpts:\n{context_block}\n\n"
        f"Write the WHY explanation now."
    )

    return system_prompt, user_prompt


def generate_text(system_prompt, user_prompt):
    response = ollama.chat(
        model=OLLAMA_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": OLLAMA_TEMPERATURE, "num_predict": OLLAMA_MAX_TOKENS},
    )
    return response["message"]["content"].strip()


def load_existing_results(path):
    if not os.path.exists(path):
        return [], set()
    with open(path, "r") as f:
        results = json.load(f)
    done_ids = {r["image_id"] for r in results}
    print(f"Resuming: {len(done_ids)} explanations already generated, will be skipped.")
    return results, done_ids


def save_results(results, json_path, csv_path):
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    flat_rows = [
        {
            "image_id": r["image_id"],
            "patient_id": r.get("patient_id"),
            "predicted_grade": r["predicted_grade"],
            "grade_label": r["grade_label"],
            "triage_decision": r["triage_decision"],
            "why_explanation": r["why_explanation"],
            "timeframe_fact": r["timeframe_fact"],
            "has_citation": r["has_citation"],
            "no_criteria_found": r.get("no_criteria_found", False),
        }
        for r in results
    ]
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Loading vector store from: {VECTOR_STORE_DIR}")
    vector_store = Chroma(persist_directory=str(VECTOR_STORE_DIR), embedding_function=embeddings)

    with open(STRUCTURED_RECORDS_JSON_PATH, "r") as f:
        structured_records = json.load(f)
    print(f"Loaded {len(structured_records)} structured records.")

    saliency_df = pd.read_csv(SALIENCY_MANIFEST_PATH)
    saliency_by_image = {row["image"]: row for _, row in saliency_df.iterrows()}
    print(f"Loaded saliency data for {len(saliency_by_image)} images.")

    results, done_ids = load_existing_results(OUTPUT_JSON_PATH)

    no_citation_count = 0
    newly_processed = 0

    for record in structured_records:
        image_id = record["image_id"]
        if image_id in done_ids:
            continue

        predicted_grade = record["diagnosis"]["predicted_grade"]
        class_probabilities = record["diagnosis"]["class_probabilities"]

        timeframe_fact = build_timeframe_sentence(predicted_grade)
        alt_grade_note = build_alt_grade_note(class_probabilities, predicted_grade)
        saliency_row = saliency_by_image.get(image_id)

        query = build_retrieval_query(record)
        docs = vector_store.similarity_search(QUERY_INSTRUCTION + query, k=TOP_K)
        context_block = format_context(docs)

        system_prompt, user_prompt = build_prompt(record, context_block, saliency_row, alt_grade_note)

        try:
            why_explanation = generate_text(system_prompt, user_prompt)
        except Exception as e:
            print(f"  WARNING: generation failed for {image_id} ({e}) - skipping, will retry next run.")
            continue

        no_criteria_found = why_explanation.strip().startswith("No additional guideline-specific clinical criteria")
        has_citation = bool(CITATION_PATTERN.search(why_explanation))
        # Catches generation cut off mid-sentence by the token budget (seen in practice:
        # a quote truncated to "...standard photograph" with no closing punctuation) -
        # the citation regex alone can pass this if an EARLIER sentence happened to cite,
        # even though the response is actually malformed.
        ends_cleanly = bool(re.search(r'[.!?]"?$', why_explanation.strip()))
        if (not has_citation and not no_criteria_found) or not ends_cleanly:
            no_citation_count += 1

        results.append({
            "image_id": image_id,
            "patient_id": record.get("patient_id"),
            "predicted_grade": predicted_grade,
            "grade_label": record["diagnosis"]["grade_label"],
            "triage_decision": record["referral"]["decision"],
            "retrieval_query": query,
            "retrieved_context": context_block,
            "attention_evidence": saliency_row.to_dict() if saliency_row is not None else None,
            "alt_grade_note": alt_grade_note,
            "why_explanation": why_explanation,
            "timeframe_fact": timeframe_fact,
            "has_citation": has_citation,
            "no_criteria_found": no_criteria_found,
        })
        newly_processed += 1

        if newly_processed % SAVE_EVERY == 0:
            save_results(results, OUTPUT_JSON_PATH, OUTPUT_CSV_PATH)
            print(f"  ...{newly_processed} new explanations generated this run (saved checkpoint).")

    save_results(results, OUTPUT_JSON_PATH, OUTPUT_CSV_PATH)

    print(f"\nDone. {newly_processed} new explanations generated this run.")
    print(f"Total explanations: {len(results)}")
    print(f"WHY-explanations making a claim WITHOUT a citation: {no_citation_count} "
          f"({no_citation_count / max(newly_processed, 1) * 100:.1f}% of this run) - "
          f"review these for ungrounded/hallucinated claims.")
    print(f"Saved to {OUTPUT_JSON_PATH} and {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()