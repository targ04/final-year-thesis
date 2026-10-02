"""
Explanation Agent - Corpus Ingestion.

Chunks and embeds the clinical guideline PDFs (ICO Guidelines for Diabetic
Eye Care, AAO Preferred Practice Pattern) into a persisted local vector
store. This is a ONE-TIME step (rerun only if you change the source PDFs)
- the actual explanation-generation script (built separately) queries this
vector store at inference time, it does not re-embed anything.

No model training happens here or anywhere in the Explanation Agent. This
script only prepares retrieval data; the LLM used later for generation is
used entirely zero-shot / off-the-shelf, grounded via retrieved context.

--- Chunking strategy ---
Clinical guideline PDFs are structured (numbered sections, named tables,
annexes) - splitting on those structural cues rather than arbitrary
character windows keeps each chunk semantically complete (e.g. never
splits a grade's referral rule or a table row in half). This uses
LangChain's RecursiveCharacterTextSplitter with separators ordered to
prefer breaking on structural markers first, falling back to paragraph/
sentence boundaries only when a section is too long to keep as one chunk.

--- Line-wrap de-hyphenation (fix) ---
pdfplumber extracts text exactly as it's laid out on the page, including
words wrapped across a line break with a hyphen (e.g. "photocoagu-\nlation").
Since "\n" is one of the chunking separators, a chunk boundary could
previously land INSIDE that hyphenated word, corrupting both the end of one
chunk and the start of the next (observed in practice: a PRP query's top
hit began mid-word with "lation at the severe NPDR stage..."). This is
fixed by rejoining "word-\nword" -> "wordword" before any other whitespace
cleanup, so hyphenated line-wraps never become chunk-boundary artifacts.

--- Chunk size (tuning) ---
CHUNK_SIZE/CHUNK_OVERLAP were increased slightly (800->1100, 120->180) to
reduce fragmentation of multi-row referral tables and multi-sentence
clinical rules, which were previously prone to being split across two
chunks and only half-retrieved.

--- Embedding model ---
Upgraded from all-MiniLM-L6-v2 (22M params, 384-dim) to BAAI/bge-base-en-v1.5
(109M params, 768-dim). MiniLM is fast but weak on dense, table-heavy
clinical semantic matching - this was visible in sanity-check retrievals
landing on title pages/section headers instead of actual guidance content.
BGE still runs fine on CPU or the A4000 and needs no API key.

IMPORTANT: BGE models are tuned to expect an instruction prefix on the
QUERY (not the passage) at retrieval time for best results - see
QUERY_INSTRUCTION below. The sanity-check block applies it; the separate
explanation-generation script that queries this vector store at inference
time MUST apply the same prefix to its retrieval queries, or retrieval
quality will regress back toward unprefixed-MiniLM-era behavior.

NOTE: the embedding dimension changed (384 -> 768), so any previously
persisted vector store at the old MiniLM path is NOT compatible with this
one. VECTOR_STORE_DIR below points at a new directory for that reason -
do not point it back at an old MiniLM-built store.

Install dependencies first:
    pip install langchain langchain-community chromadb sentence-transformers pdfplumber

Usage:
    python explanation_agent_ingest.py
"""

import os
import re
import pdfplumber

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path

# ---------------- CONFIG ----------------
# Point these at your actual PDF files.
basepath = Path(__file__).parent
GUIDELINE_PDFS = {
    "ICO Guidelines for Diabetic Eye Care (2017)": basepath  / "ICOGuidelinesforDiabeticEyeCare.pdf",
    "AAO Preferred Practice Pattern: Diabetic Retinopathy (2024)": basepath / "Diabetic_Retinopathy_PPP_2024.pdf",
}

# New directory - NOT the old MiniLM-built store. Embedding dimension
# changed (384 -> 768), so the two stores are not interchangeable.
VECTOR_STORE_DIR = basepath / "vector_store_bge"

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# BGE-specific: prefix applied to QUERIES ONLY at retrieval time (not to
# passages during ingestion). The downstream explanation-generation script
# must replicate this exact prefix on every retrieval query it issues.
QUERY_INSTRUCTION = "Represent this question for retrieving supporting documents: "

CHUNK_SIZE = 1100       # characters, not tokens - raised from 800 to reduce
                        # mid-rule/mid-table-row fragmentation
CHUNK_OVERLAP = 180     # raised from 120 proportionally with CHUNK_SIZE

# Separators in priority order: try splitting on structural markers first,
# only fall back to plain paragraph/sentence/word splits if a section is
# still too long. is_separator_regex=True lets us use the numbered-heading
# and "Table N." / "Annex" patterns actually used in these guideline docs.
SEPARATORS = [
    r"\n(?=\d\.\d\.\d\s)",     # numbered sub-subsection headers, e.g. "3.1.1 "
    r"\n(?=\d\.\d\s)",         # numbered subsection headers, e.g. "1.2 "
    r"\n(?=\d\s[A-Z])",        # numbered top-level section headers, e.g. "4 Treatment"
    r"\n(?=Table\s\d)",        # table titles, e.g. "Table 1."
    r"\n(?=Annex)",            # annex sections
    r"\n\n",                   # paragraph breaks
    r"\n",                     # line breaks
    r"(?<=\. )",               # sentence boundaries
    r" ",                      # last resort: word boundaries
]
# -----------------------------------------


def dehyphenate_line_wraps(text):
    """Rejoins words that pdfplumber extracted as hyphenated across a line
    break, e.g. "photocoagu-\nlation" -> "photocoagulation". Must run
    BEFORE whitespace/newline collapsing, since it specifically targets the
    "-\n" pattern that plain whitespace collapsing would otherwise leave
    behind as a chunk-boundary landmine ("\n" is a chunking separator, so
    an un-rejoined hyphenated word can be split in half across two chunks).

    Restricted to lowercase-to-lowercase joins (word-\nword, both lowercase)
    to avoid incorrectly merging genuine hyphenated compound terms at a
    line break followed by a new sentence/heading (e.g. "NPDR-\nTreatment").
    """
    return re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)


def looks_like_noise_page(text):
    """Flags table-of-contents and bibliography/reference-list pages for
    exclusion before chunking. Both add no grounding value and were
    observed polluting retrieval (e.g. a PRP query surfacing a numbered
    journal citation instead of actual treatment guidance).

    Per-line pattern matching was tried first and missed most bibliography
    pages, because reference entries wrap across multiple lines (author
    names on one line, journal/year on the next) - no single line matches
    often enough to trip a per-line ratio. This version instead looks for
    signatures that are reliable at the WHOLE-PAGE level:

      1. Bibliography: academic citations have a distinctive
         "year;volume:pages" or "year:ID" signature (e.g. "1998;352:837-53",
         "2015:CD010009") that essentially never appears in clinical
         guideline prose, plus repeated "et al." - both counted page-wide.
      2. Table of contents: numbered/titled lines ending in a lone page
         number, which may or may not use dot-leaders (this document's ToC
         does not).

    Appendices (e.g. "Major Study Results") are deliberately NOT filtered
    here - unlike ToC/bibliography they contain real clinical trial
    findings, just not always the first thing a referral-interval query
    should surface. That's a ranking/retrieval-quality question (addressed
    via chunking + embedding model), not a page-exclusion one.
    """
    citation_signature = re.compile(r"\d{4};\d+(\(\d+\))?:\d+", re.IGNORECASE)
    cochrane_signature = re.compile(r"\d{4}:[A-Z]{2,4}\d+")
    et_al_pattern = re.compile(r"\bet al\.", re.IGNORECASE)

    citation_hits = len(citation_signature.findall(text)) + len(cochrane_signature.findall(text))
    et_al_hits = len(et_al_pattern.findall(text))

    if citation_hits >= 3 or et_al_hits >= 3:
        return True

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 4:
        return False

    # ToC entries: a short heading followed (possibly much later on the
    # same line, with or without dot-leaders) by a lone 1-3 digit page
    # number at the very end.
    toc_line_pattern = re.compile(r"^\d*\.?\d*\.?\s*[A-Za-z].{3,80}\s\d{1,3}$")
    toc_hits = sum(1 for ln in lines if toc_line_pattern.search(ln))

    if (toc_hits / len(lines)) >= 0.3:
        return True

    return False


# Manual override: exact (1-indexed) page numbers to always exclude, for
# cases the automatic heuristic above still misses (e.g. a document's ToC
# or reference section that doesn't match either pattern). Fill in only if
# you spot bad pages still coming through after the automatic filter.
MANUAL_EXCLUDE_PAGES = {
    "ICO Guidelines for Diabetic Eye Care (2017)": {3, 4},  # confirmed table-of-contents pages
    "AAO Preferred Practice Pattern: Diabetic Retinopathy (2024)": set(),
}


def extract_pages(pdf_path, source_name):
    """Returns a list of (page_number, page_text) tuples, skipping pages
    that look like table-of-contents or bibliography/reference lists."""
    pages = []
    skipped_pages = []
    manual_excludes = MANUAL_EXCLUDE_PAGES.get(source_name, set())

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if i in manual_excludes:
                skipped_pages.append(i)
                continue

            text = page.extract_text() or ""
            # De-hyphenate line-wrapped words FIRST, before any whitespace
            # collapsing - see dehyphenate_line_wraps() docstring for why
            # ordering matters here.
            text = dehyphenate_line_wraps(text)
            # Collapse excessive whitespace but keep paragraph breaks
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            if not text.strip():
                continue
            if looks_like_noise_page(text):
                skipped_pages.append(i)
                continue
            pages.append((i, text))

    if skipped_pages:
        print(f"  Skipped {len(skipped_pages)} likely TOC/bibliography page(s): {skipped_pages}")

    return pages


def build_documents(guideline_pdfs):
    """Extracts and chunks every configured PDF, returning a flat list of
    LangChain Document objects with source/page metadata preserved."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        is_separator_regex=True,
    )

    all_documents = []

    for source_name, pdf_path in guideline_pdfs.items():
        if not os.path.exists(pdf_path):
            print(f"WARNING: {pdf_path} not found - skipping '{source_name}'.")
            continue

        print(f"Extracting: {source_name}")
        pages = extract_pages(pdf_path, source_name)
        print(f"  {len(pages)} pages with text extracted.")

        doc_chunk_count = 0
        for page_num, page_text in pages:
            chunks = splitter.split_text(page_text)
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) < 30:
                    # Skip near-empty fragments (e.g. stray headers/footers)
                    continue
                all_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": source_name,
                            "page": page_num,
                        },
                    )
                )
                doc_chunk_count += 1

        print(f"  {doc_chunk_count} chunks produced from '{source_name}'.")

    return all_documents


def main():
    documents = build_documents(GUIDELINE_PDFS)

    if not documents:
        print("No documents were produced - check GUIDELINE_PDFS paths. Aborting.")
        return

    print(f"\nTotal chunks across all guideline PDFs: {len(documents)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    # normalize_embeddings=True: BGE models are trained/evaluated with
    # cosine similarity on normalized vectors - leaving this off would
    # silently degrade ranking quality against Chroma's default distance.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Embedding and building Chroma vector store at: {VECTOR_STORE_DIR}")
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )
    # Note: no explicit .persist() call needed/available - current Chroma
    # (0.4+) auto-persists to disk as soon as persist_directory is set.
    print("Vector store built and persisted.")

    # --- Quick sanity check: run a couple of test retrievals ---
    # NOTE: the QUERY_INSTRUCTION prefix below is required for BGE models
    # to retrieve well - it is applied to the query text only, never to
    # the passages that were embedded above. Replicate this exact prefix
    # in the separate explanation-generation script.
    print("\n--- Sanity check retrievals ---")
    test_queries = [
        "Moderate NPDR referral and follow-up recommendations",
        "Proliferative diabetic retinopathy treatment panretinal photocoagulation",
        "No diabetic retinopathy screening interval",
    ]
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.similarity_search(QUERY_INSTRUCTION + query, k=4)
        for r in results:
            print(f"  [{r.metadata['source']}, p.{r.metadata['page']}] {r.page_content[:150]}...")


if __name__ == "__main__":
    main()