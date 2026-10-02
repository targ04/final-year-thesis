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

--- Embedding model ---
Uses a local sentence-transformers model (all-MiniLM-L6-v2) - free, small,
runs fine on CPU or the A4000, and needs no API key/external dependency at
retrieval time. Swap EMBEDDING_MODEL_NAME for a larger model later if
retrieval quality needs improvement.

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

VECTOR_STORE_DIR = basepath / "vector_store"  # where the embedded chunks will be persisted

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800        # characters, not tokens - keep chunks table/section-sized
CHUNK_OVERLAP = 120     # small overlap so a chunk boundary doesn't orphan context
 
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
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
 
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
    print("\n--- Sanity check retrievals ---")
    test_queries = [
        "Moderate NPDR referral and follow-up recommendations",
        "Proliferative diabetic retinopathy treatment panretinal photocoagulation",
        "No diabetic retinopathy screening interval",
    ]
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.similarity_search(query, k=4)
        for r in results:
            print(f"  [{r.metadata['source']}, p.{r.metadata['page']}] {r.page_content[:150]}...")
 
 
if __name__ == "__main__":
    main()
 