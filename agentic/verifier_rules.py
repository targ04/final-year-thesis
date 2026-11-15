"""
Deterministic rule checks for Explanation Agent output. Kept free of heavy imports so
evaluate_v2.py can re-score saved explanations offline.

Rules v2.1 (fixes two false positives in v2.0, which ran inside the first full v2 run):
  - the "backbone detected/found" rule no longer matches the word "model" alone - the
    lesion segmentation model may legitimately be said to have "found no regions"
    (v2.0's own lesion evidence text said "found", so copying it failed the check)
  - the lesion-count consistency check only applies to sentences that present the
    segmentation estimates (mention 'segmentation', 'estimat' or 'region'), and skips numbers
    that sit inside a quotation or follow a guideline qualifier ("at least", "more than",
    ">", ...), so guideline criteria (AAO mild NPDR: "at least 1 microaneurysm"; ICO severe
    NPDR: "more than 20 intraretinal haemorrhages in each of 4 quadrants") are not compared
    against measured counts, while "estimated 15 microaneurysm regions" still is
  - sentences ending in a closing quote ('...DR." The next') are split correctly
  - new rule: no ETDRS-scale criteria ('standard photograph', 'fields 4 through 7', 'ETDRS') - the
    labels use the International Classification (ICO Table 1), a different scale
"""

import re

MAX_SENTENCES = 4
CITATION_PATTERN = re.compile(r"(?:ICO|AAO)[^.]{0,100}?p\.\s?\d+", re.IGNORECASE)   # same as v1
NO_CRITERIA_PREFIX = "No additional guideline-specific clinical criteria"

SENT_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"\u201d']))\s+(?=[A-Z(\"\u201c])")
FORBIDDEN = {
    "restates the predicted grade": re.compile(r"\bpredicted grade\b", re.I),
    "uses 'indicating/suggests' (interprets instead of reporting)": re.compile(r"\b(indicat\w*|suggest\w*)\b", re.I),
    "uses ETDRS-scale criteria (labels use the International Classification, ICO Table 1)": re.compile(
        r"standard\s+photo|\bfields?\s+\d|\bETDRS\b", re.I),
    "claims a backbone detected something": re.compile(
        r"\b(backbones?|resnet[\w-]*|vit[\w/-]*|classifiers?)\b[^.]{0,40}\b(detect\w*|identif\w*|found|saw)\b", re.I),
}
LESION_WORDS = {"microaneurysm": ["MA"], "haemorrhage": ["HE"], "hard exudate": ["EX"], "exudate": ["EX", "SE"]}
NUM_LESION = re.compile(r"\b(\d+)\s+(?:[\w-]+\s+){0,3}?(microaneurysm|ha?emorrhage|hard exudate|exudate)", re.I)
ESTIMATE_SENTENCE = re.compile(r"segmentation|estimat|region", re.I)
QUALIFIER = re.compile(r"(\bat\s+least|\bmore\s+than|\bfewer\s+than|\bless\s+than|\bgreater\s+than|\bover|"
                       r"\bunder|\bexceeding|[<>\u2264\u2265]=?)\s*$", re.I)
QUOTE = re.compile(r'["\u201c\u201d]')

RULES_VERSION = "v2.1"


def check_explanation(text, lesion_counts, version=RULES_VERSION):
    """text: explanation. lesion_counts: {"MA": int, "HE": int, "EX": int, "SE": int}
    (missing types = 0). Returns a list of failures; empty list = pass.
    version="v2.0" reproduces the rules used during the first full run."""
    fails = []
    t = (text or "").strip()
    if not (t.startswith(NO_CRITERIA_PREFIX) or CITATION_PATTERN.search(t)):
        fails.append("no guideline citation in the form (SOURCE, p.N)")
    if not re.search(r'[.!?]"?\)?$', t):
        fails.append("does not end with a complete sentence")
    sentences = [s for s in SENT_SPLIT.split(t) if s.strip()]
    if len(sentences) > MAX_SENTENCES:
        fails.append(f"{len(sentences)} sentences (limit {MAX_SENTENCES})")

    forbidden = dict(FORBIDDEN)
    if version == "v2.0":
        forbidden["claims a backbone detected something"] = re.compile(
            r"\b(model|backbone|resnet|vit)[^.]{0,40}\b(detect\w*|identif\w*|found|saw)\b", re.I)
    for msg, pat in forbidden.items():
        if pat.search(t):
            fails.append(msg)

    for sent in sentences:
        if version != "v2.0" and not ESTIMATE_SENTENCE.search(sent):
            continue
        for m in NUM_LESION.finditer(sent):
            num, word = m.group(1), m.group(2)
            before = sent[:m.start()]
            after = sent[m.start(1) + len(num):]
            if version != "v2.0" and (QUALIFIER.search(before) or len(QUOTE.findall(before)) % 2 == 1
                                      or re.match(r"\s*(\+|or\s+more|or\s+fewer)", after, re.I)):
                continue
            word = word.lower().replace("hemorrhage", "haemorrhage")
            allowed = {int(lesion_counts.get(l, 0)) for l in LESION_WORDS.get(word, [])}
            if allowed and int(num) not in allowed:
                fails.append(f"states {num} {word}(s) but the lesion model measured {sorted(allowed)}")
    return fails
