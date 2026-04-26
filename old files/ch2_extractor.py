"""
HealthNav AI – Chapter 2 Symptom Extractor (TMT 2022)
=====================================================

WHAT THIS SCRIPT DOES
---------------------
This script extracts ONLY *Chapter 2: Common Symptoms* from the CURRENT
Medical Diagnosis & Treatment (TMT 2022) textbook. Chapter 2 is critical
to the triage pipeline because it contains symptom-based clinical 
reasoning information such as:

 • What questions to ask when a patient presents with a symptom
 • What systems (cardiac, respiratory, GI...) each symptom may relate to
 • Red flags and urgency indicators
 • Context clues that help distinguish dangerous vs. benign conditions

Chapter 2 is NOT organized by disease. It is organized by SYMPTOM.
This chapter is therefore the foundation of our “Symptom Intelligence Layer.”

WHY WE DO NOT FINE-TUNE ON CHAPTER 2
------------------------------------
Chapter 2 is not suitable for fine-tuning because:
 • It is not labeled training data—it's reference text.
 • It's too small (30–40 pages).
 • LLMs do not learn clinical reasoning by reading a chapter; they need structured examples.
 • Fine-tuning on raw clinical prose increases hallucination risk and causes 
   catastrophic forgetting of general medical knowledge.
 • Retrieval (RAG) is safer, cheaper, and more controllable.

Therefore:
The correct pipeline is:
   Deterministic parsing → LLM structuring → RAG retrieval

WHAT THIS SCRIPT PRODUCES
-------------------------
For each SYMPTOM (e.g., “COUGH”, “CHEST PAIN”, “DYSPNEA”), this script produces
a raw text chunk extracted directly from Chapter 2. Each chunk will later be
passed to an LLM to produce a structured JSON object:

{
    "symptom": "COUGH",
    "essential_questions": [...],
    "system_differentials": [...],
    "red_flags": [...],
    "triage_urgency_logic": "...",
    "specialty_routing": [...],
    "triage_summary": "300–400 token RAG summary"
}

CHAPTER 2 PAGE RANGE
--------------------
You provided the correct page boundaries:

Printed page numbers:
    Chapter 2 starts at: 15  
    Chapter 2 ends   at: 40  
    Chapter 3 starts at: 41  

PDF page numbers:
    Chapter 2 starts at: 35  
    Chapter 2 ends   at: 61  

IMPORTANT:
PyMuPDF uses ZERO-based page indexing internally, which is why we slice using:

    pages = pages[START_PAGE - 1 : END_PAGE]

DETECTING SYMPTOM HEADINGS
--------------------------
In Chapter 2, symptom headings have these characteristics:

 • FULL UPPERCASE
 • Short (1–4 words)
 • Not meta-headings like “GENERAL CONSIDERATIONS”
 • Not numbered or table headers

So we use heuristics:
    line.isupper()
    1 < len(line.split()) < 5
    no digits / punctuation
    not in a blacklist of unwanted headings

This reliably detects symptom names.

PIPELINE SUMMARY
----------------
1. Load PDF with PyMuPDF
2. Slice pages 35–61 to isolate Chapter 2
3. Normalize text (remove hyphen line breaks, normalize whitespace)
4. Scan lines and detect symptom headings
5. Accumulate text under each symptom
6. Produce a list of raw symptom chunks
7. Save chunks to 'outputs/tmt_symptoms_raw.jsonl'
8. (Next step in a separate script) Feed each raw chunk to GPT-4 to produce
   structured JSON for our triage system.

USAGE
-----
python chapter2_extract.py

Ensure:
 • The PDF exists at data/TMT_2022.pdf
 • You have a venv with PyMuPDF installed

This script is deterministic, safe, and the foundation of the AI triage system.
"""

import fitz
import re
import json
import os

# ============================================================
# CONFIGURATION
# ============================================================
PDF_PATH = "data/TMT_2022.pdf"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Printed pages: 15–40
# PDF pages: 35–61 (1-based)
START_PAGE = 35
END_PAGE = 61


# ============================================================
# HELPER – Clean up text
# ============================================================
def normalize_text(t: str) -> str:
    # Remove hyphenated line breaks: "symp-\ntom" -> "symptom"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # Normalize whitespace
    t = re.sub(r"\n+", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


# ============================================================
# HEADING DETECTION – Symptom names
# ============================================================
BLACKLIST = [
    "GENERAL CONSIDERATIONS",
    "DIFFERENTIAL DIAGNOSIS",
    "ESSENTIAL INQUIRIES",
    "APPROACH TO",
    "TABLE",
    "FIGURE",
    "KEY POINTS",
]

def is_symptom_heading(line: str) -> bool:
    line = line.strip()

    # Must be uppercase and not too long
    if not line.isupper():
        return False
    if len(line.split()) < 1 or len(line.split()) > 4:
        return False

    # Skip unwanted meta-headings
    for bad in BLACKLIST:
        if bad in line:
            return False

    # Avoid numeric table headers or figure labels
    if re.search(r"[0-9]", line):
        return False

    # No punctuation
    if re.search(r"[.:;/]", line):
        return False

    return True


# ============================================================
# MAIN EXTRACTION PIPELINE
# ============================================================
if __name__ == "__main__":
    print("[+] Loading PDF...")
    doc = fitz.open(PDF_PATH)

    # Extract only the chapter 2 pages
    pages = []
    for i in range(START_PAGE - 1, END_PAGE):
        text = doc[i].get_text("text")
        pages.append(text)
    print(f"[+] Extracted {len(pages)} pages for Chapter 2")

    # Concatenate text and normalize
    full_text = "\n".join(normalize_text(p) for p in pages)
    lines = full_text.split("\n")

    symptoms = []
    current_symptom = None
    buffer = []

    print("[+] Parsing symptom sections...")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect new symptom heading
        if is_symptom_heading(stripped):
            # Flush previous symptom
            if current_symptom and buffer:
                symptoms.append({
                    "symptom": current_symptom,
                    "raw_text": " ".join(buffer).strip()
                })
                buffer = []

            current_symptom = stripped
            continue

        # Accumulate text under current symptom
        if current_symptom:
            buffer.append(stripped)

    # Flush last symptom
    if current_symptom and buffer:
        symptoms.append({
            "symptom": current_symptom,
            "raw_text": " ".join(buffer).strip()
        })

    # Save raw symptom chunks
    out_path = os.path.join(OUT_DIR, "tmt_symptoms_raw.jsonl")
    with open(out_path, "w") as f:
        for s in symptoms:
            json.dump(s, f)
            f.write("\n")

    print(f"[+] Extracted {len(symptoms)} symptom sections.")
    print(f"[+] Saved to {out_path}")
    print("[✓] Chapter 2 raw symptom extraction complete.")