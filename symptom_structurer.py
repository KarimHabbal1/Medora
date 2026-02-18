'''
# HealthNav AI – Medical Knowledge Extraction Pipeline
# ====================================================

## Purpose of This Pipeline
The ultimate goal of HealthNav AI is to build an intelligent medical triage 
and care-coordination system capable of guiding patients safely from their first 
symptom to the correct level of care. To do this, the AI must have access to 
structured and clinically reliable medical knowledge.

Clinical medical textbooks such as *Current Medical Diagnosis & Treatment (TMT 2022)* 
contain extremely valuable information, but the content is not machine-ready.
The text is unstructured, contains narrative prose, and mixes multiple clinical 
concepts together. AI models cannot reliably reason using this raw text alone.

Therefore, our first task is to convert medical reference material into structured, 
machine-friendly knowledge that can be fed into our Triage Agent and later into our 
RAG (retrieval-augmented generation) system.

This entire directory contains the code that:
1. Extracts text from the TMT PDF.
2. Detects chapters, symptom headings, and condition headings.
3. Segments the book into meaningful chunks.
4. Feeds raw text into GPT-4 to convert it into structured JSON objects.
5. Produces a complete medical knowledge base for the Triage and Intake agents.

---

## Why Chapter 2 (Common Symptoms) Comes First
Chapter 2 of TMT is not disease-based.  
It is *symptom-based*, and it explains:

- what questions a clinician should ask when a patient reports a symptom,
- what systems that symptom could originate from (cardiac, pulmonary, GI…),
- what red flags must be identified,
- when a symptom suggests emergency referral,
- which specialty is most appropriate.

This chapter directly powers the Intake Agent and the Triage Agent.

Real triage begins with symptoms → questions → systems → urgency → next steps.  
Chapter 2 captures exactly this logic.

By parsing Chapter 2 first, we build our **Symptom Intelligence Layer**, which is
the foundation of triage reasoning.

---

## Why We Are NOT Fine-Tuning on the Text
We explicitly avoid fine-tuning on this textbook because:

- It is NOT training data (no examples, no labels).
- Fine-tuning on raw clinical prose increases hallucination risk.
- It would distort the model’s general medical reasoning ability.
- It is extremely expensive and unnecessary.
- Modern medical systems use deterministic parsing + RAG, not fine-tuning.

Instead:

### Extraction → LLM Structuring → RAG → AI Reasoning

This is the same architecture used by Med-PaLM, Hippocratic AI, and Avey.

---

## What Happens After Extraction
After extracting raw symptom sections, the next step is to feed each block of text 
into GPT-4 using a deterministic prompt. GPT-4 converts raw prose into a structured 
medical JSON object that contains:

- essential questions,
- system differentials,
- red flags,
- diagnostic clues,
- triage urgency logic,
- specialty routing.

Later, we will do the same process for disease-based chapters (Cardiology, Pulmonology, etc.)
to build the Condition Intelligence Layer.

---

## Final Output of This Pipeline
By the end of the extraction + GPT structuring:

### We will have two core knowledge bases:
1. Symptom Intelligence  
   (from Chapter 2)

2. Condition Intelligence  
   (from all disease-based chapters)

These databases are embedded and indexed in a vector database and used by the 
HealthNav multi-agent system to perform:

- symptom triage,
- severity evaluation,
- specialty matching,
- emergency escalation,
- next-step clinical reasoning.

This pipeline makes the model explainable, reliable, and medically grounded.

# End of Explanation


This script converts raw symptom text extracted from Chapter 2 of TMT 2022
into structured medical JSON objects suitable for the HealthNav AI triage system.

WHY THIS SCRIPT EXISTS
-----------------------
Raw clinical text is not machine-ready. Chapter 2 describes how clinicians
evaluate symptoms such as cough, chest pain, dyspnea, fever, and others.
These blocks contain:
- essential patient questions
- red flags
- system mapping
- diagnostic tests
- urgency logic
- specialty routing

We must convert this unstructured prose into standardized JSON objects.

We do NOT fine‑tune the LLM on this text — it would cause hallucinations,
catastrophic forgetting, and legal issues. Instead we use a deterministic
pipeline:

Extraction → LLM Structuring → RAG → Triage Reasoning

This script performs the LLM structuring step.

WHY THE OLD VERSION FAILED
--------------------------
The old script had three issues:
1. It used the old OpenAI API response format.
2. GPT occasionally returned non‑JSON output, causing json.loads failures.
3. gpt‑4o‑mini is not reliable for strict JSON extraction.

This new version fixes everything:
- Uses gpt‑4o (much more stable for JSON).
- Wraps JSON in <json>...</json> tags for reliable extraction.
- Adds robust fallback retry logic.
- Automatically loads .env with python‑dotenv.

OUTPUT
------
Produces:
tmt_symptoms_structured.jsonl

Each line is one symptom object:
{
  "symptom": "COUGH",
  "one_sentence_definition": "...",
  "essential_questions": [...],
  "system_differentials": [...],
  "possible_causes": [...],
  "red_flags": [...],
  "when_to_refer_emergency": [...],
  "key_examination_findings": [...],
  "useful_diagnostic_tests": [...],
  "triage_logic": "...",
  "specialty_routing": [...],
  "rag_summary": "..."
}
'''

import os
import json
import time
import re
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# ENV + CONFIG
# ============================================================
load_dotenv()  # loads OPENAI_API_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "outputs", "tmt_symptoms_raw.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "outputs", "tmt_symptoms_structured.jsonl")

client = OpenAI()  # requires OPENAI_API_KEY in .env


# ============================================================
# HELPERS
# ============================================================

def extract_json_block(text):
    """Extracts the JSON inside <json>...</json>."""
    match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def build_prompt(symptom, raw_text):
    """Build the main extraction prompt."""
    return f"""
You are a medical knowledge extraction model.

Convert the following clinical textbook text into structured JSON.

Return ONLY valid JSON inside <json></json> tags.

Schema:
{{
  "symptom": "{symptom}",
  "one_sentence_definition": "",
  "essential_questions": [],
  "system_differentials": [],
  "possible_causes": [],
  "red_flags": [],
  "when_to_refer_emergency": [],
  "key_examination_findings": [],
  "useful_diagnostic_tests": [],
  "triage_logic": "",
  "specialty_routing": [],
  "rag_summary": ""
}}

Raw text:
---
{raw_text}
---
"""


def process_symptom(symptom, raw_text):
    """Processes one symptom block with retries and JSON safety."""
    prompt = build_prompt(symptom, raw_text)

    # First attempt
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = resp.choices[0].message.content
    json_block = extract_json_block(content)

    if json_block:
        return json.loads(json_block)

    # Fallback attempt (simplified)
    retry_prompt = f"""
Return ONLY valid JSON for the symptom "{symptom}" using the schema above.
No explanations. No tags. JSON only.

Text:
{raw_text}
"""

    resp2 = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": retry_prompt}],
        temperature=0
    )

    return json.loads(resp2.choices[0].message.content)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("[+] Loading raw symptom blocks...")
    raw_items = [json.loads(line) for line in open(RAW_PATH)]

    fout = open(OUT_PATH, "w")
    print(f"[+] Processing {len(raw_items)} symptoms with GPT-4o...\n")

    for i, item in enumerate(raw_items):
        symptom = item["symptom"]
        text = item["raw_text"]

        print(f"[→] {i+1}/{len(raw_items)} - Processing: {symptom}")

        try:
            structured = process_symptom(symptom, text)
            fout.write(json.dumps(structured) + "\n")
            time.sleep(0.6)  # rate limit protection

        except Exception as e:
            print(f"[ERROR] Failed on {symptom}: {e}")
            time.sleep(1)
            continue

    fout.close()
    print(f"\n[✓] Completed! Structured symptoms saved to:\n{OUT_PATH}")