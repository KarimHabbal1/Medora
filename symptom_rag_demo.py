"""
HealthNav AI – Symptom RAG Query Demo (OSS Model via Ollama + ChromaDB)
========================================================================

This file is a **branch variant** of the HealthNav AI triage reasoning pipeline
that swaps the closed-source GPT-4o generation model for a locally-served
open-source model via Ollama, while keeping OpenAI embeddings for vector search
(the ChromaDB index was built with text-embedding-3-small and must stay consistent).

1. The **structured symptom knowledge base** (JSONL)
2. The **multi‑chunk RAG index** created by `symptom_indexer.py`
3. OpenAI **embeddings** for vector search  ← unchanged
4. ChromaDB as a **persistent local vector database**
5. A locally-served OSS model via Ollama as the **clinical reasoning engine**
6. A complete RAG loop:
       User symptoms → embedding → k‑NN search → retrieved chunks → OSS model reasoning

This script is NOT the final AI agent.  
It is a **diagnostic tool** to verify that the foundation of the medical RAG system
works exactly as expected before we build the full multi‑agent architecture.

---------------------------------------------------------------------------
BACKGROUND — WHY THIS RAG SYSTEM EXISTS
---------------------------------------------------------------------------

Clinical textbooks like *Current Medical Diagnosis & Treatment (TMT 2022)* contain
dense, unstructured prose. They are impossible for a model to “memorize” safely,
and you **must NOT fine‑tune** a foundational model on them because:

• Fine‑tuning causes hallucinations  
• It destroys the base model’s calibrated medical reasoning  
• It mixes authoritative + non‑authoritative text  
• It is expensive and unnecessary  
• Modern medical AI systems use **RAG**, not fine‑tuning

Instead, we transformed Chapter 2 (Common Symptoms) into **structured JSON**:
- definition
- essential questions
- system differentials
- possible causes
- red flags
- triage logic
- emergency criteria
- specialty routing
- RAG summary

Then we built a **multi‑chunk index**, meaning:
Each symptom is split into ~10 independent semantic chunks like:

    COUGH::red_flags  
    CHEST PAIN::triage_logic  
    DYSPNEA::possible_causes  
    FEVER::essential_questions  
    …

Each chunk is embedded independently and stored in Chroma with metadata:
    { "symptom": "COUGH", "section": "red_flags", "title": "RED FLAGS", ... }

This allows **fine‑grained retrieval** so that a user query like:
    "I have chest pain that radiates to my left arm"
returns chunks specifically related to chest‑pain red flags, ACS emergency criteria,
and triage thresholds — not the entire chapter.

This is how real clinical RAG systems like Med‑PaLM, Hippocratic, and Avey work.

---------------------------------------------------------------------------
HOW THIS DEMO SCRIPT WORKS (FULL PIPELINE)
---------------------------------------------------------------------------

1. **User enters symptoms**
   Example: “shortness of breath and chest tightness on exertion”

2. We embed the user text using:
   model="text-embedding-3-small"
   (small, fast, cheap, high‑quality for semantic search)

3. We query ChromaDB for the **top‑k = 8** closest chunks
   Top-k = 8 gives:
      - enough clinical breadth
      - avoids hallucination from unrelated chunks
      - ensures coverage of red flags + causes

4. We display which chunks were retrieved (for debugging + transparency)

5. We construct a consolidated **context block** that contains:
      === chest_pain::red_flags ===
      ...
      === dyspnea::triage_logic ===
      ...
      === cough::possible_causes ===
      ...
   Only content from the textbook.

6. We pass the context + user symptoms to the local OSS model with a strict instruction:
      "You MUST ground all reasoning strictly in the provided clinical chunks."

The model performs:
    • origin-system identification  
    • red-flag detection  
    • urgency classification  
    • next-step recommendations  
    • a short explanation grounded in the retrieved chunks  

7. The triage result is shown.

This is **Phase B completion** of the entire medical pipeline:
Extraction → JSON Structuring → Chunk Indexing → Vector DB → RAG Query

From here, we proceed to Phase C:
Building the full triage agent workflow.

---------------------------------------------------------------------------
WHY WE ARE TESTING AN OSS MODEL (THIS BRANCH)
---------------------------------------------------------------------------

Goals of this experiment:
• Evaluate whether a locally-served OSS model can match GPT-4o triage quality
• Eliminate API costs and latency from external calls during development
• Test full data-privacy — no patient data leaves the local machine
• Benchmark output structure, red-flag detection, and urgency classification

The generation model is configured via OLLAMA_MODEL in your .env file.
Embeddings still use OpenAI (text-embedding-3-small) to stay consistent
with the existing ChromaDB index.

---------------------------------------------------------------------------
TOP-K RETRIEVAL — WHY K = 8?
---------------------------------------------------------------------------

Top‑k controls how many chunks are fed to the model:
- Too small (k ≤ 3): might miss a red flag or system differential  
- Too large (k ≥ 15): adds noise and weakens grounding  

Empirical best practices (and your data density) place the sweet spot at:
        k = 8

At this granularity, each symptom has ~10 chunks, and retrieval typically returns
the right coverage: red flags, causes, triage logic, and exam findings.

---------------------------------------------------------------------------
WHAT THIS SCRIPT IS FOR
---------------------------------------------------------------------------

This file lets you test the system end-to-end:
- Does RAG retrieve correct content?
- Does the OSS model produce clinically safe decisions?
- Does the model stay grounded and not hallucinate?
- Do red flags trigger correctly?
- Does triage level match textbook criteria?

Once this is validated, we can build the **actual multi-agent architecture**:
- Intake Agent  
- Triage Agent  
- Logistics Agent  
- Insurance Agent  
- Reinforcement Learning feedback loop  

This RAG layer is their shared “medical memory.”

"""

import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_symptoms")
COLLECTION_NAME = "symptom_chunks"

# Retrieval parameter
TOP_K = 8  # clinically tuned

# OpenAI client — used only for embeddings (must match the index)
openai_client = OpenAI()

# Ollama client — used for LLM generation (local OSS model)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss-120b")

ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",  # required by the SDK but ignored by Ollama
)

# Chroma persistent client
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)


# ============================================================
# EMBEDDING UTIL
# ============================================================

def embed_query(text: str) -> List[float]:
    """Convert user text → embedding vector."""
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(user_text: str, k: int = TOP_K):
    """Return top-k most relevant clinical chunks."""
    query_vec = embed_query(user_text)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
    )

    # Flatten
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(ids, docs, metas))


# ============================================================
# LLM TRIAGE REASONING
# ============================================================

def generate_triage_response(user_text: str, retrieved_chunks):
    """Feed retrieved chunks + user text to the local OSS model for medical reasoning."""

    # Build context block
    context = ""
    for cid, doc, meta in retrieved_chunks:
        context += f"\n\n=== {cid} ({meta['title']}) ===\n{doc}"

    prompt = f"""
You are a clinical triage and reasoning model for HealthNav AI.
You MUST ground all reasoning strictly in the provided clinical chunks.

Do NOT hallucinate any diseases, tests, or red flags outside the retrieved context.

# USER SYMPTOMS
{user_text}

# RETRIEVED CLINICAL CONTEXT (from textbook, Chapter 2)
{context}

Using ONLY the above context:
1. Identify likely body-system origin(s)
2. List matching red flags (if any)
3. Determine urgency:
   - Home care
   - Primary care within 24–72 hours
   - Urgent care today
   - Emergency department
4. Provide recommendation for next step
5. Provide a SHORT explanation referencing the chunks

Return a clear, structured output.
"""

    resp = ollama_client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return resp.choices[0].message.content


# ============================================================
# DRIVER FUNCTION
# ============================================================

def run_demo():
    print("\n=== HealthNav AI Symptom RAG Demo ===\n")
    user_text = input("Describe your symptoms: ").strip()

    print("\n[1] Retrieving relevant clinical chunks...")
    chunks = retrieve_chunks(user_text, TOP_K)

    for cid, doc, meta in chunks:
        print(f" → {cid}  (symptom={meta['symptom']}, section={meta['section']})")

    print(f"\n[2] Generating triage reasoning using {OLLAMA_MODEL} (Ollama)...\n")
    answer = generate_triage_response(user_text, chunks)

    print("=== TRIAGE RESULT ===\n")
    print(answer)
    print("\n=======================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_demo()