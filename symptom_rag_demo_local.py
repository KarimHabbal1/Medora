"""
HealthNav AI – Symptom RAG Query Demo (GPT-4o + ChromaDB)
==========================================================

This file is the **first working prototype** of the HealthNav AI triage reasoning
pipeline. It connects everything we have built so far:

1. The **structured symptom knowledge base** (JSONL)
2. The **multi‑chunk RAG index** created by `symptom_indexer.py`
3. OpenAI **embeddings** for vector search
4. ChromaDB as a **persistent local vector database**
5. GPT‑4o as the **clinical reasoning engine**
6. A complete RAG loop:
       User symptoms → embedding → k‑NN search → retrieved chunks → GPT‑4o reasoning

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

6. We pass the context + user symptoms to GPT‑4o with a strict instruction:
      “You MUST ground all reasoning strictly in the provided clinical chunks.”

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
WHY WE CHOSE GPT‑4o FOR TRIAGE REASONING
---------------------------------------------------------------------------

GPT‑4o:
• Has excellent medical reasoning  
• Low latency (needed for real-time triage)  
• Supports long context for chunk aggregation  
• More deterministic than the 4o-mini models  
• Perfect for safe RAG-based clinical reasoning  

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
- Does GPT‑4o produce clinically safe decisions?
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
import requests
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import traceback
import sys
sys.excepthook = lambda exctype, value, tb: __import__("traceback").print_exception(exctype, value, tb)

# ============================================================
# CONFIG
# ============================================================
print("[DBG] RUNNING FILE:", __file__)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_symptoms")
COLLECTION_NAME = "symptom_chunks"

# Retrieval parameter
TOP_K = 8  # clinically tuned

# OpenAI API key (loaded from .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError("[ERROR] OPENAI_API_KEY not found in .env file.")
client = OpenAI()

# Chroma persistent client
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)


# ============================================================
# EMBEDDING UTIL
# ============================================================

def embed_query(text: str) -> List[float]:
    """Convert user text → embedding vector."""
    try:
        print("[DBG] Calling OpenAI embedding...")
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            timeout=30  # force fail instead of hanging forever
        )
        print("[DBG] Embedding length:", len(resp.data[0].embedding))

        print("[DBG] Embedding received.")
        return resp.data[0].embedding
    except Exception as e:
        print("[DBG] Embedding failed.")
        raise RuntimeError(
            f"[ERROR] OpenAI embedding call failed: {e}\n"
            "Check your OPENAI_API_KEY in .env and your internet connection."
        ) from e

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(user_text: str, k: int = TOP_K):
    print("[DBG] ENTERED retrieve_chunks()")

    print("[DBG] about to embed query...")
    query_vec = embed_query(user_text)
    print("[DBG] got query embedding, len =", len(query_vec))

    print("[DBG] about to run chroma query...")
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
   #     include=["documents", "metadatas", "distances"]
    )

    print("[DBG] chroma query returned keys:", results.keys())

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    print("[DBG] retrieved counts:", len(ids), len(docs), len(metas))

    return list(zip(ids, docs, metas))


# ============================================================
# LLM TRIAGE REASONING
# ============================================================

def generate_triage_response(user_text: str, retrieved_chunks):
    """Feed retrieved chunks + user text to a LOCAL Ollama medical LLM (Med42)."""

    # Build context block
    context = ""
    for cid, doc, meta in retrieved_chunks:
        context += f"\n\n=== {cid} ({meta['title']}) ===\n{doc}"

    system_msg = (
        "You are a clinical triage assistant for HealthNav AI.\n"
        "You MUST stay grounded in the provided clinical chunks.\n"
        "If the retrieved context does not contain an answer, say you don't have enough info.\n"
        "Be cautious: if severe red flags appear, recommend urgent/emergency care.\n"
        "Do not invent diagnoses, tests, or thresholds that are not in the context."
    )

    user_prompt = f"""
# USER SYMPTOMS
{user_text}

# RETRIEVED CLINICAL CONTEXT (from textbook, Chapter 2)
{context}

Using ONLY the above context:
1) Identify likely body-system origin(s)
2) List matching red flags (if any)
3) Determine urgency:
   - Home care
   - Primary care within 24–72 hours
   - Urgent care today
   - Emergency department
4) Provide next-step recommendation
5) Give a SHORT explanation referencing the chunk IDs (like cough::red_flags)
"""

    # Ollama local chat API
    payload = {
        "model": "thewindmom/llama3-med42-8b",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False
    }

    try:
        resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
    except Exception as e:
        return f"[ERROR] Failed to call local Ollama model. Details: {e}"



# ============================================================
# DRIVER FUNCTION
# ============================================================

def run_demo():
    print("\n=== HealthNav AI Symptom RAG Demo ===\n")
    user_text = input("Describe your symptoms: ").strip()

    print("\n[1] Retrieving relevant clinical chunks...")

    try:
        chunks = retrieve_chunks(user_text, TOP_K)
        print("[DBG] retrieve_chunks returned", type(chunks), "len=", len(chunks))
    except Exception:
        print("\n[ERROR] retrieve_chunks crashed. Traceback:\n")
        traceback.print_exc()
        return


    for cid, doc, meta in chunks:
        print(f" → {cid}  (symptom={meta['symptom']}, section={meta['section']})")

    print("\n[2] Generating triage reasoning using LOCAL Med42 (Ollama).\n")
    answer = generate_triage_response(user_text, chunks)

    print("=== TRIAGE RESULT ===\n")
    print(answer)
    print("\n=======================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_demo()