"""
HealthNav AI - Symptom RAG Query Demo (FAISS version)
=====================================================
Replaces ChromaDB with FAISS for local vector search.
Same RAG pipeline: embed -> kNN search -> GPT-4o / Ollama reasoning.
"""

import os
import pickle
import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_symptoms")
INDEX_FILE = os.path.join(FAISS_DIR, "symptom_index.faiss")
META_FILE  = os.path.join(FAISS_DIR, "symptom_meta.pkl")

TOP_K = 8

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# LOAD INDEX + METADATA
# ============================================================

def load_index():
    try:
        import faiss
    except ImportError:
        raise RuntimeError("faiss not installed. Run: pip install faiss-cpu")

    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_FILE}. Run symptom_indexer_faiss.py first.")

    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)

    print(f"[+] Loaded FAISS index with {index.ntotal} vectors.")
    return index, meta


# ============================================================
# EMBED QUERY
# ============================================================

def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    print(f"[DBG] Query embedding shape: {vec.shape}")
    return vec


# ============================================================
# RETRIEVE CHUNKS
# ============================================================

def retrieve_chunks(index, meta, user_text: str, k: int = TOP_K):
    query_vec = embed_query(user_text)
    distances, indices = index.search(query_vec, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "id":       meta["ids"][idx],
            "document": meta["documents"][idx],
            "metadata": meta["metadatas"][idx],
            "distance": float(distances[0][rank]),
        })
    return results


# ============================================================
# TRIAGE REASONING (Ollama local model)
# ============================================================

def generate_triage_response(user_text: str, chunks):
    context = ""
    for chunk in chunks:
        cid  = chunk["id"]
        doc  = chunk["document"][:500]
        meta = chunk["metadata"]
        context += f"\n\n=== {cid} ({meta['title']}) ===\n{doc}"

    system_msg = (
        "You are a clinical triage assistant for HealthNav AI.\n"
        "You MUST stay grounded in the provided clinical chunks.\n"
        "If the retrieved context does not contain an answer, say you don't have enough info.\n"
        "Be cautious: if severe red flags appear, recommend urgent/emergency care.\n"
        "Do not invent diagnoses, tests, or thresholds not in the context."
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
   - Primary care within 24-72 hours
   - Urgent care today
   - Emergency department
4) Provide next-step recommendation
5) Give a SHORT explanation referencing the chunk IDs (like cough::red_flags)
"""

    payload = {
        "model": "thewindmom/llama3-med42-8b",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        return f"[ERROR] Failed to call local Ollama model. Details: {e}"


# ============================================================
# DRIVER
# ============================================================

def run_demo():
    print("\n=== HealthNav AI Symptom RAG Demo (FAISS) ===\n")

    index, meta = load_index()

    user_text = input("Describe your symptoms: ").strip()

    print("\n[1] Retrieving relevant clinical chunks...")
    chunks = retrieve_chunks(index, meta, user_text)

    for c in chunks:
        print(f"  -> {c['id']}  (symptom={c['metadata']['symptom']}, section={c['metadata']['section']}, dist={c['distance']:.4f})")

    print("\n[2] Generating triage reasoning using LOCAL Med42 (Ollama).\n")
    answer = generate_triage_response(user_text, chunks)

    print("=== TRIAGE RESULT ===\n")
    print(answer)
    print("\n=======================")


if __name__ == "__main__":
    run_demo()
