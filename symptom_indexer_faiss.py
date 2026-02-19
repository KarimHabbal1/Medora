"""
HealthNav AI - Symptom RAG Index Builder (FAISS version)
=========================================================
Replaces ChromaDB with FAISS + pickle for zero-dependency local storage.
Same multi-chunk embedding logic, same OpenAI embeddings.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRUCTURED_PATH = os.path.join(BASE_DIR, "outputs", "tmt_symptoms_structured.jsonl")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_symptoms")
INDEX_FILE = os.path.join(FAISS_DIR, "symptom_index.faiss")
META_FILE  = os.path.join(FAISS_DIR, "symptom_meta.pkl")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# LOAD STRUCTURED SYMPTOMS
# ============================================================

def load_structured_symptoms(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON line: {line[:80]}")
    print(f"[+] Loaded {len(records)} structured symptom records.")
    return records


# ============================================================
# BUILD CHUNKS
# ============================================================

def build_chunks_for_symptom(symptom_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    symptom_name = symptom_obj.get("symptom", "").strip() or "UNKNOWN"
    symptom_slug = symptom_name.lower().replace(" ", "_")
    chunks = []

    def add_chunk(section_key, title, value):
        if isinstance(value, str):
            content = value.strip()
        elif isinstance(value, list):
            content = "\n".join(f"- {item}" for item in value if str(item).strip())
        else:
            content = ""
        if not content:
            return
        chunks.append({
            "id":       f"{symptom_slug}::{section_key}",
            "text":     f"Symptom: {symptom_name}\nSection: {title}\n\n{content}",
            "metadata": {
                "symptom": symptom_name,
                "section": section_key,
                "title":   title,
                "source":  "TMT_2022_CH2",
                "type":    "symptom_chunk",
            },
        })

    add_chunk("definition",             "DEFINITION",             symptom_obj.get("one_sentence_definition", ""))
    add_chunk("essential_questions",    "ESSENTIAL QUESTIONS",    symptom_obj.get("essential_questions", []))
    add_chunk("system_differentials",   "SYSTEM DIFFERENTIALS",   symptom_obj.get("system_differentials", []))
    add_chunk("possible_causes",        "POSSIBLE CAUSES",        symptom_obj.get("possible_causes", []))
    add_chunk("red_flags",              "RED FLAGS",              symptom_obj.get("red_flags", []))
    add_chunk("when_to_refer_emergency","WHEN TO REFER - EMERGENCY", symptom_obj.get("when_to_refer_emergency", []))
    add_chunk("key_examination_findings","KEY EXAMINATION FINDINGS", symptom_obj.get("key_examination_findings", []))
    add_chunk("useful_diagnostic_tests","USEFUL DIAGNOSTIC TESTS", symptom_obj.get("useful_diagnostic_tests", []))
    add_chunk("triage_logic",           "TRIAGE LOGIC",           symptom_obj.get("triage_logic", ""))
    add_chunk("specialty_routing",      "SPECIALTY ROUTING",      symptom_obj.get("specialty_routing", []))
    add_chunk("rag_summary",            "RAG SUMMARY",            symptom_obj.get("rag_summary", ""))
    return chunks


# ============================================================
# EMBED TEXTS
# ============================================================

def embed_texts(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_symptom_index():
    try:
        import faiss
    except ImportError:
        print("[ERROR] faiss not installed. Run: pip install faiss-cpu")
        return

    os.makedirs(FAISS_DIR, exist_ok=True)

    # Step 1: load
    symptoms = load_structured_symptoms(STRUCTURED_PATH)

    # Step 2: build chunks
    all_chunks = []
    for s in symptoms:
        all_chunks.extend(build_chunks_for_symptom(s))
    print(f"[+] Built {len(all_chunks)} chunks from {len(symptoms)} symptoms.")

    if not all_chunks:
        print("[!] No chunks found. Did you run symptom_structurer.py first?")
        return

    documents = [c["text"]     for c in all_chunks]
    ids       = [c["id"]       for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    # Step 3: embed
    print("[+] Embedding all chunks with OpenAI...")
    embeddings = embed_texts(documents)
    print(f"[+] Embedding complete. Shape: {embeddings.shape}")

    # Step 4: build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"[+] FAISS index built with {index.ntotal} vectors.")

    # Step 5: save index + metadata
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"ids": ids, "documents": documents, "metadatas": metadatas}, f)

    print(f"[OK] Saved FAISS index to: {INDEX_FILE}")
    print(f"[OK] Saved metadata to:    {META_FILE}")

    # Sanity check
    print(f"\n[Sample chunk] {ids[0]}")
    print(documents[0][:300])


if __name__ == "__main__":
    build_symptom_index()
