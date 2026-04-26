"""
HealthNav AI – Symptom RAG Index Builder
========================================

This script takes the structured symptom JSONL file produced by
`symptom_structurer.py` and builds a vector index for RAG.

Why this exists
---------------
We now have a small but high-quality "Symptom Intelligence Layer" from
Chapter 2 of TMT 2022. To actually USE it in a triage agent, we need
to:

1. Turn each symptom object into a text representation.
2. Embed that text with an OpenAI embedding model.
3. Store embeddings + metadata in a vector database (ChromaDB).
4. Later: query this index given a user’s symptom description.

This script does steps (1)–(3).

Output
------
Creates a persistent ChromaDB directory (./chroma_symptoms_db) containing
a collection named "symptoms". Each document is one symptom.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# ============================================================
# ENV + PATHS
# ============================================================
load_dotenv()  # loads OPENAI_API_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRUCTURED_PATH = os.path.join(BASE_DIR, "outputs", "tmt_symptoms_structured.jsonl")

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_symptoms_db")
COLLECTION_NAME = "symptoms"

EMBEDDING_MODEL = "text-embedding-3-large"  # good quality; you can switch to -small if needed

client = OpenAI()  # uses OPENAI_API_KEY from .env


# ============================================================
# HELPERS
# ============================================================

def load_structured_symptoms(path):
    """Load JSONL file: one symptom JSON per line."""
    symptoms = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            symptoms.append(json.loads(line))
    return symptoms


def symptom_to_document(sym):
    """
    Convert one structured symptom object into a text blob suitable
    for embedding. We include the most important fields.
    """
    parts = []
    parts.append(f"Symptom: {sym.get('symptom', '')}")
    parts.append(f"Definition: {sym.get('one_sentence_definition', '')}")

    eq = sym.get("essential_questions", [])
    if eq:
        parts.append("Essential questions:")
        for q in eq:
            parts.append(f"- {q}")

    sd = sym.get("system_differentials", [])
    if sd:
        parts.append("System differentials:")
        for d in sd:
            parts.append(f"- {d}")

    pc = sym.get("possible_causes", [])
    if pc:
        parts.append("Possible causes:")
        for c in pc:
            parts.append(f"- {c}")

    rf = sym.get("red_flags", [])
    if rf:
        parts.append("Red flags:")
        for r in rf:
            parts.append(f"- {r}")

    wtr = sym.get("when_to_refer_emergency", [])
    if wtr:
        parts.append("When to refer to emergency:")
        for r in wtr:
            parts.append(f"- {r}")

    # Keep it compact but rich enough for semantic search
    return "\n".join(parts)


def embed_text(text):
    """Get embedding vector from OpenAI."""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return resp.data[0].embedding


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"[+] Loading structured symptoms from:\n    {STRUCTURED_PATH}")
    symptoms = load_structured_symptoms(STRUCTURED_PATH)
    print(f"[+] Loaded {len(symptoms)} symptom objects.")

    # Init Chroma persistent client
    print(f"[+] Initializing ChromaDB at: {CHROMA_DIR}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # To avoid duplicates if you rerun, you can (optionally) reset:
    # chroma_client.delete_collection(COLLECTION_NAME)
    # collection = chroma_client.create_collection(COLLECTION_NAME)

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for idx, sym in enumerate(symptoms):
        symptom_name = sym.get("symptom", f"SYMPTOM_{idx}")
        doc_id = f"symptom-{idx}-{symptom_name.replace(' ', '_')}"

        print(f"[→] Embedding {idx+1}/{len(symptoms)}: {symptom_name}")

        text = symptom_to_document(sym)
        vector = embed_text(text)

        ids.append(doc_id)
        documents.append(text)
        # store full JSON as a string in metadata (handy for retrieval)
        metadatas.append({
            "symptom": symptom_name,
            "raw_json": json.dumps(sym)
        })
        embeddings.append(vector)

    print("[+] Inserting into Chroma collection...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("\n[✓] Done. Symptom index built in:")
    print(f"    {CHROMA_DIR}")
    print(f"    Collection: {COLLECTION_NAME}")
    print("\nYou can now query this index from a triage/RAG script.")