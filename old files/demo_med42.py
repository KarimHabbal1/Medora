"""
HealthNav AI - Conversational Triage Demo (Med42)
=================================================
Model: thewindmom/llama3-med42-8b (via Ollama)
Pipeline: FAISS RAG + conversational memory + triage decision
"""

import os
import json
import pickle
import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR  = os.path.join(BASE_DIR, "faiss_symptoms")
INDEX_FILE = os.path.join(FAISS_DIR, "symptom_index.faiss")
META_FILE  = os.path.join(FAISS_DIR, "symptom_meta.pkl")

MODEL_NAME = "thewindmom/llama3-med42-8b"
TOP_K      = 8
OLLAMA_URL = "http://localhost:11434/api/chat"
TIMEOUT    = 300

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# LOAD FAISS INDEX
# ============================================================

def load_index():
    import faiss
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    return index, meta


# ============================================================
# EMBED + RETRIEVE
# ============================================================

def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)


def retrieve_chunks(index, meta, text: str, k: int = TOP_K):
    vec = embed_query(text)
    distances, indices = index.search(vec, k)
    return [{
        "id":       meta["ids"][i],
        "document": meta["documents"][i],
        "metadata": meta["metadatas"][i],
    } for i in indices[0]]


# ============================================================
# BUILD CLINICAL CONTEXT FROM CHUNKS
# ============================================================

def build_context(chunks) -> str:
    context = ""
    for c in chunks:
        context += f"\n\n=== {c['id']} ({c['metadata']['title']}) ===\n{c['document']}"
    return context


# ============================================================
# CALL OLLAMA
# ============================================================

def call_ollama(messages: list) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        return f"[ERROR] Ollama call failed: {e}"


# ============================================================
# CONVERSATIONAL LOOP
# ============================================================

SYSTEM_PROMPT = """You are Medi, a compassionate and professional AI triage nurse for HealthNav AI.

Your job is to have a natural, caring conversation with the patient to understand their symptoms before making a triage decision.

RULES:
- Talk like a caring nurse, NOT like a medical textbook
- Ask ONE clarifying question at a time
- Never dump a list of diagnoses at the patient
- Gather: symptom description, duration, severity (1-10), associated symptoms, medical history if relevant
- After 3-4 exchanges when you have enough information, give a clear triage recommendation
- Always ground your triage decision in the clinical context provided
- If you see RED FLAG symptoms (chest pain + arm radiation, difficulty breathing, signs of stroke), escalate to emergency IMMEDIATELY without waiting for more questions
- Be warm, reassuring, and clear

TRIAGE LEVELS you can recommend:
- "You can manage this at home" 
- "Please see your doctor within 24-72 hours"
- "Go to urgent care today"
- "Call emergency services / go to the ER immediately"

Clinical reference context will be provided to ground your reasoning. Use it silently — do not quote chunk IDs to the patient."""


def run_demo():
    print("\n" + "="*55)
    print("   HealthNav AI - Triage Assistant (Med42)")
    print("="*55)
    print("Type 'quit' to exit\n")

    index, meta = load_index()

    # Conversation history (for memory)
    chat_history = []
    full_symptom_text = ""  # accumulates what patient says for RAG retrieval
    chunks_retrieved = False
    clinical_context = ""

    print("Medi: Hello! I'm Medi, your HealthNav triage assistant.")
    print("      How are you feeling today? What brings you in?\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("\nMedi: Take care and feel better soon. Goodbye!")
            break
        if not user_input:
            continue

        # Accumulate symptoms for RAG
        full_symptom_text += " " + user_input

        # Retrieve RAG chunks once we have enough symptom info (after first message)
        if not chunks_retrieved:
            chunks = retrieve_chunks(index, meta, full_symptom_text)
            clinical_context = build_context(chunks)
            chunks_retrieved = True
        
        # Re-retrieve if patient mentions new symptoms
        elif len(chat_history) % 4 == 0:
            chunks = retrieve_chunks(index, meta, full_symptom_text)
            clinical_context = build_context(chunks)

        # Build messages for the model
        system_with_context = SYSTEM_PROMPT + f"\n\n--- CLINICAL REFERENCE CONTEXT ---{clinical_context}"

        messages = [{"role": "system", "content": system_with_context}]
        messages += chat_history
        messages.append({"role": "user", "content": user_input})

        # Get response
        print("\nMedi: ", end="", flush=True)
        response = call_ollama(messages)
        print(response)
        print()

        # Save to history
        chat_history.append({"role": "user",      "content": user_input})
        chat_history.append({"role": "assistant",  "content": response})


if __name__ == "__main__":
    run_demo()
