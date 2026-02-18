# HealthNav AI – Clinical Reasoning Engine

## 🏥 Overview
**HealthNav AI** is an intelligent medical triage and care-coordination system designed to guide patients safely from initial symptoms to the correct level of care. 

This repository contains the **Data Processing & RAG Engine** that powers the AI's clinical reasoning capabilities. Unlike generic LLM applications, this system is built on a **Symptom Intelligence Layer** derived directly from authoritative medical textbooks (specifically *Chapter 2: Common Symptoms* of *Current Medical Diagnosis & Treatment 2022*).

By avoiding direct fine-tuning and instead using a **Retrieval-Augmented Generation (RAG)** approach with structured medical knowledge, we ensure high explainability, reduce hallucinations, and maintain strict clinical grounding.

---

## 🚀 Pipeline Architecture

The system processes raw medical text into a searchable vector database through a deterministic pipeline:

1.  **Extraction**: Raw text is extracted from the source PDF (TMT 2022).
2.  **Structuring**: GPT-4o converts unstructured prose into structured JSON objects (e.g., identifying Red Flags, Essential Questions).
3.  **Indexing**: Structured data is split into semantic chunks and embedded into a Vector Database (ChromaDB).
4.  **Retrieval (RAG)**: The Triage Agent queries this database for relevant clinical rules to reason about user symptoms.

---

## 📂 File Descriptions

### 1. Data Extraction
-   **`ch2_extractor.py`**:
    -   **Purpose**: Extracts *Chapter 2: Common Symptoms* from the TMT 2022 PDF.
    -   **Logic**: Uses heuristics (font size, capitalization) to identify symptom headings and extract relevant text blocks.
    -   **Output**: `outputs/tmt_symptoms_raw.jsonl`

-   **`tmt_extract.py`**:
    -   **Purpose**: A broader extraction script for the entire textbook (Chapter 3+).
    -   **Logic**: Detects condition-level headings (e.g., *PNEUMONIA*, *ASTHMA*) and sections (*Treatment*, *Diagnosis*).
    -   **Output**: `outputs/tmt_chunks.jsonl`

### 2. Knowledge Structuring
-   **`symptom_structurer.py`**:
    -   **Purpose**: Converts raw text chunks into machine-readable JSON.
    -   **Logic**: Uses GPT-4o to extract specific fields:
        -   *Definition*
        -   *Essential Questions*
        -   *Red Flags*
        -   *System Differentials*
        -   *Emergency Criteria*
    -   **Output**: `outputs/tmt_symptoms_structured.jsonl`

### 3. Vector Indexing
-   **`symptom_indexer.py`** (Recommended):
    -   **Purpose**: Builds the RAG index.
    -   **Technique**: **Multi-Chunk Embedding**. Instead of embedding a whole symptom as one blob, it splits each symptom into smaller, focused chunks (e.g., `COUGH - RED FLAGS`, `CHEST PAIN - TRIAGE LOGIC`). This improves retrieval precision.
    -   **Output**: `chroma_symptoms/` (Persistent ChromaDB)

-   **`build_symptom_index.py`** (Legacy/Alternative):
    -   **Purpose**: An earlier version of the indexer that embeds entire symptom objects as single documents. Less precise for retrieval than the multi-chunk approach.

### 4. RAG & Triage Demo
-   **`symptom_rag_demo.py`**:
    -   **Purpose**: End-to-end demonstration of the triage system.
    -   **Flow**:
        1.  Accepts user symptom description.
        2.  Embeds query using OpenAI (`text-embedding-3-small`).
        3.  Retrieves top-k relevant chunks from ChromaDB.
        4.  Generates a clinical triage assessment using GPT-4o grounded in retrieved chunks.

---

## ⚙️ Setup & Usage

### Prerequisites
-   Python 3.8+
-   `pip install openai chromadb pymupdf python-dotenv`
-   A valid `.env` file with `OPENAI_API_KEY`.

### Running the Pipeline

1.  **Extract Data**:
    ```bash
    python ch2_extractor.py
    ```
    *Extracts raw text from the PDF.*

2.  **Structure Data**:
    ```bash
    python symptom_structurer.py
    ```
    *Converts text to structured JSON using GPT-4o.*

3.  **Build Index**:
    ```bash
    python symptom_indexer.py
    ```
    *Creates the ChromaDB vector store.*

4.  **Test Triage**:
    ```bash
    python symptom_rag_demo.py
    ```
    *Run an interactive session to test symptom analysis.*

---

## 📊 Data Flow Summary

| Stage | Input File | Script | Output File |
| :--- | :--- | :--- | :--- |
| **1. Extract** | `data/TMT_2022.pdf` | `ch2_extractor.py` | `outputs/tmt_symptoms_raw.jsonl` |
| **2. Structure** | `outputs/tmt_symptoms_raw.jsonl` | `symptom_structurer.py` | `outputs/tmt_symptoms_structured.jsonl` |
| **3. Index** | `outputs/tmt_symptoms_structured.jsonl` | `symptom_indexer.py` | `chroma_symptoms/` (Directory of vector store) |
| **4. Query** | User Input | `symptom_rag_demo.py` | Terminal Output (Triage) |

---

## 🔮 Future Improvements
-   **Agent Integration**: Connect this RAG engine to the Intake and Logistics agents.
-   **Condition Layer**: Implement similar structuring for disease-specific chapters (handled by `tmt_extract.py`).
-   **Evaluation**: Build an automated eval set to measure retrieval accuracy and triage safety.
