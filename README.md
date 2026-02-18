# HealthNav AI – Clinical Reasoning Engine
## Branch: `llms/ragSystem/meditron-7b`

> **Experiment:** Using `meditron:7b` — a Llama 2 model fine-tuned specifically on medical papers, PubMed, and clinical guidelines — as the local generation engine.

---

## Why Meditron?

Meditron is purpose-built for clinical reasoning. Unlike general-purpose models, it was adapted to the medical domain through training on:
- PubMed abstracts and full-text papers
- Medical guidelines (WHO, clinical protocols)
- Medical exam Q&A datasets

It outperforms GPT-3.5 and Llama 2 on medical reasoning benchmarks, making it the strongest medically-aligned option at the 7B scale.

---

## Model Details

| Property | Value |
|---|---|
| **Model** | `meditron:7b` |
| **Base** | Llama 2 7B |
| **Parameters** | 7 billion |
| **Download size** | ~4 GB |
| **Min RAM** | 8 GB |
| **Specialization** | Medical reasoning, clinical guidelines |

---

## Setup

### Pull the model
```bash
ollama pull meditron:7b
```

### `.env` file
```
OPENAI_API_KEY=sk-...

OLLAMA_MODEL=meditron:7b
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

1. **Extract → Structure → Index** (if not already done — see `closedSourceModel` branch)

2. **Run triage demo:**
```bash
python symptom_rag_demo.py
```

---

## What to Evaluate

- Does medical fine-tuning improve red-flag detection vs general models?
- Does it understand clinical terminology in retrieved chunks better than `llama3.2`?
- How does structured output quality compare to GPT-4o?
- Does it stay grounded or hallucinate beyond retrieved context?

---

## Future Improvements
- **Agent Integration**: Connect this RAG engine to the Intake and Logistics agents.
- **Evaluation**: Build an automated eval set to benchmark meditron vs other models on the same queries.
