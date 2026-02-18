# HealthNav AI – Clinical Reasoning Engine
## Branch: `llms/ragSystem/llama3.1-8b`

> **Experiment:** Using `llama3.1:8b` — Meta's highly capable general-purpose model with strong instruction-following — as the local generation engine.

---

## Why Llama 3.1 8B?

Llama 3.1 8B is a strong general-purpose baseline. While not medically fine-tuned, it excels at:
- Following complex structured prompts precisely
- Staying grounded in provided context (critical for RAG)
- Producing clean, well-formatted outputs

This branch tests whether a high-quality general model can match medically fine-tuned models when the clinical knowledge is supplied entirely through RAG retrieval.

---

## Model Details

| Property | Value |
|---|---|
| **Model** | `llama3.1:8b` |
| **Base** | Meta Llama 3.1 |
| **Parameters** | 8 billion |
| **Download size** | ~5 GB |
| **Min RAM** | 8 GB |
| **Specialization** | General — strong instruction following |

---

## Setup

### Pull the model
```bash
ollama pull llama3.1:8b
```

### `.env` file
```
OPENAI_API_KEY=sk-...

OLLAMA_MODEL=llama3.1:8b
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

- Can a general model match medical-fine-tuned models when clinical knowledge comes from RAG?
- Does it follow the structured triage output format more reliably than `meditron`?
- How does it handle edge cases where retrieved chunks are ambiguous?
- Is there a meaningful quality gap vs the medically fine-tuned models?

---

## Future Improvements
- **Agent Integration**: Connect this RAG engine to the Intake and Logistics agents.
- **Evaluation**: Build an automated eval set to benchmark llama3.1:8b vs medical models on the same queries.
