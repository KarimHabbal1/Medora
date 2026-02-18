# HealthNav AI – Clinical Reasoning Engine
## Branch: `llms/ragSystem/mistral-7b`

> **Experiment:** Using `mistral:7b` — Mistral AI's efficient 7B model known for strong structured output and reasoning — as the local generation engine.

---

## Why Mistral 7B?

Mistral 7B punches well above its weight class. It is known for:
- Excellent structured output adherence (important for triage format)
- Strong reasoning relative to its parameter count
- Very fast inference — lowest latency of all models in this benchmark series
- Efficient attention mechanisms (sliding window attention)

This branch tests whether Mistral's output discipline and speed make it a practical real-time triage engine despite having no medical fine-tuning.

---

## Model Details

| Property | Value |
|---|---|
| **Model** | `mistral:7b` |
| **Base** | Mistral 7B v0.3 |
| **Parameters** | 7 billion |
| **Download size** | ~4 GB |
| **Min RAM** | 8 GB |
| **Specialization** | General — strong structured output, fast inference |

---

## Setup

### Pull the model
```bash
ollama pull mistral:7b
```

### `.env` file
```
OPENAI_API_KEY=sk-...

OLLAMA_MODEL=mistral:7b
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

- Does Mistral's structured output discipline produce cleaner triage responses than medical models?
- Is inference speed noticeably faster than 7B medical models?
- Does it stay grounded in retrieved chunks or hallucinate beyond the clinical context?
- Is the lack of medical fine-tuning a meaningful disadvantage when knowledge is RAG-supplied?

---

## Future Improvements
- **Agent Integration**: Connect this RAG engine to the Intake and Logistics agents.
- **Evaluation**: Build an automated eval set to benchmark mistral:7b vs medical models on speed and quality.
