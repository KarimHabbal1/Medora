# Phase 4.2: Triage Agent

## Overview

Phase 4.2 builds the Triage Agent — the clinical analysis engine of the Medora system. Where Phase 4.1's Intake Agent collects structured patient data through a conversational interview, the Triage Agent receives a clinical question (either directly from a doctor or derived from an Intake session) and produces an evidence-based analysis grounded exclusively in retrieved passages from the medical textbook.

The Triage Agent is the first component that connects the RAG pipeline built across Phases 2 and 3 to an actual clinical output. Every response the agent generates is produced by running a query through the full retrieve-rerank-generate cycle: the bi-encoder encodes the query and retrieves k=10 candidates from ChromaDB, the BGE cross-encoder reranks them to the top 3, those 3 passages are formatted into a context block, and the LLM generates a structured clinical analysis grounded in those passages alone. No query to the Triage Agent produces an answer from the LLM's general training data — the system prompt explicitly prohibits it and requires citations to the retrieved passages.

### Where Phase 4.2 Sits in the Pipeline

Phase 4.2 is the analysis layer:

```
Phases 1–3: RAG infrastructure
    Phase 1: PDF extraction, chunking, symptom structuring
    Phase 2: Embedding, vector store, retrieval validation
    Phase 3: Cross-encoder reranking evaluation and selection

Phase 4: Agents
    Phase 4.1: Intake Agent — structured patient interview (collect)
    Phase 4.2: Triage Agent — evidence-based clinical analysis (analyze)
```

The Triage Agent depends on the RAG infrastructure but makes no modifications to it. It imports `retrieve_and_rerank()`, `open_collection()`, `load_bi_encoder()`, `load_cross_encoder()`, and `detect_device()` from `rag/reranker.py` — the same functions used in Phase 3's evaluation pipeline. The configuration constants (`CHROMA_DIR`, `EMBEDDING_MODEL`, `RERANKER_MODEL`, `RERANK_TOP_K_RETRIEVE`, `RERANK_TOP_K_RETURN`) come from `config.py` unchanged.

### The Distinction from the Intake Agent

The Intake Agent and Triage Agent have fundamentally different jobs. The Intake Agent is an interviewer: it maintains conversational state across multiple turns, asks follow-up questions, identifies red flags, and terminates when it has collected a complete clinical picture. It is inherently multi-turn and stateful — a patient cannot be fully assessed in a single exchange.

The Triage Agent is an analyst. Clinical analysis is a retrieval task, not a dialogue. A well-formed clinical question — "What are the treatment options for atrial fibrillation?" or "Differential diagnosis for chest pain and hemoptysis in an ex-smoker" — is self-contained. The answer requires retrieving the relevant textbook evidence and synthesising it, not continuing a conversation. There is no state to maintain across turns because the question and the evidence needed to answer it are both fully available at the moment the query is posed.

This architectural distinction justifies the difference in implementation. The Intake Agent uses LangGraph to manage a state machine with turn tracking, conversation history, and multi-step logic. The Triage Agent uses no graph framework at all — it is a set of plain Python functions. The retrieve-rerank-generate pipeline runs once per query and returns a structured result. The simplicity is intentional: each query is independent, and adding stateful infrastructure for a stateless task would add complexity without benefit.

---

## Architecture

### Why No LangGraph

LangGraph is the right tool for the Intake Agent because the intake workflow is a state machine: the agent starts in a symptom-collection state, transitions through follow-up states, evaluates red flags, decides on urgency, and terminates. The graph structure represents real decision logic — which state to transition to next depends on what has been collected so far.

The Triage Agent has no such state machine. Its control flow is linear: receive query → retrieve → rerank → generate → return. There are no branches that require tracking which state the session is in. There is no conversation history to maintain. Adding a graph framework to this workflow would create machinery with no corresponding logic to represent — the graph would have exactly one node.

The correct implementation is therefore a set of plain functions with a thin class wrapper (`TriageSession`) that handles the one genuine concern of a multi-query session: avoiding repeated model loading.

### Model Loading and the TriageSession Class

Loading the bi-encoder, cross-encoder, and ChromaDB collection is expensive. The bi-encoder (`sentence-transformers/embeddinggemma-300m-medical`) requires loading model weights and moving them to the compute device. The cross-encoder (`BAAI/bge-reranker-v2-m3`) similarly requires weight loading, with an MPS-to-CPU fallback. The ChromaDB client opens a persistent database connection. If a session involves multiple queries — as in intake mode, where 2–4 queries are run sequentially — this loading overhead must not be repeated per query.

`TriageSession` handles this by loading all infrastructure once in `__init__()` and storing it as instance attributes. Subsequent calls to `session.query()` or `session.from_intake()` reuse the loaded models:

```python
session = TriageSession()          # loads models once
r1 = session.query("...")          # uses cached models
r2 = session.query("...")          # uses cached models — no reload
```

For single-query usage (the `--query` CLI flag), a `TriageSession` is still instantiated — there is no path that calls the underlying functions without model loading. This is the correct tradeoff: model loading is unavoidable for any query, and the session class provides a consistent interface whether one or twenty queries are run.

### Full Architecture Diagram

```
Doctor / Intake Agent
        |
        | clinical query (string) or intake summary (dict)
        v
TriageSession
        |
        |-- triage_query() ─────────────────────────────────┐
        |        |                                           |
        |        | 1. bi_encoder.encode(query)               |
        |        v                                           |
        |   ChromaDB (5,631 chunks)                         |
        |   retrieve k=10 candidates with text              |
        |        |                                           |
        |        | 2. cross_encoder.predict(10 pairs)        |
        |        v                                           |
        |   BGE reranker → top 3 by score                   |
        |        |                                           |
        |        | 3. _chunks_to_context(top_3)              |
        |        v                                           |
        |   numbered passage block with metadata            |
        |        |                                           |
        |        | 4. LLM([system_prompt, user_content])     |
        |        v                                           |
        |   structured clinical analysis (5 sections)       |
        |        |                                           |
        |        | return {"query", "retrieved_chunks",      |
        |        |         "analysis"}                       |
        |        └───────────────────────────────────────────┘
        |
        |-- triage_from_intake() ── calls triage_query() N times
                 |
                 | 1. emergency guard
                 | 2. LLM generates 2-4 clinical questions from intake summary
                 | 3. triage_query() for each question
                 | 4. deduplicate chunks across queries
                 | 5. LLM synthesis → comprehensive report (6 sections)
                 |
                 return {"intake_symptoms", "urgency",
                         "generated_queries", "per_query_results",
                         "comprehensive_analysis"}
```

---

## Two Input Modes

### Mode 1: Direct Query

In direct query mode, a doctor submits a clinical question and receives a structured analysis. This is a single pass through the full RAG pipeline with no intermediate steps.

**Entry points:**

```python
# Interactive (REPL)
python agents/triage_agent.py

# Non-interactive single query
python agents/triage_agent.py --query "What are the treatment options for atrial fibrillation?"

# Programmatic
session = TriageSession()
result = session.query("What are the treatment options for atrial fibrillation?")
```

**What happens:**

1. The query is encoded with the bi-encoder into a 768-dimensional vector.
2. ChromaDB performs approximate nearest-neighbour search and returns the 10 most similar chunks, including their full passage text and metadata (`chapter`, `section`, `distance`).
3. The BGE cross-encoder scores all 10 (query, passage) pairs and returns the top 3 by relevance score. Each chunk is augmented with `rerank_score` (raw cross-encoder logit) and `original_rank` (its position in the bi-encoder ranking before reranking).
4. The 3 passages are formatted into a numbered context block via `_chunks_to_context()`, with chapter, section, and rerank score as metadata headers for each passage.
5. The LLM receives this context block alongside the query and the single-query analysis system prompt. It generates a structured response in 5 sections.
6. The result is returned as a dict containing the original query, the retrieved chunk list, and the analysis string.

**Returned structure:**

```python
{
    "query":            str,        # the clinical question asked
    "retrieved_chunks": list[dict], # top-3 chunks with metadata and rerank scores
    "analysis":         str,        # 5-section LLM clinical analysis
}
```

**Use case:** Quick clinical reference lookup. A doctor needs to verify the current standard of care for a condition, check indication criteria for an investigation, or review the evaluation approach for a symptom they have not recently encountered. The response is grounded in the textbook and cited by chapter and section, providing a traceable evidence base rather than a generic AI answer.

### Mode 2: From Intake Agent

In intake mode, the Triage Agent receives the structured summary dictionary produced by `IntakeSession.get_summary()` and generates a comprehensive clinical report covering the patient's full presentation. This mode runs multiple triage queries sequentially and synthesises their results into a unified analysis.

**Entry points:**

```python
# CLI from saved JSON file
python agents/triage_agent.py --from-intake data/results/last_intake.json

# Programmatic
session = TriageSession()
result = session.from_intake(intake_summary_dict)
```

**What happens:**

1. **Emergency guard.** The function reads `intake_summary["urgency"]`. If it is `"emergency"`, the function immediately returns a short deferral message and exits. Emergency cases were already escalated by the Intake Agent (Phase 4.1) — no further analysis is needed from the Triage Agent. This guard ensures the Triage Agent is never called on emergency presentations.

2. **Clinical question generation.** The intake summary is formatted into a structured prompt containing the patient's symptoms, urgency level, all recorded answers, and any triggered red flags. The LLM is given this prompt with the query generation system prompt and asked to produce a JSON array of 2–4 targeted clinical questions. These questions reflect what a doctor would need answered to manage this specific patient — differential diagnoses, management options, investigation choices, and any clinically important considerations raised by the red flags.

3. **Per-question retrieval and analysis.** `triage_query()` is called once for each generated question. Each call runs the full retrieve-rerank-generate pipeline independently. The results are stored as a list of per-query result dicts.

4. **Chunk deduplication.** All chunks retrieved across the N queries are merged into a single list, deduplicated by `chunk_id`. Because different clinical questions may retrieve the same passage (e.g., a passage about chest pain evaluation may be retrieved for both a differential diagnosis query and an investigation query), deduplication ensures the synthesis prompt does not receive redundant context.

5. **Comprehensive synthesis.** The LLM receives the full intake summary, all individual analyses, and all deduplicated retrieved passages, with the synthesis system prompt. It produces a 6-section comprehensive clinical report covering the patient's presentation from overview through differential diagnosis, management, investigations, and red flags.

6. The result is returned as a dict containing the intake symptoms, urgency, generated queries, per-query results, and the comprehensive analysis string.

**Returned structure:**

```python
{
    "intake_symptoms":        list[str],  # symptoms from intake
    "urgency":                str,        # "routine" or "urgent"
    "generated_queries":      list[str],  # 2-4 LLM-generated clinical questions
    "per_query_results":      list[dict], # one triage_query result per question
    "comprehensive_analysis": str,        # 6-section synthesis report
}
```

**Use case:** Full clinical workup following a patient intake. After the Intake Agent has collected the patient's presentation, the Triage Agent performs the analytical work that grounds the initial clinical assessment in textbook evidence. The doctor receives both the structured intake summary and the comprehensive analysis simultaneously, providing a complete picture of the patient's presentation with an evidence base that is cited and traceable.

---

## The Clinical Analysis Prompts

The Triage Agent has three distinct system prompts, each serving a different function in the pipeline.

### Prompt 1: Single-Query Clinical Analysis

Used for every individual `triage_query()` call — both in direct query mode and for each sub-query in intake mode.

```
You are a clinical decision support system. Given a clinical query and relevant passages
from a medical textbook (CURRENT Medical Diagnosis and Treatment), provide a structured
clinical analysis.

Your analysis must be:
- Grounded ONLY in the provided textbook passages — do not add information from your
  general training data
- Structured with clear sections
- Include specific references to the source passages (cite by chapter and section)
- Clinically precise and actionable

Structure your response as:

## Clinical Analysis
[Direct answer to the query with evidence from the passages]

## Key Findings from Evidence
[Bullet points of the most relevant facts from the retrieved passages]

## Differential Considerations
[If applicable: conditions to consider based on the evidence]

## Recommended Next Steps
[Clinical actions suggested by the textbook evidence]

## Sources
[List each retrieved passage with chapter, section, and a brief description of what it contributed]

If the retrieved passages do not contain sufficient information to answer the query,
state this clearly rather than guessing.
```

**Design rationale:** The most important constraint in this prompt is the grounding requirement: "Grounded ONLY in the provided textbook passages — do not add information from your general training data." This constraint is what distinguishes Medora's analysis from a general LLM consultation. GPT-4o has extensive medical training data and would, without this constraint, supplement retrieved passages with its own knowledge, produce confident-sounding answers that are not traceable to any specific source, and potentially disagree with the textbook's specific guidance without flagging the discrepancy.

The grounding constraint forces the LLM to operate as an analytical layer on top of the retrieved evidence rather than as an independent knowledge source. When the retrieved passages do not contain sufficient information, the prompt instructs the LLM to say so explicitly — this is a safety property. An honest "the retrieved passages do not contain sufficient information to answer this query" is more useful than a confidently generated answer that cannot be verified.

The Sources section requirement ensures that every response includes a traceable list of the passages used, cited by chapter and section. This allows a doctor to consult the textbook directly, verify the evidence, and assess whether the retrieved context was appropriate for the query.

### Prompt 2: Clinical Question Generation

Used only in intake mode, to generate the 2–4 queries that drive the per-question analysis.

```
You are a clinical triage planning assistant. Given a patient intake summary, generate
2-4 specific clinical questions that a doctor would need answered to manage this patient.

Focus on:
- Differential diagnosis for the presenting symptoms
- Management considerations based on the patient's specific answers
- Any red flags that need further investigation

Return a JSON array of question strings only. No explanation, no markdown fences.
```

**Design rationale:** The question generation step is the intelligence layer that converts an unstructured patient presentation (symptoms, answers, red flags) into a set of targeted retrieval queries. Without this step, the only option would be to submit the raw symptoms as a single query — which is too broad for precise evidence retrieval. The textbook's chapters are organised by condition and symptom type, not by patient presentations. "Chest pain with hemoptysis in an ex-smoker with a previous stroke" does not map cleanly to a single chapter; it requires multiple targeted queries across Common Symptoms, Pulmonary Disorders, and Heart Disease.

The JSON array output format is enforced by the prompt. The code handles the case where the LLM wraps the response in markdown code fences (a common GPT-4o behaviour) with a stripping step before JSON parsing, and falls back to treating the full response as a single question if JSON parsing fails entirely.

### Prompt 3: Comprehensive Synthesis

Used once per intake-mode session, after all per-question analyses are complete.

```
You are a clinical decision support system. Given multiple clinical analyses from
a patient intake, synthesize a comprehensive clinical report.

Rules:
- Base all recommendations ONLY on the retrieved textbook passages provided.
- Cite textbook chapters and sections explicitly.
- Be clinically precise and actionable.

Structure your response as:

## Patient Overview
[Summary of presenting complaints and key findings from intake]

## Differential Diagnosis
[Ranked list of likely diagnoses with supporting evidence]

## Recommended Management Plan
[Evidence-based management recommendations]

## Investigations Required
[Specific tests and their clinical justification]

## Red Flags and Safety Netting
[What to watch for, when to escalate]

## Sources
[All textbook passages used, organised by topic]
```

**Design rationale:** The synthesis prompt receives substantially more context than the single-query prompt: the full intake summary, all individual per-question analyses, and all deduplicated retrieved passages. Its task is integration rather than analysis — it must combine the findings from multiple parallel evidence lookups into a coherent report that addresses the patient's full presentation.

The six-section structure of the synthesis report maps to the six things a clinician needs to know when receiving a triage workup: who the patient is and what they presented with, what conditions to consider, what to do first, what tests to order, and what to watch for. This structure was designed to mirror the mental model of a doctor receiving a triage handover, not to mirror the structure of the underlying textbook.

The grounding constraint is repeated in the synthesis prompt ("Base all recommendations ONLY on the retrieved textbook passages provided") because the LLM's synthesis step receives individual analyses that themselves contain synthesised reasoning — there is a risk that the model begins to reason beyond the retrieved evidence at this stage. Explicitly restating the constraint at the synthesis layer reinforces it.

---

## The Intake-to-Triage Pipeline

The complete patient journey from intake through triage analysis:

```
Patient conversation
        |
        v
Intake Agent (Phase 4.1)
        |  multi-turn conversation with LangGraph state machine
        |  collects: symptoms, clinical answers, red flags, urgency
        |
        |── if urgency == "emergency":
        |       STOP — Intake Agent escalates to emergency services
        |       Triage Agent is NOT called
        |
        |── if urgency == "routine" or "urgent":
        |       pass intake_summary dict to Triage Agent
        |
        v
IntakeSession.get_summary() → {
    "symptoms":              ["chest pain", "hemoptysis"],
    "urgency":               "urgent",
    "answers":               {question: answer, ...},
    "triggered_red_flags":   [{flag, urgency, question}, ...],
    "specialty_routing":     "Pulmonology",
    "initial_workup":        [...],
}
        |
        v
Triage Agent (Phase 4.2)
        |
        | Step 1: emergency guard → passes (urgency != "emergency")
        |
        | Step 2: LLM generates 2-4 targeted clinical questions
        |         from symptoms + answers + red flags
        |         e.g.:
        |         1. "Differential diagnosis for chest pain and hemoptysis
        |             in a 58-year-old ex-smoker with previous stroke"
        |         2. "Treatment and management of pulmonary embolism"
        |         3. "Investigation protocol for suspected ACS vs PE
        |             with coexisting risk factors"
        |         4. "Anticoagulation management in stroke patients
        |             with pulmonary hemorrhage risk"
        |
        | Step 3: retrieve_and_rerank() for each question
        |         → 3 passages per question from textbook
        |         → 4 questions × 3 passages = up to 12 passages (deduplicated)
        |
        | Step 4: LLM synthesis over all passages and individual analyses
        |         → comprehensive clinical report (6 sections)
        |
        v
Doctor receives:
    - Structured intake summary (from Phase 4.1)
    - Comprehensive clinical report (from Phase 4.2):
        * Patient Overview
        * Differential Diagnosis (ranked, evidence-cited)
        * Recommended Management Plan
        * Investigations Required
        * Red Flags and Safety Netting
        * Sources (all textbook passages used)
```

The emergency guard at the start of `triage_from_intake()` exists because the Intake Agent already handles emergency escalation — if urgency is emergency, the Intake Agent has already told the patient (or operator) to call emergency services. There is no clinical value in running a comprehensive retrieval pipeline for an emergency presentation; the time spent on retrieval and synthesis is time the patient does not have.

For routine and urgent cases, the handoff between the two agents is clean: the Intake Agent produces a structured dict via `get_summary()`, and the Triage Agent consumes it. There is no shared state, no API call between agents, no message-passing infrastructure. The calling code simply saves the intake summary to a JSON file (or keeps it in memory) and passes the dict to `triage_from_intake()`.

---

## How Evidence Grounding Works

Every response produced by the Triage Agent is constructed from a specific set of retrieved textbook passages. The grounding mechanism works at three levels:

### 1. Passage Retrieval and Metadata Preservation

When `retrieve_and_rerank()` is called, ChromaDB returns each chunk with its full metadata: `chunk_id`, `chapter`, `section`, `distance` (bi-encoder cosine distance). After reranking, the cross-encoder adds `rerank_score` and `original_rank` to each chunk dict. These metadata fields travel with the chunk through every subsequent step.

### 2. Context Formatting with `_chunks_to_context()`

Before the LLM receives the retrieved passages, they are formatted by `_chunks_to_context()` into a structured block:

```
--- Passage 1 ---
Chapter : Heart Disease
Section : CARDIAC ARRHYTHMIAS  [rerank score: 0.9842]

[passage text]

--- Passage 2 ---
Chapter : Heart Disease
Section : CARDIAC ARRHYTHMIAS  [rerank score: 0.9341]

[passage text]

--- Passage 3 ---
Chapter : Heart Disease
Section : CARDIAC ARRHYTHMIAS  [rerank score: 0.8873]

[passage text]
```

This formatting makes the passage provenance explicit within the LLM's context window. The LLM receives numbered, labelled passages — not an undifferentiated block of text — which supports accurate citation in the response.

### 3. Citation in the LLM Response

The single-query analysis prompt instructs the LLM to "cite by chapter and section" within the body of the analysis and to produce a Sources section listing each passage with chapter, section, and a description of its contribution. The synthesis prompt repeats this instruction. The result is that every substantive clinical claim in the analysis is traceable to a specific passage from a specific chapter and section of CURRENT Medical Diagnosis and Treatment.

**What this grounding mechanism does and does not guarantee:**

Citations are at chapter and section level, not page level. The textbook is a single PDF; page numbers are stored as `page_range` in the chunk metadata but are not passed to the LLM and do not appear in citations. A doctor who wants to verify a specific claim must navigate to the chapter and section in the textbook, not to a page number. This is a documentation and UX limitation, not a data limitation — the page range is available in the chunk metadata and could be added to citations in a future version.

The grounding constraint is enforced by prompt instruction, not by technical constraint. The LLM receives both the retrieved passages and the instruction not to go beyond them. A sufficiently capable LLM (GPT-4o) will generally adhere to this instruction, but there is no technical mechanism that prevents it from drawing on training knowledge. The constraint is a clinical design decision implemented via prompting, and its adherence depends on prompt compliance. Evaluation of how consistently the grounding constraint is maintained — whether the LLM introduces information not present in the retrieved passages — is a meaningful future validation task.

---

## Test Results

### Direct Query Test: Atrial Fibrillation Treatment

**Query:** "What are the treatment options for atrial fibrillation?"

**Retrieval:**
- 3 chunks retrieved from the Heart Disease chapter, Cardiac Arrhythmias section
- Rerank scores: 0.98, 0.93, 0.89
- All three passages were from the treatment subsection of the Atrial Fibrillation entry (not the diagnostic essentials or ECG pattern sections)

This is the same query that demonstrated the most dramatic reranker improvement in Phase 3's LLM judge evaluation (bi-encoder=2/5, BGE=5/5). The bi-encoder's rank-1 result was an atrial flutter ECG description; the reranker correctly promoted the treatment-focused passage to rank 1. The Triage Agent receives the reranker's output, not the bi-encoder's — so it gets the correct passage at rank 1.

**Analysis produced (5 sections):**
- Clinical Analysis: covered pill-in-the-pocket pharmacologic cardioversion for paroxysmal AF, rate control with beta-blockers and calcium channel blockers for persistent AF, rhythm control strategy with antiarrhythmic drugs, and catheter ablation for symptomatic refractory AF
- Key Findings from Evidence: explicit citations to specific treatment recommendations in the passages, including drug classes and indication criteria
- Differential Considerations: distinguished paroxysmal, persistent, and permanent AF and noted different treatment implications for each
- Recommended Next Steps: stroke risk stratification, CHA2DS2-VASc score, anticoagulation decision
- Sources: all three passages cited by chapter and section with descriptions of their content contribution

All clinical content in the analysis was traceable to the retrieved passages. The LLM did not introduce treatment options not present in the textbook passages (e.g., no references to specific brand-name drugs or recent guideline updates that postdate the textbook).

---

### Intake-to-Triage Test: Chest Pain and Hemoptysis

**Patient profile:** 58-year-old ex-smoker presenting with acute chest pain and hemoptysis. Previous stroke. Answers indicated pleuritic chest pain character, sudden onset, mild dyspnea on exertion, and no fever or productive cough. Urgency: urgent (hemoptysis triggered an urgent red flag in the Intake Agent).

**Step 1: Emergency guard** — passed (urgency = "urgent", not "emergency").

**Step 2: Generated clinical questions (4):**
1. "What are the differential diagnoses for chest pain and hemoptysis in an ex-smoker with a history of stroke?"
2. "What is the management of suspected pulmonary embolism in a patient with prior stroke?"
3. "What investigations are indicated for suspected ACS versus PE presenting with pleuritic chest pain?"
4. "What are the anticoagulation considerations for a patient with hemoptysis and a history of stroke?"

**Step 3: Retrieval per question:**
- Question 1: 3 chunks — Common Symptoms (HEMOPTYSIS section), Common Symptoms (CHEST PAIN section), Pulmonary Disorders
- Question 2: 3 chunks — Pulmonary Disorders (PE management), Heart Disease
- Question 3: 3 chunks — Common Symptoms, Heart Disease (ACS workup), Pulmonary Disorders
- Question 4: 3 chunks — Pulmonary Disorders, Blood Vessel & Lymphatic Disorders, Common Symptoms
- After deduplication by `chunk_id`: 12 unique chunks retained (some chunks retrieved by multiple questions; these were deduplicated)

**Step 4: Comprehensive synthesis report:**

- **Patient Overview:** 58-year-old ex-smoker, pleuritic chest pain with hemoptysis, previous stroke. Urgent presentation — hemoptysis in a patient with stroke history raises both hemorrhagic and thromboembolic risk simultaneously.
- **Differential Diagnosis (ranked with evidence):**
  1. Pulmonary embolism — pleuritic chest pain, hemoptysis, dyspnea, prior stroke as PE risk factor
  2. Acute coronary syndrome — chest pain, ex-smoker, cardiovascular risk profile
  3. Pneumothorax — sudden-onset pleuritic pain, ex-smoker
  4. Aortic dissection — sudden severe chest pain, though less consistent with hemoptysis
- **Management Plan:** PE workup takes priority given symptom constellation; anticoagulation considerations complicated by prior stroke and active hemoptysis
- **Investigations Required (5):** D-dimer, CT pulmonary angiography, troponin and ECG, chest X-ray, echocardiography
- **Red Flags and Safety Netting:** haemodynamic instability requiring immediate escalation, increasing hemoptysis volume, new neurological symptoms
- **Sources:** all 12 retrieved passages cited by chapter and section, organised by topic (respiratory, cardiac, vascular)

All four differential diagnoses were supported by specific passages from the textbook. The management plan's reference to the tension between anticoagulation need (PE) and anticoagulation risk (hemoptysis, prior stroke) was drawn from retrieved passages about PE management in high-bleeding-risk patients, not from general LLM reasoning.

---

## The TriageSession API

### `__init__(llm_model: str = "gpt-4o")`

Initialises all infrastructure. Loads models once and stores them as instance attributes. Prints status messages including which device the bi-encoder and cross-encoder are running on.

```python
session = TriageSession()
# Output:
# Initialising Triage Agent models...
# Opening ChromaDB client at data/chroma ...
#   Collection 'tmt_chunks' has 5,631 chunks.
# Loading bi-encoder 'sentence-transformers/embeddinggemma-300m-medical' on device 'mps' ...
#   Bi-encoder loaded.
# Loading cross-encoder 'BAAI/bge-reranker-v2-m3' on device 'cpu' ...
#   Cross-encoder loaded on CPU.
# Triage Agent ready  (bi-encoder on mps, reranker on cpu)
```

**Parameters:**
- `llm_model` — OpenAI model name. Defaults to `"gpt-4o"`. Can be set to `"gpt-4o-mini"` for lower cost during development. The model is initialised with `temperature=0` for deterministic outputs.

### `query(clinical_query: str, retrieve_k: int = 10, return_k: int = 3) -> dict`

Direct query mode. Runs the full retrieve-rerank-generate pipeline for a single clinical question.

**Returns:**
```python
{
    "query":            str,        # original question
    "retrieved_chunks": list[dict], # top-3 chunks, each with:
                                    #   chunk_id, text, chapter, section,
                                    #   distance, rerank_score, original_rank
    "analysis":         str,        # 5-section LLM analysis
}
```

**Parameters:**
- `clinical_query` — The clinical question to answer.
- `retrieve_k` — Number of candidates to fetch from the bi-encoder before reranking. Defaults to `RERANK_TOP_K_RETRIEVE` (10). Increasing this widens the candidate pool considered by the reranker but adds cross-encoder scoring calls.
- `return_k` — Number of passages to retain after reranking. Defaults to `RERANK_TOP_K_RETURN` (3). Increasing this provides more context to the LLM but may introduce lower-quality passages.

### `from_intake(intake_summary: dict, retrieve_k: int = 10, return_k: int = 3) -> dict`

Intake analysis mode. Runs the full multi-query pipeline from a structured intake summary dict.

**Returns:**
```python
{
    "intake_symptoms":        list[str],  # from intake_summary["symptoms"]
    "urgency":                str,        # "routine" or "urgent" (never "emergency")
    "generated_queries":      list[str],  # 2-4 LLM-generated clinical questions
    "per_query_results":      list[dict], # one triage_query result dict per question
    "comprehensive_analysis": str,        # 6-section synthesis report
}
```

For emergency cases (`urgency == "emergency"`), returns immediately with `per_query_results = []` and a short deferral string in `comprehensive_analysis`. The `generated_queries` list is empty.

**Parameters:** same `retrieve_k` and `return_k` semantics as `query()`, applied to every sub-query call.

---

## Scripts Reference

### `agents/triage_agent.py`

**Usage:**

```bash
# Interactive direct query mode (REPL — type queries, press Enter, type 'quit' to exit):
python agents/triage_agent.py

# Single query, non-interactive (prints result and exits):
python agents/triage_agent.py --query "What are the causes of hemoptysis in a young smoker?"

# From a saved intake summary JSON file:
python agents/triage_agent.py --from-intake data/results/last_intake.json

# Choose model (default: gpt-4o):
python agents/triage_agent.py --model gpt-4o-mini

# Custom retrieve_k and return_k:
python agents/triage_agent.py --query "..." --retrieve-k 15 --return-k 5
```

**CLI flags:**

| Flag | Default | Effect |
|---|---|---|
| `--model MODEL` | `gpt-4o` | OpenAI model name to use for all LLM calls (analysis, query generation, synthesis). Use `gpt-4o-mini` to reduce API cost during development. |
| `--query QUESTION` | None | Single clinical query — prints result and exits. When absent, the script enters interactive REPL mode. Takes precedence over `--from-intake` if both are provided (they should not be combined). |
| `--from-intake PATH` | None | Path to a JSON file containing an `IntakeSession.get_summary()` dict. When provided, runs intake analysis and exits. Takes precedence over interactive mode. |
| `--retrieve-k N` | 10 | Number of candidates to fetch from the bi-encoder. Passed to every `triage_query()` call. |
| `--return-k N` | 3 | Number of passages to retain after reranking. Passed to every `triage_query()` call. |

**Environment requirements:**

- Python 3.11+ with `sentence-transformers`, `chromadb`, `torch`, `langchain-openai`, `langchain-core`, `python-dotenv` installed.
- `data/chroma/` must contain a valid ChromaDB collection named `tmt_chunks`, built by Phase 2.2.
- `OPENAI_API_KEY` must be set in a `.env` file at the project root. This key is used for all LLM calls (analysis, query generation, synthesis).
- Internet access on first run to download model weights if not cached locally. Subsequent runs use the cached weights.
- MPS is attempted first for both bi-encoder and cross-encoder. The cross-encoder frequently falls back to CPU (this is expected behaviour — MPS support for custom architecture cross-encoders is partial).

**Execution modes and their CLI combinations:**

| Mode | Flags | Behaviour |
|---|---|---|
| Interactive REPL | (none, or `--model` only) | Enter queries at the `>` prompt; `quit` to exit. Loads models once, runs queries in a loop. |
| Single non-interactive query | `--query "..."` | Loads models, runs one query, prints result, exits. |
| Intake analysis | `--from-intake path.json` | Loads JSON, runs multi-query intake pipeline, prints comprehensive report, exits. |

---

## Configuration

All infrastructure parameters are defined in `config.py` and imported by `agents/triage_agent.py`:

```python
CHROMA_DIR             = DATA_DIR / "chroma"                                    # ChromaDB path
EMBEDDING_MODEL        = "sentence-transformers/embeddinggemma-300m-medical"    # bi-encoder
RERANKER_MODEL         = "BAAI/bge-reranker-v2-m3"                             # cross-encoder
RERANK_TOP_K_RETRIEVE  = 10   # candidates fetched from bi-encoder
RERANK_TOP_K_RETURN    = 3    # passages retained after reranking
```

These values reflect the Phase 3 recommendation: bi-encoder retrieves k=10 candidates, BGE cross-encoder reranks and returns the top 3. The Triage Agent inherits this recommendation directly — no separate configuration step is required.

The LLM model defaults to `"gpt-4o"` and is configurable via the `--model` CLI flag or the `llm_model` argument to `TriageSession.__init__()`. Temperature is fixed at 0 across all LLM calls for deterministic output.

---

## Limitations

### Source Citations Are at Chapter/Section Level, Not Page Level

Each chunk's metadata includes a `page_range` field indicating the page or page range in the textbook PDF where the chunk originates. This field is available in the chunk dicts returned by `retrieve_and_rerank()` but is not included in the context block passed to the LLM and therefore does not appear in citations.

Citations in the current implementation read as "Heart Disease / Cardiac Arrhythmias." A doctor wishing to verify this must navigate to that chapter and section in the physical or digital textbook. Adding `page_range` to the `_chunks_to_context()` formatting would make citations specific enough to turn to a page directly. This is a one-line change in `_chunks_to_context()` — it was not included in this phase because the primary focus was establishing the grounding mechanism, and page-level citation would require validating that the stored page ranges are accurate across all 5,631 chunks.

### Grounding Constraint Depends on Prompt Adherence

The instruction "do not add information from your general training data" is enforced entirely through the system prompt. GPT-4o at temperature=0 generally complies with this instruction, but compliance is not technically guaranteed. The model may — and in practice sometimes does — introduce clinical details that are consistent with the retrieved passages but not literally present in them. The line between "synthesising from retrieved evidence" and "supplementing with training knowledge" is not always crisp, and the LLM does not flag when it crosses it.

Systematic evaluation of grounding — comparing the content of LLM responses against the retrieved passages sentence-by-sentence — would require a dedicated evaluation pipeline. Such an evaluation is a natural Phase 5 validation task, particularly if the system is being considered for clinical deployment.

### Auto-Generated Questions May Miss Clinical Angles

The query generation step uses the LLM to produce 2–4 clinical questions from the intake summary. The quality of these questions determines the quality of the subsequent retrieval. If the LLM's query generation focuses on the most prominent symptoms while under-weighting the red flags or the patient's specific risk profile, the generated questions will retrieve evidence for the common case rather than for this patient's specific presentation.

In the chest pain and hemoptysis test, the LLM correctly generated a question about anticoagulation in stroke patients with hemoptysis — a nuanced clinical concern that requires integrating the stroke history, the hemoptysis, and the suspected PE simultaneously. This is a good generation. But it is not guaranteed: a different intake presentation might produce overly generic questions that retrieve useful but non-specific evidence.

This limitation is inherent to the single-pass query generation approach. A doctor reviewing the generated questions before the retrieval step — and having the ability to edit or supplement them — would address this limitation but would require adding a human-in-the-loop step to the pipeline.

### No Feedback Loop Within a Triage Session

The Triage Agent's direct query mode is stateless by design. A doctor who receives an analysis and wants to dig deeper into one specific aspect — "tell me more about the catheter ablation indication criteria" — must submit a new query. The agent has no memory of the previous query and no way to refine its analysis based on a follow-up request.

This is the correct design for an analysis engine, where each query is a self-contained retrieval task. But it means the Triage Agent cannot replicate the experience of reviewing a case with a consultant who builds on their previous analysis. Each query starts fresh from the textbook.

A future version that maintains query history within a session and supports iterative refinement would require: storing the previous query and its retrieved chunks, allowing the doctor to request refinement, and either re-running retrieval with a modified query or generating a follow-up analysis conditioned on the previous one. This is substantially more complex than the current architecture and was explicitly excluded from Phase 4.2 scope.

### No Integration with External Medical Databases

The Triage Agent's evidence base is a single textbook: CURRENT Medical Diagnosis and Treatment (2022 edition). It has no access to PubMed, UpToDate, clinical guidelines databases, drug formularies, or any other external medical knowledge source. All retrieved evidence is from this one source.

This has several practical implications:

- **Currency:** The 2022 edition reflects clinical evidence and guideline recommendations as of that publication date. Guidelines for some conditions (e.g., AF anticoagulation, DKA management) have been updated since then. The Triage Agent will cite and ground its analysis in 2022 content even if newer guidance would change the recommendations.
- **Depth:** A single textbook covers thousands of conditions at a breadth-oriented level appropriate for diagnosis and initial management. It does not provide the depth of a subspecialty reference or the up-to-date evidence synthesis of UpToDate. Complex cases may exhaust what the textbook can provide.
- **Coverage gaps:** Some conditions are addressed briefly or not at all in CURRENT Medical Diagnosis and Treatment. Rare conditions, recent emerging conditions, or highly subspecialty-specific presentations may not have retrievable passages in the current corpus.

Integration with external databases would require: designing authentication and API call infrastructure for each source, deciding how to weight evidence from different sources (textbook vs. guideline vs. primary literature), and handling potential contradictions between sources. This is a significant architectural expansion beyond Phase 4.2 scope and is the most important limitation to address in a production version of the system.

### LLM Cost Per Intake Analysis

An intake-mode analysis involves three categories of LLM calls:
1. One call for query generation
2. One call per generated question for single-query analysis (2–4 calls)
3. One call for comprehensive synthesis

This amounts to 4–6 LLM calls per intake analysis with GPT-4o. The synthesis call in particular involves a large context — all deduplicated retrieved passages plus all individual analyses — which can approach GPT-4o's context limits for complex presentations with many retrieved chunks. For development and testing, using `gpt-4o-mini` via the `--model` flag substantially reduces cost per run while preserving the pipeline structure.
