# Phase 5: Triage Agent

## Overview

Phase 5 builds the Triage Agent — a clinical diagnostic engine that produces actual diagnoses grounded exclusively in the medical textbook via the full RAG pipeline. Where Phase 4.1's Intake Agent collects structured patient data through a conversational interview, the Triage Agent receives that structured summary and produces a diagnosis: a primary diagnosis with confidence level, a ranked differential, step-by-step clinical reasoning, recommended investigations, management considerations, and red flags — all cited to specific textbook passages.

The word "diagnosis" is deliberate. The Phase 4.2 Triage Agent produced "clinical analysis" — structured retrieval-grounded answers to clinical questions generated from the intake summary. Phase 5 replaces that with a genuine diagnostic output: a primary diagnosis with a stated confidence level, a ranked differential, and an explicit reasoning chain that connects the patient's findings to the conclusion. The distinction matters: an analysis tells a doctor what the textbook says about a symptom; a diagnosis tells a doctor what the patient most likely has and why.

This output is not final. The diagnosis goes to a doctor for confirmation or rejection in Phase 7. The Triage Agent is a clinical reasoning support tool, not an autonomous decision-maker. Its role is to reduce the time a doctor must spend reviewing the initial evidence base and constructing a differential, not to replace that physician judgment.

The Triage Agent operates in two distinct modes determined by the nature of the presenting complaint:

**Mode A — Common Symptoms (from the Intake Agent).** The patient has presented with a recognised symptom category that the Intake Agent handles natively. The Intake Agent has already conducted a structured interview, collected Q&A answers, identified red flags, and assigned urgency. The Triage Agent receives this structured summary and runs a single-pass RAG pipeline: one compound query, one retrieval, one diagnosis generation. No questions are asked; no state is maintained across turns.

**Mode B — Uncommon Symptoms (raw complaint).** The patient has presented with a complaint that falls outside the Intake Agent's coverage — a presentation the system has not encountered, or a symptom described in terms the intake matching logic could not classify. There is no structured clinical history. The Triage Agent receives only the raw patient complaint and must build the clinical picture through a guided multi-pass retrieval process that mirrors the way a clinician approaches an unfamiliar presentation: first understand broadly what conditions might be relevant, then ask targeted questions to narrow the differential, then synthesise those answers into a diagnosis.

Both modes share the same underlying RAG infrastructure and the same diagnosis generation prompt. The difference is in how clinical context is assembled before that infrastructure is called.

### Where Phase 5 Sits in the Pipeline

```
Phases 1–3: RAG infrastructure
    Phase 1: PDF extraction, chunking, symptom structuring
    Phase 2: Embedding (embeddinggemma-300m-medical), ChromaDB vector store,
             retrieval validation (90% Hit@1, 100% Hit@3)
    Phase 3: Cross-encoder reranking — BGE-reranker-v2-m3 selected,
             +7.4% content relevance vs bi-encoder alone

Phase 4: Agents — Patient intake
    Phase 4.1: Intake Agent — structured multi-turn interview
               Produces: symptoms, Q&A answers, red flags, urgency, routing

Phase 5: Triage Agent — clinical diagnosis
    Mode A: receives Phase 4.1 intake summary → single-pass → diagnosis
    Mode B: receives raw complaint → multi-pass with questioning → diagnosis

Phase 7 (planned): Doctor review — confirm, reject, or escalate the diagnosis
```

---

## The Diagnostic Goal

### From Analysis to Diagnosis

The Phase 4.2 Triage Agent analysed. It received a clinical question — "Differential diagnosis for chest pain and hemoptysis in an ex-smoker with a previous stroke" — and returned a structured response citing the retrieved textbook passages. That response was useful, but it was a structured summary of what the textbook says, not a clinical decision.

Phase 5 shifts the output from analysis to diagnosis. The difference is directional commitment. An analysis presents the landscape of possibilities; a diagnosis commits to a primary conclusion and ranks the alternatives. A doctor reading a Phase 4.2 analysis still has to perform the integration step: weight the evidence, assign a primary diagnosis, order the differentials. Phase 5 moves that integration step inside the Triage Agent.

The output of a Phase 5 diagnosis is a report with seven defined sections: Primary Diagnosis (with explicit confidence level), Differential Diagnoses (ranked and evidence-cited), Clinical Reasoning (step-by-step, citing textbook passages), Recommended Investigations, Management Considerations, Red Flags and Safety Netting, and Sources. The doctor receives an integrated clinical conclusion, not raw material to integrate themselves.

### The Grounding Constraint

The diagnostic output is constrained to the retrieved textbook passages. The LLM system prompt is unambiguous: "Base your diagnosis ONLY on the evidence from the provided textbook passages. Do not add diagnoses or clinical reasoning from your general training data." Every diagnosis generated by Phase 5 is traceable to specific passages from CURRENT Medical Diagnosis and Treatment. If the retrieved passages do not support a confident diagnosis, the system is required to say so — to report low confidence and explain what evidence is missing, rather than to fill the gap with general LLM knowledge.

This grounding constraint is a clinical design property, not merely a technical preference. When a doctor receives the diagnosis report, the Sources section lists exactly which passages were used and what each contributed. The doctor can verify the basis of the diagnosis directly in the textbook. An ungrounded AI diagnosis — one that draws on the model's general training rather than a cited source — cannot be verified, challenged, or corrected at the evidence level. The grounding constraint makes the Triage Agent's reasoning transparent and auditable.

The constraint is enforced by prompt instruction and depends on LLM compliance. This is documented as a limitation in the final section.

### The Diagnosis Is Provisional

The Phase 5 diagnosis is the starting point for physician review, not the endpoint of care. Phase 7 will implement the doctor-facing interface through which the diagnosing physician confirms, modifies, or rejects the triage diagnosis. The Triage Agent's role is to reduce the cognitive load on the initial reviewing physician: to have already retrieved the relevant textbook evidence, reasoned through the differential, and produced a structured starting point that the doctor can confirm quickly or challenge with specific objections.

Cases that are too complex for the automated triage pipeline — presentations that would require more than three passes of evidence retrieval to adequately characterise, or cases where the retrieved passages conflict materially — are intended to be flagged for direct clinician review rather than forced to a low-confidence diagnosis. The hard pass cap in Mode B is part of this design: it is an acknowledgment that some cases should not be automated past a certain point.

---

## Architecture

### Why LangGraph for Mode B but Not Mode A

Mode A is structurally simple. It receives a complete, structured clinical picture from the Intake Agent, constructs a single compound query, runs the RAG pipeline once, and generates the diagnosis. This is a straight-line execution: no branching, no state transitions, no waiting for user input. Adding a graph framework to a linear pipeline would create infrastructure with no logic to represent.

Mode B is structurally complex. It begins with an unknown clinical picture, must retrieve evidence before it can ask meaningful questions, waits for patient answers between passes, evaluates whether those answers reveal something that changes the clinical picture, and conditionally triggers an additional retrieval pass. This workflow has genuine branching: whether to ask the next question or proceed to retrieval, whether to trigger Pass 3 or finalise the diagnosis after Pass 2. It also involves interruption — the graph must pause at the questioning step and resume when the patient responds. This is exactly what LangGraph is designed for.

The implementation reflects this asymmetry. Mode A runs by invoking the LangGraph compiled graph in a single call that flows deterministically from `analyze_input` through `retrieve_evidence` to `generate_diagnosis` and exits. Mode B uses the same graph but traverses a different subgraph — one with `ask_question` nodes that pause at `END` to wait for user input, and conditional routing after `generate_diagnosis_pass2` that decides whether `evaluate_refinement` should trigger Pass 3. Both modes use LangGraph, but Mode B is the reason LangGraph was chosen.

### The State Object

`TriageState` is a `TypedDict` with 14 fields. Every node in the graph reads from and writes to this state object. The state persists across all passes and across the question-answer turns within Mode B.

| Field | Type | Purpose |
|---|---|---|
| `messages` | `list[BaseMessage]` (append-only) | Full conversation history — human messages (patient answers, initial complaint) and AI messages (generated questions). Annotated with `add_messages` for LangGraph's merge semantics. |
| `mode` | `str` | `"common"` (Mode A) or `"uncommon"` (Mode B). Set by `analyze_input`. |
| `intake_summary` | `dict \| None` | The structured summary dict from `IntakeSession.get_summary()`. Populated before graph invocation in Mode A; `None` in Mode B. |
| `raw_complaint` | `str` | The raw patient complaint text. Used as the initial query in Mode B Pass 1. Empty string in Mode A. |
| `retrieved_chunks` | `list[dict]` | All unique chunks retrieved across all passes, deduplicated by `chunk_id`. Grows with each pass. |
| `current_pass` | `int` | Which retrieval pass is currently executing: 1, 2, or 3. |
| `generated_questions` | `list[str]` | The 4–6 patient-facing questions generated from Pass 1 retrieval. Populated once by `generate_questions`; consumed sequentially by `ask_question`. |
| `patient_answers` | `dict` | Maps each question string to the patient's answer string. Populated incrementally by `process_answer`. |
| `current_question_idx` | `int` | Index into `generated_questions` for the next question to ask. Incremented by `process_answer`. |
| `needs_refinement` | `bool` | Whether `evaluate_refinement` determined that a critical finding warrants Pass 3. Used by the conditional edge after `generate_diagnosis_pass2`. |
| `refinement_reason` | `str` | One-sentence explanation of the critical finding that triggered Pass 3. Used for logging and transparency. |
| `refinement_search_query` | `str` | The targeted search query constructed by the LLM for Pass 3 retrieval. Specific to the critical finding (e.g., `"erythema nodosum associated conditions and causes"`). |
| `diagnosis` | `dict` | The diagnosis report dict, populated by `generate_diagnosis`. Contains `report` (the full LLM text), `mode`, `pass`, and `num_chunks_used`. |
| `diagnosis_complete` | `bool` | Whether the triage session is complete. `True` after Mode A's single pass, after Pass 2 if no refinement is needed, or after Pass 3 unconditionally. |

### The LangGraph Execution Flow

```
START
  |
  v
analyze_input
  | (always routes to retrieve_evidence)
  v
retrieve_evidence
  |── mode="common" ──────────────────────────────> generate_diagnosis ──> END
  |
  |── mode="uncommon", pass=1 ─────────────────> generate_questions
                                                        |
                                                        v
                                                   ask_question ──> END (pause for user)
                                                   (resumes at process_answer in TriageSession.respond())
                                                        |
                                                   process_answer
                                                        |
                                                        |── more questions ──> ask_question
                                                        |
                                                        |── all answered ──> retrieve_evidence_pass2
                                                                                    |
                                                                                    v
                                                                          generate_diagnosis_pass2
                                                                                    |
                                                                                    |── pass >= 3 ──> END
                                                                                    |
                                                                                    |── pass == 2 ──> evaluate_refinement
                                                                                                           |
                                                                                                           |── needs_refinement=False ──> END
                                                                                                           |
                                                                                                           |── needs_refinement=True ──> retrieve_evidence_pass3
                                                                                                                                               |
                                                                                                                                               v
                                                                                                                                     generate_diagnosis_pass3 ──> END
```

LangGraph requires unique node names. Since the same underlying retrieval and diagnosis functions are called in different passes, the graph registers wrapper nodes (`retrieve_evidence_pass2`, `retrieve_evidence_pass3`, `generate_diagnosis_pass2`, `generate_diagnosis_pass3`) that set `current_pass` in the state dict before delegating to the shared function logic. The Pass 3 diagnosis node also forces `needs_refinement = False` before calling the shared function — a safety mechanism ensuring the hard stop.

---

## Mode A: Common Symptoms (From the Intake Agent)

### Input

Mode A receives a dict produced by `IntakeSession.get_summary()`. This dict carries the full clinical picture assembled by the Intake Agent during its multi-turn patient interview:

```python
{
    "symptoms":             list[str],   # matched symptom categories, e.g. ["chest pain", "hemoptysis"]
    "urgency":              str,         # "routine", "urgent", or "emergency"
    "answers":              dict,        # {question_text: answer_text} from the intake interview
    "triggered_red_flags":  list[dict],  # [{flag, urgency, question}, ...] for each triggered red flag
    "specialty_routing":    str,         # e.g. "Pulmonology" — from Intake Agent Phase 4.1
    "initial_workup":       list[str],   # initial workup items from the intake agent
}
```

Each field is actively used in the diagnosis:

- `symptoms` appears in the summary header of the patient context block, establishing the presenting complaint.
- `urgency` is read before the graph is invoked. If `urgency == "emergency"`, `diagnose_from_intake()` returns a deferral response immediately without entering the graph.
- `answers` is formatted as a Q&A block within the patient context that the LLM uses to reason about the clinical picture. These are the specific clinical details — character of pain, timing, aggravating factors, prior history — that differentiate one diagnosis from another.
- `triggered_red_flags` appear as a separate block in the patient context, labelled and urgency-tagged, so the LLM can appropriately weight their clinical significance.
- `specialty_routing` and `initial_workup` are present in the dict but are not directly used in the Phase 5 diagnosis prompt. They are preserved for Phase 7 integration.

### The Emergency Guard

The first thing `diagnose_from_intake()` does after receiving the intake summary is read the urgency field. If `urgency == "emergency"`, it returns a short deferral response without initialising the graph at all:

```python
{
    "report":         "Emergency case — patient has been directed to emergency services. Triage Agent defers.",
    "mode":           "common",
    "pass":           0,
    "num_chunks_used": 0,
    "deferred":       True,
}
```

This guard exists for the same reason it existed in Phase 4.2: the Intake Agent has already escalated an emergency case to emergency services. Running RAG retrieval and diagnosis generation on a STEMI or tension pneumothorax presentation wastes time the patient does not have and produces output no one will use. The Triage Agent has nothing to add to an emergency case; it gets out of the way.

### Single-Pass Pipeline

For routine and urgent cases, Mode A executes a single pass:

1. **Query construction.** `_build_mode_a_query()` assembles a compound clinical query from the intake summary. The query includes the presenting symptoms and urgency, a formatted block of all clinical history answers, and any triggered red flags by name. This query is specifically designed to be richer than any individual intake answer — it combines the full clinical picture into a single dense retrieval string that can surface relevant chapters across all the presenting concerns simultaneously.

2. **Retrieval.** `retrieve_and_rerank()` is called with `retrieve_k=10`, returning the top 3 chunks after cross-encoder reranking. These chunks are stored in `retrieved_chunks`.

3. **Context formatting.** `_build_intake_context_for_prompt()` formats the intake summary into a structured block that is injected into the diagnosis prompt alongside the retrieved passages. This block presents: presenting symptoms, urgency level, all intake Q&A pairs, and triggered red flags. The LLM receives both the patient's clinical history and the textbook evidence simultaneously.

4. **Diagnosis generation.** The LLM receives the structured patient context and the three retrieved passages, with the diagnosis system prompt. It generates the full seven-section diagnosis report in one call.

5. **Return.** The diagnosis dict is returned to the caller with the report text, mode, pass number, and chunk count.

### Mode A Output Structure

```python
{
    "report":         str,   # full seven-section LLM diagnosis report
    "mode":           "common",
    "pass":           1,
    "num_chunks_used": int,  # number of unique chunks retrieved (nominally 3)
}
```

---

## Mode B: Uncommon Symptoms (Multi-Pass RAG)

Mode B is the architectural centrepiece of Phase 5. It handles presentations that arrive without a structured clinical history — raw complaints that the Intake Agent did not cover, or presentations from a different entry point into the system entirely. The challenge is that without a prior interview, there is nothing to build a retrieval query from beyond the patient's own words. Those words are often imprecise, incomplete, or framed in lay terms that do not map directly to clinical vocabulary.

The solution is to use the first retrieval pass not to answer a clinical question, but to understand the problem space well enough to ask good questions. Then ask those questions, collect the answers, and use that enriched clinical picture for the diagnostic retrieval. This three-pass architecture mirrors the way a clinician approaches an unfamiliar presentation: gather background, take a focused history, revise the assessment based on what you learn.

### Pass 1: Broad Exploration

**Input:** The raw patient complaint string. Nothing else.

**Retrieval.** The complaint is submitted directly to `retrieve_and_rerank()` as a query. This is a broad, semantically fuzzy retrieval — it will surface content from whichever chapters and sections best match the lay description. For "I have a rash on my legs that's been spreading", the retrieval surfaces skin condition content from the relevant chapters. For "my heart keeps jumping", it surfaces cardiac content.

**Question generation.** The retrieved passages are not used to generate a diagnosis at this stage — the clinical picture is still too incomplete. Instead, they are passed to the LLM alongside the original complaint with the question generation system prompt. The LLM is asked to generate 4–6 patient-friendly questions that would differentiate between the most likely conditions given what the passages reveal about the problem space. The questions are designed around onset/duration, character, aggravating and relieving factors, associated symptoms, and relevant history — the standard clinical history-taking structure.

The question generation is grounded: the LLM generates questions based on what the retrieved passages indicate are the clinically meaningful distinctions for this type of presentation. If the passages surface content about inflammatory versus infectious skin conditions, the questions will target the features that distinguish them. The questions are not generated from the LLM's general clinical knowledge; they are generated from the specific evidence the retrieval step identified as relevant to this complaint.

**Output.** A list of 4–6 question strings, stored in `generated_questions`. The first question is emitted as an `AIMessage` and the graph pauses at `END`, returning control to `TriageSession.start_uncommon()`, which returns the first question to the caller.

### Pass 2: Targeted Analysis

**Input:** The original complaint plus all patient answers collected during the questioning phase.

**Query construction.** `_build_pass2_query()` combines the raw complaint with all collected Q&A pairs into a rich clinical query. Where Pass 1 searched with "I have a rash on my legs that's been spreading", Pass 2 searches with the full clinical picture: complaint, duration, associated fever and joint pain, pain character, appearance changes, history of fungal infections, and absence of recent contact or injury. This is a fundamentally different query from Pass 1 — it is specific, clinical, and targeted to the specific differential the answers have established.

**Retrieval.** `retrieve_and_rerank()` is called again with the enriched query. The returned chunks are deduplicated against `retrieved_chunks` by `chunk_id` to avoid presenting the same passage twice in the diagnosis prompt. Both the Pass 1 and Pass 2 chunks are now available to the diagnosis step.

**Preliminary diagnosis.** All retrieved chunks and all clinical context (complaint plus all answers) are passed to the diagnosis LLM to generate a preliminary diagnosis report. This is the same seven-section format as Mode A — a primary diagnosis with confidence, ranked differentials, and clinical reasoning.

**Refinement evaluation.** After the preliminary diagnosis is generated, the LLM is called a second time with the `evaluate_refinement` prompt. This call receives the complaint, all patient answers, and the preliminary diagnosis text. It must return a structured JSON decision:

```json
{
  "needs_refinement": true,
  "reason": "Patient reported fever and joint pain alongside the leg rash, which significantly narrows the differential toward systemic inflammatory conditions rather than primary skin conditions.",
  "search_query": "erythema nodosum associated conditions and causes"
}
```

If `needs_refinement` is `false`, the preliminary diagnosis is finalised and the session is complete. If `needs_refinement` is `true`, Pass 3 is triggered with the specified `search_query`.

**What constitutes a critical finding.** The `evaluate_refinement` system prompt defines a critical finding as one that: introduces a new specific diagnosis not previously considered; strongly rules in or rules out a major condition; or reveals an important historical fact (trauma, prior procedure, family history) that redirects the clinical picture. The bar is intentionally high — routine confirmatory details should not trigger Pass 3. Pass 3 is for findings that would cause a clinician to pause and reassess the working diagnosis before finalising it.

### Pass 3: Critical Finding Refinement (Conditional)

Pass 3 fires only when the `evaluate_refinement` node returns `needs_refinement=True`. It uses the `refinement_search_query` constructed by the LLM — a targeted query specific to the critical finding — rather than repeating the Pass 2 query.

**Retrieval.** `retrieve_and_rerank()` is called with the targeted query. For the leg rash case, this is `"erythema nodosum associated conditions and causes"` — a query that will retrieve content about the systemic associations of the condition (sarcoidosis, tuberculosis, streptococcal infection, inflammatory bowel disease) that the first two passes, focused on skin presentation, may not have surfaced. New chunks are deduplicated against the existing `retrieved_chunks` pool before being added.

**Final diagnosis.** The LLM generates the final diagnosis report with all accumulated chunks: Pass 1 chunks (broad skin condition context), Pass 2 chunks (targeted inflammatory skin condition context), and Pass 3 chunks (systemic associations of the primary diagnosis). The Pass 3 diagnosis is forced final: `needs_refinement` is set to `False` before the diagnosis function is called, and `generate_diagnosis_pass3` explicitly sets `diagnosis_complete=True` in its return. The hard stop is enforced at multiple levels.

**HARD STOP.** Pass 3 never triggers Pass 4. This is a design decision, not an oversight. Its rationale is documented in the next section.

### Why Three Passes, Not More

The three-pass architecture is bounded deliberately. The bound is not arbitrary — it reflects a claim about the structure of clinical reasoning for automated triage:

**Pass 1** solves the cold-start problem: without any clinical history, the system cannot ask meaningful questions. Pass 1 retrieves enough context to make the questions intelligent rather than generic.

**Pass 2** solves the blank-slate problem: with patient answers in hand, the system now has a real clinical picture to work with. Pass 2 does the actual diagnostic retrieval — the same function Mode A performs in its single pass, but from a richer query built from the interview.

**Pass 3** solves the surprise problem: a patient answer reveals something that changes the clinical picture in a way that the Pass 2 retrieval did not anticipate. One additional targeted search on that specific finding is clinically warranted.

A presentation requiring four or more passes of targeted evidence retrieval to resolve is one where the clinical picture is changing materially with each new piece of information — where the diagnosis is genuinely unstable under the available evidence. Such cases are not appropriate for automated triage. The right response is not a fourth pass; it is flagging the case for direct clinician review. The hard stop at three passes is therefore a safety property: it forces the system to produce its best current diagnosis and acknowledge its limits, rather than iterating indefinitely on a presentation that should be escalated.

This mirrors the structure of clinical reasoning in constrained settings. A house officer triaging in an emergency department does not have unlimited time to take history. They gather an initial impression, take a focused history, note any surprises, and make a provisional call — then escalate if the picture does not settle. Three passes is the automated equivalent of that triage workflow.

---

## The Infinite Loop Problem and Its Solution

Multi-pass retrieval with conditional triggering creates a potential for unbounded iteration. The logical structure of Pass 2 — answer reveals finding → search for finding → new evidence → revise diagnosis — could in principle repeat indefinitely:

```
Patient answers reveal X → search for X → X is associated with Y and Z →
need to know more about Y → more questions? → answers reveal W →
search for W → W is associated with V → ...
```

This is not a hypothetical concern. A system without explicit pass limits would drift into successive refinements, each triggered by the previous one, until either the LLM stops finding critical findings or the recursion hits a timeout.

The solution in Phase 5 is a hard cap with distinct semantic purpose for each pass:

1. **Hard cap of 3 passes.** `current_pass >= 3` is a hard stop condition checked in `generate_diagnosis_fn` before determining `is_final`. If `current_pass >= 3`, `is_final = True` regardless of any other condition. This check exists at multiple levels: in the routing function `route_after_diagnosis`, in `generate_diagnosis_pass3` which sets `diagnosis_complete=True` explicitly, and in `TriageSession.respond()` which checks `current_pass >= 3` before calling `evaluate_refinement`.

2. **Pass 3 never evaluates refinement.** `generate_diagnosis_pass3` sets `needs_refinement=False` before calling the shared diagnosis function, ensuring the `is_final` check in that function will always return `True` for Pass 3. The routing function `route_after_diagnosis` routes `current_pass >= 3` directly to `END` without visiting `evaluate_refinement`. The cap is enforced by graph structure, not just by state logic.

3. **Manual step execution in `TriageSession.respond()`.** Because LangGraph does not natively support mid-graph resumption without checkpointing, the respond method drives Pass 2 and Pass 3 steps manually after all questions have been answered. This code path explicitly checks `current_pass >= 3` before calling `evaluate_refinement` and explicitly sets `needs_refinement=False` before running Pass 3 diagnosis. The manual path and the graph path enforce the same hard stop through independent code paths.

The result is that Pass 3 is the absolute terminal pass. The LLM is not asked to evaluate refinement after Pass 3, so it cannot trigger a fourth pass even if it believes one is warranted. The system produces its best diagnosis with the accumulated evidence and stops.

---

## The Diagnosis Report Format

The same seven-section format is used for all outputs: Mode A, Mode B Pass 2 preliminary diagnosis, and Mode B Pass 3 final diagnosis. The format is enforced by the `_DIAGNOSIS_SYSTEM` prompt.

### 1. Primary Diagnosis

The most likely diagnosis with an explicit confidence level: `high`, `moderate`, or `low`. The confidence level is assessed by the LLM based on how strongly the patient's presentation and the retrieved evidence support the leading diagnosis. High confidence means the presentation is characteristic and the textbook evidence is direct and specific. Moderate confidence means the diagnosis fits but key confirmatory findings or investigations are pending. Low confidence means multiple diagnoses remain plausible and the distinction requires further workup.

### 2. Differential Diagnoses

A ranked list of alternative diagnoses. Each entry includes: the condition name, key supporting evidence from the patient's presentation, key evidence from the textbook passages, and an explicit comparison with the primary diagnosis — why this condition is more or less likely. The ranking reflects the LLM's clinical weighting of all available evidence against the criteria the textbook provides for each condition.

### 3. Clinical Reasoning

Step-by-step reasoning connecting the patient's symptoms and answers to the diagnosis, with citations to specific textbook passages. This is the most important section for physician review: it makes the logic of the diagnosis transparent and auditable. A doctor who disagrees with the primary diagnosis can use the clinical reasoning section to identify which specific inference they would dispute.

### 4. Recommended Investigations

Specific investigations to confirm the primary diagnosis or rule out the top differentials, with clinical justification drawn from the textbook evidence. The investigation list is not generic — it reflects both the specific presentation and the textbook's guidance on diagnostic workup for the suspected condition.

### 5. Management Considerations

Initial management steps suggested by the textbook evidence. These are considerations for the reviewing physician, not prescriptions. The grounding constraint applies: management suggestions are cited to retrieved passages, not generated from general LLM medical knowledge.

### 6. Red Flags and Safety Netting

Findings that would change the diagnosis, worsen the clinical picture, or require immediate escalation. This section is the Triage Agent's explicit acknowledgment that the diagnosis is provisional — it documents the conditions under which the diagnosis should be reconsidered.

### 7. Sources

Each textbook passage used in the diagnosis, cited by chapter and section, with a description of what that passage contributed to the clinical reasoning. This section makes the grounding constraint auditable: a doctor can open the textbook to the cited chapter and section and verify that the diagnosis is derived from the stated evidence.

---

## Data Flow: Phases 2 and 3 into Phase 5

The Triage Agent is a consumer of the RAG infrastructure built across Phases 2 and 3. It imports and uses this infrastructure without modifying it:

```python
from rag.reranker import (
    retrieve_and_rerank,
    open_collection,
    load_bi_encoder,
    load_cross_encoder,
    detect_device,
)
from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RERANK_TOP_K_RETRIEVE,
    RERANK_TOP_K_RETURN,
)
```

**Phase 2.1** produced 5,631 embeddings of textbook chunks using `sentence-transformers/embeddinggemma-300m-medical`. These embeddings are stored in ChromaDB. The Triage Agent's bi-encoder (`load_bi_encoder`) loads the same model to encode query strings at inference time, so query vectors and chunk vectors occupy the same vector space.

**Phase 2.2** stored the 5,631 embeddings in a ChromaDB persistent collection at `data/chroma/`. The Triage Agent opens this collection at initialisation via `open_collection(CHROMA_DIR)`. Retrieval is performed by `retrieve_and_rerank()` against this collection.

**Phase 3** evaluated and selected `BAAI/bge-reranker-v2-m3` as the cross-encoder reranker, establishing `RERANK_TOP_K_RETRIEVE=10` and `RERANK_TOP_K_RETURN=3` as the retrieval configuration. These values are used unchanged in the Triage Agent. Every call to `retrieve_and_rerank()` in Phase 5 fetches 10 candidates and returns 3 after reranking.

**Chunk deduplication across passes.** Multiple retrieval passes against the same complaint may surface the same passages. A passage about skin inflammation relevant to the initial rash complaint may also be relevant to the targeted systemic conditions search in Pass 3. `_deduplicate_chunks()` maintains a `seen_ids` set of `chunk_id` values and filters out any chunk already present in `retrieved_chunks` before appending new results. This ensures the LLM diagnosis context does not contain repeated passages, and that `num_chunks_used` accurately reflects the number of distinct textbook passages consulted.

```python
def _deduplicate_chunks(
    existing: list[dict],
    new_chunks: list[dict],
    seen_ids: set[str],
) -> tuple[list[dict], set[str]]:
    result = list(existing)
    for chunk in new_chunks:
        cid = chunk.get("chunk_id", "")
        if cid not in seen_ids:
            result.append(chunk)
            seen_ids.add(cid)
    return result, seen_ids
```

---

## How Mode A Receives the Intake Summary

The intake summary dict is produced by `IntakeSession.get_summary()` in Phase 4.1. The Triage Agent consumes it through two distinct transformation functions:

**`_build_mode_a_query()`** constructs the retrieval query. It concatenates the symptom list, urgency level, all clinical Q&A pairs (formatted as bullet points), and any triggered red flags into a single dense query string. This string is submitted to `retrieve_and_rerank()`. The goal is to produce a query that is clinically specific enough for the BGE cross-encoder to surface the most relevant passages — richer than a bare symptom name, incorporating the clinical details that distinguish one presentation from another.

**`_build_intake_context_for_prompt()`** constructs the patient context block injected into the diagnosis LLM prompt. It formats the same information as a readable structured block: presenting symptoms header, urgency level (in uppercase), all Q&A pairs labelled, and triggered red flags with their urgency tags. This block accompanies the retrieved textbook passages in the LLM's context window. The LLM reasons over both simultaneously: the patient's specific clinical picture and the textbook's evidence about the relevant conditions.

The distinction between these two functions matters. The retrieval query is optimised for embedding similarity — it needs to be information-dense and clinically specific so that the bi-encoder and cross-encoder can identify the most relevant passages. The patient context block is optimised for LLM readability — it needs to present the information in a structured, clearly labelled format that the LLM can reason about directly. The same underlying data is transformed differently for two different consumers.

---

## Test Results

### Mode A: Chest Pain and Hemoptysis (From Intake Agent)

**Patient profile:** Urgent case. Ex-smoker. Previous stroke. Presenting complaint: tight chest pain with radiation to both arms, worsened with exertion, haemoptysis.

**Intake summary fields used:**
- Symptoms: `["chest pain", "hemoptysis"]`
- Urgency: `"urgent"`
- Answers included: exertional worsening, bilateral arm radiation, prior stroke history, ex-smoker status, no fever or productive cough
- Triggered red flags: hemoptysis (urgent)

**Emergency guard:** Passed — urgency is `"urgent"`, not `"emergency"`.

**Query constructed by `_build_mode_a_query()`:**
```
Patient presenting with: chest pain, hemoptysis. Urgency: urgent. Clinical history:
  - What is the character of your chest pain?: Tight, pressure-like
  - Does it radiate anywhere?: Yes, both arms
  - Does it worsen with exertion?: Yes
  - Any history of smoking?: Ex-smoker
  - Any prior medical history?: Previous stroke
  ... Red flags present: hemoptysis [urgent].
```

**Retrieval:** 3 chunks retrieved. Chapters: Common Symptoms (HEMOPTYSIS section), Pulmonary Disorders (PE section). Pass 1 only — single pass.

**Diagnosis generated:**
- **Primary Diagnosis:** Pulmonary Embolism — high confidence. Pleuritic character (tight, radiating), haemoptysis, exertional worsening, prior stroke as thromboembolic risk factor, ex-smoker status.
- **Differential Diagnoses (ranked):**
  1. Acute Coronary Syndrome — bilateral arm radiation, exertional worsening, ex-smoker cardiovascular risk
  2. Aortic Dissection — sudden severe chest pain, though haemoptysis less consistent
  3. Pericarditis — pleuritic component, though fever absent
- **Clinical Reasoning:** Connected stroke history to hypercoagulable state, exertional worsening to ventilation-perfusion mismatch, haemoptysis to pulmonary infarction
- **Investigations:** CT pulmonary angiography, D-dimer, troponin and ECG, chest X-ray
- **Sources:** 3 passages cited — HEMOPTYSIS section (differential diagnosis of blood-streaked sputum), Pulmonary Disorders PE section (risk factors and clinical features), Pulmonary Disorders PE section (diagnostic approach)

**Metadata:** Mode A, Pass 1, 3 chunks used.

---

### Mode B: Leg Rash (Uncommon Symptom)

**Initial complaint:** `"I have a rash on my legs that's been spreading"`

#### Pass 1 — Broad Exploration

**Query:** `"I have a rash on my legs that's been spreading"` (raw complaint, unmodified)

**Retrieved chunks:** Skin and appendages content — dermatitis sections, inflammatory skin conditions, differential approaches to lower limb rashes.

**Questions generated (6):**
1. "How long have you had this rash, and how quickly has it been spreading?"
2. "Do you have any fever, joint pain, or feel generally unwell alongside the rash?"
3. "Is the rash painful, itchy, or painless?"
4. "Has the rash changed in colour or texture since it first appeared?"
5. "Have you had any fungal skin infections in the past, or does anyone in your household have a similar rash?"
6. "Have you had any recent injuries, insect bites, or contact with new substances such as plants, chemicals, or new clothing?"

Questions target: duration and progression speed, systemic symptoms (critical for distinguishing inflammatory from infectious), sensory character, morphological change, relevant prior history, and contact/trigger history. These are directly derived from the clinical distinctions the Pass 1 passages identified as diagnostically important.

#### Patient Answers Collected

| Question | Answer |
|---|---|
| Duration and spread | About 1 week, spreading slowly |
| Systemic symptoms | Yes — fever and joint pain |
| Painful or itchy | Painful, tender to touch |
| Colour and texture changes | Yes — raised, red, nodular |
| Prior fungal infections | No |
| Contact or injury | No |

#### Pass 2 — Targeted Analysis

**Query (constructed by `_build_pass2_query()`):**
```
Patient complaint: I have a rash on my legs that's been spreading.
Patient history from follow-up questions:
  - How long have you had this rash?: About 1 week, spreading slowly
  - Do you have any fever, joint pain, or feel generally unwell?: Yes — fever and joint pain
  - Is the rash painful, itchy, or painless?: Painful, tender to touch
  - Has the rash changed in colour or texture?: Yes — raised, red, nodular
  - Prior fungal infections?: No
  - Recent contact or injury?: No
```

**Retrieval:** Targeted chunks — inflammatory skin conditions, nodular erythematous lesions, panniculitis content.

**Preliminary diagnosis:** Erythema Nodosum — moderate confidence. Raised tender red nodular lesions on lower legs, fever, joint pain, one-week duration. Clinical reasoning cited: classic clinical triad of painful nodules, fever, and arthralgia in the context of a systemic inflammatory trigger.

**Refinement evaluation:** The LLM identified fever combined with joint pain as a critical finding. The preliminary diagnosis of erythema nodosum is directionally correct, but erythema nodosum is almost always a reactive condition associated with an underlying systemic disease — sarcoidosis, tuberculosis, streptococcal infection, inflammatory bowel disease. The preliminary retrieval had not specifically targeted these associations. The LLM returned:

```json
{
  "needs_refinement": true,
  "reason": "Fever and joint pain alongside erythema nodosum pattern suggests a systemic inflammatory trigger that significantly narrows the differential — targeted search on associated conditions is warranted.",
  "search_query": "erythema nodosum associated conditions and causes"
}
```

**Pass 3 triggered.**

#### Pass 3 — Critical Finding Refinement

**Query:** `"erythema nodosum associated conditions and causes"` (the targeted refinement query, not the original complaint)

**Retrieved chunks:** Erythema nodosum section — associated conditions (sarcoidosis, tuberculosis, streptococcal pharyngitis, IBD, drug reactions), diagnostic investigations, systemic workup approach. These are new chunks not retrieved in Passes 1 or 2.

**Final diagnosis (Pass 3):**
- **Primary Diagnosis:** Erythema Nodosum — high confidence. Clinical triad confirmed: painful tender nodular lesions on lower legs, fever, arthralgia. Duration and progression consistent.
- **Differential Diagnoses:** Reactive arthritis, cellulitis (less likely — no unilateral focus), vasculitis
- **Clinical Reasoning:** Step-by-step from nodular morphology → panniculitis pattern → erythema nodosum → systemic trigger evaluation
- **Recommended Investigations:** Throat swab and ASO titres (streptococcal trigger), chest X-ray (sarcoidosis, primary TB), QuantiFERON-TB Gold (TB), ESR and CRP, ANA (autoimmune screening), review of recent medications
- **Sources:** 8 passages cited across 3 passes

**Metadata:** Mode B, Pass 3, 8 chunks used.

The Pass 3 refinement produced a materially different investigation list from what Pass 2 alone would have generated. Without the targeted search on associated conditions, the investigation recommendations would have focused on the skin manifestation. With the Pass 3 evidence, the diagnosis correctly orients the workup toward identifying the systemic trigger — which is the clinically important question for erythema nodosum management.

---

## The TriageSession API

`TriageSession` is the public interface for the Triage Agent. It manages model loading, graph construction, and conversation state for a single triage session.

### `__init__(llm_model, retrieve_k, return_k)`

```python
def __init__(
    self,
    llm_model: str = "gpt-4o",
    retrieve_k: int = RERANK_TOP_K_RETRIEVE,   # 10
    return_k: int = RERANK_TOP_K_RETURN,        # 3
):
```

Initialises all infrastructure components: the OpenAI LLM client (temperature=0), the compute device (MPS on Apple Silicon, CPU fallback), the ChromaDB collection, the bi-encoder, and the cross-encoder. Builds and compiles the LangGraph graph. Stores all components as instance attributes. Subsequent calls to `diagnose_from_intake()`, `start_uncommon()`, and `respond()` reuse the loaded models without reloading them.

Prints status messages on initialisation:
```
Initialising Triage Agent models...
Triage Agent ready  (bi-encoder on mps, reranker on cpu)
```

**Parameters:**
- `llm_model` — OpenAI model name. Default `"gpt-4o"`. Set to `"gpt-4o-mini"` for development to reduce API cost. Temperature is fixed at 0 for deterministic outputs.
- `retrieve_k` — Number of bi-encoder candidates. Inherits `RERANK_TOP_K_RETRIEVE=10` from `config.py`. Applies to every retrieval call across all passes.
- `return_k` — Number of passages kept after reranking. Inherits `RERANK_TOP_K_RETURN=3` from `config.py`. Applies to every retrieval call across all passes.

### `diagnose_from_intake(intake_summary)`

```python
def diagnose_from_intake(self, intake_summary: dict) -> dict:
```

Mode A entry point. Receives the structured intake summary dict, runs the emergency guard, initialises state, invokes the graph in a single call, and returns the diagnosis dict. Non-interactive — blocks until the diagnosis is complete and returns.

**Returns:**
```python
{
    "report":         str,   # full seven-section diagnosis report
    "mode":           "common",
    "pass":           1,
    "num_chunks_used": int,
}
```

For emergency cases:
```python
{
    "report":         str,   # short deferral message
    "mode":           "common",
    "pass":           0,
    "num_chunks_used": 0,
    "deferred":       True,
}
```

### `start_uncommon(patient_complaint)`

```python
def start_uncommon(self, patient_complaint: str) -> str:
```

Mode B entry point. Receives the raw complaint string, initialises state, invokes the graph through Pass 1 and question generation, and returns the first question string. The graph pauses at `END` after emitting the first question.

**Returns:** The first clinical question as a string.

If the graph completes without generating questions (this is the rare edge case where Pass 1 retrieval itself produces a sufficient diagnosis), `start_uncommon()` returns the diagnosis report text directly and sets `_phase = "done"`.

### `respond(patient_answer)`

```python
def respond(self, patient_answer: str) -> str:
```

Mode B answer processing. Records the patient's answer, increments the question index, and either returns the next question (if more questions remain) or runs Pass 2 and optionally Pass 3 to completion.

When all questions have been answered:
1. Runs Pass 2 retrieval and diagnosis manually (driving the nodes directly rather than reinvoking the graph, since LangGraph does not support mid-graph resumption without checkpointing).
2. If Pass 2 determines no refinement is needed (`needs_refinement=False`), finalises and returns the diagnosis report.
3. If Pass 3 is needed, runs Pass 3 retrieval and diagnosis, then finalises and returns the diagnosis report.

**Returns:** The next question string, or the final diagnosis report string when complete.

### `is_complete()`

```python
def is_complete(self) -> bool:
```

Returns `True` when the diagnosis has been generated. After `is_complete()` returns `True`, `get_diagnosis()` will return the full diagnosis dict.

### `get_diagnosis()`

```python
def get_diagnosis(self) -> dict:
```

Returns the full diagnosis dict from the current session state. Safe to call at any time; returns an empty dict if the session has not completed.

### Usage Examples

```python
# Mode A — single-pass from intake
session = TriageSession()
diagnosis = session.diagnose_from_intake(intake_summary_dict)
print(diagnosis["report"])

# Mode B — conversational
session = TriageSession()
first_question = session.start_uncommon("I have a rash on my legs that's been spreading")
print(first_question)

while not session.is_complete():
    answer = input("Patient: ")
    response = session.respond(answer)
    if session.is_complete():
        print(session.get_diagnosis()["report"])
    else:
        print(f"Agent: {response}")
```

---

## Scripts Reference

### `agents/triage_agent.py`

**Usage:**

```bash
# Mode A — from a saved intake summary JSON:
python agents/triage_agent.py --from-intake data/results/last_intake.json

# Mode B — interactive, starting from a raw complaint:
python agents/triage_agent.py --query "I have a rash on my legs that's been spreading"

# Mode B — fully interactive (prompts for complaint):
python agents/triage_agent.py

# Custom model and retrieval parameters:
python agents/triage_agent.py --model gpt-4o-mini --retrieve-k 15 --return-k 5
```

**CLI flags:**

| Flag | Default | Effect |
|---|---|---|
| `--model MODEL` | `gpt-4o` | OpenAI model name. Use `gpt-4o-mini` to reduce API cost during development. |
| `--from-intake PATH` | None | Path to a JSON file containing an `IntakeSession.get_summary()` dict. Runs Mode A non-interactively and exits. |
| `--query COMPLAINT` | None | Raw patient complaint string. Runs Mode B interactively — the agent asks follow-up questions and generates a diagnosis when complete. |
| `--retrieve-k N` | 10 | Candidates fetched from bi-encoder. Applied to all retrieval passes. |
| `--return-k N` | 3 | Passages retained after reranking. Applied to all retrieval passes. |

**Execution modes:**

| Mode | Flags | Behaviour |
|---|---|---|
| Mode A — non-interactive | `--from-intake path.json` | Loads JSON, runs single-pass diagnosis, prints report, exits. |
| Mode B — complaint from flag | `--query "complaint"` | Runs interactive Q&A loop from given complaint, prints diagnosis when complete, exits. |
| Mode B — fully interactive | (none) | Prompts for complaint, runs interactive Q&A loop, prints diagnosis when complete. |

The interactive Mode B loop (`_run_mode_b_interactive`) handles `quit`, `exit`, and `q` as session termination signals, and `KeyboardInterrupt` gracefully. Empty responses are skipped without consuming a question slot.

---

## Configuration

All infrastructure parameters are defined in `config.py` and imported by `agents/triage_agent.py`:

```python
CHROMA_DIR            = DATA_DIR / "chroma"
EMBEDDING_MODEL       = "sentence-transformers/embeddinggemma-300m-medical"
RERANKER_MODEL        = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K_RETRIEVE = 10   # candidates fetched from bi-encoder
RERANK_TOP_K_RETURN   = 3    # passages retained after reranking
```

These values reflect the Phase 3 recommendation. No separate configuration step is required — the Triage Agent inherits the Phase 3 recommendation directly from `config.py`.

The LLM defaults to `"gpt-4o"` and is configurable via `--model` or the `llm_model` argument to `TriageSession.__init__()`. Temperature is fixed at 0 across all LLM calls: question generation, refinement evaluation, and all diagnosis generation passes. Determinism is a clinical requirement — the same presentation should produce the same diagnosis on repeated runs.

---

## Limitations

### Pass 3 Trigger Depends on LLM Judgment

The decision to trigger Pass 3 is made by the LLM in the `evaluate_refinement` step. The LLM is given the complaint, all patient answers, and the preliminary diagnosis, and asked to determine whether any answer revealed a critical finding. This judgment is based on prompt instruction and the LLM's training, not on a calibrated statistical threshold.

In practice, this means the trigger may fire when it should not (over-triggering on findings that are clinically relevant but not diagnostic game-changers) or fail to fire when it should (missing subtle implications of a patient answer that a specialist would immediately recognise). There is no ground-truth dataset of "Pass 3 warranted" labels against which to calibrate this judgment.

Over-triggering adds one unnecessary retrieval pass and an additional LLM call — cost and latency, but not a clinical harm. Under-triggering is the more concerning failure mode: a critical finding is identified in the patient answers, the Pass 2 diagnosis does not reflect it fully, and the system finalises a diagnosis that would have been meaningfully refined by one additional search. The hard pass cap prevents the system from compensating for under-triggering by adding more passes.

### Mode B Questions Are Generated from Pass 1 Retrieval

The questions asked during Mode B are generated from the chunks retrieved in Pass 1. If Pass 1 retrieves content from the wrong clinical domain — because the complaint is ambiguous or because the embedding model ranks a tangential chapter highly for the specific vocabulary used — the generated questions will target the wrong differential. Questions about a misidentified problem space will produce patient answers that do not help narrow the correct differential. Pass 2 retrieval, built from those answers, will remain misdirected.

This is the most significant failure mode in Mode B. It is structurally difficult to address without either a fallback that recognises when Pass 1 retrieval is misaligned or a pre-retrieval disambiguation step that resolves the most ambiguous complaints before submitting them to the bi-encoder. Neither mechanism is implemented in Phase 5.

### Confidence Levels Are LLM-Assessed, Not Statistically Calibrated

The `high`, `moderate`, and `low` confidence designations in the Primary Diagnosis section are generated by the LLM as part of the diagnosis report. They reflect the LLM's assessment of how strongly the evidence supports the primary diagnosis, not a calibrated probability. A `high confidence` diagnosis is not a claim that the condition is present with 90% probability — it is the LLM's assertion that the patient's presentation and the retrieved evidence are strongly consistent with that diagnosis.

These confidence labels could mislead a reviewing physician if interpreted as calibrated probabilities rather than qualitative assessments. A future version that ground-truths LLM confidence labels against eventual clinical outcomes — tracking whether `high confidence` diagnoses are confirmed at a higher rate than `moderate confidence` diagnoses — would enable better calibration and more informative confidence signals.

### Evidence Base Is a Single Textbook

All retrieved evidence comes from a single source: CURRENT Medical Diagnosis and Treatment (2022 edition). The Triage Agent has no access to PubMed literature, clinical guidelines databases, drug formularies, or any other knowledge source. This has several implications:

**Currency:** The 2022 edition reflects guideline recommendations current as of that publication date. Guidelines for some conditions have been updated since then. The Triage Agent will ground its diagnosis in 2022 content even where newer guidance would change the recommendations.

**Depth:** A general medical textbook covers thousands of conditions at a breadth appropriate for diagnosis and initial management. It does not provide the subspecialty depth of a condition-specific reference. Complex presentations of rare conditions may exhaust what the textbook can provide.

**Coverage gaps:** Rare conditions, recently described syndromes, and highly subspecialty-specific presentations may not have sufficient textbook coverage to generate a specific diagnosis. The grounding constraint means the system will report low confidence or insufficient evidence rather than confabulating — but the practical result is that some presentations cannot be adequately triaged with this evidence base alone.

### Doctor Feedback Loop Not Yet Implemented

Phase 7 will implement the doctor-facing interface through which the reviewing physician confirms, modifies, or rejects the Triage Agent's diagnosis. Until Phase 7 is implemented, there is no mechanism for doctor feedback to improve the system's future performance. The diagnosis is generated, presented, and consumed — but the outcome (confirmed, rejected, modified) is not recorded.

This means the system cannot learn from its errors. A diagnosis that was confidently wrong — high confidence, primary diagnosis incorrect — leaves no trace in the system. A systematic pattern of errors in a particular clinical domain would not be detectable without the Phase 7 feedback loop. Phase 7 is therefore not merely a UX addition; it is the data collection mechanism that makes systematic evaluation and improvement possible.
