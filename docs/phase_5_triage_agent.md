# Phase 5: Triage Agent

## Overview

Phase 5 is the diagnostic engine of the Medora system. It receives clinical information — either a fully structured intake summary from the Phase 4.1 Intake Agent, or a raw patient complaint that bypassed structured intake — and produces an actual diagnosis: a primary diagnosis with confidence level, a ranked differential, step-by-step clinical reasoning cited to specific textbook passages, recommended investigations, management considerations, and red flags. Every element of the output is grounded in retrieved passages from CURRENT Medical Diagnosis and Treatment.

The word "diagnosis" is deliberate. The Triage Agent does not produce "analysis" or "clinical notes". It produces a committed diagnostic conclusion — the condition the patient most likely has, why, and what should happen next. That conclusion is provisional: it goes to a physician in Phase 7 for confirmation or rejection. The Triage Agent's role is to reduce the time a doctor spends constructing an initial differential from scratch by providing a textbook-grounded starting point they can confirm quickly or challenge with specific objections.

The Triage Agent operates in two distinct modes:

**Mode A — Common Symptoms.** The patient presented with a symptom the Intake Agent recognised. A full structured interview has already been conducted. The Triage Agent receives the structured summary (symptoms, Q&A answers, red flags, urgency), parses it into a clean clinical picture, retrieves relevant textbook passages, checks whether the intake answers are sufficient for diagnosis, optionally asks follow-up questions if critical gaps exist, and generates the diagnosis.

**Mode B — Uncommon Symptoms.** The patient presented with a complaint the Intake Agent could not classify on the first detection attempt. No structured interview exists. The Triage Agent receives only the raw complaint and builds the clinical picture through a three-pass conversational process: broad retrieval to understand the problem space, targeted questions to the patient, and progressive diagnosis refinement.

Both modes share the same RAG infrastructure (Phases 2 and 3), the same diagnosis generation prompt, and the same seven-section output format.

### Where Phase 5 Sits in the Pipeline

| Phase | Component | Function |
|---|---|---|
| 1.1–1.2 | PDF extraction and chunking | 5,631 searchable text chunks from TMT textbook |
| 1.3 | Symptom structuring | 11 structured clinical symptom objects |
| 2.1–2.3 | Embedding and retrieval | ChromaDB vector store; 90% Hit@1, 100% Hit@3 validated |
| 3 | Reranking | BGE-reranker-v2-m3; +7.4% content relevance over bi-encoder alone |
| 4.1 | Intake Agent | Multi-turn patient interview; produces structured summary |
| **5** | **Triage Agent** | **Diagnostic engine — produces grounded diagnosis report** |
| 7 (planned) | Doctor review | Confirm, reject, or escalate the triage diagnosis |

---

## Architecture

### The LangGraph StateGraph

The Triage Agent is built as a LangGraph `StateGraph` with the following nodes:

| Node | Mode | Purpose |
|---|---|---|
| `analyze_input` | Both | Determines mode; parses intake summary to clinical picture (Mode A) |
| `retrieve_evidence` | Both | Pass 1 retrieval — bi-encoder + cross-encoder reranking |
| `check_sufficiency` | Mode A | Criteria-based gap analysis against intake answers |
| `ask_followup_mode_a` | Mode A | Emits follow-up question when gaps identified |
| `process_followup_answer` | Mode A | Records follow-up answer; decides if more follow-ups needed |
| `retrieve_evidence_enriched` | Mode A | Re-retrieves with enriched context after follow-up answers |
| `generate_questions` | Mode B | Pass 1 — generates 4–6 clinical questions from retrieved passages |
| `ask_question` | Mode B | Emits the next question from `generated_questions` |
| `process_answer` | Mode B | Records patient answer; signals when all questions answered |
| `retrieve_evidence_pass2` | Mode B | Pass 2 — retrieves with enriched query (complaint + answers) |
| `generate_diagnosis_pass2` | Mode B | Generates preliminary diagnosis after Pass 2 |
| `evaluate_refinement` | Mode B | Evaluates whether a critical finding warrants Pass 3 |
| `retrieve_evidence_pass3` | Mode B | Pass 3 — targeted retrieval on critical finding |
| `generate_diagnosis_pass3` | Mode B | Final diagnosis after Pass 3; HARD STOP |
| `generate_diagnosis` | Mode A | Generates diagnosis from clinical picture + retrieved passages |

LangGraph requires unique node names. Since the same retrieval and diagnosis functions run in multiple passes, the graph registers wrapper nodes (`retrieve_evidence_pass2`, `retrieve_evidence_pass3`, `generate_diagnosis_pass2`, `generate_diagnosis_pass3`) that set `current_pass` in the state dict before delegating to the shared function logic.

### The TriageState

`TriageState` is a `TypedDict` with all fields persisted across turns by LangGraph's state management.

| Field | Type | Purpose |
|---|---|---|
| `messages` | `list[BaseMessage]` | Full conversation history. `add_messages` reducer — append-only. |
| `mode` | `str` | `"common"` (Mode A) or `"uncommon"` (Mode B). Set by `analyze_input`. |
| `intake_summary` | `dict \| None` | Raw `IntakeSession.get_summary()` dict. Kept as reference; clinical decisions use `clinical_picture`. |
| `clinical_picture` | `dict \| None` | Parsed clean version of `intake_summary` — key-value clinical findings stripped of question text and metadata. |
| `raw_complaint` | `str` | Raw patient complaint text (Mode B). Empty string in Mode A. |
| `retrieved_chunks` | `list[dict]` | All unique chunks retrieved across all passes, deduplicated by `chunk_id`. Grows with each pass. |
| `current_pass` | `int` | Active retrieval pass: 1, 2, or 3. |
| `generated_questions` | `list[str]` | 4–6 clinical questions generated from Pass 1 (Mode B only). |
| `patient_answers` | `dict` | Question → answer (Mode B only). Populated incrementally by `process_answer`. |
| `current_question_idx` | `int` | Index into `generated_questions` for the next question to ask. |
| `needs_refinement` | `bool` | Whether `evaluate_refinement` determined Pass 3 is warranted. |
| `refinement_reason` | `str` | One-sentence explanation of the critical finding that triggered Pass 3. |
| `refinement_search_query` | `str` | Targeted query for Pass 3 retrieval, generated by the LLM. |
| `info_sufficient` | `bool` | Mode A only — True if intake answers cover all patient-reportable diagnostic criteria. |
| `followup_questions` | `list[str]` | Mode A only — follow-up questions targeting identified gaps (max 3). |
| `followup_answers` | `dict` | Mode A only — follow-up question → patient answer. |
| `followup_question_idx` | `int` | Mode A only — index into `followup_questions`. |
| `followup_phase` | `bool` | Mode A only — True while follow-up questioning is active. |
| `diagnosis` | `dict` | The final diagnosis report: `report`, `mode`, `pass`, `num_chunks_used`. |
| `diagnosis_complete` | `bool` | True after Mode A single pass, Pass 2 with no refinement, or Pass 3 unconditionally. |

### Graph Flow Diagrams

**Mode A (common symptoms):**

```
START → analyze_input → retrieve_evidence → check_sufficiency
                                                    |
                                    info_sufficient=True → generate_diagnosis → END
                                                    |
                                    info_sufficient=False → ask_followup_mode_a → END (pause)
                                                              (resumes at process_followup_answer)
                                                                      |
                                                          more follow-ups → ask_followup_mode_a
                                                                      |
                                                          all answered → retrieve_evidence_enriched
                                                                                    |
                                                                             generate_diagnosis → END
```

**Mode B (uncommon symptoms):**

```
START → analyze_input → retrieve_evidence (Pass 1) → generate_questions
                                                              |
                                                        ask_question → END (pause)
                                                        (resumes at process_answer)
                                                              |
                                                    more questions → ask_question
                                                              |
                                                    all answered → retrieve_evidence_pass2
                                                                          |
                                                               generate_diagnosis_pass2
                                                                          |
                                                           pass >= 3 → END
                                                                          |
                                                           pass == 2 → evaluate_refinement
                                                                              |
                                                              no refinement → END
                                                                              |
                                                              needs_refinement → retrieve_evidence_pass3
                                                                                          |
                                                                               generate_diagnosis_pass3 → END (HARD STOP)
```

---

## The Clinical Picture Parser

### Why It Exists

The Intake Agent's `get_summary()` dict contains valuable clinical data, but it is embedded in a format shaped by the interview process rather than clinical reasoning. The `answers` field, for example, is a mapping of question strings to answer strings:

```python
{
    "What is the character of your chest pain?": "Extremely tight, like a vice around my chest",
    "Does it radiate anywhere?": "Yes, both arms",
    "Does it worsen with exertion?": "Yes, definitely gets worse when I walk",
    ...
}
```

This format has two problems for the Triage Agent. First, the question text is noisy — "What is the character of your chest pain?" is not a clinical finding; "pain quality: extremely tight" is. Second, the intake summary also carries `specialty_routing`, `initial_workup`, and `key_exam_findings` that were generated by the Intake Agent from Phase 1.3 symptom objects — not from diagnostic reasoning. If these fields were passed directly to the Triage Agent's LLM, the LLM might anchor its reasoning on the pre-computed specialty routing rather than forming an independent diagnostic conclusion from the evidence.

### What `parse_intake_to_clinical_picture()` Produces

The function strips the intake summary down to pure clinical signal: a dict of short, descriptive keys and concise values extracted from the patient's answers only. The question text is discarded. Specialty routing, initial workup, and key exam findings from the Intake Agent are not included.

**Before (raw intake summary answers):**
```python
{
    "What is the character of your chest pain?": "Extremely tight, like a vice",
    "Does the pain radiate anywhere?": "Yes, both arms",
    "Does it worsen with exertion?": "Yes",
    "Any prior medical history?": "Previous stroke 5 years ago",
    "Any smoking history?": "Yes, 10 years, quit 2 years ago"
}
```

**After (clinical picture):**
```python
{
    "pain_quality": "extremely tight, vice-like",
    "pain_radiation": "bilateral arms",
    "triggers": "exertional worsening",
    "cardiovascular_history": "previous stroke (5 years ago)",
    "smoking_history": "10 pack-years, quit 2 years ago"
}
```

The full `clinical_picture` dict returned by the function:

```python
{
    "symptoms": ["Chest Pain", "Hemoptysis"],
    "urgency": "urgent",
    "clinical_findings": {
        "pain_quality": "extremely tight, vice-like",
        "pain_radiation": "bilateral arms",
        "triggers": "exertional worsening",
        "cardiovascular_history": "previous stroke (5 years ago)",
        "smoking_history": "10 pack-years, quit 2 years ago"
    },
    "red_flags": [
        {"flag": "Hemoptysis in context of acute chest pain", "urgency": "urgent"}
    ]
}
```

Note that red flags are preserved but stripped to only `flag` and `urgency` — the `implication` field from the Intake Agent's symptom objects is dropped so the Triage Agent forms its own clinical interpretation rather than inheriting one.

### How It Prevents Bias

The cleanse is specifically designed so that the Triage Agent never sees:
- The `specialty_routing` field (e.g., "Pulmonology") — the agent should form its own specialty recommendation
- The `initial_workup` field — the agent should derive investigations from the retrieved textbook evidence
- The question text — questions contain clinical framing that may bias the LLM's interpretation of the answer
- The `clinician_note` field — the Intake Agent's full nine-section summary should not be recycled as Triage Agent input

The Triage Agent's diagnosis is grounded in the retrieved textbook passages, not in the Intake Agent's pre-computed outputs.

---

## Mode A: Common Symptoms

### Full Flow

**Step 1: Parse intake to clinical picture.**
`analyze_input` calls `parse_intake_to_clinical_picture()` and prints the extracted findings to stdout for transparency. The number of clinical findings extracted is logged.

**Step 2: Retrieve evidence (Pass 1).**
`_clinical_picture_to_query()` builds a RAG query from the clinical picture. The query concatenates the presenting symptoms, urgency level, and all clinical findings in a dense natural-language string. This query is submitted to `retrieve_and_rerank()`: bi-encoder retrieval of `retrieve_k` candidates (default 10), cross-encoder reranking, return of top `return_k` chunks (default 3).

**Step 3: Criteria-based sufficiency check.**
`check_sufficiency` runs a two-step analysis to determine whether the intake answers are sufficient for diagnosis or whether targeted follow-up questions are needed. This is the most distinctive part of Mode A — it is grounded in the textbook, not LLM judgment (see dedicated section below).

**Step 4a: If sufficient — generate diagnosis.**
`generate_diagnosis` receives the full clinical picture and all retrieved chunks. Generates the seven-section diagnosis report in a single LLM call.

**Step 4b: If insufficient — ask follow-up questions.**
`ask_followup_mode_a` emits the first follow-up question. The graph pauses at END. `TriageSession.respond_followup()` handles subsequent patient answers. After all follow-up answers are collected, `retrieve_evidence_enriched` re-retrieves with the enriched query (clinical picture plus follow-up answers), then `generate_diagnosis` generates the final report.

---

## The Criteria-Based Sufficiency Check

### Why It Is Grounded in the Textbook

The naive approach to sufficiency checking would be to ask an LLM: "Given these patient answers, do you have enough information to diagnose them?" This approach has a structural flaw: the LLM's judgment is informed by its general clinical training, not by what the retrieved textbook passages specifically say is needed to make this particular diagnosis. The result is arbitrary — the LLM might consider the answers sufficient based on medical knowledge it was trained on that has nothing to do with the evidence the system is actually using.

The Phase 5 approach grounds the sufficiency check in the retrieved textbook passages themselves. The check answers the question: "What does this textbook say is needed to diagnose these conditions, and do the intake answers provide that?"

### The Two-Step Process

**Step 1: Extract diagnostic criteria from retrieved passages.**

`check_sufficiency` sends the retrieved textbook passages to the LLM with the `extract_criteria_prompt`. The LLM is asked to identify the key diagnostic criteria from sections like "Essentials of Diagnosis", "Clinical Findings", "Symptoms and Signs", and "General Considerations". Each criterion is returned as a JSON object with four fields:

```json
{
    "criterion": "Onset: sudden vs gradual, relation to exertion",
    "why": "Differentiates PE from angina from aortic dissection",
    "category": "history",
    "patient_reportable": true
}
```

The `patient_reportable` field is the key distinction: `true` means a patient can describe it in conversation (onset, duration, character, triggers, prior history); `false` means it requires a clinician (physical exam findings, ECG results, D-dimer levels, imaging). The prompt provides explicit examples to ensure consistent categorisation.

**Step 2: Gap analysis — compare criteria against intake answers.**

The LLM receives the extracted criteria list and the patient's clinical picture, and runs a structured gap analysis. The output is a JSON object with four components:

```json
{
    "covered": [{"criterion": "onset", "covered_by": "pain_quality field confirms sudden onset"}],
    "patient_gaps": [{"criterion": "pleuritic character (worsens with breathing)", "why_critical": "Distinguishes PE from ACS"}],
    "clinician_gaps": [{"criterion": "D-dimer and troponin levels", "note": "Lab results — include in recommended investigations"}],
    "sufficient": false,
    "followup_questions": ["Does your chest pain get worse when you breathe in deeply?"]
}
```

**Patient gaps** become follow-up questions — things the patient can answer that the intake didn't cover.
**Clinician gaps** feed into the Recommended Investigations section of the diagnosis report — they cannot be answered in conversation, so they are flagged as next steps for the doctor.

### The Patient-Reportable vs Clinician-Only Distinction

This distinction is fundamental to the system's design. The Triage Agent must never ask a patient about X-ray findings, D-dimer levels, troponin results, heart murmurs, or lung crackles. These are clinician-side findings that the patient cannot report. The `patient_reportable: false` categorisation ensures these criteria are routed to Recommended Investigations, not to follow-up questions.

Conversely, things the patient absolutely can report — whether the pain is pleuritic (worsens with inspiration), whether there was a triggering event, whether they have a prior history of DVT — are patient-reportable and become targeted follow-up questions if they are not covered by the intake answers.

### Example: Full Chain

Textbook says: *"Pleuritic chest pain (worsening with inspiration) is a distinguishing feature of PE vs ACS"*

Intake doesn't cover: Whether pain worsens with deep breathing

Follow-up question generated: *"Does your chest pain get worse when you take a deep breath?"*

Patient answers: *"Yes, much worse when I breathe in"*

Diagnosis uses: Pleuritic character as key discriminating criterion for PE over ACS

The Recommended Investigations section also includes: *"D-dimer and CT pulmonary angiography to confirm PE — [clinician gap: lab and imaging confirmation required]"*

### Limits on Follow-Up Questions

Follow-up questions are capped at 3. The `followup_questions` list is sliced to 3 entries: `followups[:3]`. This prevents the Mode A flow from becoming a second extended interview session. Three targeted questions covering critical diagnostic gaps represent the appropriate balance between completeness and patient burden.

---

## Mode B: Uncommon Symptoms

### Full Three-Pass Flow

Mode B handles presentations the Intake Agent could not classify on the first detection attempt. The only input is the patient's raw complaint string — captured immediately at the point of handoff, with no clarification iterations.

### Pass 1: Broad Retrieval and Question Generation

**Input:** Raw complaint string.

**Retrieval:** The complaint is submitted directly to `retrieve_and_rerank()`. This is a broad, semantically fuzzy retrieval — it surfaces content from whichever chapters and sections best match the lay description. For "I have a rash on my legs", this surfaces dermatology chapters. For "my heart keeps skipping beats", it surfaces cardiac chapters.

**Question generation:** The retrieved passages are not used to generate a diagnosis at this stage. They are passed to the LLM with `_QUESTION_GENERATION_SYSTEM` prompt, which asks for 4–6 patient-friendly questions that would differentiate between the most likely conditions revealed by the passages. The questions target: onset/duration, character, aggravating/relieving factors, associated symptoms, and relevant history.

The questions are grounded in the retrieved evidence, not generated from general LLM clinical knowledge. If the passages reveal that inflammatory vs infectious skin conditions are the key distinction, the questions will target the features that differentiate them.

**Output:** `generated_questions` — a list of 4–6 question strings. The first question is emitted and the graph pauses at END.

### Pass 2: Targeted Retrieval, Preliminary Diagnosis, and Differentiating Questions

**Input:** Original complaint + all patient answers from Pass 1 questioning.

**Query construction:** `_build_pass2_query()` builds an enriched clinical query combining the complaint with all collected Q&A pairs. This is a fundamentally different query from Pass 1 — specific, clinical, and targeted to the differential the answers have established.

**Retrieval:** `retrieve_and_rerank()` is called with the enriched query. New chunks are deduplicated against existing `retrieved_chunks` by `chunk_id`.

**Preliminary diagnosis:** All accumulated chunks (Pass 1 + Pass 2) and all clinical context are passed to the diagnosis LLM. A full seven-section diagnosis report is generated.

**Differentiating questions (conditional):** After the preliminary diagnosis, the system evaluates whether there is diagnostic uncertainty — multiple plausible competing diagnoses where the primary does not have high confidence. If uncertain, the LLM generates 2–4 targeted differentiating questions designed to distinguish between the top competing diagnoses. These questions are asked to the patient one by one. Once answered, the system re-retrieves with the enriched context and regenerates the diagnosis. This mirrors the Mode A sufficiency check pattern: gather more patient information to resolve diagnostic ambiguity rather than relying solely on additional textbook searches.

If the primary diagnosis has high confidence with no meaningful competing differentials, no differentiating questions are asked and the diagnosis proceeds directly to refinement evaluation.

**Refinement evaluation:** After the Pass 2 diagnosis (with or without differentiating questions), `evaluate_refinement` sends the complaint, all patient answers, and the diagnosis to the LLM. The LLM must return:

```json
{
    "needs_refinement": true,
    "reason": "Patient reported fever and joint pain — significantly narrows differential toward systemic inflammatory conditions",
    "search_query": "erythema nodosum associated conditions and causes"
}
```

If `needs_refinement=false`, the diagnosis is final. If `needs_refinement=true`, Pass 3 fires.

A critical finding is defined as one that: introduces a new specific diagnosis not previously considered; strongly rules in or rules out a major condition; or reveals an important historical fact that redirects the clinical picture. The bar is intentionally high — routine confirmatory details should not trigger Pass 3.

### Pass 3: Critical Finding Refinement with Differentiating Questions (Conditional)

**Input:** The targeted `refinement_search_query` constructed by the LLM — specific to the critical finding.

**Retrieval:** `retrieve_and_rerank()` is called with the targeted query. For the leg rash case: `"erythema nodosum associated conditions and causes"` — retrieves content about systemic associations (sarcoidosis, tuberculosis, streptococcal infection, IBD) that the first two passes may not have surfaced.

**Preliminary diagnosis:** All accumulated chunks (Passes 1 + 2 + 3) are passed to the diagnosis LLM.

**Differentiating questions (conditional):** Same as Pass 2 — if the Pass 3 diagnosis is uncertain between competing conditions, 2–4 targeted differentiating questions are asked. Patient answers are used to re-retrieve and regenerate the final diagnosis. If the diagnosis is confident, no questions are asked.

**Final diagnosis:** The final diagnosis has access to the broadest possible evidence base: initial broad context, targeted differential context, specific associated-condition context, and all patient answers including differentiating responses.

**HARD STOP:** Pass 3 never triggers Pass 4. The cap is enforced at multiple levels (see Infinite Loop section).

### The Infinite Loop Problem and How It Is Solved

Multi-pass retrieval with conditional triggering creates a potential for unbounded iteration. The solution is a hard cap with distinct semantic purpose for each pass:

**Hard cap:** `current_pass >= 3` is checked at multiple points — in `route_after_diagnosis`, in `generate_diagnosis_pass3`, and in `TriageSession.respond()` before calling `evaluate_refinement`. The cap is enforced by graph structure and by the session manager independently.

**Pass 3 never evaluates refinement:** `generate_diagnosis_pass3` forces `needs_refinement=False` before the shared diagnosis function runs. The LLM is never asked whether Pass 4 is warranted — it cannot trigger one.

**Manual step execution:** Because LangGraph does not natively support mid-graph resumption without checkpointing, `TriageSession.respond()` drives Pass 2 and Pass 3 steps manually after all questions are answered. This code path explicitly checks `current_pass >= 3` before calling `evaluate_refinement` as an additional enforcement layer.

A presentation requiring four or more passes is one where the clinical picture is genuinely unstable. The right response is not a fourth pass — it is flagging the case for direct clinician review. The hard stop at Pass 3 is a safety property.

---

## The Diagnosis Report Format

All outputs — Mode A, Mode B Pass 2 preliminary, Mode B Pass 3 final — use the same seven-section format enforced by `_DIAGNOSIS_SYSTEM`.

### 1. Primary Diagnosis

The most likely diagnosis with an explicit confidence level: `high`, `moderate`, or `low`. High confidence means the presentation is characteristic and the textbook evidence is direct and specific. Moderate confidence means the diagnosis fits but key confirmatory findings are pending. Low confidence means multiple diagnoses remain plausible and the distinction requires further workup.

### 2. Differential Diagnoses

A ranked list of alternative diagnoses. Each entry includes: condition name, key supporting evidence from the patient's presentation, key evidence from the textbook passages, and an explicit comparison with the primary diagnosis — why this condition is more or less likely.

### 3. Clinical Reasoning

Step-by-step reasoning connecting the patient's symptoms and answers to the diagnosis, with citations to specific textbook passages. This is the most important section for physician review: it makes the logic of the diagnosis transparent and auditable. A doctor who disagrees can use this section to identify which specific inference they would dispute.

### 4. Recommended Investigations

Specific tests to confirm or rule out diagnoses, with clinical justification from the textbook evidence. In Mode A, this section also incorporates the clinician-side gaps identified by the sufficiency check — tests and exams that the intake answers revealed as needed but that the patient cannot self-report.

### 5. Management Considerations

Initial management steps suggested by the textbook evidence. These are considerations for the reviewing physician, not prescriptions. The grounding constraint applies: suggestions are cited to retrieved passages.

### 6. Red Flags and Safety Netting

Findings that would change the diagnosis, worsen the clinical picture, or require immediate escalation. The Triage Agent's explicit acknowledgment that the diagnosis is provisional.

### 7. Sources

Each textbook passage used, cited by chapter and section, with a description of what it contributed to the clinical reasoning. This section makes the grounding constraint auditable — a doctor can verify the basis of the diagnosis directly in the textbook.

---

## Data Flow Between Agents

The Triage Agent is a consumer of both the Intake Agent's output and the Phase 2/3 RAG infrastructure.

```
Intake Agent
    │
    │── IntakeSession.get_summary() ──────────────────────────────────────────>
    │                                                                    Triage Agent (Mode A)
    │                                                                           │
    │                                                    parse_intake_to_clinical_picture()
    │                                                                           │
    │                                                                  clinical_picture dict
    │                                                                           │
    │                                                           _clinical_picture_to_query()
    │                                                                           │
    │                                                                  RAG query string
    │                                                                           │
    │                                                                 retrieve_and_rerank()
    │                                                                           │
    │                                                               check_sufficiency()
    │                                                                           │
    │                                                          (optional follow-up questions)
    │                                                                           │
    │                                                                generate_diagnosis()
    │                                                                           │
    │                                                              Diagnosis Report → Doctor
    │
    └── IntakeSession.get_raw_complaint() ────────────────────────────────────>
                                                                    Triage Agent (Mode B)
                                                                           │
                                                              retrieve_and_rerank() [Pass 1]
                                                                           │
                                                               generate_questions()
                                                                           │
                                                              [Patient Q&A — up to 6 questions]
                                                                           │
                                                              retrieve_and_rerank() [Pass 2]
                                                                           │
                                                              generate_diagnosis_pass2()
                                                                           │
                                                              evaluate_refinement()
                                                                           │
                                                        (optional) retrieve_and_rerank() [Pass 3]
                                                                           │
                                                        (optional) generate_diagnosis_pass3()
                                                                           │
                                                              Diagnosis Report → Doctor
```

**Nothing is lost across the flow.** The clinical picture carries all intake findings. Each retrieval pass accumulates chunks into `retrieved_chunks` without discarding prior passes. Follow-up answers are merged with the original clinical picture before the final retrieval and diagnosis. The diagnosis report has access to all information gathered across every stage.

---

## Test Results

### Mode A: Chest Pain and Hemoptysis

**Patient profile:** Urgent case. Ex-smoker. Previous stroke. Presenting complaint: tight chest pain with radiation to both arms, worsened with exertion, haemoptysis.

**Intake summary fields used:**
- Symptoms: `["Chest Pain", "Hemoptysis"]`
- Urgency: `"urgent"`
- Clinical findings (after parse): pain quality, radiation, triggers, cardiovascular history, smoking history
- Red flags: hemoptysis in context of acute chest pain [urgent]

**Routing:** Intake was not escalated mid-conversation (`escalated=False`), so the Triage Agent was invoked. Urgency level is `"urgent"` — the Triage Agent processes all urgency levels.

**Sufficiency check result:** Sufficient — exertional pattern, radiation, and prior stroke history covered. Pleuritic character and leg swelling flagged as clinician-assessable gaps, routed to Recommended Investigations.

**Diagnosis:**
- Primary: Pulmonary Embolism — high confidence. Pleuritic character, haemoptysis, exertional worsening, prior stroke as thromboembolic risk factor, ex-smoker status.
- Differentials: ACS (bilateral arm radiation, exertional worsening), Aortic Dissection (sudden onset, though haemoptysis less consistent), Pericarditis (pleuritic component, fever absent)
- Investigations: CT pulmonary angiography, D-dimer, troponin and ECG, chest X-ray
- Chunks used: 3 (Mode A, Pass 1)

### Mode B: Leg Rash (Uncommon Symptom)

**Initial complaint:** `"I have a rash on my legs that's been spreading"`

**Pass 1 questions generated (6):**
1. How long have you had this rash, and how quickly has it been spreading?
2. Do you have any fever, joint pain, or feel generally unwell alongside the rash?
3. Is the rash painful, itchy, or painless?
4. Has the rash changed in colour or texture since it first appeared?
5. Have you had any fungal skin infections in the past, or does anyone in your household have a similar rash?
6. Have you had any recent injuries, insect bites, or contact with new substances?

**Patient answers:** 1 week duration, spreading slowly; fever and joint pain present; painful, tender; raised, red, nodular; no fungal history; no contact or injury.

**Pass 2 preliminary diagnosis:** Erythema Nodosum — moderate confidence. Classic clinical triad: painful tender nodular lesions on lower legs, fever, arthralgia.

**Refinement evaluation:** Fever combined with joint pain identified as critical finding — erythema nodosum is almost always reactive, with a systemic trigger that the Pass 2 retrieval had not specifically targeted. `needs_refinement=true`, `search_query="erythema nodosum associated conditions and causes"`.

**Pass 3 retrieval:** New chunks on associated conditions — sarcoidosis, tuberculosis, streptococcal pharyngitis, IBD, drug reactions.

**Pass 3 final diagnosis:**
- Primary: Erythema Nodosum — high confidence. Clinical triad confirmed.
- Differentials: Reactive arthritis, cellulitis (less likely), vasculitis
- Investigations: Throat swab and ASO titres, chest X-ray (sarcoidosis/TB), QuantiFERON-TB Gold, ESR and CRP, ANA, medication review
- Chunks used: 8 across 3 passes

The Pass 3 refinement produced a materially different investigation list — orienting the workup toward identifying the systemic trigger rather than just managing the skin manifestation.

---

## Cost Analysis

### LLM Calls per Patient Session

A full Mode A session with the sufficiency check makes approximately 25–35 GPT-4o calls:

| Stage | Calls | Notes |
|---|---|---|
| Intake Agent (Phase 4.1) | ~15–22 | Detect, merge, prefill, questions, red flag checks, adequacy checks, urgency, summary |
| Triage: `analyze_input` (parse clinical picture) | 1 | Extracts clinical findings from Q&A pairs |
| Triage: `check_sufficiency` (extract criteria) | 1 | Extracts diagnostic criteria from textbook passages |
| Triage: `check_sufficiency` (gap analysis) | 1 | Compares criteria against clinical picture |
| Triage: follow-up questions (if needed, per question) | 0–3 | One call per follow-up question asked |
| Triage: `generate_diagnosis` | 1 | Full seven-section diagnosis report |
| **Total (Mode A, no follow-ups)** | **~19–25** | |
| **Total (Mode A, 3 follow-ups)** | **~22–28** | |

Mode B adds:
- 1 call for question generation (Pass 1)
- 1 call for Pass 2 diagnosis
- 1 call for refinement evaluation
- 1 call for Pass 3 diagnosis (conditional)

**Total Mode B with Pass 3: ~27–35 calls per session.**

### Task Complexity vs Model Needed

Not all LLM calls require GPT-4o-level capability. A tiered model approach would reduce cost substantially:

| Task | Complexity | Model needed | Currently used |
|---|---|---|---|
| Symptom detection (JSON array) | Low | GPT-4o-mini | GPT-4o |
| Question rephrasing | Low | GPT-4o-mini | GPT-4o |
| Adequacy check (JSON object) | Low | GPT-4o-mini | GPT-4o |
| Red flag detection | Medium | GPT-4o-mini | GPT-4o |
| Merging question lists | Medium | GPT-4o-mini | GPT-4o |
| Pre-filling from initial message | Medium | GPT-4o-mini | GPT-4o |
| Urgency classification | Medium | GPT-4o | GPT-4o |
| Clinical picture parsing | Medium | GPT-4o | GPT-4o |
| Criteria extraction | High | GPT-4o | GPT-4o |
| Gap analysis | High | GPT-4o | GPT-4o |
| Diagnosis generation | High | GPT-4o | GPT-4o |
| Refinement evaluation | High | GPT-4o | GPT-4o |
| Clinician handover summary | High | GPT-4o | GPT-4o |

A tiered approach routing simple structured output tasks to GPT-4o-mini and reserving GPT-4o for diagnostic reasoning could reduce cost by 40–60% without meaningful quality loss on the high-stakes tasks. This is planned but not yet implemented — all calls currently use GPT-4o.

---

## The TriageSession API

`TriageSession` is the public interface for the Triage Agent. It manages model loading, graph construction, and conversation state.

### `__init__(llm_model, retrieve_k, return_k)`

```python
def __init__(
    self,
    llm_model: str = "gpt-4o",
    retrieve_k: int = RERANK_TOP_K_RETRIEVE,   # 10
    return_k: int = RERANK_TOP_K_RETURN,        # 3
):
```

Initialises all infrastructure: the OpenAI LLM client (temperature=0), compute device (MPS on Apple Silicon, CPU fallback), ChromaDB collection, bi-encoder, and cross-encoder. Builds and compiles the LangGraph graph. All components are shared across subsequent calls within the same session — models are loaded once at initialisation.

Prints status on initialisation:
```
Initialising Triage Agent models...
Triage Agent ready  (bi-encoder on mps, reranker on cpu)
```

### `diagnose_from_intake(intake_summary) → dict | str`

Mode A entry point. Initialises state and invokes the graph. The Triage Agent processes all intake cases regardless of urgency level — including cases classified as "emergency" by the `assess_urgency` node — because a doctor still needs a diagnosis. The only cases that do NOT reach the Triage Agent are those where `escalated=True` (mid-conversation red flag escalation where the patient was already directed to call 911); that routing decision is made by the Intake Agent CLI before calling `diagnose_from_intake()`, not inside the Triage Agent itself.

If intake answers are sufficient: returns the diagnosis dict directly (non-interactive).
If insufficient: returns the first follow-up question as a string. Caller should then use `respond_followup()` for each answer.

```python
# Returns on success:
{
    "report": str,           # full seven-section diagnosis report
    "mode": "common",
    "pass": 1,
    "num_chunks_used": int,
}
```

### `respond_followup(patient_answer) → dict | str`

Mode A follow-up: processes patient's answer to a follow-up question.

Returns the next follow-up question string if more remain, or the full diagnosis dict when all follow-ups are answered and the diagnosis is generated.

### `start_uncommon(patient_complaint) → str`

Mode B entry point. Receives the raw complaint string, runs Pass 1 retrieval and question generation, returns the first clinical question. The graph pauses at END.

### `respond(patient_answer) → str`

Mode B answer processing. Records the patient's answer and either:
- Returns the next question string (if more questions remain)
- Runs Pass 2 retrieval + diagnosis, then optionally Pass 3, and returns the final diagnosis report as a string

### `is_complete() → bool`

Returns True when the diagnosis has been generated. Safe to call at any time.

### `get_diagnosis() → dict`

Returns the full diagnosis dict from the current session state. Returns an empty dict if the session has not completed.

### Usage Examples

```python
# Mode A — single-pass from intake (no follow-ups needed)
session = TriageSession()
result = session.diagnose_from_intake(intake_summary_dict)
if isinstance(result, dict):
    print(result["report"])

# Mode A — with follow-up questions
session = TriageSession()
result = session.diagnose_from_intake(intake_summary_dict)
if isinstance(result, str):
    # result is the first follow-up question
    print(result)
    while not session.is_complete():
        answer = input("Patient: ")
        result = session.respond_followup(answer)
        if session.is_complete():
            print(session.get_diagnosis()["report"])
        else:
            print(result)  # next follow-up question

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
        print(response)
```

---

## Scripts Reference

### `agents/triage_agent.py`

**CLI flags:**

| Flag | Default | Effect |
|---|---|---|
| `--model MODEL` | `gpt-4o` | OpenAI model name. Use `gpt-4o-mini` to reduce API cost during development. |
| `--from-intake PATH` | None | Path to a JSON file containing an `IntakeSession.get_summary()` dict. Runs Mode A non-interactively and exits. |
| `--query COMPLAINT` | None | Raw patient complaint string. Runs Mode B interactively — asks follow-up questions and generates diagnosis. |
| `--retrieve-k N` | 10 | Candidates fetched from bi-encoder. Applied to all retrieval passes. |
| `--return-k N` | 3 | Passages retained after reranking. Applied to all retrieval passes. |

**Execution modes:**

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

**Full pipeline — Intake Agent + Triage Agent in one command:**

```bash
# Runs intake interview, then auto-routes to Triage Agent based on outcome:
python agents/intake_agent.py

# With a pre-specified symptom (skips detection, always routes Mode A):
python agents/intake_agent.py --symptom "Chest Pain,Hemoptysis"

# With a specific model for both agents:
python agents/intake_agent.py --model gpt-4o-mini
```

---

## Configuration

All RAG infrastructure parameters are defined in `config.py`:

```python
CHROMA_DIR            = DATA_DIR / "chroma"
EMBEDDING_MODEL       = "sentence-transformers/embeddinggemma-300m-medical"
RERANKER_MODEL        = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K_RETRIEVE = 10   # candidates fetched from bi-encoder
RERANK_TOP_K_RETURN   = 3    # passages retained after reranking
```

These values reflect the Phase 3 recommendation. No separate configuration step is required. The Triage Agent inherits the Phase 3 configuration directly from `config.py`.

---

## Design Evolution

The Triage Agent reached its current design through seven iterations. Each iteration identified a structural flaw in the prior approach and introduced a targeted fix. This section documents the full progression — from the initial advisory engine that produced no diagnosis, to the grounded diagnostic system with bias prevention and pass-count caps that exists in the final implementation.

### Iteration 1: Clinical Analysis Engine (not a diagnostic engine)

The initial Triage Agent produced "clinical analysis" — advisory text about what the textbook says. It did not produce an actual diagnosis. Problems:

- No primary diagnosis with confidence level
- No ranked differential diagnoses
- No clinical reasoning chain
- The output read like a literature review, not a diagnostic report

### Iteration 2: Diagnostic Engine with Two Modes

Complete redesign to produce actual diagnoses:

- **Mode A (common symptoms)**: Single-pass RAG from intake summary
- **Mode B (uncommon symptoms)**: Multi-pass RAG with questioning
- Diagnosis report format: primary diagnosis, ranked differentials, clinical reasoning, investigations, management, red flags, sources

### Iteration 3: The Infinite Loop Problem (Mode B)

- **Risk identified**: In Mode B, patient answers could reveal new findings → new RAG search → new questions → new answers → infinite loop
- **Solution**: Hard cap of 3 passes with clear purpose for each:
  - Pass 1: understand the problem (broad retrieval → generate questions)
  - Pass 2: analyze with patient input (targeted retrieval → preliminary diagnosis)
  - Pass 3: handle surprises (critical finding refinement → final diagnosis)
- Pass 3 only triggers if the LLM identifies a critical finding that significantly changes the differential
- No Pass 4, ever — cases needing more than 3 passes should be flagged for direct clinician review

### Iteration 4: LLM-Judgment Sufficiency Check → Criteria-Based Gap Analysis

- **Initial approach**: After retrieving evidence for Mode A, asked the LLM "do you have enough info for a diagnosis?" — a yes/no judgment call
- **Flaw identified**: The LLM could say "sufficient" by filling gaps from its training data rather than the textbook. The check was ungrounded
- **Final design**: Two-step criteria-based approach:
  1. Extract diagnostic criteria from the retrieved textbook passages (what the textbook says you need to know)
  2. Structured gap analysis comparing criteria against intake answers (what is covered vs what is missing)
- Follow-up questions target specific textbook-grounded gaps, not LLM intuition

### Iteration 5: Patient-Reportable vs Clinician-Only Criteria

- **Flaw identified**: The diagnostic criteria extracted from the textbook include things like "ECG changes", "D-dimer level", and "CT angiography findings" — information a patient cannot provide
- **Problem**: The system would generate follow-up questions asking patients about X-ray results or blood tests
- **Final design**: Each criterion is categorised as `patient_reportable: true/false`. Only patient-reportable gaps generate follow-up questions. Clinician-only gaps are routed to the "Recommended Investigations" section of the diagnosis report

### Iteration 6: Clinical Picture Parser

- **Flaw identified**: The raw intake summary passed to the Triage Agent contained noise — question text, clinician notes, specialty routing lists, workup recommendations from Phase 1.3 data
- **Problem**: Specialty routing from the Intake Agent could bias the Triage Agent's diagnosis (e.g., if the intake summary says "route to Cardiology", the Triage Agent might anchor on cardiac diagnoses)
- **Final design**: `parse_intake_to_clinical_picture()` distills the intake summary to pure clinical signal — key-value pairs of findings, stripped of all artifacts and pre-computed recommendations. The Triage Agent forms its own diagnostic conclusions from the evidence

### Iteration 7: Emergency Guard Removal

- **Initial approach**: Triage Agent had an internal emergency guard — if `urgency == "emergency"`, return immediately with "Triage Agent defers"
- **Flaw identified**: This meant emergency cases (e.g., sudden-onset headache classified as emergency by the Intake Agent's final assessment) received no diagnosis — the doctor got only the intake summary with no evidence-based analysis
- **Final design**: The emergency guard was removed from the Triage Agent entirely. The routing decision lives in the Intake Agent: only skip Triage when `escalated=True` (mid-conversation red flag). All other cases, including emergency-urgency from the final assessment, receive a full diagnosis

---

## Limitations

### All LLM Calls Use GPT-4o (No Tiered Model Yet)

Every LLM call in both the Intake Agent and the Triage Agent currently uses GPT-4o at temperature=0. Many tasks — symptom detection, question rephrasing, adequacy checks — could be handled by GPT-4o-mini at substantially lower cost without meaningful quality loss. A tiered model approach is planned but not yet implemented. The cost analysis section above shows the expected savings from tiering.

### Sufficiency Check Adds 2 Extra LLM Calls That May Not Always Be Needed

The sufficiency check (criteria extraction + gap analysis) runs on every Mode A session regardless of whether the intake answers are clearly comprehensive. For presentations with rich intake data where sufficiency is obvious — a complete medical history with no ambiguity — the two extra LLM calls add latency and cost without providing value. A heuristic pre-filter (e.g., checking whether a minimum number of clinical finding fields are populated before running the sufficiency check) could skip the check in clear-cut cases. This optimisation is not yet implemented.

### Mode B Questions Are Only as Good as the Initial Retrieval

Pass 1 questions are generated from the chunks retrieved in Pass 1. If Pass 1 retrieves content from the wrong clinical domain — because the complaint is ambiguous or the bi-encoder ranks a tangential chapter highly — the generated questions will target the wrong differential. Pass 2 retrieval, built from those misdirected answers, will compound the error. There is no fallback that recognises when Pass 1 retrieval is misaligned. This is the most significant failure mode in Mode B.

### Pass 3 Trigger Depends on LLM Judgment

The decision to trigger Pass 3 is made by the LLM in `evaluate_refinement`. The LLM may over-trigger (adding an unnecessary retrieval pass) or under-trigger (missing a subtle implication that a specialist would recognise). There is no calibrated threshold — only prompt instruction and LLM compliance. Over-triggering adds cost and latency but no clinical harm. Under-triggering finalises a diagnosis that would have been meaningfully refined, which is the more concerning failure mode.

### No External Medical Database Integration

All retrieved evidence comes from a single source: CURRENT Medical Diagnosis and Treatment (2022 edition). The Triage Agent has no access to PubMed literature, clinical guidelines databases, drug formularies, or any other knowledge source. Guidelines current as of 2022 may not reflect subsequent updates. Rare conditions, recently described syndromes, and highly subspecialty-specific presentations may not have sufficient textbook coverage to generate a specific diagnosis — the grounding constraint means the system will report low confidence rather than confabulating, but some presentations cannot be adequately triaged with this evidence base alone.

### Doctor Feedback Loop Not Yet Implemented (Phase 7)

Phase 7 will implement the doctor-facing interface through which the reviewing physician confirms, modifies, or rejects the Triage Agent's diagnosis. Until Phase 7 is implemented, there is no mechanism for recording diagnostic outcomes. A systematically wrong diagnosis in a particular clinical domain would not be detectable without the feedback loop. Phase 7 is not merely a UX addition — it is the data collection mechanism that makes systematic evaluation and improvement possible.
