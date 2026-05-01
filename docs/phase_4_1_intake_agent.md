# Phase 4.1: Intake Agent

## Overview

Phase 4.1 builds the Intake Agent — the conversational front-end of the Medora system that conducts structured clinical intake with patients before they are seen by a clinician. The agent receives a patient's initial complaint in natural language, conducts a targeted multi-turn conversation to collect the clinical information a care team needs, and produces a structured nine-section clinician handover note as output.

The Intake Agent occupies a specific position in the Medora pipeline: it sits at the patient-facing boundary, between the patient's unstructured description of their symptoms and the clinical reasoning layer. Its output is a standardised artefact that can be handed directly to a clinician or consumed by the downstream Triage Agent (Phase 4.2). The agent does not diagnose, does not recommend treatment, and does not draw on the RAG knowledge base built in Phases 2 and 3. Its sole function is information collection and structuring.

### Where Phase 4.1 Sits in the Pipeline

| Phase | Component | Function |
|---|---|---|
| 1.1–1.2 | PDF extraction and chunking | Converts TMT textbook to 5,631 searchable text chunks |
| 1.3 | Symptom structuring | Extracts 11 structured clinical symptom objects from the textbook |
| 2.1–2.3 | Embedding and retrieval | Builds a semantic search index over the chunk corpus; validates retrieval quality |
| 3 | Reranking | Adds cross-encoder reranking to retrieval; improves passage relevance |
| **4.1** | **Intake Agent** | **Conducts patient intake; produces clinician handover note** |
| 4.2 | Triage Agent (Phase 4.2) | Analyses the handover note; retrieves supporting evidence from the RAG system |

The Intake Agent does not use RAG. It does not call ChromaDB, does not embed queries, and does not retrieve chunks. Instead, it uses the 11 structured symptom objects produced in Phase 1.3 directly — as a structured lookup table rather than a retrieval corpus. The symptom objects provide the essential questions to ask, the red flags to watch for, and the clinical metadata needed to populate the clinician summary.

### The Distinction from the Triage Agent

The Intake Agent **collects** information. The Triage Agent (Phase 4.2) **analyses** it.

The Intake Agent speaks with the patient in plain language, asks clinical questions rephrased for patient comprehension, detects urgency signals in real time, and assembles a clinical record. It produces a handover note structured for clinician consumption. At no point does it draw clinical conclusions, establish diagnoses, or consult the medical literature.

The Triage Agent will receive the Intake Agent's structured output and query the RAG knowledge base — the 5,631 embedded chunks from the TMT textbook, reranked by the Phase 3 cross-encoder — to enrich the clinical picture with evidence-based content. The clean separation between collection (Phase 4.1) and analysis (Phase 4.2) mirrors standard clinical practice: a nurse or intake coordinator collects the history; the physician analyses it. The two agents play analogous roles in the Medora system.

---

## Why an Agent Architecture

### Why Not a Script With If/Else

The naive approach to clinical intake automation is a branching script: if the patient says "chest pain," show questions 1 through 7; if they say "cough," show questions 8 through 12. This approach fails for three reasons intrinsic to the clinical intake problem.

**Clinical conversations are stateful.** The appropriate next question depends not just on the patient's current complaint but on everything said so far. If a patient mentions arm radiation, the follow-up question about shortness of breath becomes more clinically urgent. If they say they have a history of heart disease, the cardiology-specific questions should be asked with greater depth. A branching script with fixed question sequences cannot encode this kind of state-sensitive routing without becoming an unmaintainable nest of conditional logic.

**Clinical conversations branch on multiple signals simultaneously.** A single patient turn can trigger a red flag check, an adequacy assessment, a follow-up question, and an urgency update — all before the next question is chosen. The routing decision at the end of each turn depends on the joint state of all these signals. Scripting this logic imperatively leads to code that is correct for specific cases but fragile to any deviation from the anticipated flow.

**Clinical conversations are inherently incomplete in advance.** The intake conversation for a patient presenting with both chest pain and blood in their sputum needs questions from both the Chest Pain and Hemoptysis question sets, deduplicated and ordered sensibly. No static branching script can anticipate all possible multi-symptom combinations and pre-author appropriate question sequences for each. The agent needs to reason about its question list dynamically.

A state machine architecture solves all three problems: it tracks what has been asked and answered, it routes between nodes based on the current accumulated state, and it assembles its question list dynamically by merging and deduplicating across however many symptoms were detected. The complexity of the clinical intake problem is a natural fit for a state machine.

### Why LangGraph

LangGraph is a framework for building stateful, multi-step LLM applications as explicit directed graphs. Nodes in the graph represent discrete actions — detecting a symptom, asking a question, checking for red flags — and edges represent routing decisions between those actions. The framework manages state persistence across turns, conversation history accumulation, and the control flow between nodes.

Four properties of LangGraph make it the right tool for this agent.

**Natural fit for clinical intake as a state machine.** The intake process has a clear structure: detect the symptom, plan the questions, iterate through them with checks at each step, and generate a summary. Each of these is a node. The routing conditions — whether to follow up, whether to escalate, whether the intake is complete — are edge conditions. The graph structure makes the clinical protocol explicit and inspectable rather than buried in conditional logic.

**Visual graph export for thesis diagrams.** LangGraph graphs can be exported as image files via `graph.get_graph().draw_mermaid_png()`. The compiled graph of the Intake Agent produces a Mermaid-formatted diagram showing all nodes and their conditional routing edges. This is a direct research artefact: the thesis can include the compiled graph as a visual representation of the clinical protocol, not merely a description of it.

**State persistence and conversation history management.** LangGraph's `StateGraph` manages the accumulated state object across the full conversation. The `messages` field uses the `add_messages` reducer, which appends new messages to the conversation history without overwriting previous turns. This provides automatic conversation context management — every LLM call in the graph has access to the full conversation history through the state object without any manual bookkeeping.

**LLM-agnostic design.** The graph is built around `langchain_openai.ChatOpenAI`, which exposes a standard interface that can be substituted for any LangChain-compatible LLM provider. The default configuration uses GPT-4o at temperature=0, but swapping to a locally deployed model — for example, a quantised Llama-3 instance on an EC2 GPU — requires only changing the model identifier passed to the `--model` flag. This is directly relevant to the thesis's planned comparison between cloud-hosted and local LLM performance on the intake task.

---

## Architecture: The LangGraph State Machine

The Intake Agent is compiled as a directed graph with nine nodes and conditional routing edges. The overall flow is:

```
START → detect_symptom → merge_questions → prefill → ask_question → [PAUSE]
                                                         ↑
                                              process_answer → escalate → generate_summary → END
                                                         ↓         ↑
                                               ask_followup → [PAUSE]
                                                         ↓
                                               ask_question → [PAUSE]
                                                         ↓
                                               assess_urgency → generate_summary → END
```

The `[PAUSE]` markers indicate where the graph reaches END and waits for the patient's next input. The `IntakeSession` class (documented in the Scripts Reference section below) manages this pausing behaviour: it invokes the graph or calls individual node functions depending on the current conversation phase, threading patient responses back into the state after each pause.

### Node 1: `detect_symptom`

The entry node. Receives the patient's first message and maps it to one or more of the 11 canonical symptom names defined in Phase 1.3.

The node presents the LLM with the full list of 11 symptom names and asks it to return a JSON array of matched symptoms from the patient's text. The system prompt requires exact capitalisation matching (e.g., `"Chest Pain"` not `"chest pain"`) to enable direct dictionary lookup against `_SYMPTOM_MAP`. The prompt instructs the LLM to return multiple symptoms when the patient's message clearly describes more than one — for example, `"chest pain and coughing blood"` returns `["Chest Pain", "Hemoptysis"]`.

If no symptom is matched — that is, the LLM returns no valid names from the patient's message — the node immediately routes to Triage Agent Mode B. There is no clarification loop. The node sets `uncommon_symptom = True`, records the patient's raw complaint in `raw_complaint`, sets `intake_complete = True`, and appends a handoff message informing the patient they will be connected to the diagnostic system. The graph reaches END and the session manager routes to the Triage Agent.

On successful detection, the node loads all matching symptom objects from `_SYMPTOM_MAP`, pools their red flags and urgency rules (deduplicating by flag text and rule criteria respectively), and populates the state with `symptom_names`, `symptom_data_list`, `all_red_flags`, and `all_urgency_rules`. An introductory AI message acknowledging the detected symptom(s) is appended to the conversation history.

If detection succeeds, the conditional edge `route_after_detect` routes forward to `merge_questions`. If detection fails (no valid names returned), the edge routes to END — the uncommon path is triggered immediately without any further clarification attempts.

### Node 2: `merge_questions`

Deduplicates and orders the essential questions from all detected symptom objects into a single unified question list.

Each of the 11 symptom objects in Phase 1.3 includes an `essential_questions` array — typically 4 to 6 questions specific to that symptom. For single-symptom presentations, the merge step is a passthrough: `merged_questions` is set to the symptom's `essential_questions` list unchanged.

For multi-symptom presentations, the raw combined list contains overlapping questions. A patient with chest pain and dyspnea might otherwise be asked "How long have you had the chest pain?" and "How long have you been experiencing shortness of breath?" as two separate questions, when a single "How long have you been experiencing these symptoms?" would serve both clinically. The `merge_questions` node sends the combined question list to the LLM with instructions to remove near-duplicate questions, preserve all unique questions, and order the result logically: general before specific, onset before character, symptom features before medical history.

The LLM is instructed to return a JSON array of 5 to 8 questions. Fewer is better — a shorter intake is less fatiguing for patients and more likely to be completed with full answers. The node falls back to the combined raw list if the LLM response cannot be parsed, ensuring the merge step never blocks the intake flow.

### Node 3: `prefill`

Extracts information already stated in the patient's initial message and records it as pre-filled answers, so those questions are skipped during the intake.

This node addresses a failure mode in naive intake implementations: a patient who says "I have chest pain and I'm coughing up blood" should never subsequently be asked "are you coughing blood?" The prefill node prevents this by reviewing the patient's opening message against the full question list before any questions are asked.

The LLM receives the patient's first message and the merged question list, and is instructed to return a JSON object mapping question strings to extracted answer strings — but only for questions that the patient's message clearly and specifically answers. The instruction explicitly prohibits inference: if the patient did not state something, it should not be pre-filled. The extracted answers are merged into the `answers` dict and the `prefilled_answers` dict simultaneously. The `ask_question` node checks `prefilled_answers` before presenting each question and skips any question whose key appears in the pre-filled set.

### Node 4: `ask_question`

Rephrases the next unanswered clinical question into patient-friendly language, referencing previous answers for conversational continuity.

The node identifies the next question index by scanning from `current_question_idx` forward, skipping any question already present in `prefilled_answers`. When all questions have been asked or pre-filled, it sets `intake_complete = True` and returns without generating a new message.

For each question it does ask, the node passes three inputs to the LLM: the clinical question text from the symptom object, the name(s) of the detected symptom(s) for clinical framing, and the full Q&A history accumulated so far. The system prompt instructs the LLM to rephrase the clinical question into warm, plain English while preserving the clinical intent exactly. Crucially, it is instructed to reference relevant details from previous answers naturally: "You mentioned the pain spreads to your arms — are you also noticing any shortness of breath or nausea?" This produces a conversation that feels responsive rather than scripted.

The rephrased question is appended as an AI message, the current question index is recorded in state, and the graph reaches END, pausing for the patient's response.

### Node 5: `process_answer`

The central routing node. Records the patient's answer, checks all accumulated answers for red flags, assesses whether the current answer is clinically adequate, and updates urgency.

**Answer recording.** The patient's most recent human message is recorded in the `answers` dict keyed by the current clinical question string.

**Red flag detection (Fix 3).** The node sends the full accumulated Q&A history (all questions and all answers recorded so far, not just the current turn) along with the pooled `all_red_flags` list to the LLM. The red flag system prompt uses "clinically permissive" framing: it explicitly instructs the model to translate everyday patient language into clinical concepts, and provides mapping examples. The prompt acknowledges that patients do not use medical terminology, and that the agent must bridge the gap. The LLM returns a JSON array of triggered red flag objects copied from the input list. Newly triggered flags are merged with any previously triggered flags, deduplicating by flag text to prevent double-counting.

**Urgency update.** For each newly triggered red flag, the flag's `urgency` field is compared against the current urgency level using the priority order `routine < urgent < emergency`. Urgency only escalates — it is never downgraded based on a subsequent answer, matching clinical practice.

**Adequacy check (Fix 2 and Fix 5).** If this question has not already had a follow-up (checked via `followup_question_idx`), the node sends the question and answer to a separate LLM call with an adequacy assessment prompt. The adequacy prompt checks for two distinct problems: vague answers ("sometimes", "maybe", "I don't know") that provide no useful clinical signal, and clinically incomplete answers that are clear but missing critical detail ("yes" to a smoking question without quantity or duration, "yes" to a heart disease question without naming the conditions). Either failure sets `pending_followup = True` and holds the question index at its current value.

**Routing.** After all checks, `route_after_process_answer` evaluates the updated state:
- If urgency has reached "emergency", route to `escalate`.
- If `pending_followup` is set, route to `ask_followup`.
- If `intake_complete` is True (all questions answered and no follow-up pending), route to `assess_urgency`.
- Otherwise, route to `ask_question` for the next question.

### Node 6: `ask_followup`

Generates a targeted follow-up question for a vague or clinically incomplete answer. Only fires once per question.

The node reads the current clinical question and the patient's most recent (vague or incomplete) answer, and asks the LLM to acknowledge what the patient said and ask for the specific missing information. The system prompt provides concrete examples: if the answer was "yes" to smoking, ask for duration, quantity, and cessation date. If the answer was "sometimes," ask how often specifically. If the answer was "arms," ask which arm and whether the symptom is constant or intermittent.

After the follow-up question is presented, `followup_question_idx` is set to the current question index and `pending_followup` is reset to False. This marks the question as "already followed up" — the next time `process_answer` runs for this question, it will skip the adequacy check and accept whatever the patient provides, preventing infinite follow-up loops. Any answer, however incomplete, is accepted on the second attempt.

The graph reaches END after the follow-up, pausing for the patient's response. The `IntakeSession.respond()` method routes the subsequent answer back to `process_answer`.

### Node 7: `escalate`

Fires when `urgency` reaches "emergency". Generates an immediate safety message directing the patient to emergency services.

The escalation message is fixed-text rather than LLM-generated, ensuring consistency and preventing any variation in the emergency directive. It informs the patient that their symptoms require immediate medical attention, instructs them to call 911 or go to the nearest emergency department, and states that a clinical summary is being prepared for their care team. The `escalated` flag is set to True in state.

After escalation, the graph routes unconditionally to `generate_summary`. The intake does not continue — no further questions are asked. The clinician summary is generated immediately so the care team has a record of what was collected before the emergency directive was issued.

### Node 8: `assess_urgency`

Evaluates the complete answer set against the pooled urgency rules from all detected symptoms, and takes the higher of the rule-based urgency and the flag-based urgency accumulated during questioning.

This node runs after all questions have been answered and no emergency escalation was triggered. It gives the LLM the full Q&A history, the complete `all_urgency_rules` list, and the current urgency level derived from red flag triggering. The LLM is asked to evaluate which urgency level best fits the overall clinical picture and return a JSON object with `urgency` (one of "routine", "urgent", "emergency") and a one-sentence clinical rationale.

The final urgency is the maximum of the LLM's judgment and the flag-triggered urgency. This dual mechanism ensures that urgency can be set by either hard red flag triggers during questioning or by holistic assessment of the complete answer set after all questions are answered. A patient who does not trigger any individual red flag but whose answers collectively suggest a concerning presentation can still be classified as urgent.

### Node 9: `generate_summary`

Produces the nine-section clinician handover note from the complete intake state.

The node assembles a context dict containing all information collected during the intake: detected symptom names, all Q&A pairs, triggered red flags, urgency level and action, and the deduplicated specialty routing, initial workup, key exam findings, admission criteria, and referral criteria extracted from all detected symptom objects. For multi-symptom presentations, these lists are pooled and deduplicated across all symptom objects before being passed to the LLM.

The summary LLM call receives this structured context dict and a system prompt specifying the nine required sections in order with markdown headers. The LLM produces the actual narrative content of each section, integrating the structured data with clinical judgment about how to present it concisely. The generated text is stored in `state["summary"]` and displayed as an AI message formatted with a visual separator.

---

## The State Object

The full conversation state is defined by the `IntakeState` TypedDict. All 16 fields are preserved across turns by LangGraph's state management.

| Field | Type | Purpose |
|---|---|---|
| `messages` | `list[BaseMessage]` | Full conversation history (HumanMessage and AIMessage objects). Uses `add_messages` reducer — new messages are appended, not overwritten. |
| `symptom_names` | `list[str]` | Canonical names of all detected symptoms (e.g., `["Chest Pain", "Hemoptysis"]`). Empty until `detect_symptom` succeeds. |
| `symptom_data_list` | `list[dict]` | Full symptom objects from Phase 1.3 for all detected symptoms. Contains the raw structured data consumed by all downstream nodes. |
| `merged_questions` | `list[str]` | Deduplicated, ordered question list produced by `merge_questions`. The authoritative list of questions the intake will ask. |
| `all_red_flags` | `list[dict]` | Pooled red flag objects from all detected symptom objects. Each dict contains `flag`, `implication`, and `urgency` fields. |
| `all_urgency_rules` | `list[dict]` | Pooled urgency rule objects from all detected symptom objects. Each dict contains `criteria`, `urgency`, and `action` fields. |
| `current_question_idx` | `int` | Index into `merged_questions` tracking which question the agent is currently asking. Increments after each accepted answer. |
| `answers` | `dict` | Maps question text (string) to patient answer text (string). Accumulates throughout the intake. Also contains pre-filled answers from the initial message. |
| `pending_followup` | `bool` | True when the most recent answer was deemed vague or clinically incomplete and a follow-up has not yet been asked. Triggers the `ask_followup` branch. |
| `followup_question_idx` | `int \| None` | Records the question index for which a follow-up has already been issued. Prevents multiple follow-ups on the same question. |
| `prefilled_answers` | `dict` | Subset of `answers` containing only answers extracted from the patient's initial message by the `prefill` node. Used by `ask_question` to determine which questions to skip. |
| `triggered_red_flags` | `list[dict]` | Accumulates red flag dicts that have been triggered during the intake. Deduplication is applied on merge to prevent double-counting. |
| `urgency` | `str` | Current urgency level: `"routine"`, `"urgent"`, or `"emergency"`. Monotonically non-decreasing — never lowered once raised. |
| `escalated` | `bool` | True if the `escalate` node has fired. Indicates emergency directive was issued to the patient. |
| `intake_complete` | `bool` | True when all questions in `merged_questions` have been answered or pre-filled and no follow-up is pending. Triggers the `assess_urgency` → `generate_summary` path. |
| `summary` | `str` | The generated clinician handover note text. Populated by `generate_summary`. Empty string until intake completes. |
| `clarification_attempts` | `int` | Legacy field. Always 0 in the current implementation — symptom detection now routes immediately to Triage Agent Mode B on failure rather than cycling through clarification attempts. Retained in state for schema compatibility. |

---

## The Six Improvements Over Naive Implementation

The Intake Agent underwent six targeted improvements over a baseline implementation. Each improvement addresses an identified failure mode in the original design. These are documented here as identified flaws and their corresponding fixes, because the progression from flaw to fix is itself an engineering finding — it demonstrates that the clinical correctness requirements of an intake agent exceed what a straightforward LLM pipeline delivers without deliberate design choices.

### Fix 1: Context-Aware Questions

**Identified flaw.** The original `ask_question` node called the LLM with only the current clinical question and the symptom name. Each question was rephrased in isolation, without awareness of what the patient had already said. The resulting conversation felt scripted and robotic — the agent would ask "do you have any shortness of breath?" immediately after the patient had mentioned "I feel like I can't catch my breath at all."

**Fix.** The `ask_question` node now passes all previous Q&A pairs to the LLM along with the current question. The system prompt explicitly instructs the model to reference relevant details from previous answers naturally. If the patient mentioned their pain spreads to the arms, the rephrased next question might say: "You mentioned the pain spreads to your arms — are you also noticing any shortness of breath or nausea?" The clinical intent of the question is unchanged; the delivery is personalised to the conversation history.

**Impact.** Conversations feel responsive and coherent. Patients are not asked to repeat information they have already provided. The intake experience is more consistent with how a skilled human intake nurse would conduct the same interview — building on what has already been shared rather than running through a fixed list.

### Fix 2: Follow-Up on Vague Answers

**Identified flaw.** The original implementation moved to the next question unconditionally after each patient response. Answers like "maybe," "sometimes," "I'm not sure," and "I don't know" were accepted and recorded. These answers are clinically uninformative — a recorded answer of "maybe" to "do you have a history of heart disease" provides no useful clinical signal and could lead to a summary that misleads the clinician.

**Fix.** After each answer is recorded, a separate LLM call assesses answer adequacy. The adequacy check looks for genuinely vague responses that cannot support clinical decision-making. If the answer is deemed inadequate, `pending_followup` is set to True and the `ask_followup` node fires. The follow-up acknowledges what the patient said and asks for the specific missing information. The follow-up fires at most once per question: `followup_question_idx` records which question has already received a follow-up, and any subsequent answer to that question is accepted regardless of content. This prevents infinite follow-up loops — a patient who cannot or will not provide more detail will not be kept in a loop.

**Impact.** Clinically uninformative answers are reduced in the recorded dataset. Patients who initially give vague responses often provide more useful information when asked a targeted, specific follow-up that acknowledges their previous answer rather than simply repeating the original question.

### Fix 3: Better Red Flag Detection

**Identified flaw.** The original red flag detection prompt instructed the LLM to "be conservative — only flag if clearly described." The red flag definitions in the Phase 1.3 symptom objects use clinical terminology: "Prolonged chest pain episodes," "Sustained palpitations," "Differential blood pressures between arms." Patients never use this language. The result was that red flag detection almost never triggered, even when patients described symptoms that clearly warranted urgent escalation.

A secondary flaw compounded this: the original implementation checked only the current answer against the red flag list, ignoring all previously recorded answers. A red flag that became apparent only in combination with two earlier answers would never be detected.

**Fix.** The red flag detection prompt was rewritten with a "clinically permissive" framing. The prompt explicitly instructs the LLM to interpret everyday patient language as clinical concepts, and provides illustrative mappings: "The pain has been going on for hours" → "Prolonged chest pain episodes"; "I feel a tearing sensation in my back" → aortic dissection-consistent presentation; "My heart has been pounding non-stop" → "Sustained palpitations." The entire accumulated Q&A history is sent with each red flag check, not just the current answer.

**Impact.** Red flag detection functions as intended. The agent correctly escalates patients whose conversational descriptions match clinically significant presentations, even when their language bears no resemblance to the medical terminology in the red flag definitions.

### Fix 4: Multi-Symptom Support

**Identified flaw.** The original `detect_symptom` node returned a single symptom name. A patient presenting with "chest pain and coughing blood" would have one symptom detected and the other ignored. This is clinically dangerous: combined presentations are frequently more serious than either symptom in isolation, and the combination of chest pain with hemoptysis (possible pulmonary embolism) is a distinct clinical picture from either alone.

**Fix.** `detect_symptom` now returns a JSON array of symptom names. The prompt provides explicit multi-symptom examples. All matched symptom objects are loaded, and their essential questions, red flags, and urgency rules are all pooled. The `merge_questions` node then deduplicates and orders the combined question list. Throughout the intake, `all_red_flags` and `all_urgency_rules` contain the union of all symptom-specific signals.

**Impact.** No symptoms stated by the patient are ignored. Multi-symptom presentations — which are clinically common and often more diagnostically significant than isolated symptoms — are handled correctly. The clinician summary correctly reflects all presenting symptoms and their associated clinical metadata.

### Fix 5: Clinical Depth Probing

**Identified flaw.** The original adequacy check (once added for Fix 2) detected only vague answers. It did not distinguish between vague and clinically incomplete. A patient answering "yes" to "do you have a history of heart disease?" provides a clear, non-vague answer — but it is clinically incomplete. The clinician needs to know whether the patient has ischemic heart disease, valvular disease, heart failure, or an arrhythmia. Similarly, "yes" to a smoking question without quantity or duration is factually clear but practically useless for risk stratification.

**Fix.** The adequacy system prompt was extended to detect two distinct failure modes: vague answers (category: "vague") and clinically incomplete answers (category: "incomplete"). The prompt provides concrete examples of what "incomplete" means for common clinical questions: "yes" to smoking requires duration, amount per day, and cessation date; "yes" to heart disease requires the specific diagnoses; "yes" to medications requires which medications. The follow-up node then asks specifically for the missing detail, not just for more information generally.

**Impact.** The clinician summary contains actionable detail rather than binary yes/no flags. A smoking history of "20 pack-years, quit 5 years ago" is clinically meaningful; "yes" is not.

### Fix 6: Pre-Fill from Initial Message

**Identified flaw.** A patient who opens with "I have severe chest pain that started two hours ago and I'm also coughing up blood" provides answers to at least two of the standard intake questions immediately. Without pre-filling, the agent would subsequently ask "How long have you had the chest pain?" and "Are you coughing blood?" — information the patient already provided. This is frustrating for patients and signals that the system is not actually listening to what they said.

**Fix.** The `prefill` node runs immediately after `merge_questions`, before any questions are asked. It sends the patient's opening message and the full merged question list to the LLM, asking it to extract answers that are explicitly present in the opening message. Only explicitly stated information is extracted — the prompt prohibits inference. The extracted answers are recorded in both `answers` and `prefilled_answers`. The `ask_question` node skips any question whose key appears in `prefilled_answers`.

**Impact.** Intake conversations are shorter and more efficient. Patients who provide rich initial descriptions are not forced to repeat themselves. The agent implicitly communicates that it read and understood the patient's opening statement, which builds conversational trust. In testing, a detailed initial message can eliminate two to three questions from the intake, reducing a seven-question intake to four or five actual exchanges.

---

## The Clinician Handover Summary

The final output of the Intake Agent is a nine-section structured document intended for direct handover to the clinical team. The sections are generated by the LLM using the accumulated intake state as input. The LLM is instructed to produce precise, clinical content using bullet points within sections and to omit sections gracefully when there is nothing to report.

### Section 1: Presenting Complaint

A one- or two-sentence summary of why the patient is seeking care, using the clinical presentation vocabulary. Derived from the detected symptom names and the patient's own description of their chief complaint. This section frames the entire handover note — the clinician knows immediately what the patient is there for before reading any further.

### Section 2: History of Presenting Complaint

A structured account of each intake question and the patient's answer, organised chronologically and by clinical relevance. This is the primary clinical data section — it captures all information collected during the intake, including pre-filled answers from the initial message, standard question-and-answer pairs, and any follow-up answers obtained after the adequacy check. The LLM integrates these Q&A pairs into a narrative-style history rather than a raw list.

### Section 3: Red Flags Identified

Lists any triggered red flags with their clinical implications. If no red flags were triggered, this section is omitted or explicitly states "none identified." For each triggered flag, the implication drawn from the Phase 1.3 symptom object is included — for example, "Prolonged chest pain episodes: possible acute coronary syndrome." This section directly informs the urgency recommendation in Section 4.

### Section 4: Urgency Classification and Recommended Action

States the final urgency level (routine, urgent, or emergency) and the recommended action associated with that level. For emergency classifications, the action is "EMERGENCY — activate emergency response immediately." For urgent classifications, the action is drawn from the matched urgency rule in the symptom object (e.g., "Refer for same-day cardiology evaluation"). For routine classifications, "Routine clinical assessment" is the default.

### Section 5: Specialty Routing Recommendation

Lists the clinical specialties most appropriate for follow-up, drawn from the `specialty_routing` field of all detected symptom objects and deduplicated across symptoms. For a patient with chest pain, this might list Cardiology, Pulmonology, and Emergency Medicine. The routing recommendation guides triage decisions about where in the health system the patient should be directed.

### Section 6: Suggested Initial Workup

Lists the initial investigations or tests indicated for the presenting symptoms, drawn from the `initial_workup` field of all symptom objects. For chest pain this might include ECG, troponin levels, chest X-ray, and CBC. These are drawn directly from the Phase 1.3 structured data, not generated by the LLM, ensuring they are grounded in the TMT textbook rather than invented.

### Section 7: Key Examination Findings to Elicit

Lists the physical examination findings most relevant to the presenting symptoms, drawn from the `key_exam_findings` field. This section supports the examining clinician by highlighting what to look for specifically, given what the patient has reported. For dyspnea, for example, this might include respiratory rate, oxygen saturation, accessory muscle use, and auscultation findings.

### Section 8: Admission Criteria

Lists the criteria that would warrant hospital admission, drawn from the `when_to_admit` field of all symptom objects. This section is included regardless of the current urgency level — even a routine presentation can benefit from the clinician knowing in advance what findings would change the disposition. Criteria are presented as clinical conditions rather than as a recommendation.

### Section 9: Referral Criteria

Lists the criteria that would warrant specialist referral if inpatient admission is not required, drawn from the `when_to_refer` field. This section closes the clinical handover by informing the treating clinician of the thresholds for escalating the patient's care to a specialist.

---

## Data Flow: Phase 1.3 to Phase 4.1

The 11 structured symptom objects produced in Phase 1.3 are the direct data source for the Intake Agent. They are consumed at module load time and used throughout the intake without any additional transformation beyond pooling and deduplication.

```
Phase 1.3 output: tmt_symptoms_gpt4o.json
        │
        ├── symptom.essential_questions
        │       └── merge_questions node → merged_questions (deduplicated)
        │               └── ask_question node → rephrased intake questions
        │
        ├── symptom.red_flags
        │       └── pooled → all_red_flags
        │               └── process_answer node → triggered_red_flags, urgency
        │
        ├── symptom.urgency_rules
        │       └── pooled → all_urgency_rules
        │               └── assess_urgency node → final urgency classification
        │
        ├── symptom.specialty_routing
        │       └── deduplicated across symptoms → generate_summary (Section 5)
        │
        ├── symptom.initial_workup
        │       └── deduplicated across symptoms → generate_summary (Section 6)
        │
        ├── symptom.key_exam_findings
        │       └── deduplicated across symptoms → generate_summary (Section 7)
        │
        ├── symptom.when_to_admit
        │       └── deduplicated across symptoms → generate_summary (Section 8)
        │
        └── symptom.when_to_refer
                └── deduplicated across symptoms → generate_summary (Section 9)
```

The Phase 1.3 symptom objects contain 15 fields each. The Intake Agent uses 8 of them directly. The remaining fields — `body_systems`, `differential_diagnosis`, `key_history_points`, `treatment_overview`, `etiology`, `epidemiology` — are not consumed by Phase 4.1. They are available for consumption by the Triage Agent (Phase 4.2), which may use the differential diagnosis and treatment overview fields to structure its RAG-backed clinical reasoning.

The design decision to use the structured symptom objects directly rather than via vector retrieval is deliberate. With only 11 entries, a lookup-by-name approach — case-insensitive string matching against a pre-loaded dictionary — is faster, more reliable, and more interpretable than embedding-based retrieval. There is no scenario in which vector similarity search over 11 items would outperform a dictionary lookup. The Phase 2.1 embedding pipeline explicitly deferred the question of whether to embed the symptom objects until evaluation revealed it was necessary; Phase 4.1 confirms it is not.

---

## LLM Integration

### Model Configuration

The Intake Agent uses `langchain_openai.ChatOpenAI` as its LLM interface. All nodes in the graph receive the same LLM instance, created once at session initialisation and shared across the full intake. The LLM is configured with `temperature=0` for all calls, producing deterministic outputs and preventing the kind of creative variation that is appropriate in open-ended generation tasks but inappropriate in a clinical intake tool where consistency is a correctness requirement.

The default model is `gpt-4o`. The choice of GPT-4o is driven by two requirements: instruction following fidelity on structured output tasks (JSON array and JSON object responses are required from multiple nodes), and clinical language understanding sufficient to map patient descriptions to clinical concepts. Both requirements benefit from a frontier model at inference time.

### LLM-Agnostic Design

The `--model` CLI flag passes the model name directly to `ChatOpenAI(model=model, temperature=0)`. Because LangChain's `ChatOpenAI` wrapper accepts any OpenAI-compatible model identifier, swapping to a different model requires only changing this flag — no code changes. This is architecturally significant for the thesis: a planned experiment will compare the intake quality of GPT-4o against a locally hosted model (e.g., Llama-3-70B via vLLM on EC2) on the same intake tasks. The model-agnostic design means this comparison can be conducted without modifying the agent code.

The underlying `ChatOpenAI` interface can also be replaced with other LangChain provider classes (e.g., `ChatAnthropic`, `ChatOllama`) for local model deployment without changing the graph structure or node implementations, because all nodes interact with the LLM through the standard LangChain message interface.

### LLM Call Budget per Intake

A typical intake session makes approximately 15 to 20 LLM calls:

| Node / Call | LLM calls | Notes |
|---|---|---|
| `detect_symptom` | 1 | Returns JSON array of symptom names; routes immediately to Mode B on failure (no clarification loop) |
| `merge_questions` | 0 or 1 | Only called if multiple symptoms detected |
| `prefill` | 1 | Extracts pre-filled answers from initial message |
| `ask_question` (per question) | 1 each | Typically 4–7 calls for a 5–8 question intake |
| `process_answer` — red flag check (per answer) | 1 each | One call per answer turn |
| `process_answer` — adequacy check (per answer) | 1 each | One call per answer turn (skipped on second attempt) |
| `ask_followup` (conditional) | 0 or 1 per question | Fires only on inadequate answers |
| `assess_urgency` | 1 | Final urgency classification |
| `generate_summary` | 1 | Produces the nine-section handover note |

For a 6-question intake with no follow-ups and one symptom, the typical call count is: 1 (detect) + 1 (prefill) + 6 (questions) + 6 (red flag checks) + 6 (adequacy checks) + 1 (urgency) + 1 (summary) = 22 calls. Multi-symptom presentations add 1 for the merge step; follow-ups add 1 per vague answer plus 1 additional adequacy check. The practical range is 15 to 25 LLM calls per complete intake session.

---

## Scripts Reference

### `agents/intake_agent.py`

The single script that implements the full Phase 4.1 agent.

**CLI usage:**

```bash
# Standard interactive session — symptom detection from patient's first message:
python agents/intake_agent.py

# Use a different model (e.g., cheaper or local):
python agents/intake_agent.py --model gpt-4o-mini

# Skip symptom detection — begin intake for a pre-specified symptom:
python agents/intake_agent.py --symptom "Chest Pain"

# Skip symptom detection for multiple symptoms (comma-separated):
python agents/intake_agent.py --symptom "Chest Pain,Hemoptysis"
```

**CLI flags:**

| Flag | Default | Effect |
|---|---|---|
| `--model MODEL` | `gpt-4o` | OpenAI model name passed to `ChatOpenAI`. Accepts any OpenAI-compatible identifier. |
| `--symptom NAME` | `None` | Bypasses the `detect_symptom` node and sets the symptom directly. Accepts comma-separated values for multi-symptom testing (e.g., `"Chest Pain,Dyspnea"`). Must match canonical symptom names exactly (case-insensitive). Valid values: Cough, Dyspnea, Hemoptysis, Chest Pain, Palpitations, Lower extremity edema, Fever, Involuntary Weight Loss, Fatigue, Acute Headache, Dysuria. |

The `--symptom` flag is primarily used for testing: it allows development of the questioning and summary nodes without running the detection node, and it supports testing specific multi-symptom combinations directly.

### `IntakeSession` API

The `IntakeSession` class exposes the agent as a programmatic API, suitable for integration into a web application, a test harness, or the downstream Triage Agent.

```python
from agents.intake_agent import IntakeSession

# Create a session with default model (gpt-4o)
session = IntakeSession()

# Create a session with a specific model
session = IntakeSession(llm_model="gpt-4o-mini")

# Create a session with a pre-specified symptom (bypasses detection)
session = IntakeSession(skip_to_symptom="Chest Pain")

# Start the intake with the patient's first message
response = session.start("I have chest pain that started two hours ago")
print(response)  # Agent's first question

# Process subsequent patient responses
response = session.respond("The pain is crushing and goes down my left arm")
print(response)  # Next question or follow-up

# Check whether the intake is complete
if session.is_complete():
    summary = session.get_summary()
    # summary is a structured dict, not a string

# Get the structured summary dict when complete
summary = session.get_summary()
```

**`IntakeSession.start(patient_message: str) → str`**

Begins the intake session with the patient's first message. Returns the agent's first response as a string — either an introductory message followed by the first question (for a successful symptom detection), or a handoff message informing the patient they are being routed to the diagnostic system (if no common symptom was matched). Must be called before any `respond()` calls.

**`IntakeSession.respond(patient_message: str) → str`**

Processes the patient's answer to a question. Returns the agent's next message, which may be the next question, a follow-up question for a vague answer, an emergency escalation directive, or the final clinician handover summary. After the summary is returned, subsequent calls to `respond()` return a static "intake complete" message.

**`IntakeSession.is_complete() → bool`**

Returns True when the intake has concluded — either all questions have been answered and the summary has been generated, emergency escalation has triggered the early-exit path, or the symptom was not recognised and the session has been routed to Triage Agent Mode B. Returns False during an active intake.

**`IntakeSession.get_summary() → dict`**

Returns the structured summary as a Python dict when `is_complete()` is True. Returns an empty dict if the intake is still in progress.

The summary dict has the following structure:

```python
{
    "symptoms": ["Chest Pain", "Hemoptysis"],        # list of detected symptom names
    "urgency": "urgent",                              # "routine" | "urgent" | "emergency"
    "escalated": False,                               # bool — True if 911 directive was issued
    "answers": {                                      # dict of question → answer pairs
        "What is the character of the pain?": "Crushing, radiates to left arm",
        ...
    },
    "triggered_red_flags": [                          # list of triggered red flag dicts
        {"flag": "Prolonged chest pain episodes", "implication": "...", "urgency": "urgent"},
        ...
    ],
    "specialty_routing": ["Cardiology", "Emergency Medicine", ...],
    "initial_workup": ["ECG", "Troponin", "Chest X-ray", ...],
    "key_exam_findings": ["Blood pressure both arms", "Heart sounds", ...],
    "when_to_admit": ["ACS with ongoing chest pain", ...],
    "when_to_refer": ["Stable angina for outpatient cardiology evaluation", ...],
    "clinician_note": "## Presenting Complaint\n...",  # full nine-section markdown note
}
```

---

---

## Routing to the Triage Agent

Once an intake session completes, the CLI automatically routes the patient to the Triage Agent without any manual intervention. The routing logic lives in `intake_agent.py`'s `main()` function and follows three distinct paths based on the intake outcome.

### The Three Routing Paths

```
Intake completes
        |
        |── uncommon_symptom=True ──────────────────────────────────────────────>
        |                                                                         Triage Agent Mode B
        |                                                              (session.get_raw_complaint() passed)
        |
        |── escalated=True ─────────────────────────────────────────────────────>
        |                                                                         STOP — no triage
        |                                                              (patient directed to emergency services)
        |
        └── common symptom, not escalated ──────────────────────────────────────>
                                                                          Triage Agent Mode A
                                                               (session.get_summary() passed)
```

**Path 1: Common symptom, not escalated — Mode A**

The most common path. When the Intake Agent successfully classified the symptom and completed the interview without triggering emergency escalation, the CLI calls `session.get_summary()` and passes the resulting structured dict to `TriageSession.diagnose_from_intake()`. The Triage Agent receives the full clinical picture — symptoms, Q&A pairs, red flags, urgency — and runs a single-pass RAG diagnosis.

**Path 2: Uncommon symptom — Mode B**

When the Intake Agent's first detection attempt fails to match any of the 11 canonical symptoms, the agent routes immediately to Mode B without any clarification attempts. `uncommon_symptom` is set to True and `intake_complete` is set to True simultaneously. The CLI detects `session.is_uncommon()` and routes to `TriageSession.start_uncommon()`, passing only the raw complaint string captured from the patient's first message. The Triage Agent then runs its multi-pass conversational Mode B flow, asking its own follow-up questions.

**Path 3: Emergency escalation — stop**

When urgency reaches `"emergency"` during questioning, the Intake Agent issues the emergency directive and generates the summary, but the CLI detects `summary.get("escalated") == True` and does not invoke the Triage Agent at all. Routing the patient to the Triage Agent after they have already been directed to emergency services would be clinically inappropriate and wastes resources.

### How Uncommon Symptoms Are Detected

The detection mechanism is in `_build_detect_symptom_node`. On the patient's very first message, if the LLM returns no valid symptom names, the node immediately triggers the uncommon path — there are no clarification attempts:

1. `uncommon_symptom` is set to True in state
2. `raw_complaint` is set to the patient's message (the original complaint, verbatim)
3. `intake_complete` is set to True to signal that the intake flow should exit
4. A handoff message is appended explaining the transfer to the diagnostic system

The patient's original message is used as the raw complaint because it contains their spontaneous, unguided description of their complaint. The Triage Agent's Mode B Pass 1 retrieval works best with the patient's unfiltered initial complaint.

### What Data Is Passed at Each Routing Point

| Routing path | Data passed | Method called |
|---|---|---|
| Mode A | `IntakeSession.get_summary()` dict — full structured summary | `TriageSession.diagnose_from_intake(summary)` |
| Mode B | Raw complaint string — patient's first message | `TriageSession.start_uncommon(raw_complaint)` |
| Emergency | Nothing | Triage Agent not invoked |

### The Full Pipeline Flow (CLI)

```bash
python agents/intake_agent.py
```

This single command runs the complete patient journey:

1. Patient types their complaint
2. Intake Agent conducts multi-turn interview (or routes immediately to uncommon path if symptom not recognised)
3. Intake Agent generates clinician handover summary
4. CLI auto-routes to Triage Agent (Mode A or Mode B depending on outcome)
5. Triage Agent produces diagnosis report
6. Session ends

No manual handoff is required. The `--symptom` flag pre-specifies the symptom and always routes to Mode A (since the symptom is known). The `--model` flag is passed through to both the Intake Agent and the Triage Agent, so model selection is uniform across the full pipeline.

---

## Updated State Fields

The `IntakeState` TypedDict contains two fields added specifically to support Triage Agent routing that were not part of the original six-fix implementation:

### `uncommon_symptom: bool`

Default: `False`. Set to `True` by `_build_detect_symptom_node` when the patient's first message does not match any of the 11 canonical symptoms. The routing is immediate — no clarification attempts are made. Once True, this field signals to the session manager (`IntakeSession`) that the intake has concluded via the uncommon path and that the Triage Agent's Mode B should be invoked. The `is_uncommon()` method reads this field indirectly via `self._phase == "uncommon"`.

This field is distinct from `intake_complete`. Both become True simultaneously on the uncommon path. `intake_complete` signals that no further questions should be asked; `uncommon_symptom` signals which triage path to take.

### `raw_complaint: str`

Default: `""` (empty string). Set to the text of the patient's message when the uncommon routing is triggered. This is the verbatim text of the patient's opening message — because the routing is immediate (no clarification loop), this is always the patient's first and only message at the point of handoff.

The Triage Agent's Mode B uses this string as the seed for its first retrieval pass, so fidelity to the patient's original spontaneous presentation matters. Since there are no clarification iterations that could reshape the description, the raw complaint is always the patient's unguided initial statement.

The full updated state table including these two fields:

| Field | Type | Purpose |
|---|---|---|
| `messages` | `list[BaseMessage]` | Full conversation history. Uses `add_messages` reducer. |
| `symptom_names` | `list[str]` | Canonical names of all detected symptoms. |
| `symptom_data_list` | `list[dict]` | Full symptom objects for all detected symptoms. |
| `merged_questions` | `list[str]` | Deduplicated, ordered question list. |
| `all_red_flags` | `list[dict]` | Pooled red flags from all detected symptom objects. |
| `all_urgency_rules` | `list[dict]` | Pooled urgency rules from all detected symptom objects. |
| `current_question_idx` | `int` | Index into `merged_questions`. |
| `answers` | `dict` | Question text → patient answer text. |
| `pending_followup` | `bool` | True when last answer was vague or clinically incomplete. |
| `followup_question_idx` | `int \| None` | Question index that has already received a follow-up. |
| `prefilled_answers` | `dict` | Answers extracted from the patient's initial message. |
| `triggered_red_flags` | `list[dict]` | Accumulated triggered red flag dicts. |
| `urgency` | `str` | `"routine"`, `"urgent"`, or `"emergency"`. Monotonically non-decreasing. |
| `escalated` | `bool` | True if emergency directive was issued. |
| `intake_complete` | `bool` | True when all questions answered or uncommon path triggered. |
| `summary` | `str` | The generated nine-section clinician handover note. |
| `clarification_attempts` | `int` | Legacy field. Always 0 — uncommon path is triggered immediately on detection failure, not after repeated attempts. |
| `uncommon_symptom` | `bool` | True if symptom not in the 11 canonical symptom categories. Triggers Triage Agent Mode B. |
| `raw_complaint` | `str` | Patient's first message, captured for Triage Agent Mode B handoff. |

---

## The IntakeSession API (Updated)

The full `IntakeSession` public API, including the two methods added for Triage Agent integration:

**`IntakeSession.start(patient_message: str) → str`**

Begins the intake with the patient's first message. Returns the agent's first response.

**`IntakeSession.respond(patient_message: str) → str`**

Processes a patient response. Returns the next question, follow-up, emergency directive, or final summary.

**`IntakeSession.is_complete() → bool`**

Returns True when the intake has concluded — either via the full question-answer flow (common path) or via the uncommon symptom detection (immediate routing when no common symptom matched). Returns True for both the `"done"` and `"uncommon"` phases.

**`IntakeSession.is_uncommon() → bool`**

Returns True when the intake concluded via the uncommon symptom path. This is the signal the CLI uses to choose between Triage Agent Mode A and Mode B.

```python
def is_uncommon(self) -> bool:
    """True when the symptom was not recognised and should be routed to Triage Agent Mode B."""
    return self._phase == "uncommon"
```

**`IntakeSession.get_raw_complaint() → str`**

Returns the raw patient complaint string for Triage Agent Mode B handoff. Returns the patient's first message as a verbatim string. Returns an empty string if the session did not conclude via the uncommon path.

```python
def get_raw_complaint(self) -> str:
    """Return the raw patient complaint for Triage Agent Mode B handoff."""
    return self._state.get("raw_complaint", "")
```

**`IntakeSession.get_summary() → dict`**

Returns the structured summary dict when `is_complete()` is True and the session did not take the uncommon path. Returns an empty dict if the intake is still in progress. The summary dict structure is unchanged from the original specification.

### Full Integration Example

```python
from agents.intake_agent import IntakeSession
from agents.triage_agent import TriageSession

# Run the intake
intake = IntakeSession()
print(intake.start("I have severe chest pain and I'm coughing blood"))

while not intake.is_complete():
    answer = input("Patient: ")
    print(intake.respond(answer))

# Route to Triage Agent
triage = TriageSession()

if intake.is_uncommon():
    # Mode B — uncommon symptom
    raw_complaint = intake.get_raw_complaint()
    first_question = triage.start_uncommon(raw_complaint)
    print(first_question)

    while not triage.is_complete():
        answer = input("Patient: ")
        response = triage.respond(answer)
        print(response)

else:
    # Mode A — common symptom
    summary = intake.get_summary()
    if not summary.get("escalated"):
        diagnosis = triage.diagnose_from_intake(summary)
        print(diagnosis["report"])
    else:
        print("Emergency case — Triage Agent not invoked.")
```

---

## Remaining Limitations

These limitations are documented explicitly as areas for future work. Each limitation represents a known gap between the current implementation and a production clinical intake system.

### LLM Urgency Assessment May Exceed Structured Data

The `assess_urgency` node gives the LLM broad clinical judgment authority over the final urgency classification. It receives the urgency rules from the Phase 1.3 symptom objects but is not constrained to only those rules. The LLM may classify a presentation as urgent based on its general clinical training rather than on the structured rules — effectively hallucinating a clinical rationale beyond what the structured data contains. This is controlled hallucination in the sense that the output direction (urgency level) is constrained to three options, but the rationale may not be directly traceable to the structured data. The red flag-triggered urgency from `process_answer` provides a structured lower bound; the LLM assessment can only add urgency, not override a structured emergency trigger.

### No Conversation Memory Across Sessions

`IntakeSession` is a single-session object. When the session ends, the state is discarded. There is no mechanism for persisting intake history across sessions — a returning patient, or a patient who disconnects and reconnects mid-intake, starts from scratch. LangGraph's built-in checkpointing and persistence features could address this, but they require a backend store (e.g., SQLite or Redis) that has not been configured.

### Single Language (English Only)

All prompts, system messages, and the Phase 1.3 symptom data are in English. The agent has no multilingual capability. Real-world clinical populations frequently include patients who are not fluent in English, and intake quality for non-English speakers would be substantially degraded. LangGraph's multilingual capabilities and the multilingual support of models like GPT-4o could support a multilingual intake with appropriate prompt engineering, but this is not implemented.

### One Follow-Up Attempt per Question

The `followup_question_idx` mechanism allows at most one follow-up per question. A patient who gives a vague answer followed by another vague answer will have the second vague answer accepted without further probing. This is a deliberate design choice to prevent conversational loops, but it means that clinical depth probing is limited. In practice, a patient who cannot provide more detail after a targeted follow-up is unlikely to provide more detail on a third attempt, and the clinical risk of infinite loops outweighs the marginal gain from additional attempts.

### No Clinician Review Step Before Summary Finalisation

The clinician handover note is generated automatically from the intake data without any human review. A production system would benefit from a brief clinical review step — either automated (a second LLM pass checking the summary for factual consistency with the recorded answers) or human (a clinician reviewing and approving the summary before it becomes the official intake record). The current implementation generates and presents the summary immediately, with no opportunity for correction before handoff.
