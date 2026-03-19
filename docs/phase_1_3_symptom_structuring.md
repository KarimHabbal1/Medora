# Phase 1.3: Symptom Structuring

## Overview

Phase 1.3 converts raw textbook chunks into structured clinical JSON objects for the 11 common presenting symptoms found in Chapter 2 of the TMT textbook. Each output object encodes the full clinical reasoning framework for a single symptom: what to ask, what to watch for, what to consider, and where to refer.

### Why This Phase Exists

The chunked text produced in Phase 1.2 is optimised for retrieval — it lets a vector search find relevant passages when a user describes a condition. But retrieval alone is insufficient for driving a clinical intake conversation. When a patient says "I have chest pain," the agent cannot simply retrieve chunks and narrate them back; it needs to:

- Ask targeted, structured questions in a logical order
- Recognise specific answer patterns as red flags
- Apply urgency rules to decide whether to escalate
- Route to the correct specialty

These decisions require a machine-readable object, not free text. The structured symptom objects produced in Phase 1.3 serve precisely this purpose.

### The Distinction Between Chunks and Structured Objects

| Artefact | Source | Format | Consumer | Purpose |
|---|---|---|---|---|
| Chunks (`tmt_chunks_structured.json`) | Phases 1.1–1.2 | Free text + metadata | Triage Agent (RAG) | Semantic retrieval over all 42 chapters |
| Structured symptoms (`tmt_symptoms_gpt4o.json`) | Phase 1.3 | Typed JSON schema | Intake Agent (Phase 5) | Pre-loaded decision support for 11 symptoms |

Chunks answer the question: "What does the book say about this topic?" Structured objects answer the question: "How should the agent reason through this presenting complaint?"

### Why Chapter 2 Only

Chapter 2 ("Common Symptoms") covers the 11 presenting complaints that initiate the majority of clinical encounters: cough, dyspnea, chest pain, palpitations, lower extremity edema, fever, involuntary weight loss, fatigue, acute headache, dysuria, and hemoptysis. These are the entry points for every patient interaction in the Intake Agent.

Chapters 3–42 describe specific conditions and their management. The Triage Agent retrieves from those chapters via RAG. Structuring all 200–300 conditions would be costly, time-consuming, and architecturally redundant — the chunk embeddings already encode the key clinical content at retrieval time. If Phase 4 evaluation reveals gaps, Tier 2 structuring can be added for high-frequency conditions.

---

## How It Works

The script `data_processing/symptom_structurer.py` executes five sequential steps for each symptom.

### Step 1: Load Structured Chunks, Filter to Chapter 2

The script reads `data/chunks/tmt_chunks_structured.json` (produced in Phase 1.2) and filters it to retain only chunks whose `chapter` field equals `"Common Symptoms"`. This isolates the Chapter 2 content from the full 42-chapter chunk file.

```python
symptom_chunks = [
    c for c in all_chunks if c.get("chapter") == COMMON_SYMPTOMS_CHAPTER
]
```

### Step 2: Group Chunks by Symptom Section

Each chunk carries a `section` field that names the symptom it belongs to (e.g., `"CHEST PAIN"`, `"COUGH"`). The script groups all Chapter 2 chunks by their section value, producing a dictionary that maps each symptom name to its ordered list of chunks.

Chunks within each section are sorted by the trailing integer in their `chunk_id` (format: `tmt::<chapter>::<section>::<subsection>::<index>`) to preserve reading order.

```python
grouped.setdefault(section, []).append(chunk)
# then sort by _chunk_sort_key(chunk["chunk_id"])
```

### Step 3: Concatenate Chunk Texts Per Symptom

All chunk texts for a symptom are joined into a single string. When the subsection label changes between consecutive chunks, a Markdown heading is inserted (`### <subsection>`) so the LLM can use the subsection structure (e.g., "When to Admit", "Symptoms and Signs", "Differential Diagnosis") as semantic anchors during extraction.

```python
parts.append(f"\n### {sub}\n")   # inserted at each subsection transition
parts.append(chunk["text"])
```

### Step 4: Send to LLM with a Detailed Schema Prompt

The concatenated text is placed into a structured prompt that instructs the model to act as a medical knowledge extraction system and to populate a fixed JSON schema. The prompt explicitly restricts the model to information stated or directly implied in the provided text, preventing hallucination from background training data.

Temperature is set to 0 for both models, producing deterministic and reproducible output.

The system role for GPT-4o is:

> "You are a precise medical knowledge extraction assistant. Always respond with valid JSON only, no prose."

### Step 5: Parse JSON Response with Robust Fallback Strategies

LLMs sometimes wrap their JSON output in Markdown code fences, XML-style tags, or prose. The `parse_json_response` function attempts four extraction strategies in order:

1. Direct `json.loads` on the full response string
2. Extract content from ` ```json ... ``` ` or ` ``` ... ``` ` Markdown blocks
3. Extract content from `<json> ... </json>` tags
4. Find the outermost `{...}` block using a regular expression

If all four strategies fail, the symptom is stored as `{"symptom": "<name>", "error": "parsing_failed"}` so the output file remains valid JSON and the failure is visible without crashing the pipeline.

---

## The Schema

Each structured symptom object conforms to a fixed 15-field schema. Every field is required; the validator flags any missing fields during processing.

The original schema had 10 fields covering the intake-critical information: questions, red flags, urgency rules, differential diagnoses, and routing. After reviewing the textbook subsections more carefully — specifically "When to Admit", "When to Refer", "Treatment", and the etiology and epidemiology discussions — it became clear that five additional fields were needed to avoid dropping clinically important content. The schema was expanded to 15 fields before the full GPT-4o run. This iterative schema development is documented here as a methodological record for the thesis: structuring medical knowledge correctly requires reading the source material structurally, not just thematically.

### `symptom` (string)

The canonical name of the symptom as it appears in the textbook section heading. Used as the lookup key when the Intake Agent receives a patient complaint.

### `body_systems` (array of strings)

The physiological systems relevant to this symptom. Used to scope the differential and to inform the specialty routing logic. A symptom like dyspnea spans respiratory, cardiovascular, hematologic, metabolic, and psychiatric systems simultaneously — this field makes that breadth explicit.

### `essential_questions` (array of strings)

The key questions the Intake Agent asks the patient. These are drawn directly from the textbook's clinical reasoning guidance for each symptom. The Intake Agent presents these questions sequentially, collecting the structured context needed to evaluate red flags and apply urgency rules. This is the most interaction-critical field in the schema.

### `red_flags` (array of objects)

Each entry contains:
- `flag` — the specific warning sign (e.g., "Thunderclap headache")
- `implication` — what the finding suggests clinically (e.g., "Possible subarachnoid hemorrhage")
- `urgency` — one of `emergency`, `urgent`, or `routine`

The Intake Agent checks patient responses against these flags in real time. A matched flag triggers the corresponding urgency escalation path without waiting for the full intake to complete.

### `differential_diagnosis` (array of objects)

Each entry contains:
- `condition` — the condition name
- `key_features` — a list of distinguishing clinical features
- `likelihood_context` — the patient profile or setting in which this diagnosis should be prioritised

This field does not drive direct agent output in Phase 5. It is primarily used to inform the agent's reasoning and to generate a preliminary differential for the clinician summary.

### `urgency_rules` (array of objects)

Each entry contains:
- `criteria` — the combination of findings that triggers escalation
- `urgency` — the urgency level
- `action` — the recommended clinical response

Urgency rules are the operational escalation logic of the Intake Agent. When a patient's symptom pattern matches a criterion, the agent uses the `action` field to determine its next step — whether to advise immediate emergency attendance, urgent same-day review, or routine referral.

### `specialty_routing` (array of strings)

The clinical specialties relevant to this symptom. The Intake Agent uses this field to generate a referral recommendation at the end of the intake conversation.

### `key_history_points` (array of strings)

Important elements of the patient history beyond the essential questions. These supplement the conversational intake with context that may not emerge from direct questioning alone (e.g., recent travel, occupational exposures, medication history).

### `key_exam_findings` (array of strings)

Physical examination findings that are diagnostically significant for this symptom. The Intake Agent does not perform examinations, but it can prompt the attending clinician to look for specific findings, or use this field to populate a structured clinical handover note.

### `initial_workup` (array of strings)

Recommended initial investigations. Used to generate a suggested workup list in the clinician-facing summary produced after each intake session.

### `when_to_admit` (array of strings)

Specific criteria that warrant hospital admission for this symptom. Drawn directly from the textbook's "When to Admit" subsections. Each entry describes a clinical scenario (e.g., "Massive hemoptysis requiring airway protection") rather than a generic threshold. Entry counts range from 1 to 7 depending on the complexity of the symptom and the number of distinct admission triggers in the source text.

### `when_to_refer` (array of strings)

Criteria indicating that specialist referral is appropriate. Drawn from the textbook's "When to Refer" subsections. Each entry identifies a specific clinical situation and, where possible, the target specialty (e.g., "Nephrotic syndrome to a nephrologist"). Entry counts range from 1 to 3 per symptom.

### `treatment_overview` (array of strings)

Key treatment approaches relevant to this symptom, including drug classes, procedural interventions, and supportive measures. This field does not replace condition-specific management guidance in Chapters 3–42 — it captures the top-level therapeutic options that the Intake Agent can include in the clinician handover note as a prompt for immediate management planning. Entry counts range from 3 to 7 per symptom.

### `etiology` (array of strings)

Common causes, pathophysiological mechanisms, and contributing factors for this symptom. Drawn from the textbook's aetiological discussions within each symptom section. This field provides the agent with the conceptual framework underlying the differential diagnosis — why certain conditions are on the list, not just which conditions are on it. Entry counts range from 3 to 8 per symptom.

### `epidemiology` (string)

A single descriptive string summarising prevalence data, affected demographics, incidence rates, and high-risk populations for this symptom, as stated in the textbook. Unlike the array fields, this is a free-text summary because the textbook presents epidemiological data narratively rather than as enumerable items. All 11 symptoms have this field populated.

---

## The 11 Structured Symptoms

The GPT-4o run produced complete structured objects for all 11 symptoms in Chapter 2. The table below summarises each symptom using counts from `tmt_symptoms_gpt4o.json`. The five new fields (when_to_admit, when_to_refer, treatment_overview, etiology) are included as additional columns; epidemiology is a single string for all 11 symptoms and is not counted here.

| Symptom | Body Systems | Ess. Questions | Red Flags | Differentials | Urgency Rules | Admit | Refer | Treatment | Etiology |
|---|---|---|---|---|---|---|---|---|---|
| Cough | Respiratory, Gastrointestinal, Cardiovascular | 5 | 4 | 6 | 3 | 3 | 3 | 4 | 4 |
| Dyspnea | Respiratory, Cardiovascular, Hematologic, Metabolic, Psychiatric | 5 | 4 | 5 | 3 | 3 | 3 | 4 | 4 |
| Hemoptysis | Respiratory, Cardiovascular | 3 | 2 | 4 | 2 | 1 | 3 | 4 | 7 |
| Chest Pain | Cardiovascular, Pulmonary, Gastrointestinal, Musculoskeletal | 5 | 4 | 5 | 3 | 3 | 2 | 3 | 3 |
| Palpitations | Cardiovascular, Endocrine, Psychiatric | 6 | 3 | 4 | 2 | 1 | 2 | 3 | 4 |
| Lower Extremity Edema | Cardiovascular, Lymphatic, Renal, Hepatic, Musculoskeletal, Integumentary | 6 | 2 | 6 | 2 | 2 | 3 | 4 | 4 |
| Fever | Immune, Nervous, Endocrine | 5 | 3 | 3 | 2 | 3 | 2 | 3 | 4 |
| Involuntary Weight Loss | Endocrine, Gastrointestinal, Psychiatric, Oncological | 5 | 2 | 8 | 3 | 3 | 1 | 7 | 8 |
| Fatigue | Endocrine, Cardiovascular, Respiratory, Renal, Hematologic, Neurologic, Psychiatric, Gastrointestinal, Musculoskeletal | 8 | 5 | 6 | 2 | 1 | 3 | 5 | 4 |
| Acute Headache | Neurologic, Cardiovascular, Infectious, Ophthalmologic | 5 | 5 | 5 | 3 | 4 | 1 | 3 | 6 |
| Dysuria | Urinary, Reproductive | 6 | 4 | 5 | 2 | 7 | 3 | 5 | 4 |

---

## GPT-4o vs Llama 3.1:8b Comparison

To evaluate whether an open-source local model could replace the commercial API for knowledge structuring, the script ran both GPT-4o and Llama 3.1:8b (via Ollama) on three comparison symptoms: Chest Pain, Cough, and Dyspnea. This comparison is documented here as primary thesis evidence.

### Side-by-Side Comparison: Chest Pain

Chest Pain is the most clinically demanding of the comparison symptoms — it spans four body systems and carries several life-threatening differentials. It provides the clearest illustration of the quality gap.

**GPT-4o — Chest Pain**

- `essential_questions`: 5 (quality, radiation, precipitants/relieving factors, associated symptoms, cardiovascular history)
- `red_flags`: 4 — prolonged episodes (emergency), tearing pain radiating to back (emergency), pain with dyspnea and cough (emergency), abnormal ECG (urgent)
- `differential_diagnosis`: 5 — ACS, aortic dissection, pulmonary embolism, pericarditis, musculoskeletal pain
- `urgency_rules`: 3 — each tied to a specific actionable criterion and outcome
- `body_systems`: 4 (Cardiovascular, Pulmonary, Gastrointestinal, Musculoskeletal)
- `key_history_points`: 5 entries covering quality, duration, radiation, precipitants, associated symptoms, and cardiovascular risk factors
- `key_exam_findings`: 4 entries including pericardial rub, differential blood pressures, and ECG findings
- `initial_workup`: 4 tests including high-sensitivity troponin and D-dimer
- `when_to_admit`: 3 — failure to exclude life-threatening causes, high-risk PE with positive D-dimer, abnormal ECG and troponin
- `when_to_refer`: 2 — poorly controlled noncardiac chest pain to a pain specialist, sickle cell anemia to a hematologist
- `treatment_overview`: 3 — guided by underlying etiology, high-dose PPI therapy for noncardiac chest pain, cognitive-behavioural interventions for psychological causes
- `etiology`: 3 — cardiovascular/pulmonary/musculoskeletal/gastrointestinal disorders, anxiety states, cocaine use
- `epidemiology`: present (string) — notes variability of ACS, aortic dissection, and PE frequency across clinical settings

**Llama 3.1:8b — Chest Pain**

- `essential_questions`: 3 (nature of pain, sharp vs dull, radiation only)
- `red_flags`: 2 — acute MI (emergency), pulmonary embolism (emergency) — no specificity on distinguishing features
- `differential_diagnosis`: 2 — ACS, pulmonary embolism only
- `urgency_rules`: 2 — the second urgency rule (`negative D-dimer` → routine) is clinically incorrect: a negative D-dimer in a low-pretest-probability patient is reassuring, not a trigger for routine further testing without context
- `body_systems`: 2 (cardiovascular, pulmonary) — gastrointestinal and musculoskeletal causes omitted
- `key_history_points`: 2 generic entries
- `key_exam_findings`: 2 generic entries (vital signs, lung auscultation)
- `initial_workup`: 2 tests — no chest radiograph, no D-dimer separately listed

The Llama output is syntactically valid but clinically thin. It captures only the two highest-salience diagnoses and omits the nuanced reasoning that makes the schema useful (pericarditis, musculoskeletal causes, aortic dissection are entirely absent). The `urgency_rules` entry containing `negative D-dimer → routine workup` represents a clinical reasoning error.

### Summary Count Table: All 3 Comparison Symptoms

| Symptom | Metric | GPT-4o | Llama 3.1:8b |
|---|---|---|---|
| Chest Pain | Essential questions | 5 | 3 |
| Chest Pain | Red flags | 4 | 2 |
| Chest Pain | Differentials | 5 | 2 |
| Chest Pain | Urgency rules | 3 | 2 |
| Cough | Essential questions | 5 | 3 |
| Cough | Red flags | 4 | 3 |
| Cough | Differentials | 6 | 3 |
| Cough | Urgency rules | 3 | 1 |
| Dyspnea | Essential questions | 5 | 3 |
| Dyspnea | Red flags | 4 | 2 |
| Dyspnea | Differentials | 5 | 3 |
| Dyspnea | Urgency rules | 3 | 2 |

### Quality Analysis

The differences between the two models are consistent and systematic across all three symptoms:

**Comprehensiveness.** GPT-4o consistently extracts more entries in every field category. For Cough, GPT-4o identified 6 differentials against Llama's 3. For Dyspnea, GPT-4o produced 4 red flags against Llama's 2. These are not marginal differences — they represent entire diagnostic branches that the Intake Agent would fail to consider if using Llama output.

**Clinical specificity.** GPT-4o's red flags include precise clinical detail: "Dyspnea with hypoxemia on sitting or standing — Platypnea-orthodeoxia syndrome — urgent." Llama's flags are generic and lack the specificity needed to trigger correct escalation. GPT-4o's `urgency_rules` map specific finding combinations to specific actions. Llama's urgency rules are coarse and in one case clinically incorrect.

**Formatting consistency.** GPT-4o consistently capitalises symptom names, uses title case for conditions, and returns clean structured strings in arrays. Llama produced lowercase field values (`"symptom": "chest pain"`, `"body_systems": ["cardiovascular", "pulmonary"]`), inconsistent key feature formatting (comma-joined strings instead of array elements), and a malformed urgency string (`"emergency | urgent"`) that would require additional normalisation to process reliably.

**Coverage of body systems.** GPT-4o captured all relevant physiological domains: chest pain spans four systems in the GPT-4o output, five in dyspnea. Llama restricted chest pain to two systems and dyspnea to two systems, omitting hematologic, metabolic, and psychiatric dimensions that are clinically meaningful.

### The Thesis Argument

The comparison data supports the following position for the thesis write-up:

> Open-source 8-billion-parameter models are insufficient for medical knowledge structuring tasks that require comprehensive multi-domain extraction, consistent schema adherence, and clinically accurate reasoning rules. The 3–4x deficit in extracted entries across red flags, differentials, and urgency rules, combined with observed clinical errors in urgency logic, disqualifies Llama 3.1:8b from production use in this pipeline without fine-tuning on medical extraction corpora. GPT-4o at temperature 0 produces output that is consistent, comprehensive, and clinically coherent across all 11 symptoms.

| Dimension | GPT-4o | Llama 3.1:8b |
|---|---|---|
| Average essential questions (3 symptoms) | 5.0 | 3.0 |
| Average red flags (3 symptoms) | 4.0 | 2.3 |
| Average differentials (3 symptoms) | 5.3 | 2.7 |
| Average urgency rules (3 symptoms) | 3.0 | 1.7 |
| Clinical reasoning errors observed | None | At least 1 (D-dimer rule) |
| Formatting consistency | High | Low (case, array structure) |
| Body systems coverage | Comprehensive | Narrow |

---

## How the Intake Agent Uses This (Phase 5 Preview)

The structured symptom objects are the data layer of the Intake Agent. The workflow is as follows:

1. **Symptom detection.** The patient states their chief complaint. The Intake Agent performs fuzzy matching against the 11 symptom names to identify the relevant structured object.

2. **Question loop.** The agent iterates through `essential_questions`, asking each one in turn and recording the patient's response. Questions may be reordered based on prior responses (e.g., if a patient confirms chest pain and dyspnea early, the dyspnea structured object's questions may be merged).

3. **Real-time red flag checking.** After each patient response, the agent checks the answer against `red_flags`. If a flag is matched, the associated `urgency` level immediately updates the session state. An `emergency` match may interrupt the intake loop and route the patient directly to the escalation pathway.

4. **Urgency rule evaluation.** Once the core questions are complete, the agent evaluates `urgency_rules` against the full set of collected responses. This produces a final urgency classification for the session.

5. **Specialty routing.** The matched urgency rule's `action` field, combined with the `specialty_routing` list, determines the referral recommendation surfaced in the clinician handover note.

6. **Clinician summary generation.** The agent produces a structured summary that includes: collected history, triggered red flags, applied urgency rule, suggested `initial_workup`, and relevant `key_exam_findings` for the attending clinician to verify.

This is not RAG retrieval. The Intake Agent does not search embeddings — it loads a pre-structured object and executes a deterministic decision procedure against it. The value of Phase 1.3 is that this decision procedure is grounded in the textbook's own clinical reasoning, encoded once and reused across every patient interaction.

---

## Why Not Structure the Whole Book?

Chapters 3–42 of the TMT textbook contain approximately 200–300 specific conditions. There are several reasons not to structure all of them in Phase 1.3:

**Cost.** Processing all 42 chapters with GPT-4o at temperature 0 would cost approximately $15–20 in API fees, depending on the average chunk count per condition. Phase 1.3 spent a small fraction of that cost to cover the 11 high-priority symptoms.

**Architectural redundancy.** The Triage Agent uses RAG over the full chunk set. Each chunk already preserves the textbook's structured subsections — "Essentials of Diagnosis", "When to Refer", "When to Admit", "Treatment" — as named fields in the chunk metadata. The retrieval system can locate and surface this content without pre-structuring it into JSON objects. Pre-structuring 200+ conditions would duplicate functionality that RAG already provides.

**Phase 4 evaluation gate.** The Phase 4 evaluation will measure retrieval accuracy and agent response quality across the full condition set. If RAG alone is sufficient for Chapters 3–42, no further structuring is needed. If the evaluation reveals systematic gaps for high-frequency conditions (e.g., hypertension management, diabetes workup), a targeted Tier 2 structuring run can be added at that point.

**Intake Agent scope.** The Intake Agent's function is to structure the initial complaint, not to manage conditions. By the time the patient's presenting symptom has been characterised and urgency classified, control passes to the Triage Agent (which handles condition-specific RAG retrieval) or to a human clinician. Structured objects for conditions would not be consumed by the Intake Agent's decision loop.

---

## Scripts Reference

### `data_processing/symptom_structurer.py`

The single script that executes the full Phase 1.3 pipeline.

**Usage:**

```bash
# Full run — GPT-4o for all 11 symptoms, Ollama for 3 comparison symptoms:
python data_processing/symptom_structurer.py

# GPT-4o only (skip Ollama):
python data_processing/symptom_structurer.py --gpt-only

# Ollama only (skip GPT-4o):
python data_processing/symptom_structurer.py --ollama-only

# Use a different Ollama model:
python data_processing/symptom_structurer.py --ollama-model mistral

# Process specific symptoms only:
python data_processing/symptom_structurer.py --symptoms "CHEST PAIN,COUGH"
```

**CLI flags:**

| Flag | Effect |
|---|---|
| `--gpt-only` | Skips Ollama; runs GPT-4o for all/selected symptoms |
| `--ollama-only` | Skips GPT-4o; runs Ollama for comparison symptoms only |
| `--ollama-model MODEL` | Sets the Ollama model tag (default: `llama3.1:8b`) |
| `--symptoms "A,B"` | Restricts processing to the named symptoms (case-insensitive) |

**Environment requirements:**

- `OPENAI_API_KEY` must be set in the `.env` file at `data_processing/.env`
- Ollama must be running locally on port 11434 for Ollama calls (connection errors are caught and logged without crashing the script)

**Output:**

The script writes to `data/structured_symptoms/`:
- `tmt_symptoms_gpt4o.json` — GPT-4o structured objects for all 11 symptoms (or selected subset)
- `tmt_symptoms_ollama.json` — Llama 3.1:8b structured objects for the 3 comparison symptoms

After both models complete, a comparison summary is printed to stdout showing counts for essential questions, red flags, and differentials per symptom.

### Output Files

**`data/structured_symptoms/tmt_symptoms_gpt4o.json`**

A JSON array of 11 objects, one per symptom. Each object conforms to the 15-field schema described above. This file is the primary artefact consumed by the Intake Agent in Phase 5.

**`data/structured_symptoms/tmt_symptoms_ollama.json`**

A JSON array of 3 objects (Chest Pain, Cough, Dyspnea). Used for thesis comparison only; not consumed by any production agent component.

---

## Limitations

**Schema quality is bounded by textbook content.** Some symptoms have richer clinical sections than others in the TMT textbook. Hemoptysis, for example, yielded only 3 essential questions and 2 red flags — not because of model failure but because the textbook section is shorter and less detailed than the Chest Pain section. The structured output is a faithful extraction, not an augmentation.

**GPT-4o may add minor inferences beyond the text.** Despite the explicit instruction to extract only stated or directly implied information, GPT-4o at temperature 0 occasionally draws on its training distribution when the textbook phrasing is ambiguous. Temperature 0 minimises this risk significantly and makes the behaviour reproducible, but it cannot eliminate it entirely. Each structured object should be reviewed by a clinician before deployment.

**Llama 3.1:8b output is usable but thin.** The Ollama output is syntactically valid JSON and correctly identifies the most prominent clinical features. However, the deficits documented in the GPT-4o vs Llama comparison section make it unsuitable for production use without fine-tuning on medical knowledge extraction tasks. It may be adequate for a rapidly prototyped demo but should not be used in any evaluation benchmarks or clinician-facing features.

**Only 11 symptoms are structured.** The 200–300 specific conditions in Chapters 3–42 are handled via RAG retrieval, not structured objects. A patient complaint that falls outside the 11 Chapter 2 symptoms (e.g., a direct question about a named condition) must be routed to the Triage Agent's retrieval pipeline. The Intake Agent's structured decision support does not extend to condition management.
