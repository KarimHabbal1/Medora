# Phase 5: Patient Memory System

## Overview

The Patient Memory System gives Medora a persistent clinical identity for each patient that survives across sessions. After every intake and triage session, the system extracts clinically relevant information — conditions, medications, allergies, demographics, smoking history, surgical history, family history — and merges it into a persistent profile stored on disk. When the same patient returns, the Intake Agent loads their profile and uses it to confirm known facts rather than re-eliciting them from scratch.

The problem it solves is direct: a patient who told Medora three months ago that they have asthma, take a salbutamol inhaler, and are allergic to penicillin should not have to repeat all of this on their next visit. The Intake Agent should acknowledge what is already known, briefly confirm it is still accurate, and focus its questions on the new presenting complaint.

The system also benefits the Triage Agent. A diagnosis made with knowledge of a patient's prior conditions, previous diagnoses, and current medications is materially different from one made from scratch. A patient with a history of DVT presenting with chest pain and leg swelling has a different prior probability distribution than a first-time presenter with identical acute findings.

### Where the Patient Memory System Sits in the Pipeline

| Phase | Component | Function |
|---|---|---|
| 1.1–1.2 | PDF extraction and chunking | 5,631 searchable text chunks from TMT textbook |
| 1.3 | Symptom structuring | 11 structured clinical symptom objects |
| 2.1–2.3 | Embedding and retrieval | ChromaDB vector store; 90% Hit@1, 100% Hit@3 validated |
| 3 | Reranking | BGE-reranker-v2-m3; +7.4% content relevance over bi-encoder alone |
| 4.1 | Intake Agent | Multi-turn patient interview; produces structured summary |
| 4.2 | Triage Agent | Diagnostic engine — produces grounded diagnosis report |
| **5** | **Patient Memory** | **Persistent cross-session profiles; injects history into Intake and Triage** |
| 7 (planned) | Doctor review | Confirm, reject, or escalate the triage diagnosis |

The Patient Memory System is not a pipeline stage in the linear sense — it runs alongside sessions rather than between them. It reads at the start of every session (loading context) and writes at the end of every session (updating the profile). Between sessions, it is the connective tissue that makes Medora's clinical agents stateful over time.

---

## The Patient Profile Schema

Each patient has a single JSON file. The full schema:

```json
{
  "patient_name": "Karim Habbal",
  "created": "2025-10-01T14:32:00.123456+00:00",
  "last_updated": "2025-10-22T09:17:44.654321+00:00",
  "sessions": 2,
  "demographics": {
    "age": 34,
    "gender": "male"
  },
  "known_conditions": [
    {
      "condition": "Asthma",
      "since": "childhood",
      "source_session": 1
    }
  ],
  "medications": [
    {
      "medication": "Salbutamol inhaler",
      "for": "Asthma",
      "source_session": 1
    }
  ],
  "allergies": [
    {
      "allergen": "Penicillin",
      "reaction": "rash",
      "source_session": 1
    }
  ],
  "smoking_history": {
    "status": "former",
    "duration": "10 years",
    "quantity": "2 packs/day",
    "quit": "2 years ago"
  },
  "substance_use": {},
  "family_history": [
    {
      "condition": "Heart disease",
      "relation": "father",
      "source_session": 1
    }
  ],
  "surgical_history": [],
  "session_history": [
    {
      "session_number": 1,
      "date": "2025-10-01T14:32:00.123456+00:00",
      "symptoms": ["Cough"],
      "urgency": "routine",
      "diagnosis": "Pertussis (Whooping Cough)",
      "key_findings": "2-week dry cough with pertussis exposure, low-grade fever, and known asthma; antibiotic treatment initiated.",
      "outcome": null
    }
  ]
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `patient_name` | `str` | Display name exactly as entered by the patient. |
| `created` | `str` (ISO-8601) | UTC timestamp of first profile creation. |
| `last_updated` | `str` (ISO-8601) | UTC timestamp of the most recent update. |
| `sessions` | `int` | Count of completed sessions. Incremented by `update_from_session()`. |
| `demographics.age` | `int \| null` | Extracted from session data. Set once; not overwritten if already present. |
| `demographics.gender` | `str \| null` | Same extraction logic as age. |
| `known_conditions` | `list` | Each entry is a dict with `condition`, optional `since`, and `source_session`. The LLM occasionally stores entries as plain strings — the context formatters handle both. |
| `medications` | `list` | Each entry is a dict with `medication`, optional `for` (indication), and `source_session`. Same string fallback handling. |
| `allergies` | `list` | Each entry is a dict with `allergen` or `allergy`, optional `reaction`, and `source_session`. |
| `smoking_history` | `dict` | Object with keys `status`, `duration`, `quantity`, `quit`. Replaced entirely if updated. |
| `substance_use` | `dict` | Free-form object for any substance use history. Replaced entirely if updated. |
| `family_history` | `list` | Each entry is a dict with `condition`, optional `relation`, and `source_session`. |
| `surgical_history` | `list` | Each entry is a dict with `procedure`, optional `date`, and `source_session`. |
| `session_history` | `list` | One entry per completed session. See session history schema below. |

### Session History Entry Schema

Each entry in `session_history` captures the clinical facts of a single session:

| Field | Type | Description |
|---|---|---|
| `session_number` | `int` | Sequential session counter, starting at 1. |
| `date` | `str` (ISO-8601) | UTC timestamp of the session. |
| `symptoms` | `list[str]` | Presenting symptoms from the intake summary. |
| `urgency` | `str` | Urgency classification: `"routine"`, `"urgent"`, or `"emergency"`. |
| `diagnosis` | `str` | Primary diagnosis extracted from the triage report (first non-empty line under `## Primary Diagnosis`). |
| `key_findings` | `str` | One-sentence LLM-generated summary of the session's clinical highlights. |
| `outcome` | `null` | Reserved for the treating clinician to record the actual outcome (Phase 7). Always `null` in the current implementation. |

---

## How Profiles Are Updated

After each session, `update_from_session()` merges new clinical information into the profile. The flow has seven steps:

### Step 1: Load the current profile

The existing profile is loaded from disk, or a blank one is created if this is the patient's first session. The current session number is `profile["sessions"] + 1`.

### Step 2: Build the LLM input

The LLM receives three pieces of information:

- The current session number
- The existing profile's clinical sections (demographics, conditions, medications, allergies, smoking history, substance use, family history, surgical history) — not the full profile, just the fields that may need updating
- The new intake summary dict (from `IntakeSession.get_summary()`)
- The new diagnosis dict (from `TriageSession.get_diagnosis()`)

### Step 3: LLM extracts the delta

The LLM is given explicit instructions to extract only NEW information — not to duplicate existing entries. If the patient reported a condition that already appears in the profile, it should not be added again. If existing information has changed (the patient quit smoking since the last visit, a medication was discontinued), the LLM should return the updated value.

The LLM's response is a JSON object containing only the fields that changed, with list fields returning only the new items to append rather than the full list.

### Step 4: Apply the delta

The update is applied field by field with type-specific merge strategies:

- **Demographics**: merged key by key — only null/missing fields are overwritten. An age recorded in Session 1 is never replaced by a different value in Session 2.
- **List fields** (`known_conditions`, `medications`, `allergies`, `family_history`, `surgical_history`): new items from the delta are appended to the existing list. The existing list is never replaced.
- **Object fields** (`smoking_history`, `substance_use`): the entire object is replaced if the LLM returns a non-empty update.

### Step 5: Build the session history entry

A session history entry is constructed from:
- The intake summary's `symptoms` and `urgency` fields
- The primary diagnosis, extracted by parsing the triage report for the `## Primary Diagnosis` section header and pulling the first non-empty line beneath it
- The LLM's one-sentence `session_summary` from the delta response

The entry is appended to `profile["session_history"]`.

### Step 6: Atomic file write

The updated profile is written atomically:

```python
tmp_path = path.with_suffix(".tmp")
# Write full JSON to .tmp
tmp_path.replace(path)
```

Writing to a `.tmp` file and renaming it means a crash mid-write leaves the original profile intact. The patient's history is never partially overwritten.

### Step 7: Thread-safe serialisation

Each patient has a dedicated `threading.Lock`. Two concurrent sessions for the same patient are serialised — neither can read-modify-write the profile simultaneously. Sessions for different patients do not block each other. This is sufficient for single-server deployment; horizontal scaling would require a distributed lock.

---

## The Confirmation Approach

### The Problem with Skipping Known Questions

The initial design skipped any question that had already been answered in a prior session. If the patient's profile said they were a smoker, the Intake Agent would not ask about smoking. If the profile listed asthma as a known condition, the Intake Agent would not ask about medical history.

This approach has a fundamental flaw: patient circumstances change. A patient may have quit smoking since their last visit. A medication may have been discontinued by their GP. A condition may have resolved. Silently skipping these questions means the profile becomes stale with no mechanism for correction.

### The Final Design: Briefly Confirm Known Facts

The final design confirms known facts rather than skipping them. The Intake Agent's context string instructs the agent to ask one brief confirmation per known item — a yes/no question — rather than re-eliciting the full history.

Example of a confirmation generated from the context:

> "I see you were taking salbutamol for your asthma — is that still the case?"

> "My records show you quit smoking about 2 years ago — still smoke-free?"

If the patient confirms: the agent acknowledges and moves on immediately.

If the patient reports a change: the agent notes the update, and the change will be captured in the next `update_from_session()` call.

This approach recovers stale information in roughly one exchange per item rather than one full re-interview per session. The marginal time cost is small. The benefit — catching a discontinued medication before the Triage Agent factors it into a diagnosis — is clinically material.

### How the Context String Drives This

`get_context_for_intake()` constructs the context string with two distinct sections:

**Section 1 — Known patient history:** A bulleted summary of all known clinical facts. The Intake Agent can reference these facts when responding to the patient's answers.

**Section 2 — Confirmation instructions:** An explicit list of items to confirm with the patient, generated only for items that are actually present in the profile:

```
Briefly CONFIRM these known facts with the patient (one sentence each,
accept yes/no — if anything changed, record the update):
  - smoking history (former smoker — 10 years — 2 packs/day — quit 2 years ago)
  - conditions (Asthma)
  - medications (Salbutamol inhaler)
  - allergies
```

The context string closes with: `Focus your detailed questions on the NEW presenting complaint.`

---

## Integration with the Intake Agent

The patient name and profile are resolved at the very start of a session, before any clinical questioning begins.

### Step 1: Patient identification

```python
patient_name = input("Patient name: ").strip()
profile = memory.get_or_create(patient_name)
intake_context = memory.get_context_for_intake(patient_name)
```

If the patient has prior sessions, a welcome-back message is printed. If not, the profile is created blank.

### Step 2: Context injection into IntakeSession

```python
session = IntakeSession(
    llm_model=args.model,
    skip_to_symptom=args.symptom,
    patient_context=intake_context,
)
```

`patient_context` is passed to `IntakeSession.__init__()` and stored in the session object. The context is available to every node in the LangGraph graph for the duration of the session.

### Step 3: Context reaches the detect_symptom node

The `detect_symptom` node uses the patient context when generating its opening response. If the patient has known conditions, the context instructs the agent to briefly confirm them before proceeding to the new complaint. This happens naturally in the first one or two exchanges — the agent confirms what it knows, then focuses on the presenting symptom.

### Step 4: Context reaches the ask_question node

The `ask_question` node also has access to the patient context when rephrasing each clinical question. If a question overlaps with something already in the patient's history, the agent can reference the known fact rather than asking from scratch:

> "You mentioned you have asthma — when your chest feels tight like this, does it feel like your usual asthma or different in some way?"

### Step 5: Profile updated after session completes

After both the Intake and Triage sessions complete, the profile is updated:

```python
if session.is_complete() and patient_name != "anonymous":
    llm_for_memory = _make_llm(args.model)
    memory.update_from_session(patient_name, summary_data, diagnosis_data, llm_for_memory)
```

Anonymous sessions are never persisted. This is a deliberate design choice — patients who decline to provide a name do not accumulate a history.

---

## Integration with the Triage Agent

Patient history reaches the Triage Agent through the intake summary dict, which serves as the handoff between agents.

### Step 1: Triage context retrieved

Before passing the intake summary to the Triage Agent, the Intake Agent's `main()` retrieves a separate triage-oriented context string:

```python
triage_ctx = memory.get_context_for_triage(patient_name)
if triage_ctx:
    summary["patient_history_context"] = triage_ctx
```

`get_context_for_triage()` produces a compact clinical summary — known conditions, medications, allergies, smoking history, family history, surgical history, and previous diagnoses (last 3). The format is terse and clinical, matched to what the Triage Agent needs for differential diagnosis rather than the confirmation-oriented format used for the Intake Agent.

### Step 2: `parse_intake_to_clinical_picture()` preserves the history

When the Triage Agent parses the intake summary into a clean clinical picture, it explicitly carries the patient history through:

```python
patient_history = intake_summary.get("patient_history_context", "")
if patient_history:
    result["patient_history"] = patient_history
```

This is one of the few fields that `parse_intake_to_clinical_picture()` does not strip. The Triage Agent's clinical picture parser is otherwise designed to remove pre-computed signals that might bias the diagnostic LLM — specialty routing, initial workup suggestions, and intake agent notes are all removed. Patient history from the memory system is an exception because it represents ground truth about the patient, not a prior agent's inference.

### Step 3: `_clinical_picture_to_context()` includes the history section

When the clinical picture is formatted for injection into diagnosis prompts, the patient history appears as a dedicated section:

```
Patient background (from previous sessions):
Patient history:
- Known conditions: Asthma (since childhood)
- Medications: Salbutamol inhaler
- Allergies: Penicillin
- Smoking: former smoker — 10 years — 2 packs/day — quit 2 years ago
- Previous diagnosis: Pertussis (Whooping Cough) (2025-10-01)

Consider this history when forming your differential diagnosis.
```

### Step 4: Diagnosis integrates patient history

The Triage Agent's diagnosis prompt receives this full context. Known comorbidities factor into the differential. Prior diagnoses are visible when evaluating recurrence likelihood. Current medications are visible when the LLM generates management considerations (the system already knew about the penicillin allergy and correctly routed to azithromycin in the test case).

---

## Storage

### File Layout

```
data/
  patient_profiles/
    karim_habbal.json
    jane_doe.json
    maría_josé.json
```

Patient profiles are stored as JSON files under `data/patient_profiles/`. The filename is the normalised form of the patient's display name.

### Name Normalisation

```python
def _normalise_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")
```

- `"Karim Habbal"` → `"karim_habbal.json"`
- `"  Jane  Doe  "` → `"jane_doe.json"`
- `"María José"` → `"maría_josé.json"` (non-ASCII characters are preserved)

The display name is preserved inside the profile's `patient_name` field — normalisation is only for filesystem naming.

### Privacy

Patient profile files are listed in `.gitignore`. They contain Protected Health Information (PHI) and must not be committed to the repository. The storage directory is created automatically on first use if it does not exist.

### Scalability

JSON files on the local filesystem are sufficient for development and single-machine deployment. Concurrent access within a single Python process is safe due to per-patient threading locks. Concurrent access from multiple processes (e.g., multiple gunicorn workers) is not safe — the last write wins without cross-process coordination. The planned migration to a database in Phase 7 (EC2 deployment) will resolve this.

---

## The PatientMemory API

`PatientMemory` is a stateless class — every public method reads from disk before operating and writes back to disk on completion. There is no in-memory cache; each call reflects the current state of the file.

### `__init__(storage_dir=None)`

```python
memory = PatientMemory()
# or
memory = PatientMemory(storage_dir=Path("/custom/path"))
```

Creates the storage directory if it does not exist. Initialises the per-patient lock registry (empty at startup).

---

### `get_or_create(patient_name) → dict`

```python
profile = memory.get_or_create("Karim Habbal")
```

Loads an existing profile from disk, or creates and saves a blank profile if none exists. Returns a deep copy of the profile dict — the caller may mutate it freely without affecting the stored file. To persist changes, use `update_from_session()`.

The blank profile has all list fields as empty lists, all object fields as empty dicts, and `sessions=0`.

---

### `update_from_session(patient_name, intake_summary, diagnosis, llm) → dict`

```python
updated_profile = memory.update_from_session(
    "Karim Habbal",
    session.get_summary(),
    triage.get_diagnosis(),
    llm,
)
```

Merges new clinical information from a completed session into the patient's profile. Uses the LLM to extract the delta (new information only). Appends a session history entry. Writes the updated profile atomically. Returns the updated profile dict.

| Argument | Type | Description |
|---|---|---|
| `patient_name` | `str` | Patient display name. |
| `intake_summary` | `dict` | Dict returned by `IntakeSession.get_summary()`. |
| `diagnosis` | `dict` | Dict returned by `TriageSession.get_diagnosis()`. |
| `llm` | `ChatOpenAI` | LLM instance for delta extraction. Temperature should be 0. |

---

### `get_context_for_intake(patient_name) → str`

```python
ctx = memory.get_context_for_intake("Karim Habbal")
```

Returns a formatted string summarising the patient's clinical history, with confirmation instructions for known facts. Designed to be injected into the Intake Agent as a `patient_context` argument.

Returns an empty string if the patient has no prior sessions (first visit or no prior data). In that case, the Intake Agent behaves as normal with no modification.

The context string includes:
- Known conditions, medications, allergies, smoking history, substance use, family history, surgical history
- Last 3 sessions from `session_history` with date, symptoms, and diagnosis
- A list of facts to briefly confirm with the patient

---

### `get_context_for_triage(patient_name) → str`

```python
ctx = memory.get_context_for_triage("Karim Habbal")
```

Returns a compact clinical summary formatted for the Triage Agent. Terse and clinical — not conversation-ready. Designed to be embedded in the `patient_history_context` key of the intake summary dict before it is passed to `TriageSession.diagnose_from_intake()`.

Returns an empty string if the patient has no prior sessions.

The context string includes:
- Known conditions (with `since` dates where available)
- Current medications
- Allergies
- Smoking history
- Family history
- Surgical history
- Previous diagnoses from `session_history` (last 3, with dates)
- A closing instruction: `Consider this history when forming your differential diagnosis.`

---

### `list_patients() → list[str]`

```python
names = memory.list_patients()
# e.g. ["Jane Doe", "Karim Habbal"]
```

Returns a list of display names for all patients with stored profiles, sorted alphabetically by filename. If a profile file cannot be parsed, the filename stem is returned instead of the display name.

---

## CLI Reference

The module has a standalone CLI for development and debugging. Requires `OPENAI_API_KEY` in the environment only when `--simulate` is used.

```bash
python agents/patient_memory.py --patient "Karim Habbal" [flags]
```

| Flag | Description |
|---|---|
| `--patient NAME` | **Required.** Patient display name. |
| `--show` | Load and print the patient's current profile as formatted JSON. |
| `--simulate` | Run a simulated session update using built-in test intake and diagnosis data. Requires `OPENAI_API_KEY`. Prints the updated profile and both context strings after the update. |
| `--intake-context` | Print the intake agent context string for this patient. |
| `--triage-context` | Print the triage agent context string for this patient. |
| `--list` | List all patients with stored profiles. The `--patient` flag is still required but its value is not used when `--list` is the only active flag. |
| `--model MODEL` | OpenAI model for `--simulate`. Default: `gpt-4o`. |

### Usage Examples

```bash
# Inspect a patient's stored profile:
python agents/patient_memory.py --patient "Karim Habbal" --show

# Simulate a full session update (creates the profile if new):
python agents/patient_memory.py --patient "Karim Habbal" --simulate

# See what the Intake Agent would receive as context:
python agents/patient_memory.py --patient "Karim Habbal" --intake-context

# See what the Triage Agent would receive as context:
python agents/patient_memory.py --patient "Karim Habbal" --triage-context

# List all stored patient profiles:
python agents/patient_memory.py --patient "-" --list

# Simulate with a cheaper model:
python agents/patient_memory.py --patient "Karim Habbal" --simulate --model gpt-4o-mini
```

---

## Design Evolution

### Iteration 1: Skip Known Questions Entirely

The first design skipped any intake question that was already answered in a prior session. The Intake Agent received a list of topics to omit, and those questions were removed from the question flow.

**Flaw identified:** Patient history is not static. A patient who was a smoker at their last visit may have quit. A medication may have been discontinued. A condition may have been resolved or reclassified. Silently skipping these questions means the profile becomes a snapshot of the patient's state at session 1, not their current state. The Triage Agent would factor in stale clinical information — a patient listed as a current smoker who quit two years ago would receive management recommendations calibrated to an active smoker.

### Iteration 2: Briefly Confirm Known Facts

The final design changed the instruction from "skip these questions" to "briefly confirm these facts."

The context string instructs the Intake Agent to ask one yes/no confirmation per known item. If the patient confirms, the agent moves on in one exchange. If the patient reports a change, the agent acknowledges it and the new information is captured in the `update_from_session()` call at the end of the session.

This adds at most a few exchanges to the beginning of a returning patient's session — a minor overhead that prevents the much more serious failure mode of operating on stale clinical data.

---

## Limitations

### LLM-Dependent Extraction Quality

The accuracy of profile updates depends on the LLM correctly identifying what is new versus what already exists. The LLM receives the current profile's clinical sections and the new session data, and is instructed to return only the delta. In practice, the LLM generally executes this correctly for clearly structured data. Edge cases — conditions stated ambiguously, medications referenced by brand name in one session and generic name in another — may produce missed updates or incorrect merges. There is no automated validation of the delta output beyond JSON schema conformance.

### No Deduplication Guarantee

The update strategy appends new items from the LLM delta to the existing list. If the LLM incorrectly identifies an existing condition as new across two sessions, the condition will appear twice in `known_conditions`. The context formatters do not deduplicate before rendering — a duplicate entry will appear twice in the confirmation list and in the triage context. This is a known gap; deduplication requires either a string-matching heuristic (fragile) or an additional LLM normalisation step (additional cost).

### String vs Dict Inconsistency

The profile schema expects list entries to be dicts with specific keys (`condition`, `medication`, `allergen`, etc.). The LLM occasionally returns plain strings. The context formatters handle both formats with explicit type checks:

```python
for c in conditions:
    if isinstance(c, dict):
        part = c.get("condition", "")
    else:
        part = str(c)
```

This makes the system tolerant of the inconsistency, but it means the schema is effectively loose. A profile may contain a mix of dict and string entries for the same field across sessions.

### Name-Only Authentication

Patients are identified by their display name as entered at the start of a session. There is no password, token, or identity verification. A patient who misspells their name creates a new profile. A patient who enters someone else's name gains access to that person's history. This is acceptable for development but would be a critical security gap in production. Phase 7 will implement proper authentication tied to a user account system.

### No Profile Merging

The per-patient lock prevents concurrent writes from corrupting the file, but it does not prevent logical conflicts. If a patient completes two sessions simultaneously (unlikely but not impossible in a multi-server deployment), the second write will overwrite the first. The result is a profile that reflects one session's delta and omits the other's. Resolving logical merge conflicts across concurrent sessions would require either pessimistic locking at the session level or a merge strategy in the update function. Neither is implemented.

### JSON Files Do Not Scale

The current storage backend is adequate for development with a small patient population. For production deployment with many concurrent patients and multiple server processes, a relational or document database with proper concurrent access semantics is required. The Phase 7 migration to EC2 will replace JSON file storage with a database backend. The `PatientMemory` class is designed to be replaceable — the public API does not expose any filesystem-specific concepts, so a drop-in database-backed implementation is feasible.
