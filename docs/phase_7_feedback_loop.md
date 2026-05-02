# Phase 7: Feedback Loop

## Overview

Phase 7 is the data collection infrastructure for continuous improvement of the Medora diagnostic system. It captures doctor verdicts on system-generated diagnoses, stores the complete diagnostic chain per case, and exports reviewed cases as structured training data for future fine-tuning.

The complete loop:

1. The system produces a diagnosis (Phase 5 Triage Agent)
2. The case — including every piece of information the system used — is automatically saved to `data/feedback/`
3. A doctor reviews the diagnosis through the CLI or a future frontend
4. The doctor confirms the system was correct, or rejects it and provides the correct diagnosis
5. Accumulated reviewed cases are exported as JSONL for fine-tuning

Phase 7 is a data collection layer, not a training pipeline. The actual fine-tuning — running LoRA/QLoRA on the exported JSONL — is a future step that becomes straightforward once a sufficient number of reviewed cases has accumulated. The bottleneck is not the training code; it is obtaining real doctor feedback on real cases.

### Where Phase 7 Sits in the Pipeline

| Phase | Component | Function |
|---|---|---|
| 1.1–1.2 | PDF extraction and chunking | 5,631 searchable text chunks from TMT textbook |
| 1.3 | Symptom structuring | 11 structured clinical symptom objects |
| 2.1–2.3 | Embedding and retrieval | ChromaDB vector store; 90% Hit@1, 100% Hit@3 validated |
| 3 | Reranking | BGE-reranker-v2-m3; +7.4% content relevance over bi-encoder alone |
| 4.1 | Intake Agent | Multi-turn patient interview; produces structured summary |
| 5 | Triage Agent | Diagnostic engine — produces grounded diagnosis report |
| **7** | **Feedback Loop** | **Doctor review → labeled training data → fine-tuning path** |

---

## Why Feedback Collection Matters

The Triage Agent produces diagnoses grounded in textbook evidence. But it has no mechanism to know whether those diagnoses are correct. Without a feedback loop, a diagnosis that is consistently wrong in a particular clinical domain would be undetectable — the system would continue producing the same error with no signal that anything is wrong.

Doctor confirmation and rejection creates labeled training data. Each reviewed case is a pair: the system's diagnosis, and whether it was right. Over time, patterns emerge:

- Which conditions are consistently misdiagnosed
- Which symptom presentations produce confident but incorrect primary diagnoses
- Which retrieval failures lead to wrong conclusions — cases where the retrieved chunks didn't cover the right condition, causing the LLM to reason from incomplete evidence
- Which urgency levels correlate with lower accuracy

This data enables multiple downstream improvements:

**Fine-tuning the LLM.** Rejected cases are supervision signal. The LLM can be trained to produce the doctor's diagnosis on cases structurally similar to ones it got wrong.

**Improving prompts.** Patterns of systematic error often reveal prompt failures — the diagnosis format forcing the LLM toward a premature conclusion, or the clinical picture parser dropping a finding that was diagnostically important.

**Identifying RAG gaps.** If the system consistently misses a condition, the retrieved chunks for cases in that clinical area should be inspected. The textbook may not have sufficient coverage, or the embedding model may not be retrieving the right passages.

**Calibrating confidence.** Cases where the system expressed high confidence but was rejected are the most instructive. Calibration failures — where high confidence does not correspond to high accuracy — are a systematic risk in clinical AI.

---

## The Case Schema

Each case is stored as a JSON file in `data/feedback/`. The filename uses the format `{YYYYMMDD}_{patient_name}_{uuid8}.json` — for example, `20260421_karim_habbal_a1b2c3d4.json`.

The full schema:

```json
{
  "case_id": "20260421_karim_habbal_a1b2c3d4",
  "patient_name": "Karim Habbal",
  "timestamp": "2026-04-21T09:14:32.001234+00:00",

  "symptoms": ["Cough"],
  "urgency": "routine",

  "intake_summary": {
    "symptoms": ["Cough"],
    "urgency": "routine",
    "escalated": false,
    "answers": {
      "How long have you had this cough?": "About 2 weeks, started gradually.",
      "Is the cough dry or productive?": "Mostly dry.",
      "Do you have a fever?": "Yes, low-grade, around 37.8°C.",
      "Any known exposures?": "My colleague was diagnosed with pertussis last week."
    },
    "triggered_red_flags": [],
    "clinician_note": "2-week dry cough with pertussis exposure."
  },

  "clinical_picture": {
    "symptoms": ["Cough"],
    "urgency": "routine",
    "clinical_findings": {
      "onset": "2 weeks ago",
      "character": "dry",
      "fever": "low-grade 37.8°C",
      "exposure": "pertussis contact"
    },
    "red_flags": []
  },

  "diagnosis_report": {
    "report": "## Primary Diagnosis\nPertussis (Whooping Cough) — confidence: moderate\n\n...",
    "mode": "common",
    "pass": 1,
    "num_chunks_used": 3
  },

  "retrieved_chunks": [
    {
      "chunk_id": "chunk_0042",
      "chapter": "Chapter 9 — Pulmonary Disorders",
      "section": "Pertussis",
      "text": "Pertussis is characterised by a prolonged paroxysmal cough..."
    }
  ],

  "system_primary_diagnosis": "Pertussis (Whooping Cough)",

  "review_status": "pending",
  "doctor_decision": null,
  "doctor_diagnosis": null,
  "doctor_notes": null,
  "reviewed_at": null
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `case_id` | `str` | Unique identifier; also the filename stem. Format: `{YYYYMMDD}_{name_slug}_{uuid8}` |
| `patient_name` | `str` | Display name of the patient, stripped of leading/trailing whitespace |
| `timestamp` | `str` | UTC ISO-8601 timestamp of when the case was saved |
| `symptoms` | `list[str]` | List of symptom names as classified by the Intake Agent |
| `urgency` | `str` | Urgency level: `"routine"`, `"urgent"`, or `"emergency"` |
| `intake_summary` | `dict` | Full output of `IntakeSession.get_summary()` — all Q&A pairs, red flags, clinician note |
| `clinical_picture` | `dict` | Parsed version produced by `parse_intake_to_clinical_picture()` — the clean clinical signal the Triage Agent reasoned from |
| `diagnosis_report` | `dict` | Full output of `TriageSession.get_diagnosis()` — report text, mode, pass count, chunk count |
| `retrieved_chunks` | `list[dict]` | All textbook passages retrieved and used during triage — the evidence base for the diagnosis |
| `system_primary_diagnosis` | `str` | Primary diagnosis name extracted from the report, used for analytics |
| `review_status` | `str` | Lifecycle state: `"pending"`, `"confirmed"`, or `"rejected"` |
| `doctor_decision` | `str \| null` | `"confirmed"` or `"rejected"` once reviewed; `null` until then |
| `doctor_diagnosis` | `str \| null` | The correct diagnosis provided by the doctor on rejection; `null` if confirmed or not yet reviewed |
| `doctor_notes` | `str \| null` | Optional free-text clinical notes from the doctor |
| `reviewed_at` | `str \| null` | UTC ISO-8601 timestamp of when feedback was submitted; `null` until reviewed |

The `system_primary_diagnosis` field is extracted automatically from the diagnosis report by parsing the `## Primary Diagnosis` section. This avoids scanning the full report text during analytics queries.

---

## How Cases Are Saved

Cases are saved automatically at the end of each diagnosis session, before the report is shown to the doctor. The Intake Agent CLI handles this — the doctor never needs to trigger it manually.

### The `save_case_from_session()` Convenience Function

```python
from agents.feedback_store import FeedbackStore, save_case_from_session

store = FeedbackStore()
case_id = save_case_from_session(
    feedback_store=store,
    patient_name="Karim Habbal",
    intake_summary=intake_session.get_summary(),
    clinical_picture=clinical_picture_dict,
    triage_diagnosis=triage_session.get_diagnosis(),
    retrieved_chunks=triage_session.get_retrieved_chunks(),
)
print(f"Case saved: {case_id}")
```

`save_case_from_session()` is a wrapper that extracts `symptoms` and `urgency` from the intake summary so the caller does not need to unpack them separately. It then delegates to `FeedbackStore.save_case()`.

### What Is Preserved

Every piece of information the system used to reach its diagnosis is stored:

- `intake_summary` — the full Q&A chain from the patient interview, including every question asked and every answer given, all triggered red flags, and the clinician note the Intake Agent generated
- `clinical_picture` — the parsed, distilled version of the intake summary that the Triage Agent actually reasoned from
- `diagnosis_report` — the full seven-section diagnosis text, plus metadata about which mode and pass produced it
- `retrieved_chunks` — every textbook passage retrieved across all passes of the triage session

This means a doctor reviewing a case has full visibility into the system's reasoning: not just the conclusion, but every step that led to it. A rejection can be accompanied by a note that identifies exactly which inference failed.

---

## How Doctors Submit Feedback

### Through the CLI

The primary review interface for Phase 7 is the CLI's `--review` flag:

```bash
python agents/feedback_store.py --review 20260421_karim_habbal_a1b2c3d4
```

The CLI displays the full case: patient name, symptoms, urgency, system diagnosis, and the complete diagnosis report. It then prompts for a decision, a correct diagnosis (if rejecting), and optional notes.

### Programmatically

```python
store = FeedbackStore()

# Confirm the system diagnosis was correct
store.submit_feedback(
    case_id="20260421_karim_habbal_a1b2c3d4",
    doctor_decision="confirmed",
)

# Reject with the correct diagnosis
store.submit_feedback(
    case_id="20260421_karim_habbal_a1b2c3d4",
    doctor_decision="rejected",
    doctor_diagnosis="Viral upper respiratory tract infection",
    doctor_notes="No pertussis exposure confirmed — colleague's diagnosis was not PCR-confirmed.",
)
```

### Decision Rules

| Decision | When to use | `doctor_diagnosis` required? |
|---|---|---|
| `"confirmed"` | The system's primary diagnosis was correct | No |
| `"rejected"` | The system's primary diagnosis was wrong | Yes — must provide the correct diagnosis |

`doctor_notes` is optional in both cases. It is most useful when rejecting: explaining which specific inference was wrong, or which clinical feature the system missed, produces higher-quality training data than a bare rejection.

If `doctor_decision` is `"rejected"` and `doctor_diagnosis` is empty, `submit_feedback()` raises a `ValueError`. The correct diagnosis is required — a rejection without a correction is not actionable as training data.

---

## The Training Data Export

Once a sufficient number of cases have been reviewed, `export_training_data()` produces a JSONL file ready for fine-tuning.

```bash
python agents/feedback_store.py --export
# Exported 47 reviewed case(s) to:
#   /path/to/project/data/feedback/training_data.jsonl
```

Or programmatically:

```python
output_path = store.export_training_data()
# Defaults to data/feedback/training_data.jsonl
```

### JSONL Format

Each line in the output file is a self-contained JSON object:

```json
{
  "case_id": "20260421_karim_habbal_a1b2c3d4",
  "input": {
    "clinical_picture": {
      "symptoms": ["Cough"],
      "urgency": "routine",
      "clinical_findings": {"onset": "2 weeks ago", "exposure": "pertussis contact"},
      "red_flags": []
    },
    "retrieved_chunks": [
      {
        "chunk_id": "chunk_0042",
        "chapter": "Chapter 9 — Pulmonary Disorders",
        "section": "Pertussis",
        "text": "Pertussis is characterised by a prolonged paroxysmal cough..."
      }
    ]
  },
  "expected_output": "## Primary Diagnosis\nViral upper respiratory tract infection...",
  "system_output": "## Primary Diagnosis\nPertussis (Whooping Cough) — confidence: moderate...",
  "was_correct": false,
  "patient_name": "Karim Habbal",
  "symptoms": ["Cough"],
  "urgency": "routine",
  "system_primary_diagnosis": "Pertussis (Whooping Cough)",
  "doctor_diagnosis": "Viral upper respiratory tract infection",
  "doctor_notes": "No pertussis exposure confirmed — colleague's diagnosis was not PCR-confirmed.",
  "reviewed_at": "2026-04-21T14:22:05.001234+00:00"
}
```

### Key Fields

| Field | Description |
|---|---|
| `input` | The clinical context the model received: clinical picture and retrieved chunks |
| `expected_output` | The correct diagnosis. For confirmed cases: the system's report (it was right). For rejected cases: the doctor's corrected diagnosis |
| `system_output` | What the system actually produced — the full report text |
| `was_correct` | `true` for confirmed cases, `false` for rejected cases |

### Output Logic for `expected_output`

For **confirmed** cases: `expected_output = system_output`. The system was right — we want to reinforce this output.

For **rejected** cases: `expected_output = doctor_diagnosis`. The doctor's corrected diagnosis replaces the system's output as the supervision target. If for some reason `doctor_diagnosis` is empty despite the rejection status, the system falls back to `system_output` as a safe default.

### Compatibility with Fine-Tuning Frameworks

The JSONL format is designed to work directly with standard fine-tuning toolchains. Each entry contains the full input context (what to condition on), the expected output (what to train toward), and metadata for filtering and analysis. A fine-tuning script using unsloth or the Hugging Face `transformers` `Trainer` can consume this file without preprocessing:

```python
# Example loading for fine-tuning
import json

with open("data/feedback/training_data.jsonl") as f:
    examples = [json.loads(line) for line in f]

# Filter to rejected cases only (harder examples)
hard_cases = [e for e in examples if not e["was_correct"]]
```

---

## Analytics

`get_statistics()` returns aggregate metrics across all stored cases:

```python
stats = store.get_statistics()
```

```json
{
  "total_cases": 52,
  "pending_review": 5,
  "confirmed": 38,
  "rejected": 9,
  "confirmation_rate": 0.8085,
  "most_common_corrections": [
    {"system_diagnosis": "Pulmonary Embolism", "count": 3},
    {"system_diagnosis": "Pertussis (Whooping Cough)", "count": 2},
    {"system_diagnosis": "Community-Acquired Pneumonia", "count": 1}
  ]
}
```

### Field Reference

| Field | Description |
|---|---|
| `total_cases` | All cases ever saved, regardless of review status |
| `pending_review` | Cases with `review_status == "pending"` — awaiting doctor review |
| `confirmed` | Cases where the doctor confirmed the system's diagnosis |
| `rejected` | Cases where the doctor rejected the system's diagnosis |
| `confirmation_rate` | `confirmed / (confirmed + rejected)` as a float between 0 and 1; `null` if no reviewed cases yet |
| `most_common_corrections` | Top 10 system diagnoses by rejection count — identifies systematic weaknesses |

The `most_common_corrections` list is the most actionable output. If "Pulmonary Embolism" appears with a count of 3, it means the system has produced that primary diagnosis and been rejected three times. This is a signal to inspect those cases: were the retrieved chunks for PE insufficient? Did the clinical picture parser drop a discriminating finding? Did the LLM over-index on a single feature?

```bash
python agents/feedback_store.py --stats
```

```
Feedback Statistics
========================================
  Total cases    : 52
  Pending review : 5
  Confirmed      : 38
  Rejected       : 9
  Confirmation % : 80.9%

  Most-corrected system diagnoses:
    [3x] Pulmonary Embolism
    [2x] Pertussis (Whooping Cough)
    [1x] Community-Acquired Pneumonia
```

---

## The FeedbackStore API

`FeedbackStore` is the data layer for Phase 7. It is intentionally stateless between calls: every public method reads from and writes to disk, so multiple processes sharing the same storage directory remain consistent. Thread safety is provided by per-case locks.

```python
from agents.feedback_store import FeedbackStore
store = FeedbackStore()                          # uses data/feedback/ by default
store = FeedbackStore(storage_dir="/custom/path")
```

### `save_case(patient_name, symptoms, urgency, intake_summary, clinical_picture, diagnosis_report, retrieved_chunks) -> str`

Save a completed case awaiting doctor review. Returns the `case_id` string.

```python
case_id = store.save_case(
    patient_name="Karim Habbal",
    symptoms=["Cough"],
    urgency="routine",
    intake_summary=intake_session.get_summary(),
    clinical_picture=clinical_picture_dict,
    diagnosis_report=triage_session.get_diagnosis(),
    retrieved_chunks=retrieved_chunks_list,
)
```

### `submit_feedback(case_id, doctor_decision, doctor_diagnosis="", doctor_notes="") -> dict`

Record the doctor's verdict on a case. Returns the updated case dict.

Raises `FileNotFoundError` if the case does not exist.
Raises `ValueError` if `doctor_decision` is not `"confirmed"` or `"rejected"`, or if `doctor_diagnosis` is empty when `doctor_decision` is `"rejected"`.

```python
case = store.submit_feedback(
    case_id="20260421_karim_habbal_a1b2c3d4",
    doctor_decision="rejected",
    doctor_diagnosis="Viral URTI",
    doctor_notes="No confirmed pertussis exposure.",
)
```

### `get_case(case_id) -> dict | None`

Retrieve a specific case by ID. Returns `None` if not found. Accepts either the full filename stem or a bare UUID fragment.

```python
case = store.get_case("20260421_karim_habbal_a1b2c3d4")
case = store.get_case("a1b2c3d4")  # bare UUID fragment also works
```

### `get_pending_cases() -> list[dict]`

Return all cases with `review_status == "pending"`.

### `get_reviewed_cases() -> list[dict]`

Return all cases with `review_status != "pending"` (both confirmed and rejected).

### `get_confirmed_cases() -> list[dict]`

Return all cases where the doctor confirmed the system's diagnosis.

### `get_rejected_cases() -> list[dict]`

Return all cases where the doctor rejected the system's diagnosis.

### `get_statistics() -> dict`

Return aggregate statistics. See Analytics section for field reference.

### `export_training_data(output_path=None) -> Path`

Export all reviewed cases as a JSONL file. Defaults to `data/feedback/training_data.jsonl`. Returns the path written to.

---

## CLI Reference

All CLI operations go through `agents/feedback_store.py`:

```bash
python agents/feedback_store.py [FLAG]
```

| Flag | Description |
|---|---|
| `--stats` | Display aggregate statistics: case counts, confirmation rate, most-corrected diagnoses |
| `--pending` | List all cases awaiting doctor review with one-line summaries |
| `--reviewed` | List all reviewed cases with full doctor feedback details |
| `--list` | List all cases (pending and reviewed), sorted by timestamp |
| `--review CASE_ID` | Launch an interactive review session for the specified case |
| `--export` | Export all reviewed cases to `data/feedback/training_data.jsonl` |
| `--demo` | Insert a demo case (pertussis presentation) and immediately enter interactive review — useful for end-to-end testing without running the full pipeline |

### `--review` Interactive Session

```
============================================================
Case     : 20260421_karim_habbal_a1b2c3d4
Patient  : Karim Habbal
Symptoms : Cough
Urgency  : ROUTINE
System Dx: Pertussis (Whooping Cough)
============================================================

## Primary Diagnosis
Pertussis (Whooping Cough) — confidence: moderate
...

============================================================
Doctor decision (confirm/reject): reject
Correct diagnosis: Viral upper respiratory tract infection
Notes (optional, press Enter to skip): No confirmed pertussis exposure.

Feedback saved. (case 20260421_karim_habbal_a1b2c3d4 → rejected)
```

If a case has already been reviewed, the CLI asks whether to re-review before proceeding. Re-reviewing overwrites the prior feedback.

### `--demo` Flag

The `--demo` flag is the fastest way to verify the system end-to-end without running a full intake and triage session:

```bash
python agents/feedback_store.py --demo
# Saves a synthetic pertussis case and immediately launches interactive review
```

The demo case uses a 2-week dry cough with pertussis exposure — a realistic presentation that exercises the full review flow.

---

## Integration with the Pipeline

Phase 7 is wired into the Intake Agent CLI (`agents/intake_agent.py`). After the Triage Agent generates a diagnosis, the case is automatically saved:

```python
# Inside intake_agent.py — after triage completes
from agents.feedback_store import FeedbackStore, save_case_from_session

store = FeedbackStore()
case_id = save_case_from_session(
    feedback_store=store,
    patient_name=patient_name,
    intake_summary=intake_session.get_summary(),
    clinical_picture=clinical_picture,
    triage_diagnosis=triage_session.get_diagnosis(),
    retrieved_chunks=retrieved_chunks,
)
print(f"\nCase saved for doctor review: {case_id}")
print("Review with: python agents/feedback_store.py --review", case_id)
```

From the doctor's perspective, the workflow is:

1. A patient session runs through `python agents/intake_agent.py`
2. The diagnosis report is printed to the terminal
3. A `case_id` is printed below the report
4. The doctor reviews the diagnosis at any time later: `python agents/feedback_store.py --review {case_id}`
5. Periodically, accumulated feedback is exported: `python agents/feedback_store.py --export`

No manual file management is required. Cases accumulate automatically in `data/feedback/` and are picked up by the CLI's list and export commands.

---

## Path to Fine-Tuning (Future Work)

The fine-tuning pipeline is not yet built. The steps below describe what becomes possible once a sufficient number of reviewed cases has accumulated.

### Step 1: Accumulate Reviewed Cases

The minimum viable training set is approximately 50–100 reviewed cases, with meaningful representation of both confirmed and rejected diagnoses. Rejected cases are more valuable per example — they carry a correct supervision signal. Confirmed cases reinforce what is already working.

Quality matters more than quantity. 50 cases reviewed by senior clinicians with detailed notes will produce a better fine-tuned model than 500 cases confirmed with no clinical reasoning.

### Step 2: Export Training Data

```bash
python agents/feedback_store.py --export
# Produces: data/feedback/training_data.jsonl
```

Inspect the export before training: verify that `expected_output` fields for rejected cases contain the doctor's corrected diagnosis, not the system's original output. Check that `was_correct` is correctly set for all entries.

### Step 3: Fine-Tune with LoRA/QLoRA

The recommended approach is parameter-efficient fine-tuning using LoRA or QLoRA on Llama 3.1 8B. The base model is the LLM that currently produces the diagnosis reports (or its equivalent). The fine-tuning script is approximately 100–150 lines using unsloth:

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
import json

# Load the exported training data
dataset = load_dataset("json", data_files="data/feedback/training_data.jsonl")

# Format each example as an instruction-following pair
def format_example(example):
    input_text = json.dumps(example["input"], indent=2)
    return {
        "instruction": "Given the clinical picture and retrieved textbook passages, produce a diagnosis report.",
        "input": input_text,
        "output": example["expected_output"],
    }

dataset = dataset.map(format_example)

# Load model with 4-bit quantisation
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
)

# Train and save
```

The full training script is a future deliverable. It is intentionally not included here because the bottleneck is not the code — it is accumulating the reviewed cases that make the training meaningful.

### Step 4: Evaluate Against Held-Out Cases

Before deploying the fine-tuned model, evaluate it against a held-out set of reviewed cases. Key metrics:

- Accuracy on rejected cases: does the fine-tuned model produce the doctor's correct diagnosis on cases structurally similar to the training rejections?
- Confirmation rate on held-out confirmed cases: does fine-tuning on rejected cases degrade performance on presentations the base model was already getting right?
- Primary diagnosis match rate: for cases where `was_correct=true`, does the fine-tuned model still produce the same primary diagnosis?

### Step 5: Deploy Alongside Base Model

The fine-tuned model should initially run alongside the base model, not replace it. A/B testing on new cases — routing some to the base model and some to the fine-tuned model — allows the confirmation rate difference to be measured in production before a full switch.

---

## Limitations

### No Authentication

Any caller with access to the `FeedbackStore` instance can submit feedback. There is no concept of a logged-in doctor, no identity associated with a review decision, and no audit trail beyond the `reviewed_at` timestamp. In a production deployment, feedback submission would need to be gated behind clinician authentication and the reviewer's identity recorded in the case schema.

### JSON Files Do Not Scale

Each case is one JSON file on the local filesystem. This works for tens or hundreds of cases in a research setting. For a production system handling thousands of cases across multiple hospitals, a database is required. The storage backend should be replaced with PostgreSQL or a document store before deployment at scale.

### No Conflict Resolution

If two doctors review the same case and submit different decisions, the second submission overwrites the first. There is no mechanism to record the disagreement, flag the case for adjudication, or track inter-rater reliability. For clinical AI systems, inter-rater agreement is an important quality signal — cases where two clinicians disagree are often more instructive than cases with consensus.

### Training Data Quality Depends on Doctor Expertise

The model will learn to produce whatever diagnoses doctors confirm and reject. If a doctor confirms an incorrect diagnosis, or rejects a correct one, that error propagates into the training data. There is no validation layer between doctor feedback and the training export. The `doctor_notes` field exists precisely to make feedback auditable — a second reviewer can read a rejection note and assess whether it reflects sound clinical reasoning — but this review is not enforced.

### No Automated Retraining Trigger

The export and fine-tuning process is entirely manual. There is no trigger that fires when a threshold number of new reviewed cases accumulates, no automated evaluation pipeline, and no deployment mechanism. Each cycle — export, train, evaluate, deploy — requires human intervention. Automating this loop is a future infrastructure task.
