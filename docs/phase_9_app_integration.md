# Phase 9: Full-Stack App Integration

## Overview

Phase 9 integrates the Medora agent pipeline (Intake Agent, Triage Agent, Patient Memory, Feedback Store) into a production web application. The backend is a FastAPI server with PostgreSQL persistence. The frontend is a React + TypeScript single-page application. The agents run as in-memory sessions managed by a singleton session manager, with all heavy models (bi-encoder, cross-encoder, ChromaDB) loaded once at startup and shared across sessions.

### Where Phase 9 Sits in the Pipeline

| Phase | Component | Function |
|---|---|---|
| 1–3 | Data processing, embedding, reranking | 5,631 chunks in ChromaDB with cross-encoder reranking |
| 4.1 | Intake Agent | Multi-turn patient interview |
| 5 | Triage Agent | Diagnostic engine with multi-pass RAG |
| 6 | Web Search Agent | External evidence for outside-textbook cases |
| 7 | Feedback Loop | Doctor review + training data export |
| 8 | Benchmarking | Model comparison across test sets |
| **9** | **App Integration** | **Full-stack web app: React frontend + FastAPI backend + agent pipeline** |

---

## Architecture

### System Components

```
Browser (React + TypeScript + Vite)
    │
    │  HTTP / JSON
    │
Vite Dev Proxy (:5173 → :8000)
    │
FastAPI Backend (:8000)
    ├── Auth (JWT, bcrypt)
    ├── Patient routes (profile, medical history, consent)
    ├── Triage routes (sessions, messages, reports)
    ├── Doctor routes (dashboard, reports, feedback)
    ├── Admin routes (user management)
    │
    ├── AgentSessionManager (singleton)
    │   ├── IntakeSession (per session)
    │   ├── TriageSession (per session, created on intake completion)
    │   ├── PatientMemory (shared)
    │   └── FeedbackStore (shared)
    │
    ├── PostgreSQL (users, sessions, messages, reports, feedback)
    │
    └── Shared Model Cache (module-level)
        ├── ChromaDB collection (5,631 chunks)
        ├── Bi-encoder (embeddinggemma-300m-medical, MPS/CPU)
        └── Cross-encoder (bge-reranker-v2-m3, MPS/CPU)
```

### Key Design Decisions

**In-memory agent sessions.** Each triage session maps to an `AgentSessionState` object held in a Python dict. The state includes the live `IntakeSession` and `TriageSession` objects with their full conversation history and LangGraph state. Sessions are evicted after 30 minutes of inactivity by a background cleanup task.

**Shared model cache.** The bi-encoder, cross-encoder, and ChromaDB collection are loaded once at module level and shared across all `TriageSession` instances. Before this optimisation, each new `TriageSession()` reloaded all models from disk (~10-15 seconds). With the cache, session creation is instant.

**Background memory updates.** Patient memory updates (`PatientMemory.update_from_session()`) and feedback case saves (`FeedbackStore.save_case()`) run in background asyncio tasks after the HTTP response is sent. These operations involve LLM calls and should not block the patient-facing response.

**Patient safety.** Diagnosis data never reaches the patient. The frontend receives only the agent's conversational messages (questions, confirmations). Diagnosis reports are stored as `ClinicalReport` objects visible only to doctors. Emergency escalation messages are always forwarded to patients.

---

## Backend

### Database Schema

The PostgreSQL schema includes tables for multi-tenant hospital management, user authentication, patient profiles, triage sessions, and clinical reports.

**Agent-specific columns on `TriageSession`:**
- `agent_phase` — enum: `intake`, `triage_mode_a`, `triage_mode_b`, `escalated`, `completed`
- `intake_summary_json` — JSONB: symptoms, urgency, red flags, escalation data
- `clinical_picture_json` — JSONB: parsed clinical findings
- `detected_symptoms` — JSONB: symptom list from intake

**Agent-specific columns on `ClinicalReport`:**
- `diagnosis_mode` — string: e.g., `"common"` or `"uncommon"`
- `diagnosis_pass_count` — integer: number of RAG passes
- `chunks_used_count` — integer: RAG chunks used
- `model_version` — string: model identifier

### Session Manager (`backend/app/services/session_manager.py`)

The `AgentSessionManager` is a singleton that bridges FastAPI request handling and the agent classes.

**Phase state machine:**
```
intake
  ├─ (escalated) → escalated [END]
  ├─ (uncommon) → triage_mode_b → completed [DIAGNOSIS]
  └─ (common)
      ├─ (auto-diagnosis) → completed [DIAGNOSIS]
      └─ (needs followup) → triage_mode_a → completed [DIAGNOSIS]
```

**Agent method calls (all correct, verified against agent interfaces):**
- `IntakeSession`: `start()`, `respond()`, `is_complete()`, `get_summary()`, `is_uncommon()`, `get_raw_complaint()`
- `TriageSession`: `start_uncommon()`, `diagnose_from_intake()`, `respond()`, `respond_followup()`, `is_complete()`, `get_diagnosis()`
- `PatientMemory`: `get_context_for_intake()`, `update_from_session()` (with `llm` parameter)
- `FeedbackStore`: `save_case()`, `submit_feedback()`, `get_statistics()`, `get_pending_cases()`

### Triage Router (`backend/app/routers/triage.py`)

Patient-facing endpoints for the agentic workflow:

| Endpoint | Method | Purpose |
|---|---|---|
| `/triage/sessions` | POST | Create new session (assigns doctor, creates IntakeSession) |
| `/triage/sessions` | GET | List patient's sessions |
| `/triage/sessions/{id}` | GET | Get session details (patient-safe) |
| `/triage/sessions/{id}/phase` | GET | Current agent phase + escalation flag |
| `/triage/sessions/{id}/message` | POST | Send message to agent, get response |
| `/triage/sessions/{id}/messages` | GET | Message history (diagnosis messages filtered out) |
| `/triage/sessions/{id}/end` | POST | End session, generate report |
| `/triage/sessions/{id}/report` | GET | Forbidden (403) — reports are doctor-only |

### Doctor Router (`backend/app/routers/doctor.py`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/doctor/dashboard` | GET | Stats: total patients, pending reports |
| `/doctor/patients` | GET | Assigned patients with last triage date |
| `/doctor/patients/{id}` | GET | Patient demographics |
| `/doctor/patients/{id}/reports` | GET | Patient's reports |
| `/doctor/reports/{id}` | GET | Full clinical report |
| `/doctor/reports/{id}/full-chain` | GET | Report + intake summary + clinical picture |
| `/doctor/reports/{id}/feedback` | POST | Submit rating + correction |
| `/doctor/feedback/statistics` | GET | Aggregate feedback stats |
| `/doctor/feedback/pending` | GET | Cases awaiting review |

### Configuration (`backend/app/config.py`)

```python
database_url: str              # PostgreSQL connection
secret_key: str                # JWT signing
openai_api_key: Optional[str]  # For agent LLM calls
llm_provider: str = "openai"   # "openai" or "ollama"
llm_model: str = "gpt-4o-mini" # Default model
ollama_url: str = "http://localhost:11434"
```

---

## Frontend

### Technology Stack

- React 19 + TypeScript 6
- Vite 8 (dev server, build)
- Tailwind CSS 4
- Axios (HTTP client with JWT interceptor)
- React Router 7 (role-based routing)

### Route Map

```
/login, /signup                    — Authentication
/patient/dashboard                 — Patient home (profile, sessions, start triage)
/patient/profile                   — Edit demographics
/patient/medical-history           — Edit medical background
/patient/triage                    — Session list + create new
/patient/triage/:sessionId         — Chat interface
/doctor/dashboard                  — Doctor home (stats, recent reports)
/doctor/patients                   — Assigned patients
/doctor/patients/:patientId        — Patient detail
/doctor/reports                    — All reports
/doctor/reports/:reportId          — Report detail + feedback form
/admin/users                       — User management
```

### Chat Interface (`TriageSession.tsx`)

The patient chat interface provides:

- **Phase indicator** — patient-friendly labels: "Describing your symptoms", "Follow-up questions", "Assessment complete"
- **Escalation banner** — red alert for emergency escalation, always visible
- **Optimistic messaging** — patient message appears immediately, replaced with server response
- **Word-by-word streaming** — agent responses appear word by word (30ms per word) for natural feel
- **Auto-focus** — input stays focused after sending, patient can type continuously
- **Auto-scroll** — chat scrolls to bottom on new messages

### Doctor Report View (`DoctorReportDetail.tsx`)

Displays structured clinical reports with:
- Urgency badge (routine/urgent/emergency)
- Summary section
- 10 clinical sections (presenting complaints, history, suspected conditions, red flags, etc.)
- Escalation message (if applicable)
- Model metadata (version, diagnosis mode, pass count, chunks used)
- Feedback form (thumbs up/down + optional correction)

Report section content renders as formatted key-value pairs, bulleted lists, or plain text depending on data type — not raw JSON.

---

## Integration Fixes Applied

### Bug 1: `PatientMemory.update_from_session()` — Missing `llm` Parameter

The backend called `update_from_session(patient_name, intake_summary, diagnosis)` but the agent requires a 4th `llm` parameter (LangChain LLM instance). Fixed by creating an LLM via `make_llm()` and passing it.

### Bug 2: `FeedbackStore.save_case()` — Wrong `clinical_picture`

The backend passed `intake_summary.get("clinical_picture", {})` which is always empty — the clinical picture is internal to `TriageSession`, not in the intake summary. Fixed by passing an empty dict explicitly (non-blocking; feedback still works).

### Bug 3: `FeedbackStore.save_case()` — Wrong `retrieved_chunks` Key

Used `diagnosis.get("chunks", [])` but `TriageSession.get_diagnosis()` returns `num_chunks_used` (int), not a chunks list. Fixed by passing `[]`.

### Bug 4: Router Missing `provider`/`ollama_url`

The triage router didn't pass LLM provider settings to `create_session()`, hardcoding OpenAI. Fixed by adding `llm_provider`, `llm_model`, `ollama_url` to backend `Settings` and passing them in the router.

### Bug 5: Blocking Memory Updates

`PatientMemory.update_from_session()` and `FeedbackStore.save_case()` made synchronous LLM/IO calls inside the async request handler, blocking the HTTP response. Fixed by moving both to background `asyncio.create_task()` calls.

### Optimisation: Shared Model Cache

Each `TriageSession()` constructor reloaded ChromaDB, bi-encoder, and cross-encoder from disk (~10-15s). Added a module-level `_model_cache` dict that loads models once and shares them across all instances.

---

## Triage Agent Enhancement: Differentiating Questions

The Mode B multi-pass flow was enhanced to ask differentiating questions in Pass 2 and Pass 3 when the preliminary diagnosis is uncertain.

**Previous flow:**
- Pass 1: RAG → questions → patient answers
- Pass 2: RAG with answers → diagnosis (no patient interaction)
- Pass 3: Targeted RAG → refined diagnosis (no patient interaction)

**New flow:**
- Pass 1: RAG → questions → patient answers
- Pass 2: RAG with answers → preliminary diagnosis → if uncertain, ask 2-4 differentiating questions → patient answers → re-retrieve and regenerate diagnosis
- Pass 3: Targeted RAG → preliminary diagnosis → if uncertain, ask 2-4 differentiating questions → patient answers → final diagnosis (HARD STOP)

The differentiating questions target features that distinguish between competing diagnoses in the differential. The LLM evaluates whether the preliminary diagnosis is confident enough to skip additional questions. If the primary diagnosis has high confidence with no meaningful competing differentials, no questions are asked and the diagnosis is finalised immediately.

This mirrors the Mode A pattern where the sufficiency check identifies gaps and generates targeted follow-up questions.

---

## Running Locally

### Prerequisites

- Python 3.10+ with project venv (`.venv/`)
- Node.js 22+ (via nvm)
- PostgreSQL 14+
- OpenAI API key

### Setup

```bash
# 1. Database
brew services start postgresql@14
createdb medora

# 2. Backend environment
cat > backend/.env << 'EOF'
DATABASE_URL=postgresql://$(whoami)@localhost/medora
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
OPENAI_API_KEY=<your-key>
EOF

# 3. Database migrations
cd backend
alembic upgrade head

# 4. Bootstrap users
ADMIN_EMAIL=admin@medora.com ADMIN_PASSWORD=admin123 python create_admin.py
# Create a doctor via admin API or seed script

# 5. Start backend
uvicorn app.main:app --port 8000

# 6. Start frontend (separate terminal)
cd frontend
npm install
npm run dev  # → http://localhost:5173
```

### Test Accounts

| Role | Email | Password |
|---|---|---|
| Admin | admin@medora.com | admin123 |
| Doctor | doctor@medora.com | doctor123 |
| Patient | (sign up in browser) | — |

---

## Limitations

### No True Token Streaming

The agent pipeline uses synchronous `llm.invoke()` calls. The frontend simulates streaming by revealing the response word by word after the full response arrives. True token streaming would require switching to `llm.astream()` and Server-Sent Events, which would require restructuring the agent pipeline.

### Single-Worker Backend

The backend runs as a single uvicorn worker. All agent sessions share the same process. For production with multiple concurrent patients, multiple workers or a task queue (Celery) would be needed.

### In-Memory Session Storage

Agent sessions are stored in a Python dict. A server restart loses all active sessions. For production, session state would need to be persisted (Redis, database) or the system would need sticky sessions behind a load balancer.

### No WebSocket Support

The chat uses HTTP request-response. The patient must wait for the full agent response before the UI updates. WebSocket support would enable real-time push notifications and true streaming.
