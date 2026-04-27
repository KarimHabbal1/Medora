# Phase 8: Backend API and Database Implementation

## Overview

Phase 8 implements the persistent backend infrastructure that binds together every prior phase of the Medora project. Previous phases produced discrete artefacts: structured symptom objects (Phase 1.3), a ChromaDB vector index over 5,631 textbook chunks (Phase 2.2), a validated retrieval-and-reranking pipeline (Phase 3), and a clinical report schema defined collaboratively with the team. Phase 8 takes those artefacts and places them inside a running system — one with a relational database, authenticated HTTP routes, and a well-defined data model — so that a patient can register, submit symptoms, receive a triage interaction, and have a structured clinical report stored and made available to the responsible physician.

### Why This Phase Exists

The retrieval and reasoning work done in Phases 1–3 produces intelligence, not infrastructure. A cross-encoder that correctly identifies the most relevant textbook passage for a clinical query is valuable only if there is a system that can accept a patient's message, invoke that retrieval pipeline, persist the exchange, generate a structured report, and return it to the right people with the right access controls. Without a backend, the RAG pipeline exists only as a set of Python scripts that can be run offline; it cannot serve any patient or clinician.

Phase 8 provides that infrastructure. It is not the final production system — the AI and RAG logic is stubbed with mock services at this stage — but it is the complete, correct structural foundation: every table, every route, every role check, and every field in the clinical report schema is implemented and validated. When the RAG integration is wired in, it slots into service functions that already exist, with database columns that already accept its output, and API routes that already return it in the correct shape.

### Connection to Previous Phases

| Prior Phase | Artefact | How It Connects to Phase 8 |
|---|---|---|
| Phase 1.3 | Structured symptom JSON (11 symptoms) | Symptom tables (`symptoms`, `symptom_questions`, `symptom_red_flags`, `symptom_urgency_rules`, `symptom_workup_items`) directly mirror this schema in PostgreSQL |
| Phase 2.2 | ChromaDB vector index (`tmt_chunks` collection) | `rag_queries` and `rag_retrieved_chunks` tables record every retrieval call; `chunk_id` links back to ChromaDB without duplicating textbook content |
| Phase 3 | Validated retrieve-and-rerank pipeline | `rag_retrieved_chunks` stores `original_rank`, `final_rank`, `vector_distance`, and `rerank_score` to enable full traceability of reranking decisions |
| Clinical report schema | Agreed field list from team review | `clinical_reports` table implements every agreed field; the `ClinicalReportResponse` Pydantic schema enforces the contract at the API boundary |

---

## Backend Architecture

The backend is a Python application structured around four components: the FastAPI web framework, a PostgreSQL relational database accessed through SQLAlchemy and managed through Alembic, a ChromaDB vector store that remains external to the relational layer, and a JWT-based authentication system.

### FastAPI

FastAPI is the HTTP framework. It handles request routing, automatic OpenAPI documentation generation, dependency injection (for database sessions and authentication), and request/response validation through Pydantic. The application is defined in `backend/app/main.py`, which mounts six router modules: `auth`, `patient`, `triage`, `doctor`, `admin`, and `system`.

FastAPI was selected over Flask and Django for three reasons. First, its native Pydantic integration means that every request body and response model is type-checked and validated without additional serialisation code. Second, its dependency injection system (`Depends`) allows authentication guards and database sessions to be composed cleanly across routes without repetition. Third, its automatic OpenAPI generation produces interactive API documentation at `/docs` without any additional tooling.

### PostgreSQL

PostgreSQL serves as the relational application database. It stores all user accounts, patient profiles, session records, clinical reports, feedback, and audit logs. PostgreSQL was chosen over SQLite because the project requires production-grade JSONB column support (used for structured clinical report fields, symptom metadata, and detected symptoms), native UUID generation via the `pgcrypto` extension, and proper enum types that are enforced at the database level.

The application does not store full textbook chunk text in PostgreSQL. Only retrieval metadata — chunk IDs, rank positions, similarity scores — is stored, with a foreign-key-equivalent reference back to the ChromaDB entry by `chunk_id`.

### ChromaDB

ChromaDB remains a separate, independent component. It is not part of the PostgreSQL schema. Its sole responsibility is vector search over the 5,631 textbook chunks embedded in Phase 2.2. The backend service layer will call into ChromaDB when executing real retrieval queries; at present, those calls are mocked. The database schema provides the `rag_queries` and `rag_retrieved_chunks` tables to record what was retrieved and how it was ranked, but the vector index itself lives in `data/chroma/` and is managed entirely by ChromaDB's own persistence layer.

### SQLAlchemy ORM

SQLAlchemy 2.0 is used as the ORM layer. All database tables are defined as Python classes inheriting from `Base` in `backend/app/models/models.py`. Column types, constraints, default values, and relationships are declared in Python and reflected in the Alembic migration. The ORM is used in all route handlers through a `Session` dependency injected by FastAPI's `Depends(get_db)` mechanism.

### Alembic

Database schema migrations are managed by Alembic. A single initial migration (`alembic/versions/001_initial.py`) creates the full schema in the correct dependency order: PostgreSQL enum types first, then the `pgcrypto` extension, then tables from independent to dependent. The migration is written by hand rather than auto-generated to ensure that the enum types, JSONB columns, and foreign key constraints are created precisely as intended.

### JWT Authentication

Authentication uses JSON Web Tokens. On successful sign-in, the server issues an access token (30-minute expiry) and a refresh token (7-day expiry). All protected routes require a valid Bearer token in the `Authorization` header. Token verification, password hashing (bcrypt via `passlib`), and user lookup are handled in `backend/app/auth/jwt.py` and `backend/app/auth/dependencies.py`.

---

## Design Principle: Two Separate Data Stores

The most important architectural decision in Phase 8 is the deliberate separation between PostgreSQL and ChromaDB. These two stores serve different purposes and must not be conflated.

**PostgreSQL stores operational data.** Every user account, every patient profile, every triage session, every message, every clinical report, every piece of doctor feedback, and every audit log lives in PostgreSQL. This is the data that drives the application: it is queried to answer questions like "which sessions belong to this patient?", "what is the urgency level of this report?", and "has this doctor already submitted feedback on this report?" PostgreSQL handles this data well because it is relational — entities reference each other through foreign keys — and because it benefits from ACID guarantees and structured query capabilities.

**ChromaDB stores vector search data.** The 5,631 textbook chunks, their 768-dimensional embeddings, and the HNSW index over those embeddings live in ChromaDB. This data is static from the application's perspective: it is built once (in Phase 2.2), and the application only reads from it at query time. ChromaDB handles this data well because approximate nearest-neighbour search over high-dimensional float vectors is its core function.

**PostgreSQL does not duplicate textbook content.** When the intake agent or RAG agent retrieves a chunk from ChromaDB, the chunk text is not written into any PostgreSQL column. Instead, the `rag_retrieved_chunks` table records the `chunk_id` string, the `original_rank` and `final_rank` integers, the `vector_distance` float, and the `rerank_score` float. This is sufficient for full traceability of every retrieval decision — including audit, re-evaluation, and retraining — without bloating the relational database with text that already exists in ChromaDB.

This separation keeps each store at its natural scale and prevents either from being used for a purpose it was not designed for.

---

## Database Schema Implementation

The PostgreSQL schema consists of nineteen tables organised into five logical groups: hospital and user management, patient clinical data, triage session and messaging, clinical knowledge (symptoms), and system observability. All primary keys are UUIDs generated by `gen_random_uuid()` from the `pgcrypto` extension. All timestamps default to `now()` at the database level.

### Hospital and User Management

#### `hospitals`

Stores the healthcare institution that owns a set of users. Every user account, doctor profile, patient profile, triage session, and audit log is associated with a hospital, enabling multi-tenancy in future deployments.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | `gen_random_uuid()` |
| `name` | VARCHAR | Required |
| `address` | TEXT | Optional |
| `contact_email` | VARCHAR | Optional |
| `local_server_identifier` | VARCHAR | Reserved for on-premise deployment identification |
| `created_at` | TIMESTAMP | `now()` |

#### `users`

The central identity table. Every person in the system has exactly one row in `users`. Roles are stored as a PostgreSQL enum (`userrole`).

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `hospital_id` | UUID FK → `hospitals` | Required |
| `full_name` | VARCHAR | Required |
| `email` | VARCHAR UNIQUE | Required |
| `phone` | VARCHAR | Optional |
| `password_hash` | TEXT | bcrypt hash; plaintext password is never stored |
| `role` | `userrole` enum | `patient`, `doctor`, or `admin` |
| `is_active` | BOOLEAN | `true` by default |
| `registration_method` | `registrationmethod` enum | `self_signup` or `admin_created` |
| `created_by_admin_id` | UUID FK → `users` (self) | Null for self-registered users |
| `last_login_at` | TIMESTAMP | Updated on each successful authentication |

#### `doctor_profiles`

One-to-one extension of `users` for accounts with `role = doctor`.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `user_id` | UUID FK → `users` UNIQUE | One profile per user |
| `hospital_id` | UUID FK → `hospitals` | |
| `specialty` | VARCHAR | Required; defaults to `"General"` when created by admin |
| `license_number` | VARCHAR | Optional |
| `department` | VARCHAR | Optional |

#### `patient_profiles`

One-to-one extension of `users` for accounts with `role = patient`.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `user_id` | UUID FK → `users` UNIQUE | |
| `hospital_id` | UUID FK → `hospitals` | |
| `assigned_doctor_id` | UUID FK → `doctor_profiles` | Null until assigned by hospital admin workflow |
| `date_of_birth` | DATE | Optional |
| `sex` | VARCHAR | Optional |
| `height_cm` | NUMERIC | Optional |
| `weight_kg` | NUMERIC | Optional |

### Patient Clinical Data

#### `patient_medical_history`

One-to-one with `patient_profiles`. Stores the patient's pre-existing health context as JSONB, enabling flexible list structures without requiring a separate row per item.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `patient_id` | UUID FK → `patient_profiles` UNIQUE | |
| `chronic_conditions` | JSONB | |
| `medications` | JSONB | |
| `allergies` | JSONB | |
| `surgeries` | JSONB | |
| `family_history` | JSONB | |
| `smoking_status` | VARCHAR | |
| `pregnancy_status` | VARCHAR | |
| `additional_notes` | TEXT | |
| `skipped` | BOOLEAN | `true` if the patient chose to skip this step |
| `updated_at` | TIMESTAMP | Updates on every write |

#### `patient_consents`

One patient may grant multiple consent types at different times. Each row records a single consent event. The `consenttype` enum values are: `medical_disclaimer`, `data_storage`, `ai_assistance`, and `chat_history_storage`.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `patient_id` | UUID FK → `patient_profiles` | |
| `consent_type` | `consenttype` enum | |
| `accepted` | BOOLEAN | |
| `accepted_at` | TIMESTAMP | |

### Triage Session and Messaging

#### `triage_sessions`

The central record for a single patient interaction. A session is created when the patient submits a chief complaint and remains `active` until the patient ends it, at which point the status transitions to `completed` and a clinical report is generated automatically.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `patient_id` | UUID FK → `patient_profiles` | |
| `doctor_id` | UUID FK → `doctor_profiles` | Uses assigned doctor, or first available |
| `hospital_id` | UUID FK → `hospitals` | |
| `status` | `triagesessionstatus` enum | `active`, `completed`, or `cancelled` |
| `chief_complaint` | TEXT | Free-text entry from the patient |
| `detected_symptoms` | JSONB | Populated by intake agent as symptoms are identified |
| `urgency_level` | `urgencylevel` enum | `routine`, `urgent`, `emergency`, or `unknown` |
| `escalation_type` | `escalationtype` enum | `none`, `emergency_call`, or `complex_diagnosis_agent` |
| `chat_retention_policy` | `chatretentionpolicy` enum | `keep_full_history` or `summary_only` |
| `started_at` | TIMESTAMP | |
| `ended_at` | TIMESTAMP | Null while active |

#### `session_messages`

Every message in a triage session is stored as a row here, regardless of sender. Messages are retained according to the session's `chat_retention_policy`.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `session_id` | UUID FK → `triage_sessions` | |
| `sender` | `messagesender` enum | `patient`, `intake_agent`, `rag_agent`, or `system` |
| `content` | TEXT | |
| `message_type` | `messagetype` enum | `text`, `question`, `answer`, `warning`, `summary`, or `stream_delta` |
| `is_persisted_after_summary` | BOOLEAN | |
| `is_visible_to_doctor` | BOOLEAN | |
| `is_deleted` | BOOLEAN | Soft-delete flag |

#### `clinical_reports`

One report per session (UNIQUE constraint on `session_id`). Generated on session end. `visible_to_patient` defaults to `false`; the responsible physician must explicitly release it before the patient can access it.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `session_id` | UUID FK → `triage_sessions` UNIQUE | |
| `patient_id` | UUID FK → `patient_profiles` | |
| `doctor_id` | UUID FK → `doctor_profiles` | |
| `presenting_complaints` | JSONB | |
| `history_of_presenting_complaint` | JSONB | |
| `summary_text` | TEXT | Required |
| `suspected_conditions` | JSONB | |
| `triggered_red_flags` | JSONB | |
| `urgency_level` | `urgencylevel` enum | Required |
| `recommended_action` | TEXT | Required |
| `specialty_routing` | JSONB | |
| `suggested_workup` | JSONB | |
| `key_exam_findings` | JSONB | |
| `admission_criteria` | JSONB | |
| `referral_criteria` | JSONB | |
| `external_escalation_completed` | BOOLEAN | `false` by default |
| `escalation_message` | TEXT | |
| `visible_to_patient` | BOOLEAN | `false` by default |
| `model_version` | VARCHAR | AI model version used |
| `generated_at` | TIMESTAMP | |

#### `doctor_feedback`

Doctors may submit structured feedback on any assigned report. This is the primary mechanism for capturing clinician corrections to AI-generated reports, supporting future evaluation and retraining.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `report_id` | UUID FK → `clinical_reports` | |
| `doctor_id` | UUID FK → `doctor_profiles` | One row per doctor per report |
| `rating` | `doctorfeedbackrating` enum | `thumbs_up` or `thumbs_down` |
| `correction_text` | TEXT | Optional free-text correction |
| `feedback_category` | `feedbackcategory` enum | `wrong_urgency`, `wrong_diagnosis`, `missing_info`, `unsafe_response`, `irrelevant_sources`, or `other` |

### RAG Traceability

#### `rag_queries`

Records each call to the retrieval pipeline during a session. Enables post-hoc analysis of what was queried, which models were used, and how many candidates were requested.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `session_id` | UUID FK → `triage_sessions` | |
| `query_text` | TEXT | |
| `retrieve_k` | INTEGER | Candidates requested from bi-encoder; default 10 |
| `final_k` | INTEGER | Results returned after reranking; default 3 |
| `embedding_model` | VARCHAR | |
| `reranker_model` | VARCHAR | |

#### `rag_retrieved_chunks`

Records each chunk returned by the retrieval pipeline. The `chunk_id` is the ChromaDB document ID — no chunk text is duplicated here.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `rag_query_id` | UUID FK → `rag_queries` | |
| `chunk_id` | VARCHAR | ChromaDB document ID (e.g., `ch05_chest_pain_003`) |
| `original_rank` | INTEGER | Rank from bi-encoder before reranking |
| `final_rank` | INTEGER | Rank after reranking |
| `vector_distance` | FLOAT | Cosine distance from bi-encoder |
| `rerank_score` | FLOAT | Cross-encoder relevance score |
| `chapter` | VARCHAR | |
| `section` | VARCHAR | |
| `subsection` | VARCHAR | |
| `used_in_final_answer` | BOOLEAN | Whether this chunk was passed to the LLM |

### Clinical Knowledge (Symptoms)

These five tables mirror the structured symptom schema produced in Phase 1.3 and store the clinical knowledge base in PostgreSQL for use by the intake agent.

- **`symptoms`** — One row per recognised symptom. `body_systems` is JSONB.
- **`symptom_questions`** — Intake questions per symptom, ordered by `order_index`.
- **`symptom_red_flags`** — Red-flag criteria with `flag`, `implication`, and `urgency` level.
- **`symptom_urgency_rules`** — Rule-based urgency criteria with associated action recommendations.
- **`symptom_workup_items`** — Recommended investigations per symptom.

### System Observability

- **`audit_logs`** — Records significant application events for security and compliance. `user_id` is nullable to support events that occur before or outside authentication.
- **`performance_logs`** — Records latency metrics per RAG pipeline operation: `retrieval_time_ms`, `rerank_time_ms`, `llm_time_ms`, and `total_time_ms`.

---

## API Routes

The backend exposes six route groups. All routes except `/health` and `/health/db` require a valid Bearer JWT token. Routes that are not explicitly listed here do not exist in the current implementation.

### Authentication (`/auth`)

| Method | Path | Description |
|---|---|---|
| POST | `/auth/signup` | Self-registration for patients. Creates a `users` row with `role = patient` and a corresponding `patient_profiles` row. A default hospital is created automatically for demonstration purposes. |
| POST | `/auth/signin` | Authenticates using OAuth2 password form. Returns an access token and a refresh token. |
| GET | `/auth/me` | Returns the authenticated user's profile. Requires a valid access token. |
| POST | `/auth/refresh-token` | Placeholder — returns HTTP 501 Not Implemented. |
| POST | `/auth/logout` | Placeholder — returns a success message. No server-side token invalidation is implemented. |
| PATCH | `/auth/change-password` | Verifies the old password and updates `password_hash`. |

### Patient (`/patients`)

All routes in this group require `role = patient`. Doctors and admins cannot call these routes.

| Method | Path | Description |
|---|---|---|
| GET | `/patients/me` | Returns the authenticated patient's profile row from `patient_profiles`. |
| PATCH | `/patients/me` | Updates demographic fields (`date_of_birth`, `sex`, `height_cm`, `weight_kg`). |
| GET | `/patients/me/medical-history` | Returns the patient's `patient_medical_history` row, creating an empty record if none exists. |
| PUT | `/patients/me/medical-history` | Writes or updates the patient's medical history. Accepts all JSONB fields. |
| GET | `/patients/me/consents` | Returns all consent records for the authenticated patient. |
| POST | `/patients/me/consents` | Grants a consent type. Returns HTTP 400 if the same consent type has already been granted. |

### Triage (`/triage`)

All routes in this group require `role = patient`.

| Method | Path | Description |
|---|---|---|
| POST | `/triage/sessions` | Creates a new triage session with `status = active`. Assigns the patient's designated doctor, or the first available doctor if none is assigned. |
| GET | `/triage/sessions` | Returns all sessions belonging to the authenticated patient. |
| GET | `/triage/sessions/{session_id}` | Returns a single session, scoped to the authenticated patient. |
| POST | `/triage/sessions/{session_id}/message` | Stores the patient's message, invokes the intake agent service, stores the agent's response, and returns the agent message. |
| GET | `/triage/sessions/{session_id}/messages` | Returns all messages for a session, scoped to the authenticated patient. |
| POST | `/triage/sessions/{session_id}/end` | Marks the session as `completed`, records `ended_at`, generates and stores a clinical report, and returns the report ID. |
| GET | `/triage/sessions/{session_id}/report` | Returns HTTP 403 unconditionally. Clinical reports are accessible only through doctor routes. |

### Doctor (`/doctor`)

All routes in this group require `role = doctor` or `role = admin`.

| Method | Path | Description |
|---|---|---|
| GET | `/doctor/dashboard` | Returns aggregate counts: total assigned patients and number of unreleased reports. |
| GET | `/doctor/patients` | Returns all patients assigned to the authenticated doctor. |
| GET | `/doctor/patients/{patient_id}` | Returns a single patient's demographic data, scoped to the authenticated doctor. |
| GET | `/doctor/patients/{patient_id}/reports` | Returns all clinical report summaries for a specific patient, scoped to the authenticated doctor. |
| GET | `/doctor/reports` | Returns all clinical report summaries for all patients assigned to the authenticated doctor. |
| GET | `/doctor/reports/{report_id}` | Returns the full clinical report, scoped to the authenticated doctor. |
| POST | `/doctor/reports/{report_id}/feedback` | Submits a feedback record for a report. Returns HTTP 400 if feedback has already been submitted. |

### Admin (`/admin`)

All routes in this group require `role = admin`.

| Method | Path | Description |
|---|---|---|
| POST | `/admin/users` | Creates a new user of any role. Requires a valid `hospital_id`. Creates the corresponding profile row (`doctor_profiles` or `patient_profiles`) automatically. |
| GET | `/admin/users` | Returns all users across all hospitals. |
| PATCH | `/admin/users/{user_id}` | Updates any user field except `password_hash`. |

### System Health (`/health`)

These routes require no authentication.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}`. Used by load balancers and deployment checks. |
| GET | `/health/db` | Executes `SELECT 1` against PostgreSQL. Returns `{"status": "ok"}` on success or HTTP 503 on failure. |

---

## Authentication and Role Access Control

### JWT Token Issuance

On a successful `POST /auth/signin`, the server produces two tokens. The access token encodes the user's email address as the `sub` claim and expires after 30 minutes. The refresh token uses the same claim structure and expires after 7 days. Both tokens are signed with the `SECRET_KEY` from the application's environment using the HS256 algorithm.

Every protected route extracts the Bearer token from the `Authorization` header, verifies the signature and expiry, resolves the `sub` claim to a `users` row, and checks that `is_active = true`. A token signed with a different key, an expired token, or a token whose subject email no longer exists in the database all result in HTTP 401.

### Role Guards

Role enforcement is implemented as three FastAPI dependency functions in `backend/app/auth/dependencies.py`.

| Dependency | Allowed roles | Effect if denied |
|---|---|---|
| `get_current_patient_user` | `patient` only | HTTP 403 |
| `get_current_doctor_user` | `doctor` or `admin` | HTTP 403 |
| `get_current_admin_user` | `admin` only | HTTP 403 |

These dependencies are composed: `get_current_patient_user` calls `get_current_active_user`, which in turn calls `get_current_user`. A route that declares `Depends(get_current_patient_user)` performs the full chain: token verification → user lookup → active check → role check.

### Access Isolation

Patients cannot access doctor routes. A patient attempting to call `GET /doctor/reports` will receive HTTP 403 at the role guard, before any database query is executed.

Doctors can only access data for patients assigned to them. Every doctor route filters by `doctor_profile.id` when querying `patient_profiles`, `triage_sessions`, and `clinical_reports`. A doctor cannot retrieve another doctor's patients or reports even if they possess a valid token.

Clinical reports are not accessible through the patient-facing triage routes. `GET /triage/sessions/{session_id}/report` returns HTTP 403 unconditionally for all callers, regardless of role. The report is accessible only through `GET /doctor/reports/{report_id}` and only to the assigned physician.

---

## Triage Session Flow

The following sequence describes the complete lifecycle of a patient triage interaction as implemented in the current backend.

**1. Session creation.** The patient sends `POST /triage/sessions` with an optional `chief_complaint` string. The backend resolves the responsible doctor — preferring the patient's `assigned_doctor_id`, falling back to the first available doctor profile. A `triage_sessions` row is created with `status = active`, `urgency_level = unknown`, `escalation_type = none`, and `chat_retention_policy = keep_full_history`.

**2. Message exchange.** The patient sends `POST /triage/sessions/{session_id}/message` with a `content` string. The backend verifies that the session exists, belongs to the authenticated patient, and has `status = active`. The patient message is stored as a `session_messages` row with `sender = patient`. The intake agent service is then invoked with the session ID and message content. The agent's response is stored as a second `session_messages` row with `sender = intake_agent`. The agent message is returned to the client. This exchange repeats for as many turns as the patient initiates.

**3. Session end.** The patient sends `POST /triage/sessions/{session_id}/end`. The backend sets `status = completed` and `ended_at` on the session row. The report generation service is invoked, returning a `ClinicalReportResponse` object populated with all required fields. A `clinical_reports` row is created from this object. The response to the client includes the report ID.

**4. Report availability.** The report is immediately available to the assigned doctor through the doctor routes. `visible_to_patient` is `false` and remains so unless the doctor explicitly updates it through a future mechanism.

---

## Clinical Report Storage

The `clinical_reports` table implements the complete agreed report schema. Every field listed below has a corresponding column in the database, a corresponding field in the `ClinicalReportResponse` Pydantic schema, and is populated by the report generation service.

| Field | Type | Purpose |
|---|---|---|
| `presenting_complaints` | JSONB | Structured list of the patient's reported complaints |
| `history_of_presenting_complaint` | JSONB | Timeline and context of the presentation |
| `summary_text` | TEXT | Narrative summary of the triage interaction |
| `suspected_conditions` | JSONB | Differential diagnosis candidates ranked by likelihood |
| `triggered_red_flags` | JSONB | Any red-flag symptoms identified during the session |
| `urgency_level` | enum | `routine`, `urgent`, or `emergency` |
| `recommended_action` | TEXT | Recommended clinical next step |
| `specialty_routing` | JSONB | Suggested specialist or department for referral |
| `suggested_workup` | JSONB | Investigations recommended before the clinical encounter |
| `key_exam_findings` | JSONB | Expected physical examination findings to look for |
| `admission_criteria` | JSONB | Criteria that would indicate hospital admission |
| `referral_criteria` | JSONB | Criteria for outpatient specialist referral |
| `external_escalation_completed` | BOOLEAN | Whether an emergency escalation was triggered |
| `escalation_message` | TEXT | Content of any external escalation communication |
| `visible_to_patient` | BOOLEAN | Patient visibility flag; `false` by default |
| `model_version` | VARCHAR | Version identifier of the AI model used |
| `generated_at` | TIMESTAMP | Report generation timestamp |

---

## Doctor Feedback Loop

After reviewing a clinical report, the assigned doctor may submit feedback through `POST /doctor/reports/{report_id}/feedback`. The feedback system is designed to support future model retraining and quality monitoring rather than to alter the report content directly.

A feedback record consists of three components. The `rating` field captures a binary overall judgement: `thumbs_up` (the report was clinically appropriate) or `thumbs_down` (the report contained errors or omissions). The `correction_text` field accepts free-text input for cases where the doctor wishes to describe the specific problem in their own words. The `feedback_category` field classifies the type of problem: `wrong_urgency`, `wrong_diagnosis`, `missing_info`, `unsafe_response`, `irrelevant_sources`, or `other`.

Only one feedback record per doctor per report is permitted. Submitting a second feedback record for the same report returns HTTP 400. This constraint prevents feedback loops while still allowing multiple doctors who have access to the same report to each submit their independent assessment.

The feedback table is the primary data source for any future retraining or evaluation pipeline. A collected set of `thumbs_down` records with `correction_text` and `feedback_category` provides structured ground-truth signal that can be used to benchmark model versions, identify systematic failure modes, and guide fine-tuning.

---

## Current Limitations

### AI and RAG Logic Is Mocked

The intake agent (`backend/app/services/intake_agent_service.py`) and the RAG agent (`backend/app/services/rag_agent_service.py`) are mock services. They return fixed, plausible-looking responses without performing any retrieval, reasoning, or generation. The report generation service (`backend/app/services/report_service.py`) similarly returns a static placeholder report.

These stubs exist to allow the full request-response cycle to be tested end-to-end — a patient can sign up, start a session, exchange messages, end the session, and retrieve a report — without requiring the RAG pipeline to be connected. When the RAG integration is implemented, these service functions are the integration points: the function signatures, return types, and database write logic are already in place.

### Refresh Token Is Not Implemented

`POST /auth/refresh-token` returns HTTP 501 Not Implemented. Refresh token verification, rotation, and revocation logic has not been built. Sessions expire after 30 minutes and require the user to re-authenticate.

### Doctor Assignment Requires a Manual Workflow

When a patient self-registers, their `patient_profiles.assigned_doctor_id` is null. The triage session creation falls back to the first doctor profile in the database if no doctor is assigned. A proper hospital administrator workflow — in which an admin assigns a doctor to a patient after registration — has not been implemented at the API level. The data model fully supports it (the `assigned_doctor_id` column and the admin user creation route exist), but no dedicated assignment endpoint has been built.

### No Production Security Hardening

The backend is configured for local development. CORS middleware is not configured, meaning the API will reject cross-origin browser requests. No rate limiting, request size limits, or brute-force protection are applied to the authentication routes. The `SECRET_KEY` must be set to a cryptographically strong random value before any deployment. HTTPS termination is the responsibility of a reverse proxy not included in this phase.

### ChromaDB Integration Is a Future Connection Point

The `rag_queries` and `rag_retrieved_chunks` tables are ready to receive data from the retrieval pipeline, and the service layer has the correct function signatures. However, no code in the current backend calls ChromaDB. The vector store remains independent until the intake agent and RAG agent services are replaced with real implementations.

---

## How to Run

### Prerequisites

- Python 3.11 or higher
- A running PostgreSQL instance
- The `pgcrypto` extension available in the target database (included in standard PostgreSQL installations)

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

The `requirements.txt` specifies: `fastapi==0.104.1`, `uvicorn==0.24.0`, `sqlalchemy==2.0.23`, `alembic==1.12.1`, `psycopg2-binary==2.9.9`, `python-jose[cryptography]==3.3.0`, `passlib[bcrypt]==1.7.4`, `python-multipart==0.0.6`, `pydantic==2.5.0`, and `pydantic-settings==2.1.0`.

### 2. Configure Environment

Copy `.env.example` to `.env` and populate all values:

```bash
cp .env.example .env
```

```
DATABASE_URL=postgresql://user:password@localhost/medora
SECRET_KEY=<cryptographically-random-string>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

The `.env` file is listed in `.gitignore` and must not be committed to version control.

### 3. Run the Alembic Migration

From the `backend/` directory:

```bash
alembic upgrade head
```

This executes `alembic/versions/001_initial.py`, which creates the `pgcrypto` extension, all eleven PostgreSQL enum types, and all nineteen tables in the correct dependency order.

### 4. Start the FastAPI Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables hot-reloading during development. The interactive API documentation is available at `http://localhost:8000/docs`. The health check endpoint is at `http://localhost:8000/health`.
