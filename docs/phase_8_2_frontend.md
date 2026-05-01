# Phase 8.2: Frontend UI Implementation

**Project:** Medora — AI-Assisted Clinical Triage System
**Phase:** 8.2
**Scope:** Frontend user interface
**Status:** Prototype complete

---

## 1. Overview

Phase 8.2 marks the transition from a functional backend API to a fully interactive, user-facing system. The backend implemented in earlier phases provides a complete REST API for authentication, patient data management, triage sessions, clinical report generation, and administrative control. This phase establishes the frontend application that consumes those endpoints and exposes the system's functionality to its three primary user roles: patient, doctor, and administrator.

The primary objectives of this phase are:

- Provide a role-differentiated user interface for patients, doctors, and administrators.
- Connect all approved API endpoints to corresponding UI flows.
- Enforce role-based access control at the routing level to complement backend authorization.
- Deliver a working prototype suitable for demonstration and functional evaluation.

The frontend does not implement business logic, clinical reasoning, or agent coordination. All such decisions are delegated to the backend and, in future phases, to the AI/RAG layer. The frontend's responsibility is strictly limited to presenting data, capturing input, and communicating with the API.

---

## 2. Architecture Overview

The frontend is a single-page application (SPA) that communicates exclusively with the FastAPI backend over HTTP. It is served independently of the backend and proxies all API requests through the Vite development server to avoid CORS constraints during development.

### Communication Model

All API communication uses REST with JSON payloads. Authentication is managed via Bearer tokens issued by the `/auth/signin` endpoint. The token is attached to every subsequent request via an Axios request interceptor. The backend is the single source of truth for all data, validation, and authorization decisions.

### Routing Model

Client-side routing is implemented with React Router. Routes are organized in three protected groups — patient, doctor, and admin — each wrapped in a `ProtectedRoute` guard (which verifies authentication) and a `RoleGuard` (which verifies the user's role). Unauthenticated users are redirected to `/login`. Authenticated users reaching an unauthorized route are redirected to `/unauthorized`.

### Role-Based Access

Upon login, the backend returns a user object including the `role` field (`patient`, `doctor`, or `admin`). The `AuthContext` stores this object and exposes it application-wide. Route guards consume this context to determine whether a given route is accessible without requiring a second API call.

The root path `/` resolves dynamically based on role:

| Role    | Redirect target         |
|---------|-------------------------|
| patient | `/patient/dashboard`    |
| doctor  | `/doctor/dashboard`     |
| admin   | `/admin/users`          |

### Backend Dependency

The frontend has no offline capability and no local data store. Every page load, state update, and user action that involves data requires a live connection to the FastAPI backend. All validation, business logic, and persistence are exclusively backend-controlled. The frontend is a rendering and input layer: it sends requests and presents responses. If the backend is unavailable, the application will fail to authenticate, load data, or submit forms. This is an acknowledged architectural characteristic of the prototype, not an oversight.

---

## 3. Technology Stack

| Technology       | Version  | Role                                          |
|-----------------|----------|-----------------------------------------------|
| React            | 19.x     | Component model and rendering                 |
| TypeScript       | 6.x      | Static typing and interface definitions       |
| Vite             | 8.x      | Build tooling and development server          |
| React Router     | 7.x      | Client-side routing and navigation            |
| Tailwind CSS     | 4.x      | Utility-first styling                         |
| Axios            | 1.x      | HTTP client with interceptor support          |

**React** was selected for its component model, ecosystem maturity, and compatibility with TypeScript. Version 19 introduces improved concurrent rendering, which benefits interactive UIs such as the triage chat interface.

**TypeScript** enforces type safety across the API layer, component props, and shared types. This is particularly important for a system where schema correctness affects clinical data. Interface definitions for backend response models are maintained in `/src/types`, creating a single source of truth for data shapes used across the frontend.

**Vite** provides near-instant hot module replacement during development and produces optimized bundles for production. Its plugin system supports both React Fast Refresh and Tailwind CSS 4's native Vite integration without additional configuration overhead.

**React Router v7** provides declarative nested routing with layout-level components, which maps directly to the application's role-based layout structure. Nested `<Route>` composition allows `ProtectedRoute` and `RoleGuard` to be applied at the group level rather than per-route.

**Tailwind CSS v4** is used for all styling via utility classes. It eliminates the need for separate CSS files in most components and enforces visual consistency through a shared design token system.

**Axios** is preferred over the native Fetch API for its built-in request and response interceptor support. The token attachment and 401 redirect logic are centralized in a single `apiClient` instance rather than duplicated across API modules.

---

## 3.1 API Contract (Frontend Consumption)

The following table lists every backend endpoint consumed by the frontend, grouped by domain. No endpoint outside this set is called by the application.

### Authentication

| Method | Endpoint             | Consumer         | Purpose                                      |
|--------|----------------------|------------------|----------------------------------------------|
| POST   | `/auth/signup`       | SignupPage        | Create patient account and PatientProfile    |
| POST   | `/auth/signin`       | LoginPage         | Authenticate and obtain Bearer token         |
| GET    | `/auth/me`           | AuthContext       | Validate token and retrieve current user     |
| PATCH  | `/auth/change-password` | (settings)    | Update authenticated user's password         |

### Patient

| Method | Endpoint                       | Consumer            | Purpose                              |
|--------|--------------------------------|---------------------|--------------------------------------|
| GET    | `/patients/me`                 | PatientProfile      | Retrieve own profile                 |
| PATCH  | `/patients/me`                 | PatientProfile      | Update demographic fields            |
| GET    | `/patients/me/medical-history` | MedicalHistory      | Retrieve structured medical history  |
| PUT    | `/patients/me/medical-history` | MedicalHistory      | Save or update medical history       |
| GET    | `/patients/me/consents`        | PatientDashboard    | Retrieve consent records             |
| POST   | `/patients/me/consents`        | PatientDashboard    | Record a new consent acknowledgement |

### Triage

| Method | Endpoint                              | Consumer       | Purpose                                          |
|--------|---------------------------------------|----------------|--------------------------------------------------|
| POST   | `/triage/sessions`                    | TriageList     | Create a new triage session                      |
| GET    | `/triage/sessions`                    | TriageList     | List all sessions for the current patient        |
| GET    | `/triage/sessions/{id}`               | TriageSession  | Retrieve a specific session                      |
| POST   | `/triage/sessions/{id}/message`       | TriageSession  | Send patient message and receive agent response  |
| GET    | `/triage/sessions/{id}/messages`      | TriageSession  | Retrieve full message history for a session      |
| POST   | `/triage/sessions/{id}/end`           | TriageSession  | End session and trigger report generation        |
| GET    | `/triage/sessions/{id}/report`        | TriageSession  | Returns HTTP 403 for patients (enforced backend) |

### Doctor

| Method | Endpoint                               | Consumer            | Purpose                                    |
|--------|----------------------------------------|---------------------|--------------------------------------------||
| GET    | `/doctor/dashboard`                    | DoctorDashboard     | Retrieve aggregate stats                   |
| GET    | `/doctor/patients`                     | DoctorPatients      | List assigned patients                     |
| GET    | `/doctor/patients/{id}`                | DoctorPatientDetail | Retrieve patient details                   |
| GET    | `/doctor/patients/{id}/reports`        | DoctorPatientDetail | List reports for a specific patient        |
| GET    | `/doctor/reports`                      | DoctorReports       | List all reports for the current doctor    |
| GET    | `/doctor/reports/{id}`                 | DoctorReportDetail  | Retrieve full clinical report              |
| POST   | `/doctor/reports/{id}/feedback`        | DoctorReportDetail  | Submit structured feedback on a report     |

### Admin

| Method | Endpoint              | Consumer   | Purpose                                    |
|--------|-----------------------|------------|--------------------------------------------||
| GET    | `/admin/users`        | AdminUsers | List all users in the system               |
| POST   | `/admin/users`        | AdminUsers | Create a new user with role assignment     |
| PATCH  | `/admin/users/{id}`   | AdminUsers | Update user fields inline                  |
| GET    | `/admin/hospitals`    | AdminUsers | Retrieve hospital list for dropdown        |

### System

| Method | Endpoint       | Consumer | Purpose                          |
|--------|----------------|----------|----------------------------------|
| GET    | `/health`      | (ops)    | Application liveness check       |
| GET    | `/health/db`   | (ops)    | Database connectivity check      |

---

## 4. Application Structure

```
src/
├── api/              # API module — one file per backend domain
│   ├── client.ts     # Axios instance with auth interceptor
│   ├── auth.ts
│   ├── admin.ts
│   ├── doctor.ts
│   ├── patient.ts
│   └── triage.ts
│
├── components/
│   ├── auth/         # ProtectedRoute, RoleGuard
│   ├── doctor/       # Doctor-specific composed components
│   ├── layout/       # AppLayout (authenticated shell), AuthLayout
│   └── ui/           # Reusable primitives: Button, Card, Input,
│                     #   Badge, ErrorAlert, LoadingSpinner,
│                     #   TextArea, EmptyState
│
├── context/
│   └── AuthContext.tsx  # Global auth state (user, token, login, logout)
│
├── hooks/
│   └── useAuth.ts    # Typed consumer of AuthContext
│
├── pages/
│   ├── auth/         # LoginPage, SignupPage
│   ├── patient/      # PatientDashboard, PatientProfile,
│   │                 #   MedicalHistory, TriageList, TriageSession
│   ├── doctor/       # DoctorDashboard, DoctorPatients,
│   │                 #   DoctorPatientDetail, DoctorReports,
│   │                 #   DoctorReportDetail
│   └── admin/        # AdminUsers
│
├── types/            # TypeScript interfaces for all backend models
│   ├── admin.ts
│   ├── auth.ts
│   ├── doctor.ts
│   ├── enums.ts
│   ├── patient.ts
│   └── triage.ts
│
├── App.tsx           # Route tree root
└── main.tsx          # React entry point
```

### Separation of Concerns

The `/api` layer is the exclusive point of contact with the backend. Pages and components do not construct HTTP requests directly; they call typed async functions exported from the relevant API module. This isolates API changes to a single file and prevents scattered `axios` calls across the codebase.

The `/types` directory defines TypeScript interfaces that mirror the Pydantic response schemas on the backend. Any change to a backend response model has a single corresponding location in the frontend to update.

The `/context` and `/hooks` directories handle global state. Authentication state — user object, token presence, and loading status — is managed centrally in `AuthContext` and consumed via the `useAuth` hook. No component reads directly from `localStorage`.

### State Management Strategy

The application uses React Context exclusively for global state. No third-party state management library (Redux, Zustand, Jotai, or equivalent) is included. The `AuthContext` is the only context in the application and holds only authentication-related state: the current user object, the authenticated flag, and the loading state during token verification.

All domain data — patient profiles, triage sessions, clinical reports, user lists — is fetched directly from the backend on demand within the component that needs it. Data is not cached globally, not stored in a shared store, and not propagated through context. Components are responsible for their own loading and error states.

This approach was chosen deliberately for the prototype phase:

- It minimizes architectural complexity and setup overhead.
- It avoids premature abstractions that are difficult to unwind during iterative development.
- It keeps the backend as the unambiguous source of truth, with no risk of stale client-side state diverging from the database.

If the application scales to require shared domain state, cross-component data synchronization, or optimistic updates, a lightweight store (Zustand) would be the recommended addition in a subsequent phase.

---

## 5. Authentication and Authorization

### Signup and Login

New users register via `/auth/signup` using the `SignupPage`. The backend creates the user record and an associated `PatientProfile` automatically. On successful registration, the user is redirected to the login page.

Authentication uses the OAuth2 password flow. The `LoginPage` submits credentials to `/auth/signin`, which returns an `access_token` and a `token_type`. The token is stored in `localStorage` under the key `medora_token` and is retrieved on application load to restore session state.

On load, the application calls `GET /auth/me` using the stored token to verify its validity and retrieve the current user object. If the token is missing, expired, or rejected, the user is redirected to `/login`.

### Token Management

The `apiClient` Axios instance attaches the Bearer token to every outgoing request via a request interceptor:

```
Authorization: Bearer <access_token>
```

A response interceptor handles 401 responses by clearing the stored token and redirecting to `/login`, preventing stale authenticated state.

### Role-Based Routing

Routes are protected at two levels:

1. **`ProtectedRoute`** — verifies that the user is authenticated before rendering any child routes.
2. **`RoleGuard`** — verifies that the authenticated user's role matches the allowed roles for the route group. Unauthorized access redirects to `/unauthorized`.

Backend authorization is not bypassed by frontend guards. The backend independently validates the token and role on every request. Frontend guards exist to improve user experience by preventing navigation to inaccessible routes, not to enforce security.

---

## 6. Core User Flows

### 6.1 Patient Flow

1. **Signup** — Patient submits name, email, and password via `/auth/signup`. A `PatientProfile` is created server-side.
2. **Login** — Patient authenticates and is redirected to `/patient/dashboard`.
3. **Profile** — Patient can update demographic fields (date of birth, sex, height, weight) via `PATCH /patients/me`.
4. **Medical History** — Patient fills structured medical history (chronic conditions, medications, allergies, etc.) via `PUT /patients/me/medical-history`. History is stored as JSONB and may be skipped.
5. **Consents** — Patient records consent acknowledgements via `POST /patients/me/consents`.
6. **Triage Session** — Patient initiates a triage session via `POST /triage/sessions` with an optional chief complaint.
7. **Chat** — Patient sends messages via `POST /triage/sessions/{id}/message`. The backend processes each message through the intake agent and returns a structured response. The patient sees the conversation in a chat-style UI.
8. **End Session** — Patient ends the session via `POST /triage/sessions/{id}/end`. The backend marks the session as completed and generates a clinical report internally.
9. **Report Visibility** — The generated report is not accessible to the patient. `GET /triage/sessions/{id}/report` returns HTTP 403. The patient sees only a confirmation that the session has ended.

### 6.2 Doctor Flow

1. **Login** — Doctor authenticates and is redirected to `/doctor/dashboard`.
2. **Dashboard** — Displays aggregate metrics: total assigned patients and pending reports, sourced from `GET /doctor/dashboard`.
3. **Patient List** — `GET /doctor/patients` returns all patients assigned to the doctor. The doctor can navigate to a per-patient detail view.
4. **Reports** — `GET /doctor/reports` lists all clinical reports associated with the doctor's sessions. Individual reports are fetched via `GET /doctor/reports/{id}`.
5. **Report Detail** — The full structured clinical report is rendered with labeled sections corresponding to each field (presenting complaints, urgency level, suspected conditions, recommended action, etc.).
6. **Feedback** — The doctor can submit structured feedback on a report via `POST /doctor/reports/{id}/feedback`, including a rating and optional category and correction text.

### 6.3 Admin Flow

1. **Login** — Admin authenticates and is redirected to `/admin/users`.
2. **User List** — `GET /admin/users` returns all users in the system with their role, status, and creation date.
3. **Create User** — Admin submits a form to create a new user via `POST /admin/users`. The form includes name, email, password, role, phone, and hospital selection. Hospital is selected from a dropdown populated by `GET /admin/hospitals`.
4. **Edit User** — Admin can update a user's name, email, role, and active status inline via `PATCH /admin/users/{id}`.

---

## 7. Triage Chat Interface

The triage interface is rendered as a sequential message log. Patient messages and intake agent responses appear in a chat layout, with visual differentiation by sender.

The session lifecycle is as follows:

1. A session is created with an optional chief complaint. The backend assigns a doctor, initializes the session record, and returns the session object.
2. The patient submits free-text messages. Each message is sent to the backend, which invokes the intake agent service and returns the agent's response. Both the patient message and the agent response are stored server-side as `SessionMessage` records.
3. The patient ends the session explicitly. The backend transitions the session status to `completed`, sets `ended_at`, and triggers clinical report generation.

The frontend applies no clinical logic. It does not interpret message content, assess urgency, or determine escalation paths. All such behavior is backend-controlled and opaque to the UI. The frontend renders whatever the backend returns.

The agent response object includes `sender`, `content`, `message_type`, and visibility flags (`is_visible_to_doctor`, `is_persisted_after_summary`), all of which are set by the backend.

---

## 8. Clinical Report Rendering

Clinical reports are returned as structured JSON objects with named fields. The `DoctorReportDetail` page maps each field to a labeled UI section. The complete field set rendered includes:

| Field                              | Type         |
|------------------------------------|--------------|
| `presenting_complaints`            | JSONB        |
| `history_of_presenting_complaint`  | JSONB        |
| `summary_text`                     | Text         |
| `suspected_conditions`             | JSONB        |
| `triggered_red_flags`              | JSONB        |
| `urgency_level`                    | Enum         |
| `recommended_action`               | Text         |
| `specialty_routing`                | JSONB        |
| `suggested_workup`                 | JSONB        |
| `key_exam_findings`                | JSONB        |
| `admission_criteria`               | JSONB        |
| `referral_criteria`                | JSONB        |
| `external_escalation_completed`    | Boolean      |
| `escalation_message`               | Text         |
| `visible_to_patient`               | Boolean      |
| `model_version`                    | String       |
| `generated_at`                     | Datetime     |

JSONB fields are rendered as structured displays rather than raw JSON strings. Urgency level is rendered as a color-coded badge. The `visible_to_patient` flag is displayed as a read-only indicator. No field is editable in the report view; editing is handled through the feedback mechanism.

Report access is strictly doctor-only. Patient-facing triage routes do not expose this data.

---

## 9. Error Handling Strategy

API errors are handled at two levels: the Axios interceptor layer (for authentication failures) and the component level (for domain-specific errors).

### 401 Unauthorized

Handled globally by the Axios response interceptor. The stored token is cleared and the user is redirected to `/login`. This prevents stale session state from persisting across requests.

### 422 Unprocessable Entity

FastAPI returns a 422 when request body validation fails. The response body contains a `detail` field that is an array of validation error objects with `msg` and `loc` properties. The frontend explicitly handles this structure in all form submit handlers:

```ts
if (typeof detail === 'string') {
  msg = detail;
} else if (Array.isArray(detail)) {
  msg = detail
    .map((d) => {
      const field = d.loc?.filter((l) => l !== 'body').join(' > ') ?? '';
      return field ? `${field}: ${d.msg}` : d.msg;
    })
    .join(' | ');
} else if (detail && typeof detail === 'object') {
  msg = JSON.stringify(detail);
} else {
  msg = 'Failed to submit.';
}
```

This ensures that only a `string` value is ever stored in error state. Storing a raw object in state and rendering it as a React child results in an application crash; this handler prevents that outcome unconditionally.

### Form Validation

Client-side validation guards are applied before API calls are made. Required fields are validated at the component level to avoid unnecessary requests. The hospital selection field in the admin create-user form is validated before `POST /admin/users` is called, preventing a predictable 422.

### Error Display

Errors are displayed via the `ErrorAlert` component, which accepts a `string` message and an optional dismiss callback. Components store error state as `string`, initialized to `''`. No component renders an error state variable that could contain a non-string value.

Development-mode console logging is included in error handlers:

```ts
console.error('Request failed:', error.response?.data || error);
```

---

## 10. Security and Access Control

### Frontend Enforcement

Role-based access is enforced at the routing level via `ProtectedRoute` and `RoleGuard`. A patient user navigating directly to `/doctor/dashboard` is redirected to `/unauthorized`. An unauthenticated user navigating to any protected route is redirected to `/login`.

### Backend Enforcement

Frontend guards are not a security boundary. Every API endpoint enforces authentication and role authorization independently using the decoded JWT and the user's `role` field in the database. A patient cannot retrieve clinical reports by constructing a direct API request, regardless of frontend routing state, because the backend returns HTTP 403 for any patient attempting to access `/triage/sessions/{id}/report` or any `/doctor/*` route.

### Password Handling

Passwords are transmitted over HTTPS in production. Passwords are never stored on the client. The backend stores only `password_hash` (bcrypt). No password field appears in any API response schema.

### Token Storage

Tokens are stored in `localStorage` for the prototype. This is acknowledged as a limitation for production deployment (see Section 11). No sensitive clinical data is stored on the client side; all data is fetched from the backend on demand.

### Admin Route Isolation

Admin routes (`/admin/*`) are accessible only to users with `role = admin`. The backend's `get_current_admin_user` dependency independently enforces this on every admin endpoint.

---

## 11. Limitations

The following limitations apply to the current prototype and are acknowledged as areas requiring resolution before production deployment.

**No real-time streaming.** The triage chat interface uses a request-response model. The intake agent response is returned synchronously as a complete message. Streaming token delivery (as would be typical for LLM output) is not implemented.

**No WebSocket support.** All communication uses HTTP polling or direct request-response. Real-time session updates, typing indicators, and push notifications are not available.

**localStorage token storage.** Storing the access token in `localStorage` exposes it to XSS attacks. Production deployments should use `HttpOnly` cookies with appropriate CSRF protection.

**Hospital dropdown depends on prior data.** The hospital list in the admin create-user form is populated from `GET /admin/hospitals`. In a fresh database, the list is empty because hospitals are created automatically on the first patient self-signup. The admin must ensure at least one hospital record exists before creating users.

**No pagination or filtering.** User lists, patient lists, and report lists are fetched in full. Large datasets will degrade performance and usability.

**Basic admin tooling.** The admin interface provides user creation and basic field editing. Hospital management, bulk operations, audit log review, and system configuration are not exposed in the current UI.

**No appointment or scheduling system.** The system does not model appointments. Triage sessions are initiated ad hoc by patients and are not linked to a scheduling workflow.

---

## 12. Future Improvements

The following improvements are planned for subsequent phases:

**WebSocket-based streaming.** Replace the synchronous message endpoint with a WebSocket connection to support token-by-token streaming from the LLM backend, providing a more natural conversational experience.

**Session state persistence.** Store incomplete triage sessions in a way that allows patients to resume them across browser sessions without restarting the conversation.

**Pagination and filtering.** Add server-side pagination to all list endpoints and corresponding UI controls for filtering by role, date, urgency level, and status.

**Admin hospital management.** Add a dedicated hospital management section to the admin panel, including creation, editing, and deactivation of hospital records.

**Enhanced error UX.** Add field-level inline validation feedback in forms, rather than aggregating all errors into a single alert. Improve retry mechanisms for transient network failures.

**Production-grade token management.** Replace `localStorage` token storage with `HttpOnly` cookie-based sessions and implement token rotation.

**Accessibility audit.** Audit all components against WCAG 2.1 AA criteria. Add ARIA attributes, keyboard navigation support, and screen reader compatibility across the triage and report interfaces.

**Appointment and scheduling integration.** Extend the patient and doctor flows to support appointment-based session initiation, linking triage sessions to scheduled encounters.

---

## 13. Conclusion

Phase 8.2 delivers the complete user-facing layer of the Medora system. The frontend connects the backend API to three distinct user roles — patient, doctor, and administrator — through purpose-built interfaces. All approved API endpoints are consumed, all role-based access restrictions are enforced both client-side and server-side, and the full patient-to-doctor data flow is operational: from self-registration and medical history collection through triage session interaction and clinical report generation and review.

The prototype is functional end-to-end and suitable for demonstration, user evaluation, and iterative development. The architectural separation between the frontend (presentation), backend (business logic and persistence), and future AI layer (clinical reasoning) is maintained throughout. No clinical decision-making logic resides in the frontend codebase.

The system is prepared for Phase 9, in which the mock intake agent and report generation service will be replaced by real AI reasoning and Retrieval-Augmented Generation (RAG). Because the frontend communicates exclusively through the agreed API contract, this transition requires no frontend changes: the same request-response model, the same endpoints, and the same response schemas are preserved. The shift from deterministic mock responses to AI-generated clinical output will be entirely transparent to the UI layer.

---

*Document version: 1.0 — Phase 8.2 completion*
*Generated: 2026-04-29*
