# Medora Frontend

React + TypeScript + Vite frontend for the Medora AI Medical Triage application.

## Prerequisites

- **Node.js** ≥ 18
- **npm** ≥ 9
- Backend API running at `http://localhost:8000`

## Quick Start

```bash
cd frontend
npm install
npm run dev
```

The dev server starts at **http://localhost:5173** with an API proxy forwarding `/api/*` → `http://localhost:8000/*` (avoids CORS issues).

## Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server (port 5173) |
| `npm run build` | TypeScript check + production build |
| `npm run preview` | Preview production build locally |

## Tech Stack

- **React 19** + **TypeScript**
- **Vite 8** (dev server + bundler)
- **Tailwind CSS v4** (via `@tailwindcss/vite`)
- **React Router v7** (client-side routing)
- **Axios** (HTTP client with auth interceptors)

## Project Structure

```
src/
├── api/            # API client layer (auth, patient, triage, doctor, admin)
├── components/
│   ├── auth/       # ProtectedRoute, RoleGuard
│   ├── doctor/     # UrgencyBadge, ReportSectionCard, FeedbackForm
│   ├── layout/     # AppLayout, AuthLayout, Sidebar, Navbar
│   └── ui/         # Button, Input, TextArea, Card, Badge, etc.
├── context/        # AuthContext provider
├── hooks/          # useAuth hook
├── pages/
│   ├── admin/      # AdminUsers
│   ├── auth/       # LoginPage, SignupPage
│   ├── doctor/     # Dashboard, Patients, Reports, ReportDetail
│   └── patient/    # Dashboard, Profile, MedicalHistory, Triage, TriageSession
├── types/          # TypeScript interfaces matching backend schemas
├── App.tsx         # Route configuration
├── main.tsx        # Entry point
└── index.css       # Tailwind config + global styles
```

## Authentication

- Tokens stored in `localStorage` (key: `medora_token`)
- Axios interceptor adds `Authorization: Bearer <token>` to all requests
- On 401 response, token is cleared and user is redirected to `/login`
- Sign-in uses `application/x-www-form-urlencoded` (OAuth2 form)

## Routes

| Path | Role | Page |
|------|------|------|
| `/login` | Public | Login |
| `/signup` | Public | Signup |
| `/patient/dashboard` | Patient | Dashboard |
| `/patient/profile` | Patient | Profile editor |
| `/patient/medical-history` | Patient | Medical history form |
| `/patient/triage` | Patient | Triage sessions list |
| `/patient/triage/:sessionId` | Patient | Chat interface |
| `/doctor/dashboard` | Doctor | Dashboard with stats |
| `/doctor/patients` | Doctor | Patient list |
| `/doctor/patients/:patientId` | Doctor | Patient detail |
| `/doctor/reports` | Doctor | Report list |
| `/doctor/reports/:reportId` | Doctor | Report detail + feedback |
| `/admin/users` | Admin | User management |
| `/unauthorized` | Any | 403 page |
| `*` | Any | 404 page |

## API Proxy

The Vite dev server proxies all `/api/*` requests to the backend:

```
Frontend: http://localhost:5173/api/auth/me
   ↓ proxy rewrite
Backend:  http://localhost:8000/auth/me
```

This avoids CORS issues during development without modifying the backend.
