# Medora Backend

A FastAPI-based backend for the Medora medical triage system.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your database URL and secret key.
   - For demo, use PostgreSQL. Example: `DATABASE_URL=postgresql://user:password@localhost/medora`
   - Set `SECRET_KEY` to a random string.

3. Run Alembic migrations:
   ```bash
   alembic upgrade head
   ```

4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## Database

- Full schema implemented with SQLAlchemy 2.x models.
- PostgreSQL required for JSONB and UUID support.
- Alembic migrations in `alembic/versions/`.

## API Routes

- Auth: `/auth/*` - Signup, signin, me, refresh, logout, change-password
- Patient: `/patients/*` - Profile, medical history, consents
- Triage: `/triage/*` - Sessions, messages, reports
- Doctor: `/doctor/*` - Dashboard, patients, reports, feedback
- Admin: `/admin/*` - User management
- System: `/health*` - Health checks

## Development

- Code is modular with services for AI/RAG (currently mocked).
- Dependency injection used throughout.
- Role-based access control implemented.
- AI/RAG logic abstracted in services for future implementation.