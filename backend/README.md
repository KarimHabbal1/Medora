# Medora Backend

A FastAPI-based backend for the Medora medical triage system.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your database URL and secret key.

3. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## Database

- Database schema is not yet finalized.
- Only User model is implemented for authentication.
- Use mock data for other endpoints.

## API Routes

- Auth: `/auth/*`
- Patient: `/patients/*`
- Triage: `/triage/*`
- Doctor: `/doctor/*`
- Admin: `/admin/*`
- System: `/health*`

## Development

- Keep code modular and flexible.
- Use dependency injection.
- Mock data is used where schema is unclear.