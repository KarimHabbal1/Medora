from fastapi import FastAPI
from .routers import auth, patient, triage, doctor, admin, system

app = FastAPI(title="Medora API", version="1.0.0")

app.include_router(auth.router)
app.include_router(patient.router)
app.include_router(triage.router)
app.include_router(doctor.router)
app.include_router(admin.router)
app.include_router(system.router)


@app.get("/")
def root():
    return {"message": "Welcome to Medora API"}