from fastapi import APIRouter, Depends
from ..auth.dependencies import get_current_active_user
from ..models.user import User
from ..schemas.patient import PatientUpdate, MedicalHistory, Consent
from ..utils.mocks import mock_medical_history, mock_consents

router = APIRouter(prefix="/patients", tags=["patient"])


@router.get("/me")
def get_patient_me(current_user: User = Depends(get_current_active_user)):
    return {"id": current_user.id, "email": current_user.email, "full_name": current_user.full_name}


@router.patch("/me")
def update_patient_me(update_data: PatientUpdate, current_user: User = Depends(get_current_active_user)):
    # Placeholder: update user in DB
    return {"message": "Patient updated"}


@router.get("/me/medical-history", response_model=MedicalHistory)
def get_medical_history(current_user: User = Depends(get_current_active_user)):
    # Mock data
    return mock_medical_history


@router.put("/me/medical-history")
def update_medical_history(history: MedicalHistory, current_user: User = Depends(get_current_active_user)):
    # Placeholder: save to DB
    return {"message": "Medical history updated"}


@router.get("/me/consents")
def get_consents(current_user: User = Depends(get_current_active_user)):
    # Mock data
    return mock_consents


@router.post("/me/consents")
def grant_consent(consent: Consent, current_user: User = Depends(get_current_active_user)):
    # Placeholder: save consent
    return {"message": "Consent granted"}