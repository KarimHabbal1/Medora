from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..auth.dependencies import get_current_patient_user
from ..models import User, PatientProfile, PatientMedicalHistory, PatientConsent
from ..schemas.patient import PatientProfileResponse, PatientUpdate, MedicalHistoryResponse, MedicalHistoryUpdate, ConsentResponse, ConsentCreate

router = APIRouter(prefix="/patients", tags=["patient"])


@router.get("/me", response_model=PatientProfileResponse)
def get_patient_me(current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")
    return patient_profile


@router.patch("/me", response_model=PatientProfileResponse)
def update_patient_me(update_data: PatientUpdate, current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")
    
    for key, value in update_data.model_dump(exclude_unset=True).items():
        setattr(patient_profile, key, value)
    
    db.commit()
    db.refresh(patient_profile)
    return patient_profile


@router.get("/me/medical-history", response_model=MedicalHistoryResponse)
def get_medical_history(current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")
    
    history = db.query(PatientMedicalHistory).filter(PatientMedicalHistory.patient_id == patient_profile.id).first()
    if not history:
        # Create empty history if not exists
        history = PatientMedicalHistory(patient_id=patient_profile.id)
        db.add(history)
        db.commit()
        db.refresh(history)
    return history


@router.put("/me/medical-history", response_model=MedicalHistoryResponse)
def update_medical_history(history_data: MedicalHistoryUpdate, current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")
    
    history = db.query(PatientMedicalHistory).filter(PatientMedicalHistory.patient_id == patient_profile.id).first()
    if not history:
        history = PatientMedicalHistory(patient_id=patient_profile.id)
        db.add(history)
    
    for key, value in history_data.model_dump(exclude_unset=True).items():
        setattr(history, key, value)
    
    db.commit()
    db.refresh(history)
    return history


@router.get("/me/consents", response_model=list[ConsentResponse])
def get_consents(current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")
    
    consents = db.query(PatientConsent).filter(PatientConsent.patient_id == patient_profile.id).all()
    return consents


@router.post("/me/consents", response_model=ConsentResponse)
def grant_consent(consent_data: ConsentCreate, current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")
    
    # Check if consent already exists
    existing = db.query(PatientConsent).filter(
        PatientConsent.patient_id == patient_profile.id,
        PatientConsent.consent_type == consent_data.consent_type
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Consent already granted")
    
    consent = PatientConsent(
        patient_id=patient_profile.id,
        consent_type=consent_data.consent_type,
        accepted=consent_data.accepted
    )
    db.add(consent)
    db.commit()
    db.refresh(consent)
    return consent