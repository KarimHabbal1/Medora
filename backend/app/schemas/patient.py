from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import date, datetime
from .enums import ConsentType


class PatientProfileResponse(BaseModel):
    id: UUID
    user_id: UUID
    hospital_id: UUID
    assigned_doctor_id: Optional[UUID]
    date_of_birth: Optional[date]
    sex: Optional[str]
    height_cm: Optional[float]
    weight_kg: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class PatientUpdate(BaseModel):
    date_of_birth: Optional[date] = None
    sex: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None


class MedicalHistoryResponse(BaseModel):
    id: UUID
    patient_id: UUID
    chronic_conditions: Optional[Dict[str, Any]]
    medications: Optional[Dict[str, Any]]
    allergies: Optional[Dict[str, Any]]
    surgeries: Optional[Dict[str, Any]]
    family_history: Optional[Dict[str, Any]]
    smoking_status: Optional[str]
    pregnancy_status: Optional[str]
    additional_notes: Optional[str]
    skipped: bool
    updated_at: datetime

    class Config:
        from_attributes = True


class MedicalHistoryUpdate(BaseModel):
    chronic_conditions: Optional[Dict[str, Any]] = None
    medications: Optional[Dict[str, Any]] = None
    allergies: Optional[Dict[str, Any]] = None
    surgeries: Optional[Dict[str, Any]] = None
    family_history: Optional[Dict[str, Any]] = None
    smoking_status: Optional[str] = None
    pregnancy_status: Optional[str] = None
    additional_notes: Optional[str] = None
    skipped: Optional[bool] = None


class ConsentResponse(BaseModel):
    id: UUID
    patient_id: UUID
    consent_type: ConsentType
    accepted: bool
    accepted_at: datetime

    class Config:
        from_attributes = True


class ConsentCreate(BaseModel):
    consent_type: ConsentType
    accepted: bool