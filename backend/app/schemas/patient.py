from pydantic import BaseModel
from typing import List, Optional


class PatientUpdate(BaseModel):
    full_name: Optional[str] = None
    # Add other fields as needed, but keep flexible


class MedicalHistory(BaseModel):
    conditions: List[str]
    medications: List[str]
    allergies: List[str]


class Consent(BaseModel):
    type: str
    granted: bool
    granted_at: Optional[str] = None