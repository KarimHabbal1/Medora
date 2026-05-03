from pydantic import BaseModel, EmailStr
from typing import Optional
from uuid import UUID
from datetime import datetime
from .enums import UserRole


class UserCreateAdmin(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone: Optional[str] = None
    role: UserRole
    hospital_id: UUID


class UserResponseAdmin(BaseModel):
    id: UUID
    hospital_id: UUID
    full_name: str
    email: EmailStr
    phone: Optional[str]
    role: UserRole
    is_active: bool
    registration_method: str
    created_at: datetime
    assigned_doctor_id: Optional[UUID] = None
    assigned_doctor_name: Optional[str] = None

    class Config:
        from_attributes = True


class UserUpdateAdmin(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class HospitalResponse(BaseModel):
    id: UUID
    name: str

    class Config:
        from_attributes = True


class AssignDoctorRequest(BaseModel):
    patient_id: UUID
    doctor_id: UUID