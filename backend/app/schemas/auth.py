from pydantic import BaseModel, EmailStr
from typing import Optional
from uuid import UUID
from datetime import datetime
from .enums import UserRole, RegistrationMethod


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone: Optional[str] = None
    # hospital_id removed for demo self-signup


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    email: Optional[str] = None


class UserResponse(BaseModel):
    id: UUID
    hospital_id: UUID
    full_name: str
    email: EmailStr
    phone: Optional[str]
    role: UserRole
    is_active: bool
    registration_method: RegistrationMethod
    created_at: datetime

    class Config:
        from_attributes = True


class ChangePassword(BaseModel):
    old_password: str
    new_password: str