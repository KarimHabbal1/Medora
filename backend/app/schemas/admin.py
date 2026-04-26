from pydantic import BaseModel, EmailStr
from typing import Optional


class UserCreateAdmin(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    is_admin: bool = False


class UserUpdateAdmin(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None