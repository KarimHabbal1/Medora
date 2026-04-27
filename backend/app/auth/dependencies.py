from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from .jwt import verify_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/signin")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    email = verify_token(token, credentials_exception)
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def get_current_admin_user(current_user: User = Depends(get_current_active_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user


def get_current_doctor_user(current_user: User = Depends(get_current_active_user)):
    if current_user.role not in ["doctor", "admin"]:
        raise HTTPException(status_code=403, detail="Doctor access required")
    return current_user


def get_current_patient_user(current_user: User = Depends(get_current_active_user)):
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Patient access required")
    return current_user