from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User, Hospital, PatientProfile
from ..schemas.auth import UserCreate, UserLogin, Token, UserResponse, ChangePassword
from ..auth.jwt import authenticate_user, create_access_token, create_refresh_token, get_password_hash, verify_password
from ..auth.dependencies import get_current_active_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    # For demo, use default hospital or create one
    hospital = db.query(Hospital).first()
    if not hospital:
        hospital = Hospital(name="Demo Hospital", address="123 Demo St")
        db.add(hospital)
        db.commit()
        db.refresh(hospital)
    
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(
        hospital_id=hospital.id,
        full_name=user.full_name,
        email=user.email,
        phone=user.phone,
        password_hash=hashed_password,
        role="patient",  # Default to patient for self-signup
        registration_method="self_signup"
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create patient profile
    patient_profile = PatientProfile(
        user_id=db_user.id,
        hospital_id=hospital.id
    )
    db.add(patient_profile)
    db.commit()
    
    return db_user


@router.post("/signin", response_model=Token)
def signin(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer", "refresh_token": refresh_token}


@router.get("/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@router.post("/refresh-token", response_model=Token)
def refresh_token(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/auth/signin"))):
    # Placeholder: implement refresh logic
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/logout")
def logout():
    # Placeholder: implement logout logic (e.g., token blacklist)
    return {"message": "Logged out"}


@router.patch("/change-password")
def change_password(password_data: ChangePassword, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    if not verify_password(password_data.old_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect old password")
    current_user.password_hash = get_password_hash(password_data.new_password)
    db.commit()
    return {"message": "Password changed"}