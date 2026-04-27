from fastapi import APIRouter, Depends, HTTPException
from typing import List
from sqlalchemy.orm import Session
from ..database import get_db
from ..auth.dependencies import get_current_admin_user
from ..models import User, Hospital, DoctorProfile, PatientProfile
from ..schemas.admin import UserCreateAdmin, UserResponseAdmin, UserUpdateAdmin
from ..auth.jwt import get_password_hash

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/users", response_model=UserResponseAdmin)
def create_user(user: UserCreateAdmin, db: Session = Depends(get_db), current_user: User = Depends(get_current_admin_user)):
    # Check hospital
    hospital = db.query(Hospital).filter(Hospital.id == user.hospital_id).first()
    if not hospital:
        raise HTTPException(status_code=400, detail="Invalid hospital ID")
    
    # Check email unique
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        hospital_id=user.hospital_id,
        full_name=user.full_name,
        email=user.email,
        phone=user.phone,
        password_hash=hashed_password,
        role=user.role,
        registration_method="admin_created",
        created_by_admin_id=current_user.id
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # If doctor or patient, create profile
    if user.role == "doctor":
        doctor_profile = DoctorProfile(
            user_id=db_user.id,
            hospital_id=user.hospital_id,
            specialty="General"  # Default
        )
        db.add(doctor_profile)
    elif user.role == "patient":
        patient_profile = PatientProfile(
            user_id=db_user.id,
            hospital_id=user.hospital_id
        )
        db.add(patient_profile)
    
    db.commit()
    return db_user


@router.get("/users", response_model=List[UserResponseAdmin])
def get_users(db: Session = Depends(get_db), current_user: User = Depends(get_current_admin_user)):
    users = db.query(User).all()
    return users


@router.patch("/users/{user_id}", response_model=UserResponseAdmin)
def update_user(user_id: str, update_data: UserUpdateAdmin, db: Session = Depends(get_db), current_user: User = Depends(get_current_admin_user)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    for key, value in update_data.model_dump(exclude_unset=True).items():
        setattr(user, key, value)
    
    db.commit()
    db.refresh(user)
    return user