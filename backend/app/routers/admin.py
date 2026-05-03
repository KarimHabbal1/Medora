from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_
from typing import List
from sqlalchemy.orm import Session
from ..database import get_db
from ..auth.dependencies import get_current_admin_user
from ..models import User, Hospital, DoctorProfile, PatientProfile
from ..models.enums import UserRole
from ..schemas.admin import UserCreateAdmin, UserResponseAdmin, UserUpdateAdmin, HospitalResponse, AssignDoctorRequest
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
    db.refresh(db_user)
    return {
        "id": db_user.id,
        "hospital_id": db_user.hospital_id,
        "full_name": db_user.full_name,
        "email": db_user.email,
        "phone": db_user.phone,
        "role": db_user.role,
        "is_active": db_user.is_active,
        "registration_method": db_user.registration_method,
        "created_at": db_user.created_at,
        "assigned_doctor_id": None,
        "assigned_doctor_name": None,
    }


@router.get("/users", response_model=List[UserResponseAdmin])
def get_users(db: Session = Depends(get_db), current_user: User = Depends(get_current_admin_user)):
    users = db.query(User).all()
    result = []
    for user in users:
        assigned_doctor_id = None
        assigned_doctor_name = None

        if user.role == UserRole.patient and user.patient_profile and user.patient_profile.assigned_doctor:
            assigned_doctor = user.patient_profile.assigned_doctor
            assigned_doctor_id = str(assigned_doctor.user_id)  # User.id — matches dropdown option values
            assigned_doctor_name = assigned_doctor.user.full_name

        result.append({
            "id": user.id,
            "hospital_id": user.hospital_id,
            "full_name": user.full_name,
            "email": user.email,
            "phone": user.phone,
            "role": user.role,
            "is_active": user.is_active,
            "registration_method": user.registration_method,
            "created_at": user.created_at,
            "assigned_doctor_id": assigned_doctor_id,
            "assigned_doctor_name": assigned_doctor_name,
        })
    return result


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


@router.get("/hospitals", response_model=List[HospitalResponse])
def get_hospitals(db: Session = Depends(get_db), current_user: User = Depends(get_current_admin_user)):
    return db.query(Hospital).order_by(Hospital.name).all()


@router.post("/assign-doctor")
def assign_doctor_to_patient(request: AssignDoctorRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_admin_user)):
    patient = db.query(PatientProfile).filter(
        or_(
            PatientProfile.id == request.patient_id,
            PatientProfile.user_id == request.patient_id,
        )
    ).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    doctor = db.query(DoctorProfile).filter(
        or_(
            DoctorProfile.id == request.doctor_id,
            DoctorProfile.user_id == request.doctor_id,
        )
    ).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    patient.assigned_doctor_id = doctor.id
    db.commit()
    return {"message": "Doctor assigned successfully"}