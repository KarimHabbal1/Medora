from fastapi import APIRouter, Depends, HTTPException
from typing import List
from sqlalchemy.orm import Session
from ..database import get_db
from ..auth.dependencies import get_current_admin_user
from ..models.user import User
from ..schemas.admin import UserCreateAdmin, UserUpdateAdmin
from ..schemas.auth import UserResponse

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/users", response_model=UserResponse)
def create_user(user: UserCreateAdmin, db: Session = Depends(get_db), current_user = Depends(get_current_admin_user)):
    # Similar to signup, but for admin
    from ..auth.jwt import get_password_hash
    hashed_password = get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password, full_name=user.full_name, is_admin=user.is_admin)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.get("/users", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db), current_user = Depends(get_current_admin_user)):
    users = db.query(User).all()
    return users


@router.patch("/users/{user_id}", response_model=UserResponse)
def update_user(user_id: int, update_data: UserUpdateAdmin, db: Session = Depends(get_db), current_user = Depends(get_current_admin_user)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    for key, value in update_data.dict(exclude_unset=True).items():
        setattr(user, key, value)
    db.commit()
    db.refresh(user)
    return user