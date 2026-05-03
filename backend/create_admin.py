#!/usr/bin/env python3
"""
Script to create an initial admin user for Medora.
Run this once to bootstrap the system with an admin account.

Required environment variables:
    ADMIN_EMAIL    — email address for the admin account
    ADMIN_PASSWORD — password for the admin account
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Load .env from the backend directory so ADMIN_EMAIL / ADMIN_PASSWORD are available.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # dotenv not installed — fall back to environment variables already set

from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app.models import User, Hospital, DoctorProfile
from app.auth.jwt import get_password_hash


def create_admin():
    admin_email = os.environ.get("ADMIN_EMAIL")
    admin_password = os.environ.get("ADMIN_PASSWORD")

    if not admin_email or not admin_password:
        print("Error: ADMIN_EMAIL and ADMIN_PASSWORD environment variables must be set.")
        sys.exit(1)

    db: Session = SessionLocal()

    try:
        # Check if any admin exists
        existing_admin = db.query(User).filter(User.role == "admin").first()
        if existing_admin:
            print("Admin user already exists.")
            print(f"Email: {existing_admin.email}")
            print("If you forgot the password, use the reset command.")
            return

        # Create default hospital if none exists
        hospital = db.query(Hospital).first()
        if not hospital:
            hospital = Hospital(name="Medora General Hospital", address="123 Medical Center Dr")
            db.add(hospital)
            db.commit()
            db.refresh(hospital)
            print(f"Created hospital: {hospital.name}")

        # Create admin user
        hashed_password = get_password_hash(admin_password)
        admin_user = User(
            hospital_id=hospital.id,
            full_name="System Administrator",
            email=admin_email,
            phone="+1234567890",
            password_hash=hashed_password,
            role="admin",
            registration_method="admin_created",
            is_active=True
        )
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)

        print("Admin user created successfully!")
        print(f"Email: {admin_email}")

    except Exception as e:
        print(f"Error creating admin: {e}")
        db.rollback()
    finally:
        db.close()


def reset_admin_password():
    new_password = os.environ.get("ADMIN_PASSWORD")
    if not new_password:
        print("Error: ADMIN_PASSWORD environment variable must be set.")
        sys.exit(1)

    db: Session = SessionLocal()
    try:
        admin = db.query(User).filter(User.role == "admin").first()
        if not admin:
            print("No admin user found.")
            return

        admin.password_hash = get_password_hash(new_password)
        db.commit()
        print(f"Admin password reset for {admin.email}")

    except Exception as e:
        print(f"Error resetting password: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        reset_admin_password()
    else:
        create_admin()
