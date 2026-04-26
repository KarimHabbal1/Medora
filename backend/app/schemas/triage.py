from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID


class TriageSessionCreate(BaseModel):
    # Placeholder, add fields if needed
    pass


class TriageSession(BaseModel):
    id: str  # UUID as string
    patient_id: int
    status: str
    created_at: str


class MessageCreate(BaseModel):
    content: str
    role: str = "user"  # user or assistant


class Message(BaseModel):
    id: str
    session_id: str
    content: str
    role: str
    timestamp: str


class Report(BaseModel):
    id: str
    session_id: str
    summary: str
    recommendations: List[str]
    urgency_level: str