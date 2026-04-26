from fastapi import APIRouter, Depends, HTTPException
from typing import List
from ..auth.dependencies import get_current_active_user
from ..models.user import User
from ..schemas.triage import TriageSessionCreate, TriageSession, MessageCreate, Message, Report
from ..utils.mocks import mock_sessions, mock_messages, mock_reports

router = APIRouter(prefix="/triage", tags=["triage"])


@router.post("/sessions", response_model=TriageSession)
def create_session(session: TriageSessionCreate, current_user: User = Depends(get_current_active_user)):
    # Mock: create new session
    new_session = TriageSession(id="new-session", patient_id=current_user.id, status="active", created_at="2023-10-01T10:00:00Z")
    return new_session


@router.get("/sessions", response_model=List[TriageSession])
def get_sessions(current_user: User = Depends(get_current_active_user)):
    # Mock data
    return mock_sessions


@router.get("/sessions/{session_id}", response_model=TriageSession)
def get_session(session_id: str, current_user: User = Depends(get_current_active_user)):
    # Mock: return session
    session = next((s for s in mock_sessions if s.id == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/sessions/{session_id}/message", response_model=Message)
def send_message(session_id: str, message: MessageCreate, current_user: User = Depends(get_current_active_user)):
    # Mock: add message
    new_msg = Message(id="new-msg", session_id=session_id, content=message.content, role=message.role, timestamp="2023-10-01T10:05:00Z")
    return new_msg


@router.get("/sessions/{session_id}/messages", response_model=List[Message])
def get_messages(session_id: str, current_user: User = Depends(get_current_active_user)):
    # Mock data
    return mock_messages.get(session_id, [])


@router.post("/sessions/{session_id}/end")
def end_session(session_id: str, current_user: User = Depends(get_current_active_user)):
    # Placeholder: end session
    return {"message": "Session ended"}


@router.get("/sessions/{session_id}/report", response_model=Report)
def get_report(session_id: str, current_user: User = Depends(get_current_active_user)):
    # Mock data
    report = mock_reports.get(session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report