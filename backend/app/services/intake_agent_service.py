"""
Intake agent service — bridges the real IntakeSession (via session_manager)
with the FastAPI request/response cycle.

This replaces the previous mock implementation.
Agent logic is NOT modified; this module only wraps agent calls.
"""

from datetime import datetime, timezone
from uuid import uuid4, UUID
from ..schemas.triage import MessageResponse
from ..schemas.enums import MessageSender, MessageType
from .session_manager import AgentSessionManager


async def process_patient_message(session_id: str, message: str) -> MessageResponse:
    """
    Process a patient message through the agent pipeline.

    The session_manager handles phase routing (intake → triage Mode A/B → complete)
    and ensures diagnosis data is NEVER returned in the patient-facing response.
    """
    manager = AgentSessionManager.get_instance()
    result = await manager.process_message(session_id, message)

    # Map agent sender string to the MessageSender enum
    sender_map = {
        "intake_agent": MessageSender.intake_agent,
        "triage_agent": MessageSender.triage_agent,
        "rag_agent": MessageSender.rag_agent,
        "system": MessageSender.system,
    }
    sender = sender_map.get(result["sender"], MessageSender.system)

    # Map message type string to MessageType enum
    type_map = {
        "question": MessageType.question,
        "answer": MessageType.answer,
        "warning": MessageType.warning,
        "escalation": MessageType.escalation,
        "text": MessageType.text,
    }
    msg_type = type_map.get(result["message_type"], MessageType.text)

    return MessageResponse(
        id=uuid4(),
        session_id=UUID(session_id),
        sender=sender,
        content=result["response_text"],
        message_type=msg_type,
        is_persisted_after_summary=True,
        is_visible_to_doctor=True,
        is_deleted=False,
        created_at=datetime.now(timezone.utc),
    )


async def get_process_result(session_id: str, message: str) -> dict:
    """
    Return the raw processing result dict from the session manager.

    Used by the router to access phase transitions, intake_summary, and diagnosis
    for database persistence — these are NEVER sent to the patient.
    """
    manager = AgentSessionManager.get_instance()
    return await manager.process_message(session_id, message)