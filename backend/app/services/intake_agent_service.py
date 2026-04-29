from datetime import datetime, timezone
from uuid import uuid4, UUID
from ..schemas.triage import MessageResponse
from ..schemas.enums import MessageSender, MessageType


def process_patient_message(session_id: str, message: str) -> MessageResponse:
    """
    Mock intake agent service.
    In real implementation, this would process the patient message and generate a response.
    """
    return MessageResponse(
        id=uuid4(),
        session_id=UUID(session_id),
        sender=MessageSender.intake_agent,
        content=f"Thank you for sharing that. Based on your message '{message}', I recommend seeing a doctor soon. Can you tell me more about your symptoms?",
        message_type=MessageType.answer,
        is_persisted_after_summary=True,
        is_visible_to_doctor=True,
        is_deleted=False,
        created_at=datetime.now(timezone.utc)
    )