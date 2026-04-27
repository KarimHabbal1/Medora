from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict, Any
from ..schemas.triage import MessageResponse
from ..schemas.enums import MessageSender, MessageType


def process_rag_query(session_id: str, query: str) -> MessageResponse:
    """
    Mock RAG agent service.
    In real implementation, this would query the vector store and generate a response.
    """
    return MessageResponse(
        id=uuid4(),
        session_id=session_id,
        sender=MessageSender.rag_agent,
        content=f"Based on medical knowledge, for '{query}', the recommended approach is to monitor symptoms and consult a healthcare provider.",
        message_type=MessageType.answer,
        is_persisted_after_summary=True,
        is_visible_to_doctor=True,
        is_deleted=False,
        created_at=datetime.now(timezone.utc)
    )