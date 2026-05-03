"""
In-memory agent session manager.

Manages the lifecycle of IntakeSession and TriageSession objects, mapping
each database TriageSession UUID to a live agent state. Provides TTL-based
cleanup and singleton model pre-loading for the heavyweight triage models.

IMPORTANT: This module wraps the agents — it does NOT modify their logic.
"""

import asyncio
import sys
import os
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

# Add the project root so that `agents` package is importable.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)

# Default TTL for idle sessions (30 minutes).
SESSION_TTL_MINUTES = 30

# Lazy-loaded agent modules — avoids crashing the backend if agent data files
# (e.g. tmt_symptoms_gpt4o.json) are not yet in place.
_IntakeSession = None
_AgentTriageSession = None
_PatientMemory = None
_FeedbackStore = None


def _lazy_import_agents():
    """Import agent classes on first use."""
    global _IntakeSession, _AgentTriageSession, _PatientMemory, _FeedbackStore
    if _IntakeSession is None:
        from agents.intake_agent import IntakeSession
        from agents.triage_agent import TriageSession as AgentTriageSession
        from agents.patient_memory import PatientMemory
        from agents.feedback_store import FeedbackStore
        _IntakeSession = IntakeSession
        _AgentTriageSession = AgentTriageSession
        _PatientMemory = PatientMemory
        _FeedbackStore = FeedbackStore


@dataclass
class AgentSessionState:
    """Holds in-memory agent objects for a single triage session."""
    intake: Any = None
    triage: Any = None
    phase: str = "intake"            # intake | triage_mode_a | triage_mode_b | escalated | completed
    intake_started: bool = False
    triage_started: bool = False
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    intake_summary: Optional[dict] = None


class AgentSessionManager:
    """
    Singleton that manages all live agent sessions.

    Usage:
        manager = AgentSessionManager.get_instance()
        state = manager.create_session(session_id, patient_context="...")
        response = manager.process_message(session_id, message)
    """

    _instance: Optional["AgentSessionManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._sessions: Dict[str, AgentSessionState] = {}
        self._sessions_lock = threading.Lock()
        self._patient_memory = None
        self._feedback_store = None
        self._agents_loaded = False
        self._cleanup_running = False
        logger.info("AgentSessionManager initialised")

    @classmethod
    def get_instance(cls) -> "AgentSessionManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        patient_context: str = "",
        provider: str = "openai",
        ollama_url: str = "http://localhost:11434",
    ) -> AgentSessionState:
        """Create a new agent session with an IntakeSession ready to start."""
        _lazy_import_agents()
        intake = _IntakeSession(
            patient_context=patient_context,
            provider=provider,
            ollama_url=ollama_url,
        )
        state = AgentSessionState(intake=intake)
        with self._sessions_lock:
            self._sessions[session_id] = state
        logger.info("Created agent session %s", session_id)
        return state

    def get_session(self, session_id: str) -> Optional[AgentSessionState]:
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state:
            state.last_active = datetime.now(timezone.utc)
        return state

    def remove_session(self, session_id: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(session_id, None)
        logger.info("Removed agent session %s", session_id)

    def preload_triage_models(self, provider: str = "openai", ollama_url: str = "http://localhost:11434") -> None:
        """Preload the heavyweight Triage Agent models so the first user request is not delayed."""
        if self._agents_loaded:
            return

        _lazy_import_agents()
        try:
            logger.info("Preloading Triage Agent models before first patient request...")
            _AgentTriageSession(provider=provider, ollama_url=ollama_url)
            self._agents_loaded = True
            logger.info("Triage Agent models preloaded successfully.")
        except Exception:
            logger.exception("Failed to preload Triage Agent models")

    # ------------------------------------------------------------------
    # Message processing (core orchestration)
    # ------------------------------------------------------------------

    async def process_message(self, session_id: str, message: str) -> dict:
        """
        Process a patient message through the correct agent phase.

        Returns a dict with:
            - response_text: str  (the sanitised patient-facing message)
            - sender: str         (intake_agent | triage_agent | system)
            - message_type: str   (question | answer | warning | escalation)
            - phase: str          (current phase after processing)
            - phase_changed: bool
            - intake_complete: bool
            - triage_complete: bool
            - intake_summary: dict | None
            - diagnosis: dict | None
        """
        state = self.get_session(session_id)
        if state is None:
            return {
                "response_text": "Session not found. Please start a new session.",
                "sender": "system",
                "message_type": "warning",
                "phase": "completed",
                "phase_changed": False,
                "intake_complete": False,
                "triage_complete": False,
                "intake_summary": None,
                "diagnosis": None,
            }

        old_phase = state.phase
        result = {
            "response_text": "",
            "sender": "intake_agent",
            "message_type": "question",
            "phase": state.phase,
            "phase_changed": False,
            "intake_complete": False,
            "triage_complete": False,
            "intake_summary": None,
            "diagnosis": None,
        }

        try:
            if state.phase == "intake":
                result = await self._handle_intake(state, message, result)
            elif state.phase == "triage_mode_b":
                result = await self._handle_triage_mode_b(state, message, result)
            elif state.phase == "triage_mode_a":
                result = await self._handle_triage_mode_a(state, message, result)
            elif state.phase == "escalated":
                result["response_text"] = (
                    "This session has been escalated. Please follow the emergency instructions provided earlier."
                )
                result["sender"] = "system"
                result["message_type"] = "warning"
            elif state.phase == "completed":
                result["response_text"] = (
                    "This session is complete. Your report has been sent to your doctor."
                )
                result["sender"] = "system"
                result["message_type"] = "answer"
        except Exception as exc:
            logger.exception("Error processing message for session %s", session_id)
            result["response_text"] = (
                "I apologise, an error occurred while processing your message. "
                "Please try again or start a new session."
            )
            result["sender"] = "system"
            result["message_type"] = "warning"

        result["phase"] = state.phase
        result["phase_changed"] = state.phase != old_phase
        return result

    # ------------------------------------------------------------------
    # Phase handlers — wrap agent calls without modifying agent logic
    # ------------------------------------------------------------------

    async def _handle_intake(self, state: AgentSessionState, message: str, result: dict) -> dict:
        if not state.intake_started:
            response_text = await asyncio.to_thread(state.intake.start, message)
            state.intake_started = True
        else:
            response_text = await asyncio.to_thread(state.intake.respond, message)

        if state.intake.is_complete():
            result["intake_complete"] = True
            state.intake_summary = state.intake.get_summary()
            result["intake_summary"] = state.intake_summary

            summary = state.intake_summary or {}
            if summary.get("escalated"):
                # Emergency escalation — pass the agent's escalation message through
                state.phase = "escalated"
                result["response_text"] = response_text
                result["sender"] = "system"
                result["message_type"] = "escalation"
                return result

            if state.intake.is_uncommon():
                # Uncommon symptom → Triage Mode B (conversational)
                state.phase = "triage_mode_b"
                triage = _AgentTriageSession()
                raw_complaint = state.intake.get_raw_complaint()
                triage_response = await asyncio.to_thread(triage.start_uncommon, raw_complaint)
                state.triage = triage
                state.triage_started = True
                # Patient sees: confirmation + first triage question
                result["response_text"] = (
                    "Thank you for describing your symptoms. "
                    "I need to ask you a few more clinical questions to help your doctor.\n\n"
                    + triage_response
                )
                result["sender"] = "triage_agent"
                result["message_type"] = "question"
                return result
            else:
                # Common symptom → Triage Mode A (structured handoff)
                state.phase = "triage_mode_a"
                triage = _AgentTriageSession()
                triage_result = await asyncio.to_thread(triage.diagnose_from_intake, state.intake_summary)
                state.triage = triage
                state.triage_started = True

                if isinstance(triage_result, str):
                    # Triage needs follow-up questions
                    result["response_text"] = (
                        "Thank you. Your information has been reviewed. "
                        "I have a few follow-up questions.\n\n"
                        + triage_result
                    )
                    result["sender"] = "triage_agent"
                    result["message_type"] = "question"
                    return result
                else:
                    # Diagnosis complete in one pass — do NOT show to patient
                    state.phase = "completed"
                    result["response_text"] = (
                        "Thank you. Your assessment is complete and your report "
                        "has been sent to your doctor for review."
                    )
                    result["sender"] = "system"
                    result["message_type"] = "answer"
                    result["triage_complete"] = True
                    result["diagnosis"] = triage_result
                    result["intake_summary"] = state.intake_summary
                    return result
        else:
            # Intake still in progress — return the agent's question
            result["response_text"] = response_text
            result["sender"] = "intake_agent"
            result["message_type"] = "question"
            return result

    async def _handle_triage_mode_b(self, state: AgentSessionState, message: str, result: dict) -> dict:
        response_text = await asyncio.to_thread(state.triage.respond, message)

        if state.triage.is_complete():
            state.phase = "completed"
            result["response_text"] = (
                "Thank you. Your assessment is complete and your report "
                "has been sent to your doctor for review."
            )
            result["sender"] = "system"
            result["message_type"] = "answer"
            result["triage_complete"] = True
            result["diagnosis"] = state.triage.get_diagnosis()
            result["intake_summary"] = state.intake_summary
        else:
            result["response_text"] = response_text
            result["sender"] = "triage_agent"
            result["message_type"] = "question"
        return result

    async def _handle_triage_mode_a(self, state: AgentSessionState, message: str, result: dict) -> dict:
        triage_result = await asyncio.to_thread(state.triage.respond_followup, message)

        if state.triage.is_complete():
            state.phase = "completed"
            result["response_text"] = (
                "Thank you. Your assessment is complete and your report "
                "has been sent to your doctor for review."
            )
            result["sender"] = "system"
            result["message_type"] = "answer"
            result["triage_complete"] = True
            result["diagnosis"] = state.triage.get_diagnosis()
            result["intake_summary"] = state.intake_summary
        elif isinstance(triage_result, str):
            result["response_text"] = triage_result
            result["sender"] = "triage_agent"
            result["message_type"] = "question"
        else:
            # Diagnosis complete (dict returned)
            state.phase = "completed"
            result["response_text"] = (
                "Thank you. Your assessment is complete and your report "
                "has been sent to your doctor for review."
            )
            result["sender"] = "system"
            result["message_type"] = "answer"
            result["triage_complete"] = True
            result["diagnosis"] = triage_result
            result["intake_summary"] = state.intake_summary
        return result

    # ------------------------------------------------------------------
    # PatientMemory bridge
    # ------------------------------------------------------------------

    def get_patient_context(self, patient_name: str) -> str:
        """Retrieve context from PatientMemory for a returning patient."""
        try:
            return self.patient_memory.get_context_for_intake(patient_name)
        except Exception:
            logger.exception("Failed to get patient context for %s", patient_name)
            return ""

    def update_patient_memory(
        self,
        patient_name: str,
        intake_summary: dict,
        diagnosis: Optional[dict] = None,
    ) -> None:
        """Update PatientMemory after session completion."""
        try:
            self.patient_memory.update_from_session(
                patient_name=patient_name,
                intake_summary=intake_summary,
                diagnosis=diagnosis,
            )
            logger.info("Updated PatientMemory for %s", patient_name)
        except Exception:
            logger.exception("Failed to update PatientMemory for %s", patient_name)

    # ------------------------------------------------------------------
    # FeedbackStore bridge
    # ------------------------------------------------------------------

    def save_feedback_case(
        self,
        patient_name: str,
        intake_summary: dict,
        diagnosis: dict,
    ) -> Optional[str]:
        """Save a completed case to FeedbackStore for doctor review."""
        try:
            symptoms = intake_summary.get("symptoms", [])
            urgency = intake_summary.get("urgency", "unknown")
            clinical_picture = intake_summary.get("clinical_picture", {})
            case_id = self.feedback_store.save_case(
                patient_name=patient_name,
                symptoms=symptoms,
                urgency=urgency,
                intake_summary=intake_summary,
                clinical_picture=clinical_picture,
                diagnosis_report=diagnosis,
                retrieved_chunks=diagnosis.get("chunks", []),
            )
            logger.info("Saved feedback case %s for %s", case_id, patient_name)
            return case_id
        except Exception:
            logger.exception("Failed to save feedback case for %s", patient_name)
            return None

    @property
    def feedback_store(self):
        _lazy_import_agents()
        if self._feedback_store is None:
            self._feedback_store = _FeedbackStore()
        return self._feedback_store

    @property
    def patient_memory(self):
        _lazy_import_agents()
        if self._patient_memory is None:
            self._patient_memory = _PatientMemory()
        return self._patient_memory

    # ------------------------------------------------------------------
    # TTL cleanup
    # ------------------------------------------------------------------

    def cleanup_expired_sessions(self) -> int:
        """Remove sessions that have been idle beyond the TTL. Returns count removed."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_TTL_MINUTES)
        expired = []
        with self._sessions_lock:
            for sid, state in self._sessions.items():
                if state.last_active < cutoff:
                    expired.append(sid)
            for sid in expired:
                del self._sessions[sid]
        if expired:
            logger.info("Cleaned up %d expired agent sessions", len(expired))
        return len(expired)

    @property
    def active_session_count(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)
