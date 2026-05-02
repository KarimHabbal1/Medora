"""
Phase 6 — Patient Memory System
================================
Manages persistent patient profiles stored as JSON files.

After each session, clinically relevant information is extracted from the
intake summary and diagnosis, then merged into the patient's profile using
an LLM. On subsequent sessions the profile is loaded and formatted as context
strings that are injected into the Intake and Triage agents.

Usage (library):
    from agents.patient_memory import PatientMemory

    memory = PatientMemory()
    profile = memory.get_or_create("Karim Habbal")
    intake_ctx = memory.get_context_for_intake("Karim Habbal")
    triage_ctx = memory.get_context_for_triage("Karim Habbal")

    # After a session:
    profile = memory.update_from_session(
        "Karim Habbal", intake_summary, diagnosis, llm
    )

Usage (CLI):
    python agents/patient_memory.py --patient "Karim Habbal" --show
    python agents/patient_memory.py --patient "Karim Habbal" --simulate

Design notes:
  - Profiles are stored as JSON files under data/patient_profiles/
  - Patient names are normalised (lowercase, strip, spaces → underscores) for
    file naming, but the display name is preserved inside the profile.
  - No database — JSON files only (Phase 6 scope).
  - The profile is NOT editable by the patient (write path is server-side only).
  - Thread-safety: a lightweight per-file lock is used so that two simultaneous
    session-update calls for the same patient do not clobber each other. This is
    sufficient for a single-server deployment; a distributed lock would be
    needed for horizontal scaling.
"""

import argparse
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

# ── Project root on sys.path so config.py is importable ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Environment ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_STORAGE_DIR = PROJECT_ROOT / "data" / "patient_profiles"

_BLANK_PROFILE_TEMPLATE: dict = {
    "patient_name": "",
    "created": "",
    "last_updated": "",
    "sessions": 0,
    "demographics": {
        "age": None,
        "gender": None,
    },
    "known_conditions": [],
    "medications": [],
    "allergies": [],
    "smoking_history": {},
    "substance_use": {},
    "family_history": [],
    "surgical_history": [],
    "session_history": [],
}

# System prompt used when merging a new session into an existing profile.
_UPDATE_SYSTEM_PROMPT = """\
You are a medical records assistant. Given a patient's existing profile and new
session data (intake summary + diagnosis), extract any NEW clinical information
that should be added to the patient's permanent record.

Rules:
- Only add information that is NEW — do not duplicate existing entries.
- If existing information has changed (e.g. the patient quit smoking since the
  last visit), UPDATE the existing entry rather than duplicating it.
- Extract demographics (age, gender) if mentioned anywhere in the session data.
- Extract conditions, medications, allergies, family history, surgical history.
- Summarise the session in one sentence for the session_history entry
  (field: "key_findings").
- The "source_session" field on any new entry must be set to the session number
  provided in the input.

Return a JSON object with ONLY the fields that need updating or extending.
Valid top-level keys:
  demographics, known_conditions, medications, allergies, smoking_history,
  substance_use, family_history, surgical_history

For list fields (known_conditions, medications, allergies, family_history,
surgical_history), return ONLY the NEW items to append — not the full list.

For object fields (demographics, smoking_history, substance_use), return the
full updated object if anything changed, otherwise omit the key entirely.

Also include a "session_summary" key with a one-sentence summary of this
session's key findings.

Return ONLY valid JSON. No explanation, no markdown fences.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_name(name: str) -> str:
    """Normalise a patient name to a safe filename stem.

    Examples:
        "Karim Habbal" → "karim_habbal"
        "  Jane  Doe  " → "jane_doe"
        "María José" → "maría_josé"   (non-ASCII preserved, only spaces replaced)
    """
    return name.strip().lower().replace(" ", "_")


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _strip_fence(text: str) -> str:
    """Remove markdown code fences from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        inner = parts[1] if len(parts) > 1 else text
        if inner.startswith("json"):
            inner = inner[4:]
        return inner.strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# PatientMemory
# ─────────────────────────────────────────────────────────────────────────────

class PatientMemory:
    """Manages persistent patient profiles stored as JSON files.

    Each patient maps to one JSON file under `storage_dir`.  The class is
    intentionally stateless between calls — every public method reads from and
    (when writing) writes back to disk so that multiple processes or threads
    sharing the same storage directory stay consistent.
    """

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._storage_dir: Path = Path(storage_dir) if storage_dir else _DEFAULT_STORAGE_DIR
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Per-filename threading locks so concurrent updates for the same
        # patient are serialised without blocking unrelated patients.
        self._locks: dict[str, threading.Lock] = {}
        self._locks_mutex = threading.Lock()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _profile_path(self, patient_name: str) -> Path:
        filename = _normalise_name(patient_name) + ".json"
        return self._storage_dir / filename

    def _get_lock(self, patient_name: str) -> threading.Lock:
        key = _normalise_name(patient_name)
        with self._locks_mutex:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _load_raw(self, path: Path) -> dict:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _save_raw(self, path: Path, profile: dict) -> None:
        # Write atomically: write to a temp file then rename so a crash mid-write
        # does not corrupt the existing profile.
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, ensure_ascii=False)
        tmp_path.replace(path)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_or_create(self, patient_name: str) -> dict:
        """Load an existing patient profile, or create a blank one.

        The returned dict is a deep copy — callers may mutate it freely without
        affecting the stored profile.  To persist changes, call one of the
        `update_*` methods.

        Args:
            patient_name: Display name of the patient (any capitalisation).

        Returns:
            The patient profile dict.
        """
        path = self._profile_path(patient_name)
        lock = self._get_lock(patient_name)

        with lock:
            if path.exists():
                return self._load_raw(path)

            # Create a fresh profile
            now = _now_iso()
            profile = json.loads(json.dumps(_BLANK_PROFILE_TEMPLATE))  # deep copy
            profile["patient_name"] = patient_name.strip()
            profile["created"] = now
            profile["last_updated"] = now
            self._save_raw(path, profile)
            return profile

    def update_from_session(
        self,
        patient_name: str,
        intake_summary: dict,
        diagnosis: dict,
        llm,
    ) -> dict:
        """Merge new session data into the patient's persistent profile.

        Uses the LLM to extract any NEW clinical information (conditions,
        medications, allergies, demographics, smoking history, etc.) from the
        session and appends it to the profile.  Also records a summary of the
        session in `session_history`.

        Args:
            patient_name:   Display name of the patient.
            intake_summary: The dict returned by IntakeSession.get_summary().
            diagnosis:      The dict returned by TriageSession.get_diagnosis().
            llm:            An LLM instance to use for extraction (OpenAI or Ollama).

        Returns:
            The updated profile dict (also persisted to disk).
        """
        path = self._profile_path(patient_name)
        lock = self._get_lock(patient_name)

        with lock:
            # Load (or create) current profile
            if path.exists():
                profile = self._load_raw(path)
            else:
                profile = self.get_or_create(patient_name)

            current_session_number = profile["sessions"] + 1

            # ── Build the LLM input ───────────────────────────────────────────
            existing_summary = {
                "demographics": profile.get("demographics", {}),
                "known_conditions": profile.get("known_conditions", []),
                "medications": profile.get("medications", []),
                "allergies": profile.get("allergies", []),
                "smoking_history": profile.get("smoking_history", {}),
                "substance_use": profile.get("substance_use", {}),
                "family_history": profile.get("family_history", []),
                "surgical_history": profile.get("surgical_history", []),
            }

            user_content = (
                f"Session number: {current_session_number}\n\n"
                f"Existing patient profile (clinical sections only):\n"
                f"{json.dumps(existing_summary, indent=2)}\n\n"
                f"New intake summary:\n"
                f"{json.dumps(intake_summary, indent=2)}\n\n"
                f"New diagnosis:\n"
                f"{json.dumps(diagnosis, indent=2)}"
            )

            response = llm.invoke([
                SystemMessage(content=_UPDATE_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])

            raw = _strip_fence(response.content)
            try:
                updates = json.loads(raw)
                if not isinstance(updates, dict):
                    updates = {}
            except (json.JSONDecodeError, ValueError):
                updates = {}

            # ── Apply updates to the profile ──────────────────────────────────

            # Demographics — merge key by key (only overwrite null/missing fields)
            if "demographics" in updates and isinstance(updates["demographics"], dict):
                existing_demo = profile.setdefault("demographics", {})
                for k, v in updates["demographics"].items():
                    # Only set if the existing value is null/empty
                    if not existing_demo.get(k):
                        existing_demo[k] = v

            # List fields — append only NEW items
            for field in ("known_conditions", "medications", "allergies",
                          "family_history", "surgical_history"):
                new_items = updates.get(field)
                if new_items and isinstance(new_items, list):
                    profile.setdefault(field, []).extend(new_items)

            # Object fields — replace entirely if present in updates
            for field in ("smoking_history", "substance_use"):
                if field in updates and isinstance(updates[field], dict) and updates[field]:
                    profile[field] = updates[field]

            # ── Build session_history entry ────────────────────────────────────
            symptoms = intake_summary.get("symptoms", [])
            urgency = intake_summary.get("urgency", "routine")
            primary_diagnosis = _extract_primary_diagnosis(diagnosis.get("report", ""))
            key_findings = updates.get("session_summary", "")

            session_entry = {
                "session_number": current_session_number,
                "date": _now_iso(),
                "symptoms": symptoms,
                "urgency": urgency,
                "diagnosis": primary_diagnosis,
                "key_findings": key_findings,
                "outcome": None,  # to be filled by the treating clinician (Phase 7)
            }

            profile.setdefault("session_history", []).append(session_entry)

            # ── Bookkeeping ────────────────────────────────────────────────────
            profile["sessions"] = current_session_number
            profile["last_updated"] = _now_iso()

            self._save_raw(path, profile)
            return profile

    def get_context_for_intake(self, patient_name: str) -> str:
        """Format the patient's history as a context string for the Intake Agent.

        This string is designed to be injected at the start of an intake session
        so the agent can skip questions the patient has already answered in prior
        sessions and can reference known history naturally.

        Returns an empty string if the patient has no prior sessions.
        """
        path = self._profile_path(patient_name)
        if not path.exists():
            return ""

        profile = self._load_raw(path)
        if not profile.get("sessions", 0):
            return ""

        lines: list[str] = ["Known patient history:"]

        # Conditions (handles both dict and string formats)
        conditions = profile.get("known_conditions", [])
        if conditions:
            cond_parts = []
            for c in conditions:
                if isinstance(c, dict):
                    part = c.get("condition", "")
                    if c.get("since"):
                        part += f" (since {c['since']})"
                else:
                    part = str(c)
                cond_parts.append(part)
            lines.append(f"- Conditions: {', '.join(cond_parts)}")

        # Medications (handles both dict and string formats)
        medications = profile.get("medications", [])
        if medications:
            med_parts = []
            for m in medications:
                if isinstance(m, dict):
                    part = m.get("medication", "")
                    if m.get("for"):
                        part += f" (for {m['for']})"
                else:
                    part = str(m)
                med_parts.append(part)
            lines.append(f"- Medications: {', '.join(med_parts)}")

        # Allergies
        allergies = profile.get("allergies", [])
        if allergies:
            allergy_parts = []
            for a in allergies:
                if isinstance(a, dict):
                    allergy_parts.append(a.get("allergen") or a.get("allergy") or str(a))
                else:
                    allergy_parts.append(str(a))
            lines.append(f"- Allergies: {', '.join(allergy_parts)}")

        # Smoking
        smoking = profile.get("smoking_history", {})
        if smoking and (smoking.get("status") or smoking.get("duration") or smoking.get("quantity")):
            smoke_str = _format_smoking(smoking)
            if smoke_str:
                lines.append(f"- Smoking: {smoke_str}")

        # Substance use
        substance = profile.get("substance_use", {})
        if substance:
            lines.append(f"- Substance use: {json.dumps(substance)}")

        # Family history
        family = profile.get("family_history", [])
        if family:
            fam_parts = []
            for f in family:
                if isinstance(f, dict):
                    fam_parts.append(f.get("condition") or str(f))
                else:
                    fam_parts.append(str(f))
            lines.append(f"- Family history: {', '.join(fam_parts)}")

        # Surgical history
        surgical = profile.get("surgical_history", [])
        if surgical:
            surg_parts = []
            for s in surgical:
                if isinstance(s, dict):
                    surg_parts.append(s.get("procedure") or str(s))
                else:
                    surg_parts.append(str(s))
            lines.append(f"- Surgical history: {', '.join(surg_parts)}")

        # Previous sessions
        session_history = profile.get("session_history", [])
        if session_history:
            count = len(session_history)
            session_parts = []
            for s in session_history[-3:]:  # show last 3 at most
                date_str = s.get("date", "")[:10]  # YYYY-MM-DD
                symptoms_str = ", ".join(s.get("symptoms", []))
                dx = s.get("diagnosis", "unknown")
                session_parts.append(f"{date_str} ({symptoms_str} → {dx} diagnosed)")
            sessions_str = "; ".join(session_parts)
            lines.append(
                f"- Previous sessions: {count} session{'s' if count != 1 else ''} "
                f"— {sessions_str}"
            )

        # Build the confirmation list — briefly verify known facts instead of skipping
        confirm_items: list[str] = []
        if smoking and (smoking.get("status") or smoking.get("duration") or smoking.get("quantity")):
            confirm_items.append(f"smoking history ({_format_smoking(smoking)})")
        if conditions:
            cond_summary = ", ".join(
                c.get("condition", "") if isinstance(c, dict) else str(c)
                for c in conditions
            )
            confirm_items.append(f"conditions ({cond_summary})")
        if medications:
            med_summary = ", ".join(
                m.get("medication", "") if isinstance(m, dict) else str(m)
                for m in medications
            )
            confirm_items.append(f"medications ({med_summary})")
        if allergies:
            confirm_items.append("allergies")
        if surgical:
            confirm_items.append("surgical history")

        lines.append("")
        if confirm_items:
            lines.append(
                "Briefly CONFIRM these known facts with the patient (one sentence each, "
                "accept yes/no — if anything changed, record the update):"
            )
            for item in confirm_items:
                lines.append(f"  - {item}")
            lines.append("")
        lines.append("Focus your detailed questions on the NEW presenting complaint.")

        return "\n".join(lines)

    def get_context_for_triage(self, patient_name: str) -> str:
        """Format the patient's history as a context string for the Triage Agent.

        This string gives the Triage Agent a compact view of the patient's
        background so it can factor known comorbidities, medications, and prior
        diagnoses into its differential.

        Returns an empty string if the patient has no prior sessions.
        """
        path = self._profile_path(patient_name)
        if not path.exists():
            return ""

        profile = self._load_raw(path)
        if not profile.get("sessions", 0):
            return ""

        lines: list[str] = ["Patient history:"]

        # Known conditions
        conditions = profile.get("known_conditions", [])
        if conditions:
            cond_parts = []
            for c in conditions:
                part = c.get("condition", "")
                if c.get("since"):
                    part += f" (since {c['since']})"
                cond_parts.append(part)
            lines.append(f"- Known conditions: {', '.join(cond_parts)}")

        # Medications
        medications = profile.get("medications", [])
        if medications:
            med_parts = [m.get("medication", "") for m in medications if m.get("medication")]
            if med_parts:
                lines.append(f"- Medications: {', '.join(med_parts)}")

        # Allergies
        allergies = profile.get("allergies", [])
        if allergies:
            allergy_parts = []
            for a in allergies:
                if isinstance(a, dict):
                    allergy_parts.append(a.get("allergen") or a.get("allergy") or str(a))
                else:
                    allergy_parts.append(str(a))
            lines.append(f"- Allergies: {', '.join(allergy_parts)}")

        # Smoking
        smoking = profile.get("smoking_history", {})
        if smoking and (smoking.get("status") or smoking.get("duration") or smoking.get("quantity")):
            smoke_str = _format_smoking(smoking)
            if smoke_str:
                lines.append(f"- Smoking: {smoke_str}")

        # Family history
        family = profile.get("family_history", [])
        if family:
            fam_parts = []
            for f in family:
                if isinstance(f, dict):
                    fam_parts.append(f.get("condition") or str(f))
                else:
                    fam_parts.append(str(f))
            lines.append(f"- Family history: {', '.join(fam_parts)}")

        # Surgical history
        surgical = profile.get("surgical_history", [])
        if surgical:
            surg_parts = []
            for s in surgical:
                if isinstance(s, dict):
                    surg_parts.append(s.get("procedure") or str(s))
                else:
                    surg_parts.append(str(s))
            lines.append(f"- Surgical history: {', '.join(surg_parts)}")

        # Previous diagnoses (last 3)
        session_history = profile.get("session_history", [])
        for s in session_history[-3:]:
            date_str = s.get("date", "")[:10]
            dx = s.get("diagnosis", "")
            if dx:
                lines.append(f"- Previous diagnosis: {dx} ({date_str})")

        lines.append("")
        lines.append("Consider this history when forming your differential diagnosis.")

        return "\n".join(lines)

    def list_patients(self) -> list[str]:
        """Return the display names of all patients with stored profiles."""
        names: list[str] = []
        for p in sorted(self._storage_dir.glob("*.json")):
            try:
                data = self._load_raw(p)
                names.append(data.get("patient_name", p.stem))
            except (json.JSONDecodeError, OSError):
                names.append(p.stem)
        return names


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers (not part of the public class API)
# ─────────────────────────────────────────────────────────────────────────────

def _format_smoking(smoking: dict) -> str:
    """Return a human-readable smoking history string."""
    parts = []
    status = smoking.get("status", "")
    if status:
        parts.append(f"{status} smoker")
    if smoking.get("duration"):
        parts.append(f"{smoking['duration']}")
    if smoking.get("quantity"):
        parts.append(f"{smoking['quantity']}")
    if smoking.get("quit"):
        parts.append(f"quit {smoking['quit']}")
    if not parts:
        return ""
    return " — ".join(parts)


def _extract_primary_diagnosis(report_text: str) -> str:
    """Extract the primary diagnosis name from a triage report string.

    The triage report uses a structured markdown format with a section:
        ## Primary Diagnosis
        [Most likely diagnosis with confidence level: ...]

    This function pulls the first non-empty line after that header.
    Falls back to the first 80 characters of the report if the pattern
    is not found.
    """
    if not report_text:
        return "Unknown"

    lines = report_text.splitlines()
    capture_next = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("## primary diagnosis"):
            capture_next = True
            continue
        if capture_next and stripped and not stripped.startswith("#"):
            # Strip leading bullet / confidence markers for brevity
            dx = stripped.lstrip("- *").split("(")[0].strip()
            return dx if dx else "Unknown"
        if capture_next and stripped.startswith("#"):
            break  # hit the next section without finding content

    # Fallback: use start of report
    return report_text[:80].replace("\n", " ").strip()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_simulate_intake() -> dict:
    """Return a realistic fake intake summary for CLI testing."""
    return {
        "symptoms": ["Cough"],
        "urgency": "routine",
        "escalated": False,
        "answers": {
            "How long have you had this cough?": "About 2 weeks, it started gradually.",
            "Is the cough dry or productive?": "Mostly dry, but sometimes a little mucus.",
            "Do you have a fever?": "Yes, low-grade, around 37.8°C.",
            "Do you smoke or have you ever smoked?": (
                "I used to smoke for about 10 years, 2 packs a day, "
                "but I quit 2 years ago."
            ),
            "Do you have any known medical conditions?": (
                "I have asthma since I was 7. "
                "My father had heart disease."
            ),
            "Do you take any medications?": "Just a salbutamol inhaler for asthma.",
            "Any known allergies?": "Penicillin — I get a rash.",
            "Any recent travel or sick contacts?": (
                "My colleague was diagnosed with pertussis last week."
            ),
        },
        "triggered_red_flags": [],
        "specialty_routing": ["Respiratory Medicine", "General Practice"],
        "initial_workup": ["CXR", "FBC", "Pertussis PCR"],
        "key_exam_findings": ["Auscultation", "Lymphadenopathy"],
        "when_to_admit": [],
        "when_to_refer": [],
        "clinician_note": (
            "Patient presents with a 2-week dry cough, low-grade fever, "
            "and known pertussis exposure. History of asthma (childhood onset), "
            "former heavy smoker (quit 2 years ago), penicillin allergy. "
            "Father had heart disease."
        ),
    }


def _build_simulate_diagnosis() -> dict:
    """Return a realistic fake diagnosis dict for CLI testing."""
    return {
        "report": (
            "## Primary Diagnosis\n"
            "Pertussis (Whooping Cough) — confidence: moderate\n\n"
            "## Differential Diagnoses\n"
            "- Viral upper respiratory tract infection\n"
            "- Mycoplasma pneumoniae infection\n\n"
            "## Clinical Reasoning\n"
            "The 2-week cough with whooping-like character, low-grade fever, "
            "and direct pertussis exposure in a partially immune adult strongly "
            "supports Bordetella pertussis infection.\n\n"
            "## Recommended Investigations\n"
            "- Nasopharyngeal PCR for Bordetella pertussis\n"
            "- FBC (lymphocytosis is characteristic)\n"
            "- Chest X-ray to exclude pneumonia\n\n"
            "## Management Considerations\n"
            "Macrolide antibiotic (azithromycin preferred — patient has penicillin allergy).\n\n"
            "## Red Flags & Safety Netting\n"
            "Return immediately if cyanosis, apnoeic spells, or respiratory distress.\n\n"
            "## Sources\n"
            "Current Medical Diagnosis & Treatment 2022, Chapter 9."
        ),
        "mode": "common",
        "pass": 1,
        "num_chunks_used": 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Medora Phase 6 — Patient Memory System (test CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patient",
        required=True,
        metavar="NAME",
        help="Patient display name (e.g. 'Karim Habbal').",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Load and print the patient's current profile.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Simulate a session update with test data "
            "(requires OPENAI_API_KEY in environment)."
        ),
    )
    parser.add_argument(
        "--intake-context",
        action="store_true",
        help="Print the intake agent context string for this patient.",
    )
    parser.add_argument(
        "--triage-context",
        action="store_true",
        help="Print the triage agent context string for this patient.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all patients with stored profiles.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        metavar="MODEL",
        help=(
            "Model to use for the simulate update. "
            "OpenAI example: gpt-4o-mini. Ollama example: gemma2:27b."
        ),
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "ollama"],
        help="LLM provider (auto-detected from model name if not specified).",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        metavar="URL",
        help="Ollama server URL (default: http://localhost:11434).",
    )
    args = parser.parse_args()

    memory = PatientMemory()

    if args.list:
        patients = memory.list_patients()
        if patients:
            print("Stored patient profiles:")
            for name in patients:
                print(f"  - {name}")
        else:
            print("No patient profiles found.")

    if args.show:
        profile = memory.get_or_create(args.patient)
        print(json.dumps(profile, indent=2, ensure_ascii=False))

    if args.intake_context:
        ctx = memory.get_context_for_intake(args.patient)
        if ctx:
            print("\n--- Intake Agent Context ---")
            print(ctx)
        else:
            print("(No prior history — context is empty for a new patient.)")

    if args.triage_context:
        ctx = memory.get_context_for_triage(args.patient)
        if ctx:
            print("\n--- Triage Agent Context ---")
            print(ctx)
        else:
            print("(No prior history — context is empty for a new patient.)")

    if args.simulate:
        print(f"\nSimulating a session update for '{args.patient}'...")
        from config import make_llm
        llm = make_llm(model=args.model, provider=args.provider, ollama_url=args.ollama_url)
        intake_summary = _build_simulate_intake()
        diagnosis = _build_simulate_diagnosis()
        updated_profile = memory.update_from_session(
            args.patient, intake_summary, diagnosis, llm
        )
        print("\nUpdated profile:")
        print(json.dumps(updated_profile, indent=2, ensure_ascii=False))

        print("\n--- Intake Agent Context (after simulation) ---")
        print(memory.get_context_for_intake(args.patient))

        print("\n--- Triage Agent Context (after simulation) ---")
        print(memory.get_context_for_triage(args.patient))


if __name__ == "__main__":
    main()
