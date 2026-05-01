"""
Phase 7 — Feedback Store (Data Layer)
======================================
Captures doctor verdicts on system diagnoses and provides retrieval and
analytics on that feedback. This is the pure data layer — no UI.

Storage layout:
    data/feedback/
        {YYYYMMDD}_{patient_name}_{uuid4_short}.json   — one file per case
        training_data.jsonl                             — exported training set

Case lifecycle:
    1. save_case()        → status "pending"
    2. submit_feedback()  → status "confirmed" | "rejected"
    3. export_training_data() → JSONL ready for fine-tuning

Usage (library):
    from agents.feedback_store import FeedbackStore, save_case_from_session

    store = FeedbackStore()
    case_id = store.save_case(...)
    store.submit_feedback(case_id, "confirmed")
    print(store.get_statistics())

Usage (CLI):
    python agents/feedback_store.py --stats
    python agents/feedback_store.py --pending
    python agents/feedback_store.py --reviewed
    python agents/feedback_store.py --list
    python agents/feedback_store.py --review 20260501_karim_abc12345
    python agents/feedback_store.py --export
"""

import argparse
import json
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# ── Project root on sys.path so config.py is importable ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_DEFAULT_STORAGE_DIR = PROJECT_ROOT / "data" / "feedback"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _normalise_name(name: str) -> str:
    """Normalise a patient name to a safe filename fragment.

    Examples:
        "Karim Habbal" → "karim_habbal"
        "  Jane  Doe  " → "jane_doe"
    """
    return name.strip().lower().replace(" ", "_")


def _extract_primary_diagnosis(report_text: str) -> str:
    """Extract the primary diagnosis name from a triage report string.

    The triage report uses a structured markdown format:
        ## Primary Diagnosis
        [Most likely diagnosis with confidence level: ...]

    Pulls the first non-empty line after that header.
    Falls back to the first 80 characters of the report if not found.
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
            # Strip leading bullet / confidence markers
            dx = stripped.lstrip("- *").split("(")[0].strip()
            return dx if dx else "Unknown"
        if capture_next and stripped.startswith("#"):
            break  # hit the next section without finding content

    # Fallback: use the start of the report
    return report_text[:80].replace("\n", " ").strip()


# ─────────────────────────────────────────────────────────────────────────────
# FeedbackStore
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackStore:
    """Stores and retrieves doctor feedback on system diagnoses.

    Each case is saved as a JSON file under data/feedback/
    File naming: {YYYYMMDD}_{patient_name}_{uuid4_short}.json

    The class is intentionally stateless between calls — every public method
    reads from and (when writing) writes back to disk so that multiple
    processes or threads sharing the same storage directory stay consistent.

    Thread-safety: a lightweight per-file lock is used so that concurrent
    updates for the same case do not clobber each other.
    """

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._storage_dir: Path = Path(storage_dir) if storage_dir else _DEFAULT_STORAGE_DIR
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Per-case threading locks
        self._locks: dict[str, threading.Lock] = {}
        self._locks_mutex = threading.Lock()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_lock(self, case_id: str) -> threading.Lock:
        with self._locks_mutex:
            if case_id not in self._locks:
                self._locks[case_id] = threading.Lock()
            return self._locks[case_id]

    def _case_path(self, case_id: str) -> Path | None:
        """Resolve a case_id to its file path.

        Accepts either the full filename stem (20260501_karim_abc12345) or
        a bare UUID fragment — scans for a matching file if needed.
        """
        # Try an exact match first (stem == case_id)
        direct = self._storage_dir / f"{case_id}.json"
        if direct.exists():
            return direct

        # Fallback: scan for a file whose stem ends with the given token
        for p in self._storage_dir.glob("*.json"):
            if p.stem == case_id or p.stem.endswith(f"_{case_id}"):
                return p
        return None

    def _load_raw(self, path: Path) -> dict:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _save_raw(self, path: Path, data: dict) -> None:
        """Atomic write: temp file → rename."""
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        tmp.replace(path)

    def _iter_cases(self) -> list[dict]:
        """Load all case files, skipping any that fail to parse."""
        cases: list[dict] = []
        for p in sorted(self._storage_dir.glob("*.json")):
            if p.stem == "training_data":
                continue
            try:
                cases.append(self._load_raw(p))
            except (json.JSONDecodeError, OSError):
                pass
        return cases

    # ── Public API ────────────────────────────────────────────────────────────

    def save_case(
        self,
        patient_name: str,
        symptoms: list[str],
        urgency: str,
        intake_summary: dict,
        clinical_picture: dict,
        diagnosis_report: dict,
        retrieved_chunks: list[dict],
    ) -> str:
        """Save a completed case awaiting doctor review.

        Stores the FULL diagnostic chain:
          - intake_summary: all Q&A pairs from intake
          - clinical_picture: the parsed version the Triage Agent used
          - diagnosis_report: the system's diagnosis output
          - retrieved_chunks: textbook passages retrieved during diagnosis

        Args:
            patient_name:      Display name of the patient.
            symptoms:          List of symptom names (e.g. ["Cough"]).
            urgency:           Urgency level from intake ("routine" | "urgent" | "emergency").
            intake_summary:    Dict returned by IntakeSession.get_summary().
            clinical_picture:  Dict returned by parse_intake_to_clinical_picture().
            diagnosis_report:  Dict returned by TriageSession.get_diagnosis().
            retrieved_chunks:  List of chunk dicts retrieved during triage.

        Returns:
            The case_id string (also the file stem) for later retrieval.
        """
        # Build the case_id and file path
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        name_slug = _normalise_name(patient_name)
        uid_short = uuid4().hex[:8]
        case_id = f"{date_str}_{name_slug}_{uid_short}"
        path = self._storage_dir / f"{case_id}.json"

        # Extract primary diagnosis from the report text
        report_text = diagnosis_report.get("report", "")
        system_primary_diagnosis = _extract_primary_diagnosis(report_text)

        case = {
            "case_id": case_id,
            "patient_name": patient_name.strip(),
            "timestamp": _now_iso(),

            # Diagnostic chain
            "symptoms": symptoms,
            "urgency": urgency,
            "intake_summary": intake_summary,
            "clinical_picture": clinical_picture,
            "diagnosis_report": diagnosis_report,
            "retrieved_chunks": retrieved_chunks,

            # Doctor feedback (null until reviewed)
            "review_status": "pending",
            "doctor_decision": None,
            "doctor_diagnosis": None,
            "doctor_notes": None,
            "reviewed_at": None,

            # Extracted for analytics
            "system_primary_diagnosis": system_primary_diagnosis,
        }

        lock = self._get_lock(case_id)
        with lock:
            self._save_raw(path, case)

        return case_id

    def submit_feedback(
        self,
        case_id: str,
        doctor_decision: str,
        doctor_diagnosis: str = "",
        doctor_notes: str = "",
    ) -> dict:
        """Record the doctor's feedback on a case.

        Args:
            case_id:          The case_id returned by save_case().
            doctor_decision:  "confirmed" or "rejected".
            doctor_diagnosis: The correct diagnosis (required when rejected).
            doctor_notes:     Optional free-text clinical notes.

        Returns:
            The updated case dict.

        Raises:
            FileNotFoundError: If no case with the given ID exists.
            ValueError: If doctor_decision is not "confirmed" or "rejected",
                        or if doctor_diagnosis is empty when decision is "rejected".
        """
        decision = doctor_decision.strip().lower()
        if decision not in ("confirmed", "rejected"):
            raise ValueError(
                f"doctor_decision must be 'confirmed' or 'rejected', got: {doctor_decision!r}"
            )
        if decision == "rejected" and not doctor_diagnosis.strip():
            raise ValueError(
                "doctor_diagnosis is required when doctor_decision is 'rejected'."
            )

        path = self._case_path(case_id)
        if path is None:
            raise FileNotFoundError(f"No case found with ID: {case_id!r}")

        lock = self._get_lock(case_id)
        with lock:
            case = self._load_raw(path)
            case["review_status"] = decision
            case["doctor_decision"] = decision
            case["doctor_diagnosis"] = doctor_diagnosis.strip() if doctor_diagnosis else None
            case["doctor_notes"] = doctor_notes.strip() if doctor_notes else None
            case["reviewed_at"] = _now_iso()
            self._save_raw(path, case)

        return case

    def get_case(self, case_id: str) -> dict | None:
        """Retrieve a specific case by ID.

        Returns None if not found.
        """
        path = self._case_path(case_id)
        if path is None:
            return None
        try:
            return self._load_raw(path)
        except (json.JSONDecodeError, OSError):
            return None

    def get_pending_cases(self) -> list[dict]:
        """Get all cases that haven't been reviewed yet."""
        return [c for c in self._iter_cases() if c.get("review_status") == "pending"]

    def get_reviewed_cases(self) -> list[dict]:
        """Get all cases that have been reviewed (confirmed or rejected)."""
        return [c for c in self._iter_cases() if c.get("review_status") != "pending"]

    def get_confirmed_cases(self) -> list[dict]:
        """Get cases where the doctor confirmed the system's diagnosis."""
        return [c for c in self._iter_cases() if c.get("review_status") == "confirmed"]

    def get_rejected_cases(self) -> list[dict]:
        """Get cases where the doctor rejected the system's diagnosis."""
        return [c for c in self._iter_cases() if c.get("review_status") == "rejected"]

    def get_statistics(self) -> dict:
        """Return aggregate statistics over all stored cases.

        Returns:
            {
                "total_cases": int,
                "pending_review": int,
                "confirmed": int,
                "rejected": int,
                "confirmation_rate": float | None,   # None if no reviewed cases
                "most_common_corrections": [         # top rejected system diagnoses
                    {"system_diagnosis": str, "count": int},
                    ...
                ]
            }
        """
        all_cases = self._iter_cases()
        total = len(all_cases)
        pending = sum(1 for c in all_cases if c.get("review_status") == "pending")
        confirmed = sum(1 for c in all_cases if c.get("review_status") == "confirmed")
        rejected = sum(1 for c in all_cases if c.get("review_status") == "rejected")
        reviewed = confirmed + rejected

        confirmation_rate: float | None = None
        if reviewed > 0:
            confirmation_rate = round(confirmed / reviewed, 4)

        # Count which system diagnoses were most often rejected
        rejected_system_dx = [
            c.get("system_primary_diagnosis", "Unknown")
            for c in all_cases
            if c.get("review_status") == "rejected"
        ]
        corrections_counter = Counter(rejected_system_dx)
        most_common_corrections = [
            {"system_diagnosis": dx, "count": cnt}
            for dx, cnt in corrections_counter.most_common(10)
        ]

        return {
            "total_cases": total,
            "pending_review": pending,
            "confirmed": confirmed,
            "rejected": rejected,
            "confirmation_rate": confirmation_rate,
            "most_common_corrections": most_common_corrections,
        }

    def export_training_data(self, output_path: Path | None = None) -> Path:
        """Export all reviewed cases as a JSONL file for fine-tuning.

        Each line in the output is a JSON object:
        {
            "input": {
                "clinical_picture": dict,
                "retrieved_chunks": list[dict]
            },
            "expected_output": str,   # doctor's diagnosis if rejected; system's if confirmed
            "system_output": str,     # what the system produced (full report)
            "was_correct": bool
        }

        Args:
            output_path: Optional output path. Defaults to data/feedback/training_data.jsonl.

        Returns:
            The path where the JSONL file was written.
        """
        if output_path is None:
            output_path = self._storage_dir / "training_data.jsonl"

        reviewed = self.get_reviewed_cases()

        with open(output_path, "w", encoding="utf-8") as fh:
            for case in reviewed:
                decision = case.get("doctor_decision", "confirmed")
                was_correct = decision == "confirmed"

                system_report = case.get("diagnosis_report", {}).get("report", "")

                if was_correct:
                    expected_output = system_report
                else:
                    # Use doctor's corrected diagnosis; fall back to system output
                    expected_output = case.get("doctor_diagnosis") or system_report

                entry = {
                    "case_id": case.get("case_id"),
                    "input": {
                        "clinical_picture": case.get("clinical_picture", {}),
                        "retrieved_chunks": case.get("retrieved_chunks", []),
                    },
                    "expected_output": expected_output,
                    "system_output": system_report,
                    "was_correct": was_correct,
                    # Optional metadata useful for training analytics
                    "patient_name": case.get("patient_name"),
                    "symptoms": case.get("symptoms", []),
                    "urgency": case.get("urgency"),
                    "system_primary_diagnosis": case.get("system_primary_diagnosis"),
                    "doctor_diagnosis": case.get("doctor_diagnosis"),
                    "doctor_notes": case.get("doctor_notes"),
                    "reviewed_at": case.get("reviewed_at"),
                }
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Convenience integration function
# ─────────────────────────────────────────────────────────────────────────────

def save_case_from_session(
    feedback_store: FeedbackStore,
    patient_name: str,
    intake_summary: dict,
    clinical_picture: dict,
    triage_diagnosis: dict,
    retrieved_chunks: list[dict],
) -> str:
    """Convenience function to save a case after a complete session.

    Extracts symptoms and urgency from the intake summary so the caller
    doesn't need to unpack them separately.

    Args:
        feedback_store:    An initialised FeedbackStore instance.
        patient_name:      Display name of the patient.
        intake_summary:    Dict returned by IntakeSession.get_summary().
        clinical_picture:  Dict returned by parse_intake_to_clinical_picture().
        triage_diagnosis:  Dict returned by TriageSession.get_diagnosis().
        retrieved_chunks:  List of chunk dicts used in the diagnosis.

    Returns:
        The case_id string for later retrieval.
    """
    symptoms: list[str] = intake_summary.get("symptoms", [])
    urgency: str = intake_summary.get("urgency", "routine")

    return feedback_store.save_case(
        patient_name=patient_name,
        symptoms=symptoms,
        urgency=urgency,
        intake_summary=intake_summary,
        clinical_picture=clinical_picture,
        diagnosis_report=triage_diagnosis,
        retrieved_chunks=retrieved_chunks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _print_case_summary(case: dict, verbose: bool = False) -> None:
    """Print a one-line (or verbose) summary of a case."""
    case_id = case.get("case_id", "?")
    patient = case.get("patient_name", "?")
    symptoms = ", ".join(case.get("symptoms", [])) or "?"
    dx = case.get("system_primary_diagnosis", "?")
    status = case.get("review_status", "?").upper()
    timestamp = case.get("timestamp", "")[:10]  # YYYY-MM-DD

    print(f"  [{status}] {case_id}")
    print(f"    Patient  : {patient}")
    print(f"    Date     : {timestamp}")
    print(f"    Symptoms : {symptoms}")
    print(f"    Dx       : {dx}")

    if verbose and case.get("review_status") != "pending":
        decision = case.get("doctor_decision", "?")
        doctor_dx = case.get("doctor_diagnosis") or "(same as system)"
        notes = case.get("doctor_notes") or "(none)"
        reviewed_at = case.get("reviewed_at", "")[:10]
        print(f"    Decision : {decision}")
        print(f"    Doctor Dx: {doctor_dx}")
        print(f"    Notes    : {notes}")
        print(f"    Reviewed : {reviewed_at}")


def _interactive_review(store: FeedbackStore, case_id: str) -> None:
    """Run an interactive review for a specific case."""
    case = store.get_case(case_id)
    if case is None:
        print(f"Error: case not found: {case_id!r}")
        return

    if case.get("review_status") != "pending":
        print(f"Case {case_id} has already been reviewed ({case['review_status']}).")
        ans = input("Re-review anyway? (y/N): ").strip().lower()
        if ans != "y":
            return

    # Print case header
    print()
    print("=" * 60)
    print(f"Case     : {case.get('case_id')}")
    print(f"Patient  : {case.get('patient_name')}")
    print(f"Symptoms : {', '.join(case.get('symptoms', []))}")
    print(f"Urgency  : {case.get('urgency', 'routine').upper()}")
    print(f"System Dx: {case.get('system_primary_diagnosis', '?')}")
    print("=" * 60)

    report = case.get("diagnosis_report", {}).get("report", "(no report)")
    print()
    print(report)
    print()
    print("=" * 60)

    # Get doctor decision
    while True:
        decision_raw = input("Doctor decision (confirm/reject): ").strip().lower()
        if decision_raw in ("confirm", "confirmed", "c"):
            decision = "confirmed"
            break
        if decision_raw in ("reject", "rejected", "r"):
            decision = "rejected"
            break
        print("  Please enter 'confirm' or 'reject'.")

    doctor_diagnosis = ""
    if decision == "rejected":
        while True:
            doctor_diagnosis = input("Correct diagnosis: ").strip()
            if doctor_diagnosis:
                break
            print("  A correct diagnosis is required when rejecting.")

    doctor_notes = input("Notes (optional, press Enter to skip): ").strip()

    try:
        store.submit_feedback(
            case_id=case.get("case_id", case_id),
            doctor_decision=decision,
            doctor_diagnosis=doctor_diagnosis,
            doctor_notes=doctor_notes,
        )
        print(f"\nFeedback saved. (case {case.get('case_id')} → {decision})")
    except (ValueError, FileNotFoundError) as exc:
        print(f"\nError saving feedback: {exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Medora Phase 7 — Feedback Store (doctor review CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show aggregate feedback statistics.",
    )
    parser.add_argument(
        "--pending",
        action="store_true",
        help="List all cases awaiting doctor review.",
    )
    parser.add_argument(
        "--reviewed",
        action="store_true",
        help="List all reviewed cases.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List ALL cases (pending and reviewed).",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export all reviewed cases to data/feedback/training_data.jsonl.",
    )
    parser.add_argument(
        "--review",
        metavar="CASE_ID",
        default=None,
        help="Interactively review a specific case by ID.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Insert a demo case and immediately review it "
            "(useful for end-to-end testing without running the full pipeline)."
        ),
    )
    return parser.parse_args()


def _build_demo_case() -> tuple[dict, dict, dict, list[dict]]:
    """Return (intake_summary, clinical_picture, diagnosis, retrieved_chunks) for demo."""
    intake_summary = {
        "symptoms": ["Cough"],
        "urgency": "routine",
        "escalated": False,
        "answers": {
            "How long have you had this cough?": "About 2 weeks, started gradually.",
            "Is the cough dry or productive?": "Mostly dry.",
            "Do you have a fever?": "Yes, low-grade, around 37.8°C.",
            "Any known exposures?": "My colleague was diagnosed with pertussis last week.",
        },
        "triggered_red_flags": [],
        "clinician_note": "2-week dry cough with pertussis exposure.",
    }
    clinical_picture = {
        "symptoms": ["Cough"],
        "urgency": "routine",
        "clinical_findings": {
            "onset": "2 weeks ago",
            "character": "dry",
            "fever": "low-grade 37.8°C",
            "exposure": "pertussis contact",
        },
        "red_flags": [],
    }
    diagnosis = {
        "report": (
            "## Primary Diagnosis\n"
            "Pertussis (Whooping Cough) — confidence: moderate\n\n"
            "## Differential Diagnoses\n"
            "- Viral URTI\n"
            "- Mycoplasma pneumoniae\n\n"
            "## Clinical Reasoning\n"
            "2-week paroxysmal cough with pertussis exposure in an adult.\n\n"
            "## Recommended Investigations\n"
            "- Nasopharyngeal PCR for Bordetella pertussis\n"
            "- FBC (lymphocytosis)\n\n"
            "## Management Considerations\n"
            "Macrolide antibiotic (azithromycin).\n\n"
            "## Red Flags & Safety Netting\n"
            "Return immediately if cyanosis or apnoeic spells.\n\n"
            "## Sources\n"
            "Current Medical Diagnosis & Treatment 2022, Chapter 9."
        ),
        "mode": "common",
        "pass": 1,
        "num_chunks_used": 3,
    }
    retrieved_chunks = [
        {
            "chunk_id": "demo_chunk_001",
            "chapter": "Chapter 9 — Pulmonary Disorders",
            "section": "Pertussis",
            "text": "Pertussis is characterised by a prolonged paroxysmal cough...",
        }
    ]
    return intake_summary, clinical_picture, diagnosis, retrieved_chunks


def main() -> None:
    args = _parse_args()
    store = FeedbackStore()

    if args.demo:
        intake_summary, clinical_picture, diagnosis, chunks = _build_demo_case()
        case_id = save_case_from_session(
            store,
            patient_name="Demo Patient",
            intake_summary=intake_summary,
            clinical_picture=clinical_picture,
            triage_diagnosis=diagnosis,
            retrieved_chunks=chunks,
        )
        print(f"\nDemo case saved: {case_id}")
        _interactive_review(store, case_id)
        return

    if args.stats:
        stats = store.get_statistics()
        print("\nFeedback Statistics")
        print("=" * 40)
        print(f"  Total cases    : {stats['total_cases']}")
        print(f"  Pending review : {stats['pending_review']}")
        print(f"  Confirmed      : {stats['confirmed']}")
        print(f"  Rejected       : {stats['rejected']}")
        rate = stats["confirmation_rate"]
        print(f"  Confirmation % : {f'{rate:.1%}' if rate is not None else 'N/A (no reviewed cases)'}")
        corrections = stats["most_common_corrections"]
        if corrections:
            print("\n  Most-corrected system diagnoses:")
            for entry in corrections:
                print(f"    [{entry['count']}x] {entry['system_diagnosis']}")
        print()

    if args.pending:
        cases = store.get_pending_cases()
        print(f"\nPending cases ({len(cases)}):")
        if not cases:
            print("  (none)")
        for c in cases:
            _print_case_summary(c)
        print()

    if args.reviewed:
        cases = store.get_reviewed_cases()
        print(f"\nReviewed cases ({len(cases)}):")
        if not cases:
            print("  (none)")
        for c in cases:
            _print_case_summary(c, verbose=True)
        print()

    if args.list:
        from itertools import chain
        all_cases = store.get_pending_cases() + store.get_reviewed_cases()
        # Sort by timestamp
        all_cases.sort(key=lambda c: c.get("timestamp", ""))
        print(f"\nAll cases ({len(all_cases)}):")
        if not all_cases:
            print("  (none)")
        for c in all_cases:
            _print_case_summary(c, verbose=True)
        print()

    if args.export:
        output = store.export_training_data()
        reviewed = store.get_reviewed_cases()
        print(f"\nExported {len(reviewed)} reviewed case(s) to:")
        print(f"  {output}")
        print()

    if args.review:
        _interactive_review(store, args.review)


if __name__ == "__main__":
    main()
