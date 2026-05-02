"""JSON-serializable schemas for the Phase 6 web evidence agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Decision = Literal["accept", "accept_with_caution", "reject"]
Confidence = Literal["high", "moderate", "low"]


def _clean_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly dict without mutating the original data."""
    return {key: value for key, value in data.items()}


@dataclass
class PatientContext:
    """De-identified clinical context used to make the evidence query relevant."""

    age: int | None = None
    age_range: str | None = None
    sex: str | None = None
    symptoms: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    comorbidities: list[str] = field(default_factory=list)
    pregnancy_status: str | None = None
    immunocompromised: bool | None = None
    medications: list[str] = field(default_factory=list)
    other_context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PatientContext":
        """Build a context object from a permissive dictionary."""
        if not data:
            return cls()
        known = {
            "age",
            "age_range",
            "sex",
            "symptoms",
            "red_flags",
            "comorbidities",
            "pregnancy_status",
            "immunocompromised",
            "medications",
        }
        kwargs = {key: data.get(key) for key in known if key in data}
        kwargs["other_context"] = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class WebEvidenceRequest:
    """Input contract for the web evidence agent."""

    clinical_question: str
    patient_context: PatientContext | dict[str, Any] | None = None
    reason_for_web: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WebEvidenceRequest":
        """Build a request from a JSON-like dictionary."""
        context = data.get("patient_context")
        if isinstance(context, dict):
            context = PatientContext.from_dict(context)
        return cls(
            clinical_question=str(data.get("clinical_question", "")),
            patient_context=context,
            reason_for_web=data.get("reason_for_web"),
        )

    def context_dict(self) -> dict[str, Any]:
        """Return patient context as a plain dictionary."""
        if isinstance(self.patient_context, PatientContext):
            return self.patient_context.to_dict()
        if isinstance(self.patient_context, dict):
            return _clean_dict(self.patient_context)
        return {}

    def to_dict(self) -> dict[str, Any]:
        """Return this request as a JSON-serializable dictionary."""
        return {
            "clinical_question": self.clinical_question,
            "patient_context": self.context_dict(),
            "reason_for_web": self.reason_for_web,
        }


@dataclass
class PrivacyReport:
    """Report describing what was removed and what medical context was retained."""

    removed_identifiers: list[str] = field(default_factory=list)
    kept_medical_context: list[str] = field(default_factory=list)
    deidentified: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class SearchResult:
    """Search backend result before page fetching."""

    title: str
    url: str
    snippet: str = ""
    domain: str = ""
    published_or_updated_date: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class FetchedSource:
    """Fetched and cleaned source page."""

    title: str
    url: str
    domain: str
    text: str = ""
    snippet: str = ""
    published_or_updated_date: str | None = None
    status_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class RankedSource:
    """Source after policy scoring and ranking."""

    title: str
    url: str
    domain: str
    source_tier: int
    source_type: str
    published_or_updated_date: str | None
    reliability_score: float
    snippet: str = ""
    text: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the public JSON representation for a ranked source."""
        data = asdict(self)
        data.pop("text", None)
        return data


@dataclass
class MedicalClaim:
    """Extractive medical claim with source support."""

    claim: str
    supporting_sources: list[str]
    confidence: Confidence = "low"

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class CouncilDecision:
    """Decision from one deterministic council role."""

    role: str
    decision: Decision
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class CouncilReview:
    """Structured output from all council roles."""

    source_validator: CouncilDecision
    recency_checker: CouncilDecision
    conflict_checker: CouncilDecision
    safety_critic: CouncilDecision
    final_reviewer: CouncilDecision
    claim_extractor: CouncilDecision | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return this object as a JSON-serializable dictionary."""
        data = {
            "source_validator": self.source_validator.to_dict(),
            "recency_checker": self.recency_checker.to_dict(),
            "conflict_checker": self.conflict_checker.to_dict(),
            "safety_critic": self.safety_critic.to_dict(),
            "final_reviewer": self.final_reviewer.to_dict(),
        }
        if self.claim_extractor:
            data["claim_extractor"] = self.claim_extractor.to_dict()
        return data


@dataclass
class WebEvidenceResult:
    """Final output contract for the Phase 6 agent."""

    clinical_question: str
    sanitized_query: str
    summary: str
    key_findings: list[MedicalClaim]
    conflicts: list[str]
    limitations: list[str]
    sources: list[RankedSource]
    council_review: CouncilReview
    final_decision: Decision
    privacy_report: PrivacyReport
    execution_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a JSON-serializable dictionary."""
        return {
            "clinical_question": self.clinical_question,
            "sanitized_query": self.sanitized_query,
            "summary": self.summary,
            "key_findings": [claim.to_dict() for claim in self.key_findings],
            "conflicts": self.conflicts,
            "limitations": self.limitations,
            "sources": [source.to_dict() for source in self.sources],
            "council_review": self.council_review.to_dict(),
            "final_decision": self.final_decision,
            "privacy_report": self.privacy_report.to_dict(),
            "execution_metadata": self.execution_metadata,
        }
