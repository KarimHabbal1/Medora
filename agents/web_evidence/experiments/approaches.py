"""Approach configurations for Phase 6 comparison experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WebEvidenceApproach:
    """Configuration for one web evidence reasoning approach."""

    name: str
    description: str
    use_llm_claim_extraction: bool = False
    use_llm_conflict_checking: bool = False
    use_llm_safety_review: bool = False
    use_llm_final_review: bool = False
    use_llm_synthesis: bool = False
    deterministic_safety_after_synthesis: bool = False
    llm_provider: str = "none"

    def to_dict(self) -> dict:
        """Return a JSON-serializable approach config."""
        return asdict(self)


APPROACHES: dict[str, WebEvidenceApproach] = {
    "deterministic_only": WebEvidenceApproach(
        name="deterministic_only",
        description=(
            "Deterministic search, ranking, claim extraction, council, and synthesis. "
            "No local LLM."
        ),
    ),
    "llm_synthesis_only": WebEvidenceApproach(
        name="llm_synthesis_only",
        description="Deterministic evidence pipeline with local LLM used only for final synthesis.",
        use_llm_synthesis=True,
        llm_provider="none",
    ),
    "llm_claims_and_synthesis": WebEvidenceApproach(
        name="llm_claims_and_synthesis",
        description="Local LLM extracts claims and synthesizes summary; deterministic safety still applies.",
        use_llm_claim_extraction=True,
        use_llm_synthesis=True,
        llm_provider="none",
    ),
    "full_llm_council": WebEvidenceApproach(
        name="full_llm_council",
        description=(
            "Local LLM assists claim extraction, conflict checking, safety review, final review, and synthesis. "
            "Deterministic fallbacks remain active."
        ),
        use_llm_claim_extraction=True,
        use_llm_conflict_checking=True,
        use_llm_safety_review=True,
        use_llm_final_review=True,
        use_llm_synthesis=True,
        llm_provider="none",
    ),
    "hybrid_recommended": WebEvidenceApproach(
        name="hybrid_recommended",
        description=(
            "Deterministic privacy, source validation, recency, and safety with local LLM for claim extraction "
            "and synthesis only. Recommended production tradeoff."
        ),
        use_llm_claim_extraction=True,
        use_llm_synthesis=True,
        deterministic_safety_after_synthesis=True,
        llm_provider="none",
    ),
}

TIE_BREAK_ORDER = [
    "hybrid_recommended",
    "deterministic_only",
    "llm_claims_and_synthesis",
    "llm_synthesis_only",
    "full_llm_council",
]


def list_approaches() -> list[WebEvidenceApproach]:
    """Return all supported approaches."""
    return list(APPROACHES.values())


def get_approach(name: str, provider: str | None = None) -> WebEvidenceApproach:
    """Return an approach by name, optionally overriding its LLM provider."""
    if name not in APPROACHES:
        valid = ", ".join(sorted(APPROACHES))
        raise ValueError(f"Unknown web evidence approach '{name}'. Valid approaches: {valid}")
    approach = APPROACHES[name]
    if provider is None:
        return approach
    return WebEvidenceApproach(
        name=approach.name,
        description=approach.description,
        use_llm_claim_extraction=approach.use_llm_claim_extraction,
        use_llm_conflict_checking=approach.use_llm_conflict_checking,
        use_llm_safety_review=approach.use_llm_safety_review,
        use_llm_final_review=approach.use_llm_final_review,
        use_llm_synthesis=approach.use_llm_synthesis,
        deterministic_safety_after_synthesis=approach.deterministic_safety_after_synthesis,
        llm_provider=provider,
    )
