"""Deterministic doctor-facing synthesis for web evidence results."""

from __future__ import annotations

from agents.web_evidence.llm.base import BaseLocalLLM, strict_json_retry_prompt
from agents.web_evidence.prompts import SYNTHESIS_SYSTEM, build_synthesis_prompt
from agents.web_evidence.schemas import MedicalClaim, RankedSource


def synthesize_summary(
    clinical_question: str,
    claims: list[MedicalClaim],
    sources: list[RankedSource],
    conflicts: list[str],
    limitations: list[str],
) -> str:
    """Create a cautious preliminary evidence summary for physician review."""
    if not sources:
        return (
            "No reliable web evidence could be retrieved for this clinical question. "
            "This preliminary result requires clinician judgment and should not be used as a final diagnosis."
        )

    source_phrases = []
    for source in sources[:3]:
        date = f", updated/published {source.published_or_updated_date}" if source.published_or_updated_date else ""
        source_phrases.append(f"{source.title} ({source.domain}{date})")

    if claims:
        claim_text = " ".join(claim.claim for claim in claims[:3])
        answer = (
            f"For physician review, guideline-level sources suggest the following preliminary evidence for: "
            f"{clinical_question}. {claim_text}"
        )
    else:
        answer = (
            f"For physician review, reliable sources were identified for: {clinical_question}, "
            "but deterministic extraction did not produce a strong claim."
        )

    safety_note = (
        " This is not a final diagnosis or direct patient treatment instruction; it requires clinician judgment."
    )
    if conflicts:
        answer += f" Potential conflict noted: {'; '.join(conflicts)}."
    if limitations:
        answer += f" Limitations: {'; '.join(limitations[:3])}."
    answer += f" Key reviewed sources: {'; '.join(source_phrases)}." + safety_note
    return answer


def synthesize_summary_with_optional_llm(
    clinical_question: str,
    claims: list[MedicalClaim],
    sources: list[RankedSource],
    conflicts: list[str],
    limitations: list[str],
    use_llm: bool = False,
    llm: BaseLocalLLM | None = None,
) -> tuple[str, dict]:
    """Synthesize with local LLM when selected, otherwise use deterministic text."""
    deterministic = synthesize_summary(clinical_question, claims, sources, conflicts, limitations)
    metadata = {
        "mode": "deterministic",
        "llm_used": False,
        "llm_failure": False,
        "deterministic_fallback": False,
    }
    if not use_llm:
        return deterministic, metadata
    if llm is None:
        metadata.update(
            {
                "mode": "local_llm_fallback",
                "deterministic_fallback": True,
                "error": "no_local_llm_configured",
            }
        )
        return deterministic, metadata

    schema = {"summary": "string"}
    prompt = build_synthesis_prompt(clinical_question, claims, sources, conflicts, limitations)
    payload = llm.generate_json(SYNTHESIS_SYSTEM, prompt, schema)
    if payload.get("error"):
        retry = llm.generate_json(SYNTHESIS_SYSTEM, strict_json_retry_prompt(prompt, schema), schema)
        if retry.get("error"):
            metadata.update(
                {
                    "mode": "local_llm_fallback",
                    "llm_used": True,
                    "llm_failure": True,
                    "deterministic_fallback": True,
                    "error": retry.get("error") or payload.get("error"),
                }
            )
            return deterministic, metadata
        payload = retry
        metadata["retry_used"] = True

    summary = str(payload.get("summary", "")).strip()
    if not summary:
        metadata.update(
            {
                "mode": "local_llm_fallback",
                "llm_used": True,
                "deterministic_fallback": True,
                "error": "missing_summary",
            }
        )
        return deterministic, metadata
    metadata.update({"mode": "local_llm", "llm_used": True})
    return summary, metadata
