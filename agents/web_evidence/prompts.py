"""Prompt templates for local LLM-assisted web evidence reasoning."""

from __future__ import annotations

from agents.web_evidence.schemas import MedicalClaim, RankedSource


CLAIM_EXTRACTION_SYSTEM = (
    "You are a medical evidence extraction assistant running locally inside a hospital-controlled system. "
    "Extract only claims that are explicitly supported by the provided source text. Do not use outside knowledge. "
    "Do not provide a final diagnosis. Return valid JSON only."
)

CONFLICT_CHECK_SYSTEM = (
    "You are a local medical evidence conflict checker. Use only the supplied claims. "
    "Flag disagreements, contraindications, missing evidence, and uncertainty. Return valid JSON only."
)

SAFETY_CRITIC_SYSTEM = (
    "You are a local safety reviewer for a doctor-facing clinical evidence summary. "
    "Reject direct patient instructions, unsupported medication dosing, overconfident diagnosis, or claims not tied to sources. "
    "Return valid JSON only."
)

FINAL_REVIEW_SYSTEM = (
    "You are the final reviewer in a local hospital-controlled evidence council. "
    "Use source, privacy, recency, conflict, and safety decisions to choose accept, accept_with_caution, or reject. "
    "Return valid JSON only."
)

SYNTHESIS_SYSTEM = (
    "You are a local medical evidence synthesis assistant for a physician-facing pre-visit report. "
    "Use only supplied claims and sources. Do not invent claims. Do not give direct patient instructions. "
    "Mention uncertainty and physician review. Do not provide a final diagnosis."
)


def build_claim_extraction_prompt(clinical_question: str, source: RankedSource) -> str:
    """Build a JSON-only claim extraction prompt for one source."""
    text = (source.text or source.snippet)[:5000]
    return f"""Clinical question: {clinical_question}
Source title: {source.title}
Source domain: {source.domain}
Source URL: {source.url}
Source text:
{text}

Return JSON:
{{
  "claims": [
    {{
      "claim": "...",
      "source_url": "{source.url}",
      "supporting_quote": "...",
      "confidence": "high|moderate|low"
    }}
  ]
}}"""


def build_conflict_prompt(claims: list[MedicalClaim]) -> str:
    """Build a JSON-only conflict-checking prompt."""
    rendered = "\n".join(f"- {claim.claim} Sources: {', '.join(claim.supporting_sources)}" for claim in claims)
    return f"""Claims:
{rendered}

Return JSON:
{{
  "decision": "accept|accept_with_caution|reject",
  "conflicts": ["..."],
  "reason": "..."
}}"""


def build_safety_prompt(summary: str, claims: list[MedicalClaim]) -> str:
    """Build a JSON-only safety review prompt."""
    rendered = "\n".join(f"- {claim.claim}" for claim in claims)
    return f"""Draft summary:
{summary}

Claims:
{rendered}

Return JSON:
{{
  "decision": "accept|accept_with_caution|reject",
  "reason": "...",
  "warnings": ["..."]
}}"""


def build_final_review_prompt(council_snapshot: dict) -> str:
    """Build a JSON-only final council review prompt."""
    return f"""Council snapshot:
{council_snapshot}

Return JSON:
{{
  "decision": "accept|accept_with_caution|reject",
  "reason": "..."
}}"""


def build_synthesis_prompt(
    clinical_question: str,
    claims: list[MedicalClaim],
    sources: list[RankedSource],
    conflicts: list[str],
    limitations: list[str],
) -> str:
    """Build a doctor-facing synthesis prompt."""
    claim_text = "\n".join(
        f"- {claim.claim} Sources: {', '.join(claim.supporting_sources)} Confidence: {claim.confidence}"
        for claim in claims
    )
    source_text = "\n".join(
        f"- {source.title} | {source.domain} | {source.url} | date={source.published_or_updated_date} | tier={source.source_tier}"
        for source in sources
    )
    return f"""Clinical question: {clinical_question}

Supported claims:
{claim_text}

Sources:
{source_text}

Conflicts:
{conflicts}

Limitations:
{limitations}

Write a concise doctor-facing evidence summary. Include uncertainty, source support, conflicts or missing evidence, and physician-review language. Do not give direct patient instructions or a final diagnosis."""


CLAIM_SCHEMA = {
    "claims": [
        {
            "claim": "string",
            "source_url": "string",
            "supporting_quote": "string",
            "confidence": "high|moderate|low",
        }
    ]
}

DECISION_SCHEMA = {
    "decision": "accept|accept_with_caution|reject",
    "reason": "string",
}
