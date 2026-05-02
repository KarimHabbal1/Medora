"""Main orchestration for the Phase 6 privacy-safe web evidence agent."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agents.web_evidence.config import DEFAULT_MAX_SEARCH_RESULTS, DEFAULT_MAX_SOURCES
from agents.web_evidence.council import MedicalClaimExtractor, SafetyCritic, run_council
from agents.web_evidence.experiments.approaches import WebEvidenceApproach, get_approach
from agents.web_evidence.llm.base import BaseLocalLLM, get_local_llm
from agents.web_evidence.page_fetcher import fetched_from_search_result, fetch_url
from agents.web_evidence.query_builder import build_search_query
from agents.web_evidence.schemas import (
    CouncilDecision,
    CouncilReview,
    PrivacyReport,
    WebEvidenceRequest,
    WebEvidenceResult,
)
from agents.web_evidence.search_client import BaseSearchClient, SearxNGSearchClient, StaticSearchClient
from agents.web_evidence.source_ranker import rank_sources
from agents.web_evidence.synthesizer import synthesize_summary, synthesize_summary_with_optional_llm


def _error_review(reason: str) -> CouncilReview:
    decision = CouncilDecision("final_reviewer", "reject", reason, {})
    neutral = CouncilDecision("not_run", "reject", reason, {})
    return CouncilReview(
        source_validator=neutral,
        recency_checker=neutral,
        conflict_checker=neutral,
        safety_critic=neutral,
        final_reviewer=decision,
    )


def _approach_uses_llm(approach: WebEvidenceApproach) -> bool:
    return any(
        [
            approach.use_llm_claim_extraction,
            approach.use_llm_conflict_checking,
            approach.use_llm_safety_review,
            approach.use_llm_final_review,
            approach.use_llm_synthesis,
        ]
    )


def _resolve_approach(approach: WebEvidenceApproach | str | None, provider: str | None) -> WebEvidenceApproach:
    if isinstance(approach, WebEvidenceApproach):
        resolved = approach
    else:
        resolved = get_approach(approach or "deterministic_only")
    if not _approach_uses_llm(resolved):
        return replace(resolved, llm_provider="none")
    if provider:
        return replace(resolved, llm_provider=provider)
    return resolved


def _collect_execution_metadata(
    approach: WebEvidenceApproach,
    council_review: CouncilReview,
    synthesis_metadata: dict[str, Any],
    llm_provider: str,
    llm_model: str | None,
) -> dict[str, Any]:
    review = council_review.to_dict()
    details = []
    for decision in review.values():
        if isinstance(decision, dict):
            details.append(decision.get("details", {}))
    llm_failure_count = sum(int(item.get("llm_failure_count", 0)) for item in details)
    llm_failure_count += sum(1 for item in details if item.get("llm_failure") is True)
    llm_failure_count += int(synthesis_metadata.get("llm_failure", False))
    deterministic_fallback_count = sum(int(item.get("deterministic_fallback_count", 0)) for item in details)
    deterministic_fallback_count += sum(1 for item in details if item.get("deterministic_fallback") is True)
    deterministic_fallback_count += int(synthesis_metadata.get("deterministic_fallback", False))
    return {
        "approach": approach.to_dict(),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_failure_count": llm_failure_count,
        "deterministic_fallback_count": deterministic_fallback_count,
        "synthesis": synthesis_metadata,
    }


def run_web_evidence_agent(
    request: WebEvidenceRequest,
    max_sources: int = DEFAULT_MAX_SOURCES,
    search_client: BaseSearchClient | None = None,
    use_mock_fetch: bool = False,
    approach: WebEvidenceApproach | str | None = None,
    llm: BaseLocalLLM | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> WebEvidenceResult:
    """Run the full privacy-safe web evidence pipeline.

    The function returns structured limitations instead of raising raw exceptions
    to callers. The optional search_client parameter keeps the provider pluggable.
    """
    resolved_approach = _resolve_approach(approach, llm_provider)
    provider = resolved_approach.llm_provider if _approach_uses_llm(resolved_approach) else "none"
    local_llm = llm or (get_local_llm(provider, model=llm_model) if provider != "none" else None)

    try:
        sanitized_query, privacy_report = build_search_query(request)
    except Exception as exc:  # noqa: BLE001
        privacy_report = PrivacyReport(removed_identifiers=["sanitization_error"], deidentified=False)
        return WebEvidenceResult(
            clinical_question=request.clinical_question,
            sanitized_query="",
            summary="The web evidence agent could not de-identify the request safely.",
            key_findings=[],
            conflicts=[],
            limitations=[f"privacy_sanitization_error: {exc}"],
            sources=[],
            council_review=_error_review("Privacy sanitization failed."),
            final_decision="reject",
            privacy_report=privacy_report,
            execution_metadata={
                "approach": resolved_approach.to_dict(),
                "llm_provider": provider,
                "llm_model": llm_model,
                "llm_failure_count": 0,
                "deterministic_fallback_count": 0,
            },
        )

    limitations: list[str] = []
    client = search_client or SearxNGSearchClient()
    try:
        search_results = client.search(sanitized_query, max_results=DEFAULT_MAX_SEARCH_RESULTS)
    except Exception as exc:  # noqa: BLE001
        search_results = []
        limitations.append(f"search_error: {exc}")

    for result in search_results:
        if result.error:
            limitations.append(result.error)

    ranked_search = [source for source in rank_sources(search_results, sanitized_query) if source.url]
    candidate_results = []
    urls = {source.url for source in ranked_search[: max_sources * 2]}
    for result in search_results:
        if result.url in urls:
            candidate_results.append(result)

    fetched = []
    for result in candidate_results[: max_sources * 2]:
        if use_mock_fetch or isinstance(client, StaticSearchClient):
            fetched.append(fetched_from_search_result(result))
        else:
            fetched_source = fetch_url(result.url)
            if fetched_source.error:
                limitations.append(fetched_source.error)
                fetched.append(fetched_from_search_result(result))
            else:
                fetched.append(fetched_source)

    ranked_sources = rank_sources(fetched, sanitized_query)[:max_sources]
    claims, claim_decision = MedicalClaimExtractor().extract(
        ranked_sources,
        clinical_question=request.clinical_question,
        use_llm=resolved_approach.use_llm_claim_extraction,
        llm=local_llm,
    )
    preliminary_summary = synthesize_summary(request.clinical_question, claims, ranked_sources, [], limitations)
    council_review, conflicts = run_council(
        ranked_sources,
        claims,
        privacy_report,
        preliminary_summary,
        claim_extractor_decision=claim_decision,
        use_llm_conflict_checking=resolved_approach.use_llm_conflict_checking,
        use_llm_safety_review=resolved_approach.use_llm_safety_review,
        use_llm_final_review=resolved_approach.use_llm_final_review,
        llm=local_llm,
    )
    summary, synthesis_metadata = synthesize_summary_with_optional_llm(
        request.clinical_question,
        claims,
        ranked_sources,
        conflicts,
        limitations,
        use_llm=resolved_approach.use_llm_synthesis,
        llm=local_llm,
    )

    final_decision = council_review.final_reviewer.decision
    if resolved_approach.deterministic_safety_after_synthesis:
        post_safety = SafetyCritic().review_deterministic(claims, summary)
        council_review.safety_critic.details["post_synthesis_deterministic_review"] = post_safety.to_dict()
        if post_safety.decision == "reject":
            final_decision = "reject"
            council_review.final_reviewer.decision = "reject"
            council_review.final_reviewer.reason = "Post-synthesis deterministic safety review rejected the result."

    execution_metadata = _collect_execution_metadata(
        resolved_approach,
        council_review,
        synthesis_metadata,
        provider,
        llm_model,
    )

    return WebEvidenceResult(
        clinical_question=request.clinical_question,
        sanitized_query=sanitized_query,
        summary=summary,
        key_findings=claims,
        conflicts=conflicts,
        limitations=limitations,
        sources=ranked_sources,
        council_review=council_review,
        final_decision=final_decision,
        privacy_report=privacy_report,
        execution_metadata=execution_metadata,
    )
