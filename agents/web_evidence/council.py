"""Deterministic and local-LLM-assisted source council for web evidence."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from agents.web_evidence.llm.base import BaseLocalLLM, strict_json_retry_prompt
from agents.web_evidence.prompts import (
    CLAIM_EXTRACTION_SYSTEM,
    CLAIM_SCHEMA,
    CONFLICT_CHECK_SYSTEM,
    DECISION_SCHEMA,
    FINAL_REVIEW_SYSTEM,
    SAFETY_CRITIC_SYSTEM,
    build_claim_extraction_prompt,
    build_conflict_prompt,
    build_final_review_prompt,
    build_safety_prompt,
)
from agents.web_evidence.schemas import CouncilDecision, CouncilReview, MedicalClaim, PrivacyReport, RankedSource
from agents.web_evidence.source_ranker import is_rejected_domain


CLAIM_KEYWORDS = (
    "guideline",
    "recommend",
    "should",
    "treatment",
    "diagnosis",
    "contraindication",
    "warning",
    "vaccine",
    "screening",
    "management",
    "antibiotic",
)


def _is_valid_decision(value: str | None) -> bool:
    return value in {"accept", "accept_with_caution", "reject"}


def _llm_json_with_retry(
    llm: BaseLocalLLM | None,
    system_prompt: str,
    user_prompt: str,
    schema_hint: dict | None = None,
) -> tuple[dict, dict]:
    """Call a local LLM for JSON with one strict retry before fallback."""
    metadata = {"llm_used": bool(llm), "llm_failure": False, "deterministic_fallback": False}
    if llm is None:
        metadata["deterministic_fallback"] = True
        metadata["error"] = "no_local_llm_configured"
        return {}, metadata

    first = llm.generate_json(system_prompt, user_prompt, schema_hint)
    if not first.get("error"):
        return first, metadata

    retry_prompt = strict_json_retry_prompt(user_prompt, schema_hint)
    second = llm.generate_json(system_prompt, retry_prompt, schema_hint)
    if not second.get("error"):
        metadata["retry_used"] = True
        return second, metadata

    metadata.update(
        {
            "llm_failure": True,
            "deterministic_fallback": True,
            "error": second.get("error") or first.get("error"),
            "raw_text": second.get("raw_text") or first.get("raw_text"),
        }
    )
    return {}, metadata


class SourceValidator:
    """Validate source trust against the configured source policy."""

    role = "source_validator"

    def review(self, sources: list[RankedSource]) -> CouncilDecision:
        rejected = [source.domain for source in sources if is_rejected_domain(source.domain)]
        reliable = [source for source in sources if source.source_tier in {1, 2}]
        tier_counts = {
            "tier_1": sum(1 for source in sources if source.source_tier == 1),
            "tier_2": sum(1 for source in sources if source.source_tier == 2),
            "tier_3": sum(1 for source in sources if source.source_tier == 3),
            "unknown": sum(1 for source in sources if source.source_tier == 99),
            "rejected_domains": rejected,
        }
        if not sources or not reliable:
            return CouncilDecision(self.role, "reject", "No reliable Tier 1 or Tier 2 sources were available.", tier_counts)
        if rejected and len(rejected) == len(sources):
            return CouncilDecision(self.role, "reject", "All sources came from rejected domains.", tier_counts)
        if tier_counts["tier_1"] > 0:
            return CouncilDecision(self.role, "accept", "At least one Tier 1 source supports review.", tier_counts)
        return CouncilDecision(self.role, "accept_with_caution", "Tier 2 sources are available but no Tier 1 source was found.", tier_counts)


class RecencyChecker:
    """Review source dates for guideline freshness."""

    role = "recency_checker"

    def review(self, sources: list[RankedSource]) -> CouncilDecision:
        current_year = datetime.now(timezone.utc).year
        missing = []
        old = []
        recent = []
        for source in sources:
            match = re.search(r"\b(20\d{2}|19\d{2})\b", source.published_or_updated_date or "")
            if not match:
                missing.append(source.url)
                continue
            year = int(match.group(1))
            if current_year - year <= 5:
                recent.append(source.url)
            elif current_year - year >= 10:
                old.append(source.url)

        details = {"recent_sources": recent, "missing_dates": missing, "old_sources": old}
        if old:
            return CouncilDecision(self.role, "accept_with_caution", "One or more sources appear older than 10 years.", details)
        if missing and not recent:
            return CouncilDecision(self.role, "accept_with_caution", "Source dates are missing or unclear.", details)
        return CouncilDecision(self.role, "accept", "Source recency is acceptable for preliminary evidence review.", details)


class MedicalClaimExtractor:
    """Extract candidate medical claims from source snippets and text."""

    role = "medical_claim_extractor"

    def extract_deterministic(self, sources: list[RankedSource], max_claims: int = 8) -> list[MedicalClaim]:
        """Extract claims using deterministic sentence matching."""
        claims: list[MedicalClaim] = []
        seen: set[str] = set()
        for source in sources:
            text = source.text or source.snippet
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sentence in sentences:
                cleaned = re.sub(r"\s+", " ", sentence).strip()
                if len(cleaned) < 40:
                    continue
                if not any(keyword in cleaned.lower() for keyword in CLAIM_KEYWORDS):
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                confidence = "high" if source.source_tier == 1 else "moderate" if source.source_tier == 2 else "low"
                claims.append(MedicalClaim(cleaned[:500], [source.url], confidence))
                if len(claims) >= max_claims:
                    return claims
        return claims

    def extract(
        self,
        sources: list[RankedSource],
        clinical_question: str = "",
        max_claims: int = 8,
        use_llm: bool = False,
        llm: BaseLocalLLM | None = None,
    ) -> tuple[list[MedicalClaim], CouncilDecision]:
        """Extract candidate medical claims with optional local LLM assistance."""
        deterministic_claims = self.extract_deterministic(sources, max_claims=max_claims)
        if not use_llm:
            return deterministic_claims, CouncilDecision(
                self.role,
                "accept_with_caution" if deterministic_claims else "reject",
                "Deterministic claim extraction completed.",
                {"mode": "deterministic", "claim_count": len(deterministic_claims)},
            )

        extracted: list[MedicalClaim] = []
        failures = 0
        fallbacks = 0
        for source in sources:
            payload, metadata = _llm_json_with_retry(
                llm,
                CLAIM_EXTRACTION_SYSTEM,
                build_claim_extraction_prompt(clinical_question, source),
                CLAIM_SCHEMA,
            )
            failures += int(metadata.get("llm_failure", False))
            fallbacks += int(metadata.get("deterministic_fallback", False))
            for item in payload.get("claims", []) if isinstance(payload.get("claims", []), list) else []:
                claim = str(item.get("claim", "")).strip()
                if not claim:
                    continue
                confidence = item.get("confidence", "low")
                if confidence not in {"high", "moderate", "low"}:
                    confidence = "low"
                extracted.append(
                    MedicalClaim(
                        claim=claim[:500],
                        supporting_sources=[str(item.get("source_url") or source.url)],
                        confidence=confidence,
                    )
                )
                if len(extracted) >= max_claims:
                    break
            if len(extracted) >= max_claims:
                break

        if not extracted:
            extracted = deterministic_claims
            fallbacks += 1
        decision = "accept" if extracted else "reject"
        reason = "Local LLM claim extraction completed." if failures == 0 and extracted else "Local LLM claim extraction fell back to deterministic extraction."
        return extracted[:max_claims], CouncilDecision(
            self.role,
            decision,
            reason,
            {
                "mode": "local_llm",
                "claim_count": len(extracted[:max_claims]),
                "llm_failure_count": failures,
                "deterministic_fallback_count": fallbacks,
            },
        )


class ConflictChecker:
    """Detect simple contradictions in extracted claims."""

    role = "conflict_checker"

    def review_deterministic(self, claims: list[MedicalClaim]) -> tuple[CouncilDecision, list[str]]:
        """Detect conflicts using deterministic keyword patterns."""
        text = " ".join(claim.claim.lower() for claim in claims)
        conflicts: list[str] = []
        if "recommend" in text and ("not recommend" in text or "do not recommend" in text):
            conflicts.append("Detected both recommendation and non-recommendation language.")
        if "first-line" in text and "avoid" in text:
            conflicts.append("Detected possible first-line versus avoid language.")
        if "contraindicated" in text and "should" in text:
            conflicts.append("Detected possible contraindication affecting recommended action.")

        if conflicts:
            return CouncilDecision(self.role, "accept_with_caution", "Potential conflicts require physician review.", {"conflicts": conflicts}), conflicts
        return CouncilDecision(self.role, "accept_with_caution", "No explicit conflict found; deterministic checker is limited.", {"conflicts": []}), []

    def review(
        self,
        claims: list[MedicalClaim],
        use_llm: bool = False,
        llm: BaseLocalLLM | None = None,
    ) -> tuple[CouncilDecision, list[str]]:
        """Review conflicts with optional local LLM assistance."""
        deterministic_decision, deterministic_conflicts = self.review_deterministic(claims)
        if not use_llm:
            deterministic_decision.details["mode"] = "deterministic"
            return deterministic_decision, deterministic_conflicts

        payload, metadata = _llm_json_with_retry(llm, CONFLICT_CHECK_SYSTEM, build_conflict_prompt(claims), DECISION_SCHEMA)
        if metadata.get("deterministic_fallback") or not _is_valid_decision(payload.get("decision")):
            if not _is_valid_decision(payload.get("decision")):
                metadata["deterministic_fallback"] = True
                metadata["error"] = "invalid_or_missing_decision"
            deterministic_decision.details.update(metadata)
            deterministic_decision.details["mode"] = "local_llm_fallback"
            return deterministic_decision, deterministic_conflicts

        conflicts = [str(item) for item in payload.get("conflicts", []) if str(item).strip()]
        return CouncilDecision(
            self.role,
            payload["decision"],
            str(payload.get("reason", "Local LLM conflict review completed.")),
            {"mode": "local_llm", "conflicts": conflicts, **metadata},
        ), conflicts


class SafetyCritic:
    """Check for unsafe patient-directed treatment language."""

    role = "safety_critic"

    def review_deterministic(self, claims: list[MedicalClaim], summary: str = "") -> CouncilDecision:
        """Review safety using deterministic patterns."""
        combined = f"{summary} {' '.join(claim.claim for claim in claims)}".lower()
        unsafe_patterns = [
            r"\btake\s+\d+\s*(mg|mcg|g)\b",
            r"\bstop taking\b",
            r"\bstart taking\b",
            r"\bdouble your dose\b",
        ]
        matched = [pattern for pattern in unsafe_patterns if re.search(pattern, combined)]
        if matched:
            return CouncilDecision(
                self.role,
                "reject",
                "Detected direct medication instructions unsuitable for an evidence summary.",
                {"matched_patterns": matched},
            )
        if not claims:
            return CouncilDecision(self.role, "accept_with_caution", "No extractive claims were available for safety review.", {})
        return CouncilDecision(
            self.role,
            "accept",
            "Claims are phrased as preliminary evidence for physician review, not patient instructions.",
            {},
        )

    def review(
        self,
        claims: list[MedicalClaim],
        summary: str = "",
        use_llm: bool = False,
        llm: BaseLocalLLM | None = None,
    ) -> CouncilDecision:
        """Review safety with optional local LLM assistance."""
        deterministic_decision = self.review_deterministic(claims, summary)
        if not use_llm:
            deterministic_decision.details["mode"] = "deterministic"
            return deterministic_decision

        payload, metadata = _llm_json_with_retry(llm, SAFETY_CRITIC_SYSTEM, build_safety_prompt(summary, claims), DECISION_SCHEMA)
        if metadata.get("deterministic_fallback") or not _is_valid_decision(payload.get("decision")):
            if not _is_valid_decision(payload.get("decision")):
                metadata["deterministic_fallback"] = True
                metadata["error"] = "invalid_or_missing_decision"
            deterministic_decision.details.update(metadata)
            deterministic_decision.details["mode"] = "local_llm_fallback"
            return deterministic_decision

        return CouncilDecision(
            self.role,
            payload["decision"],
            str(payload.get("reason", "Local LLM safety review completed.")),
            {
                "mode": "local_llm",
                "warnings": [str(item) for item in payload.get("warnings", [])],
                **metadata,
            },
        )


class FinalReviewer:
    """Combine council decisions into the final decision."""

    role = "final_reviewer"

    def review_deterministic(
        self,
        decisions: list[CouncilDecision],
        privacy_report: PrivacyReport,
        sources: list[RankedSource],
        conflicts: list[str],
    ) -> CouncilDecision:
        if not privacy_report.deidentified:
            return CouncilDecision(self.role, "reject", "Privacy report indicates de-identification failed.", {})
        if any(decision.decision == "reject" for decision in decisions):
            return CouncilDecision(self.role, "reject", "At least one council role rejected the evidence.", {})
        if not any(source.source_tier in {1, 2} for source in sources):
            return CouncilDecision(self.role, "reject", "No reliable source tier is present.", {})
        if conflicts or any(decision.decision == "accept_with_caution" for decision in decisions):
            return CouncilDecision(self.role, "accept_with_caution", "Evidence is usable with limitations for physician review.", {})
        return CouncilDecision(self.role, "accept", "Evidence passed source, recency, conflict, privacy, and safety checks.", {})

    def review(
        self,
        decisions: list[CouncilDecision],
        privacy_report: PrivacyReport,
        sources: list[RankedSource],
        conflicts: list[str],
        use_llm: bool = False,
        llm: BaseLocalLLM | None = None,
    ) -> CouncilDecision:
        """Combine council decisions, optionally asking a local LLM to review."""
        deterministic_decision = self.review_deterministic(decisions, privacy_report, sources, conflicts)
        if not use_llm:
            deterministic_decision.details["mode"] = "deterministic"
            return deterministic_decision

        snapshot = {
            "decisions": [decision.to_dict() for decision in decisions],
            "privacy_report": privacy_report.to_dict(),
            "source_count": len(sources),
            "conflicts": conflicts,
            "deterministic_decision": deterministic_decision.to_dict(),
        }
        payload, metadata = _llm_json_with_retry(llm, FINAL_REVIEW_SYSTEM, build_final_review_prompt(snapshot), DECISION_SCHEMA)
        if metadata.get("deterministic_fallback") or not _is_valid_decision(payload.get("decision")):
            if not _is_valid_decision(payload.get("decision")):
                metadata["deterministic_fallback"] = True
                metadata["error"] = "invalid_or_missing_decision"
            deterministic_decision.details.update(metadata)
            deterministic_decision.details["mode"] = "local_llm_fallback"
            return deterministic_decision
        return CouncilDecision(
            self.role,
            payload["decision"],
            str(payload.get("reason", "Local LLM final review completed.")),
            {"mode": "local_llm", **metadata},
        )


def run_council(
    sources: list[RankedSource],
    claims: list[MedicalClaim],
    privacy_report: PrivacyReport,
    summary: str = "",
    claim_extractor_decision: CouncilDecision | None = None,
    use_llm_conflict_checking: bool = False,
    use_llm_safety_review: bool = False,
    use_llm_final_review: bool = False,
    llm: BaseLocalLLM | None = None,
) -> tuple[CouncilReview, list[str]]:
    """Run council review roles and return the review plus conflicts."""
    source_decision = SourceValidator().review(sources)
    recency_decision = RecencyChecker().review(sources)
    conflict_decision, conflicts = ConflictChecker().review(claims, use_llm_conflict_checking, llm)
    safety_decision = SafetyCritic().review(claims, summary, use_llm_safety_review, llm)
    final_decision = FinalReviewer().review(
        [source_decision, recency_decision, conflict_decision, safety_decision],
        privacy_report,
        sources,
        conflicts,
        use_llm_final_review,
        llm,
    )
    return (
        CouncilReview(
            source_validator=source_decision,
            recency_checker=recency_decision,
            conflict_checker=conflict_decision,
            safety_critic=safety_decision,
            final_reviewer=final_decision,
            claim_extractor=claim_extractor_decision,
        ),
        conflicts,
    )
