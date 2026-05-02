"""Source filtering, classification, and ranking for web evidence."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urlparse

from agents.web_evidence.config import (
    GOVERNMENT_BONUS,
    GUIDELINE_BONUS,
    MISSING_DATE_PENALTY,
    OLD_SOURCE_PENALTY,
    RECENCY_BONUS,
    REJECTED_DOMAIN_KEYWORDS,
    REJECTED_DOMAIN_SCORE,
    REJECTED_DOMAINS,
    SOURCE_TIER_SCORES,
    SOURCE_TYPE_KEYWORDS,
    TIER_1_DOMAINS,
    TIER_2_DOMAINS,
    TIER_3_DOMAINS,
)
from agents.web_evidence.schemas import FetchedSource, RankedSource, SearchResult


def normalize_domain(url_or_domain: str) -> str:
    """Normalize a URL or domain into a lower-case hostname."""
    if "://" in url_or_domain:
        return urlparse(url_or_domain).netloc.lower().removeprefix("www.")
    return url_or_domain.lower().removeprefix("www.")


def source_tier(domain: str) -> int:
    """Return source tier, using 99 for unknown sources."""
    domain = normalize_domain(domain)
    if domain in TIER_1_DOMAINS or any(domain.endswith(f".{d}") for d in TIER_1_DOMAINS):
        return 1
    if domain in TIER_2_DOMAINS or any(domain.endswith(f".{d}") for d in TIER_2_DOMAINS):
        return 2
    if domain in TIER_3_DOMAINS or any(domain.endswith(f".{d}") for d in TIER_3_DOMAINS):
        return 3
    return 99


def is_rejected_domain(domain: str) -> bool:
    """Return True if source policy rejects this domain by default."""
    normalized = normalize_domain(domain)
    return (
        normalized in REJECTED_DOMAINS
        or any(normalized.endswith(f".{d}") for d in REJECTED_DOMAINS)
        or any(keyword in normalized for keyword in REJECTED_DOMAIN_KEYWORDS)
    )


def classify_source_type(title: str, snippet: str, domain: str) -> str:
    """Classify source type using deterministic keyword matching."""
    haystack = f"{title} {snippet} {domain}".lower()
    for source_type, keywords in SOURCE_TYPE_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return source_type
    if source_tier(domain) == 1:
        return "government"
    return "unknown"


def _date_year(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\b(20\d{2}|19\d{2})\b", value)
    return int(match.group(1)) if match else None


def _relevance_score(query: str, text: str) -> float:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
    if not query_terms:
        return 0.0
    haystack = text.lower()
    matched = sum(1 for term in query_terms if term in haystack)
    return min(0.12, matched / max(len(query_terms), 1) * 0.12)


def _base_fields(source: SearchResult | FetchedSource) -> tuple[str, str, str, str, str | None, str | None]:
    title = source.title
    url = source.url
    domain = source.domain or normalize_domain(source.url)
    snippet = source.snippet
    date = source.published_or_updated_date
    error = source.error
    return title, url, domain, snippet, date, error


def rank_sources(sources: Iterable[SearchResult | FetchedSource | RankedSource], query: str) -> list[RankedSource]:
    """Rank search/fetched sources by trust policy, recency, and relevance."""
    ranked: list[RankedSource] = []
    current_year = datetime.now(timezone.utc).year

    for source in sources:
        if isinstance(source, RankedSource):
            ranked.append(source)
            continue

        title, url, domain, snippet, date, error = _base_fields(source)
        tier = source_tier(domain)
        source_type = classify_source_type(title, snippet, domain)

        if is_rejected_domain(domain):
            score = REJECTED_DOMAIN_SCORE
        else:
            score = SOURCE_TIER_SCORES.get(tier, SOURCE_TIER_SCORES[99])
            if source_type == "guideline":
                score += GUIDELINE_BONUS
            if source_type == "government":
                score += GOVERNMENT_BONUS
            year = _date_year(date)
            if year is None:
                score -= MISSING_DATE_PENALTY
            elif current_year - year <= 5:
                score += RECENCY_BONUS
            elif current_year - year >= 10:
                score -= OLD_SOURCE_PENALTY
            score += _relevance_score(query, f"{title} {snippet}")

        text = source.text if isinstance(source, FetchedSource) else ""
        ranked.append(
            RankedSource(
                title=title,
                url=url,
                domain=domain,
                source_tier=tier,
                source_type=source_type,
                published_or_updated_date=date,
                reliability_score=round(max(0.0, min(score, 1.0)), 3),
                snippet=snippet,
                text=text,
                error=error,
            )
        )

    return sorted(ranked, key=lambda item: item.reliability_score, reverse=True)
