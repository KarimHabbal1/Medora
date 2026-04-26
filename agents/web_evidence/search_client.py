"""Pluggable search clients for the web evidence agent."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from agents.web_evidence.schemas import SearchResult


class BaseSearchClient:
    """Abstract search client interface."""

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Return search results for a sanitized query."""
        raise NotImplementedError


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


class SearxNGSearchClient(BaseSearchClient):
    """Search client for a self-hosted SearxNG instance."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("SEARXNG_BASE_URL") or "").rstrip("/")

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search a configured SearxNG backend; return an error result if unavailable."""
        if not self.base_url:
            return [
                SearchResult(
                    title="Search backend not configured",
                    url="",
                    snippet="Set SEARXNG_BASE_URL or run in mock mode.",
                    error="missing_searxng_base_url",
                )
            ]

        try:
            import requests

            response = requests.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json", "language": "en"},
                timeout=12,
                headers={"User-Agent": "MedoraWebEvidenceAgent/1.0"},
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except Exception as exc:  # noqa: BLE001
            return [
                SearchResult(
                    title="Search backend error",
                    url="",
                    snippet=str(exc),
                    error="search_backend_error",
                )
            ]

        results: list[SearchResult] = []
        for item in payload.get("results", [])[:max_results]:
            url = str(item.get("url", ""))
            if not url:
                continue
            results.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=url,
                    snippet=str(item.get("content", "") or item.get("snippet", "")),
                    domain=_domain(url),
                )
            )
        return results


class StaticSearchClient(BaseSearchClient):
    """Deterministic offline search client for tests and validation."""

    STATIC_RESULTS = [
        SearchResult(
            title="IDSA community-acquired pneumonia guideline",
            url="https://www.idsociety.org/practice-guideline/community-acquired-pneumonia-cap-in-adults/",
            domain="idsociety.org",
            published_or_updated_date="2019-10-01",
            snippet="Guideline recommendations for diagnosis and antibiotic management of community-acquired pneumonia in adults.",
        ),
        SearchResult(
            title="CDC clinical guidance for respiratory infections",
            url="https://www.cdc.gov/respiratory-viruses/hcp/clinical-guidance/",
            domain="cdc.gov",
            published_or_updated_date="2024-03-01",
            snippet="Clinical guidance and public-health recommendations for respiratory infection management and prevention.",
        ),
        SearchResult(
            title="NICE pneumonia in adults guideline",
            url="https://www.nice.org.uk/guidance/ng138",
            domain="nice.org.uk",
            published_or_updated_date="2023-10-31",
            snippet="NICE guideline for pneumonia diagnosis and management in adults, including antimicrobial prescribing considerations.",
        ),
        SearchResult(
            title="FDA drug safety communications",
            url="https://www.fda.gov/drugs/drug-safety-and-availability/drug-safety-communications",
            domain="fda.gov",
            published_or_updated_date="2024-01-15",
            snippet="Drug safety communications include warnings, contraindications, and updated safety information.",
        ),
        SearchResult(
            title="CDC adult immunization schedule",
            url="https://www.cdc.gov/vaccines/hcp/imz-schedules/adult-age.html",
            domain="cdc.gov",
            published_or_updated_date="2025-02-01",
            snippet="Adult vaccination recommendations by age, medical condition, and risk group.",
        ),
        SearchResult(
            title="USPSTF screening recommendations",
            url="https://www.uspreventiveservicestaskforce.org/uspstf/recommendation-topics",
            domain="uspreventiveservicestaskforce.org",
            published_or_updated_date="2024-12-01",
            snippet="Evidence-based screening recommendation statements for preventive care.",
        ),
    ]

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Return deterministic trusted-looking results without network access."""
        query_terms = set(query.lower().split())

        def score(result: SearchResult) -> int:
            haystack = f"{result.title} {result.snippet}".lower()
            return sum(1 for term in query_terms if term in haystack)

        ranked = sorted(self.STATIC_RESULTS, key=score, reverse=True)
        return ranked[:max_results]
