"""Page fetching and rough text extraction for web evidence sources."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from agents.web_evidence.config import DEFAULT_TIMEOUT_SECONDS, MAX_FETCHED_TEXT_CHARS
from agents.web_evidence.schemas import FetchedSource, SearchResult


MOCK_TEXT_BY_DOMAIN = {
    "idsociety.org": (
        "Guideline recommendations for community-acquired pneumonia in adults should be interpreted "
        "with clinical judgment. Recommendations discuss empiric antibiotic treatment, diagnostic "
        "assessment, and management decisions for physician review."
    ),
    "cdc.gov": (
        "CDC clinical guidance and vaccination recommendations are updated for public-health practice. "
        "Clinicians should consider risk groups, warnings, contraindications, and current surveillance."
    ),
    "nice.org.uk": (
        "NICE guideline recommendations describe diagnosis, severity assessment, antimicrobial prescribing, "
        "and management of pneumonia in adults. Recommendations require clinician judgment."
    ),
    "fda.gov": (
        "FDA drug safety communications provide warnings, contraindications, and updated safety information "
        "for medicines. Clinicians should review the full label before treatment decisions."
    ),
    "uspreventiveservicestaskforce.org": (
        "USPSTF screening recommendations summarize evidence for preventive services. Screening decisions "
        "should account for patient risk factors and clinician judgment."
    ),
}


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_date(soup) -> str | None:
    meta_names = [
        "article:published_time",
        "article:modified_time",
        "date",
        "dc.date",
        "dc.date.issued",
        "last-modified",
    ]
    for name in meta_names:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return str(tag["content"])[:10]
    time_tag = soup.find("time")
    if time_tag:
        return str(time_tag.get("datetime") or time_tag.get_text(" ", strip=True))[:10]
    return None


def fetched_from_search_result(result: SearchResult) -> FetchedSource:
    """Create a deterministic fetched source from a search result for mock mode."""
    domain = result.domain or _domain(result.url)
    text = MOCK_TEXT_BY_DOMAIN.get(domain, result.snippet)
    return FetchedSource(
        title=result.title,
        url=result.url,
        domain=domain,
        text=text,
        snippet=result.snippet,
        published_or_updated_date=result.published_or_updated_date,
        status_code=200,
    )


def fetch_url(url: str) -> FetchedSource:
    """Fetch and clean a web page, returning structured errors instead of raising."""
    domain = _domain(url)
    if not url:
        return FetchedSource(title="", url=url, domain=domain, error="empty_url")

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        return FetchedSource(
            title="",
            url=url,
            domain=domain,
            error=f"missing_dependency: {exc}",
        )

    try:
        response = requests.get(
            url,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            headers={"User-Agent": "MedoraWebEvidenceAgent/1.0"},
        )
        status_code = response.status_code
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return FetchedSource(
            title="",
            url=url,
            domain=domain,
            status_code=getattr(getattr(exc, "response", None), "status_code", None),
            error=f"fetch_error: {exc}",
        )

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    chunks = [node.get_text(" ", strip=True) for node in soup.find_all(["h1", "h2", "h3", "p", "li"])]
    text = _collapse(" ".join(chunk for chunk in chunks if chunk))
    return FetchedSource(
        title=title,
        url=url,
        domain=domain,
        text=text[:MAX_FETCHED_TEXT_CHARS],
        snippet=text[:500],
        published_or_updated_date=_extract_date(soup),
        status_code=status_code,
    )
