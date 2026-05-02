"""
Phase 6 — Web Search Agent
Searches credible medical sources via SearXNG for symptoms and produces
a diagnosis with evidence from trusted sources only.

Usage:
    python agents/web_search.py --symptoms "painful rash on legs, fever, joint pain"
    python agents/web_search.py --symptoms "chest pain, shortness of breath, arm radiation" --model llama3.1:8b
    python agents/web_search.py --symptoms "persistent cough, weight loss, night sweats" --max-sources 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Path bootstrap — make project root importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Domain lists
# ---------------------------------------------------------------------------
# Only these sources are used — everything else is dropped
WHITELISTED_DOMAINS = {
    "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "mayoclinic.org", "clevelandclinic.org", "uptodate.com",
    "medlineplus.gov", "who.int", "cdc.gov", "nice.org.uk",
    "bmj.com", "nejm.org", "thelancet.com", "nih.gov",
    "hopkinsmedicine.org", "merckmanuals.com", "medscape.com",
}

EXTRACTION_SYSTEM_PROMPT = """\
You are a clinical diagnostic assistant. Given a patient's symptoms and content
from credible medical sources, determine the most likely diagnosis and supporting
evidence.

Rules:
- Analyze the symptoms against the medical source content
- Identify the most likely diagnosis based on the evidence
- List supporting findings from each source
- Note any differential diagnoses mentioned
- For each finding, cite the source it came from
- If sources suggest different diagnoses, note the conflict
- Be concise — one sentence per finding

Return a JSON object:
{
    "primary_diagnosis": "most likely condition based on the evidence",
    "confidence": "high/moderate/low",
    "evidence_summary": "2-3 sentence summary connecting symptoms to diagnosis",
    "key_findings": [
        {"claim": "specific finding supporting the diagnosis", "source": "source name", "url": "source url"}
    ],
    "differential_diagnoses": ["other possible conditions mentioned in sources"]
}
Return ONLY valid JSON."""


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
def make_llm(model: str = "llama3.1:8b", provider: str = None, ollama_url: str = None):
    """Return an LLM client. Defaults to Ollama; pass provider='openai' for OpenAI."""
    resolved_provider = (provider or os.getenv("MEDORA_LLM_PROVIDER") or "ollama").lower()

    if resolved_provider == "openai":
        # Thin wrapper around the openai package
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)

        class _OpenAILLM:
            def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
                resp = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                )
                return json.loads(resp.choices[0].message.content)

        return _OpenAILLM()

    # Default: Ollama
    from agents.web_evidence.llm.ollama_client import OllamaLocalLLM

    return OllamaLocalLLM(model=model, base_url=ollama_url)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _is_whitelisted(domain: str) -> bool:
    """Check if domain is in our whitelist (also matches subdomains)."""
    for wd in WHITELISTED_DOMAINS:
        if domain == wd or domain.endswith("." + wd):
            return True
    return False


def _search_searxng(query: str, searxng_url: str, max_results: int) -> list[dict]:
    """Call SearXNG and return raw result dicts, sorted by domain trust."""
    import requests

    response = requests.get(
        f"{searxng_url}/search",
        params={"q": query, "format": "json", "language": "en"},
        timeout=12,
        headers={"User-Agent": "MedoraWebSearchAgent/1.0"},
    )
    response.raise_for_status()
    raw = response.json().get("results", [])

    results = []
    for item in raw:
        url = str(item.get("url", ""))
        if not url:
            continue
        domain = _domain(url)
        if not _is_whitelisted(domain):
            continue  # only whitelisted sources
        results.append({
            "title": str(item.get("title", "")),
            "url": url,
            "domain": domain,
            "snippet": str(item.get("content", "") or item.get("snippet", "")),
        })

    return results[:max_results]


def _fetch_page_text(url: str, char_limit: int = 2000) -> str:
    """Fetch a URL and extract readable text from HTML. Returns empty string on error."""
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "MedoraWebSearchAgent/1.0"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        tags = soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"])
        text = " ".join(t.get_text(separator=" ", strip=True) for t in tags)
        return text[:char_limit]
    except Exception:  # noqa: BLE001
        return ""


def _parse_llm_json(raw) -> dict:
    """Parse JSON whether the LLM returns a dict already or a string."""
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"No JSON object found in LLM response: {text[:200]}")


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------
def search_medical_evidence(
    symptoms: str,
    llm=None,
    model: str = "llama3.1:8b",
    searxng_url: str = None,
    max_sources: int = 5,
    provider: str = None,
    ollama_url: str = None,
) -> dict:
    """
    Search credible medical sources for a diagnosis based on symptoms.

    Args:
        symptoms: patient symptoms description (e.g., "painful rash on legs, fever, joint pain")

    Returns:
        {
            "symptoms": str,
            "primary_diagnosis": str,
            "confidence": str,
            "sources": [{"title", "url", "domain", "snippet", "content"}],
            "evidence_summary": str,
            "key_findings": [{"claim": str, "source": str, "url": str}],
            "differential_diagnoses": [str],
            "search_query": str,
        }
    """
    # 1. Build search query — frame symptoms as a diagnostic search
    search_query = f"diagnosis symptoms {symptoms}"

    # 2. Resolve SearXNG URL
    base_url = (searxng_url or os.getenv("SEARXNG_BASE_URL") or "").rstrip("/")
    if not base_url:
        raise ValueError("No SearXNG URL — set SEARXNG_BASE_URL or pass --searxng-url")

    # 3. Search
    raw_results = _search_searxng(search_query, base_url, max_sources)

    # 4. Fetch page content
    sources = []
    for item in raw_results:
        content = _fetch_page_text(item["url"])
        sources.append({
            "title": item["title"],
            "url": item["url"],
            "domain": item["domain"],
            "snippet": item["snippet"],
            "content": content,
        })

    # 5. Build LLM
    if llm is None:
        llm = make_llm(model=model, provider=provider, ollama_url=ollama_url)

    # 6. Build user prompt
    source_blocks = []
    for i, src in enumerate(sources, 1):
        text = src["content"] or src["snippet"] or "(no content fetched)"
        source_blocks.append(
            f"[Source {i}] {src['title']}\nURL: {src['url']}\n{text}"
        )

    user_prompt = (
        f"Patient symptoms: {symptoms}\n\n"
        + "\n\n---\n\n".join(source_blocks)
    )

    # 7. LLM extraction
    try:
        raw = llm.generate_json(EXTRACTION_SYSTEM_PROMPT, user_prompt)
        extraction = _parse_llm_json(raw)
    except Exception as exc:  # noqa: BLE001
        extraction = {
            "evidence_summary": f"LLM extraction failed: {exc}",
            "key_findings": [],
        }

    return {
        "symptoms": symptoms,
        "search_query": search_query,
        "primary_diagnosis": extraction.get("primary_diagnosis", ""),
        "confidence": extraction.get("confidence", ""),
        "sources": sources,
        "evidence_summary": extraction.get("evidence_summary", ""),
        "key_findings": extraction.get("key_findings", []),
        "differential_diagnoses": extraction.get("differential_diagnoses", []),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medora web evidence search agent")
    parser.add_argument("--symptoms", required=True, help="Patient symptoms to diagnose")
    parser.add_argument("--model", default="llama3.1:8b", help="LLM model name")
    parser.add_argument("--provider", default=None, help="LLM provider: ollama | openai")
    parser.add_argument("--ollama-url", default=None, help="Ollama base URL")
    parser.add_argument("--searxng-url", default=None, help="SearXNG base URL (overrides env)")
    parser.add_argument("--max-sources", type=int, default=5, help="Max sources to fetch")
    args = parser.parse_args()

    result = search_medical_evidence(
        symptoms=args.symptoms,
        model=args.model,
        provider=args.provider,
        ollama_url=args.ollama_url,
        searxng_url=args.searxng_url,
        max_sources=args.max_sources,
    )

    print(f"\nSymptoms: {result['symptoms']}")
    print(f"Search query: {result['search_query']}")
    print(f"\nPrimary Diagnosis: {result['primary_diagnosis']}")
    print(f"Confidence: {result['confidence']}")
    print(f"\nSources ({len(result['sources'])}):")
    for i, src in enumerate(result["sources"], 1):
        content_len = len(src.get("content") or "")
        print(f"  {i}. [{src['domain']}] {src['title']} ({content_len} chars)")

    print(f"\nEvidence Summary:\n{result['evidence_summary']}")

    if result.get("differential_diagnoses"):
        print(f"\nDifferential Diagnoses: {', '.join(result['differential_diagnoses'])}")

    print(f"\nKey Findings ({len(result['key_findings'])}):")
    for finding in result["key_findings"]:
        print(f"  - {finding.get('claim', '')}")
        print(f"    Source: {finding.get('source', '')} — {finding.get('url', '')}")
