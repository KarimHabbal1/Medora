"""Configuration and source policy for the Phase 6 web evidence agent."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent

TIER_1_DOMAINS = {
    "who.int",
    "cdc.gov",
    "nice.org.uk",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "fda.gov",
    "ema.europa.eu",
}

TIER_2_DOMAINS = {
    "nejm.org",
    "thelancet.com",
    "bmj.com",
    "jamanetwork.com",
    "cochranelibrary.com",
    "acc.org",
    "heart.org",
    "diabetes.org",
    "idsociety.org",
}

TIER_3_DOMAINS = {
    "mayoclinic.org",
    "clevelandclinic.org",
    "hopkinsmedicine.org",
    "msdmanuals.com",
}

REJECTED_DOMAINS = {
    "reddit.com",
    "quora.com",
    "wikipedia.org",
    "healthline.com",
    "webmd.com",
    "verywellhealth.com",
    "medicalnewstoday.com",
}

REJECTED_DOMAIN_KEYWORDS = {
    "blog",
    "forum",
    "forums",
    "substack",
    "medium.com",
}

SOURCE_TYPE_KEYWORDS = {
    "guideline": ("guideline", "recommendation", "practice guideline", "clinical guidance"),
    "systematic_review": ("systematic review", "meta-analysis", "cochrane"),
    "journal_article": ("journal", "trial", "study", "article", "nejm", "lancet", "jama", "bmj"),
    "government": ("cdc", "nih", "fda", "ema", "who", "public health"),
    "hospital_reference": ("mayo clinic", "cleveland clinic", "hopkins", "msd manual"),
}

SOURCE_TIER_SCORES = {
    1: 0.95,
    2: 0.82,
    3: 0.65,
    99: 0.35,
}

GUIDELINE_BONUS = 0.08
GOVERNMENT_BONUS = 0.06
RECENCY_BONUS = 0.05
MISSING_DATE_PENALTY = 0.04
OLD_SOURCE_PENALTY = 0.12
REJECTED_DOMAIN_SCORE = 0.0

DEFAULT_TIMEOUT_SECONDS = 10
MAX_FETCHED_TEXT_CHARS = 16000
DEFAULT_MAX_SEARCH_RESULTS = 10
DEFAULT_MAX_SOURCES = 5

TRUSTED_DOMAIN_HINTS = ("WHO", "CDC", "NICE", "NIH", "PubMed")
