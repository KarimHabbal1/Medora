"""
Phase 2.3 — Retrieval Validation
Validates ChromaDB retrieval quality by running manual gold-standard queries
against the `tmt_chunks` collection and computing Hit@k and MRR metrics.

Usage:
    python evaluation/retrieval_validation.py
    python evaluation/retrieval_validation.py --k 10
    python evaluation/retrieval_validation.py --verbose
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Project root on sys.path so config.py is importable ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RESULTS_DIR,
)

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "tmt_chunks"

# ── Manual gold-standard queries ──────────────────────────────────────────────
#
# 20 queries covering the 11 Common Symptoms sections, key condition chapters,
# and cross-chapter / emergency scenarios.
#
# expected_chapter: the chapter that MUST appear in top-k results.
# expected_section: the section that should appear (None = chapter-level only).
# category: one of "symptom", "condition", "emergency".

MANUAL_QUERIES: list[dict] = [
    # ── Symptom queries (11) ──────────────────────────────────────────────────
    {
        "query": "patient with persistent dry cough for 3 weeks",
        "expected_chapter": "Common Symptoms",
        "expected_section": "COUGH",
        "category": "symptom",
    },
    {
        "query": "progressive shortness of breath on exertion",
        "expected_chapter": "Common Symptoms",
        "expected_section": "DYSPNEA",
        "category": "symptom",
    },
    {
        "query": "sharp chest pain radiating to the left arm",
        "expected_chapter": "Common Symptoms",
        "expected_section": "CHEST PAIN",
        "category": "symptom",
    },
    {
        "query": "heart racing and fluttering sensation",
        "expected_chapter": "Common Symptoms",
        "expected_section": "PALPITATIONS",
        "category": "symptom",
    },
    {
        "query": "swollen ankles and legs bilateral",
        "expected_chapter": "Common Symptoms",
        "expected_section": "LOWER EXTREMITY EDEMA",
        "category": "symptom",
    },
    {
        "query": "high fever with chills and night sweats",
        "expected_chapter": "Common Symptoms",
        "expected_section": "FEVER",
        "category": "symptom",
    },
    {
        "query": "unintentional weight loss of 10 pounds in 2 months",
        "expected_chapter": "Common Symptoms",
        "expected_section": "INVOLUNTARY WEIGHT LOSS",
        "category": "symptom",
    },
    {
        "query": "extreme fatigue and tiredness for several weeks",
        "expected_chapter": "Common Symptoms",
        "expected_section": "FATIGUE",
        "category": "symptom",
    },
    {
        "query": "sudden severe headache worst of my life",
        "expected_chapter": "Common Symptoms",
        "expected_section": "ACUTE HEADACHE",
        "category": "symptom",
    },
    {
        "query": "painful urination with increased frequency",
        "expected_chapter": "Common Symptoms",
        "expected_section": "DYSURIA",
        "category": "symptom",
    },
    {
        "query": "coughing up blood streaked sputum",
        "expected_chapter": "Common Symptoms",
        "expected_section": "HEMOPTYSIS",
        "category": "symptom",
    },
    # ── Condition queries (6) ─────────────────────────────────────────────────
    {
        "query": "management of type 2 diabetes with metformin",
        "expected_chapter": "Diabetes Mellitus & Hypoglycemia",
        "expected_section": None,
        "category": "condition",
    },
    {
        "query": "treatment of atrial fibrillation",
        "expected_chapter": "Heart Disease",
        "expected_section": None,
        "category": "condition",
    },
    {
        "query": "pneumonia diagnosis and antibiotics",
        "expected_chapter": "Pulmonary Disorders",
        "expected_section": None,
        "category": "condition",
    },
    {
        "query": "acute kidney injury creatinine elevated",
        "expected_chapter": "Kidney Disease",
        "expected_section": None,
        "category": "condition",
    },
    {
        "query": "rheumatoid arthritis joint inflammation treatment",
        "expected_chapter": "Rheumatologic, Immunologic, & Allergic Disorders",
        "expected_section": None,
        "category": "condition",
    },
    {
        "query": "major depressive disorder SSRI treatment",
        "expected_chapter": "Psychiatric Disorders",
        "expected_section": None,
        "category": "condition",
    },
    # ── Cross-chapter / emergency queries (3) ─────────────────────────────────
    {
        "query": "fever in immunocompromised patient neutropenia",
        "expected_chapter": "Common Problems in Infectious Diseases & Antimicrobial Therapy",
        "expected_section": None,
        "category": "emergency",
    },
    {
        "query": "acute myocardial infarction emergency management",
        "expected_chapter": "Heart Disease",
        "expected_section": None,
        "category": "emergency",
    },
    {
        "query": "diabetic ketoacidosis DKA treatment protocol",
        "expected_chapter": "Diabetes Mellitus & Hypoglycemia",
        "expected_section": None,
        "category": "emergency",
    },
]

# ── Metadata filter tests ─────────────────────────────────────────────────────
#
# Each entry carries a short query and a ChromaDB `where` clause.
# After retrieval we verify that every returned chunk's chapter matches the
# filter value.

FILTER_QUERIES: list[dict] = [
    {
        "query": "chest pain",
        "where": {"chapter": "Common Symptoms"},
        "expected_chapter": "Common Symptoms",
    },
    {
        "query": "treatment options",
        "where": {"chapter": "Heart Disease"},
        "expected_chapter": "Heart Disease",
    },
    {
        "query": "when to admit",
        "where": {"chapter": "Pulmonary Disorders"},
        "expected_chapter": "Pulmonary Disorders",
    },
]


# ── Device detection ──────────────────────────────────────────────────────────

def detect_device() -> str:
    """Return 'mps' if Apple Silicon GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def open_collection(chroma_dir: Path):
    """Open the persistent ChromaDB client and return the tmt_chunks collection."""
    import chromadb
    from chromadb.config import Settings

    print(f"Opening ChromaDB client at {chroma_dir} …")
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"  Collection '{COLLECTION_NAME}' has {collection.count():,} chunks.")
    return collection


# ── Embedding helpers ─────────────────────────────────────────────────────────

def load_model(model_name: str, device: str):
    """Load and return a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model '{model_name}' on device '{device}' …")
    model = SentenceTransformer(model_name, device=device)
    print("  Model loaded.")
    return model


def encode_query(model, query: str) -> list[float]:
    """Encode a single query string into a normalised embedding vector."""
    vec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    return vec.tolist()


# ── Core retrieval ────────────────────────────────────────────────────────────

def query_collection(
    collection,
    query_vec: list[float],
    k: int,
    where: dict | None = None,
) -> list[dict]:
    """
    Run a vector query against the collection.
    Returns a list of result dicts with keys: chunk_id, chapter, section, distance.
    """
    kwargs = dict(
        query_embeddings=[query_vec],
        n_results=k,
        include=["metadatas", "distances"],
    )
    if where is not None:
        kwargs["where"] = where

    raw = collection.query(**kwargs)

    results = []
    for chunk_id, meta, dist in zip(
        raw["ids"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append(
            {
                "chunk_id": chunk_id,
                "chapter": meta.get("chapter", ""),
                "section": meta.get("section", ""),
                "distance": round(float(dist), 6),
            }
        )
    return results


# ── Metric helpers ────────────────────────────────────────────────────────────

def find_first_chapter_rank(results: list[dict], expected_chapter: str) -> int | None:
    """Return the 1-based rank of the first result whose chapter matches, or None."""
    for rank, r in enumerate(results, start=1):
        if r["chapter"] == expected_chapter:
            return rank
    return None


def find_first_section_rank(results: list[dict], expected_section: str) -> int | None:
    """Return the 1-based rank of the first result whose section matches, or None."""
    for rank, r in enumerate(results, start=1):
        if expected_section.upper() in r["section"].upper():
            return rank
    return None


def hits_in_top_k(rank: int | None, k: int) -> bool:
    """True if rank is defined and <= k."""
    return rank is not None and rank <= k


def compute_metrics(per_query: list[dict], k: int) -> dict:
    """
    Compute Hit@1, Hit@3, Hit@k, and MRR at the chapter level.
    Also compute section-level metrics for queries that have expected_section set.
    """
    n = len(per_query)

    # Chapter-level
    ch_hit1 = sum(1 for q in per_query if hits_in_top_k(q["chapter_rank"], 1)) / n
    ch_hit3 = sum(1 for q in per_query if hits_in_top_k(q["chapter_rank"], 3)) / n
    ch_hitk = sum(1 for q in per_query if hits_in_top_k(q["chapter_rank"], k)) / n
    ch_mrr  = sum(
        1.0 / q["chapter_rank"] if q["chapter_rank"] is not None else 0.0
        for q in per_query
    ) / n

    # Section-level (only for queries where expected_section was specified)
    sec_queries = [q for q in per_query if q["expected_section"] is not None]
    if sec_queries:
        ns = len(sec_queries)
        sec_hit1 = sum(1 for q in sec_queries if hits_in_top_k(q["section_rank"], 1)) / ns
        sec_hit3 = sum(1 for q in sec_queries if hits_in_top_k(q["section_rank"], 3)) / ns
        sec_hitk = sum(1 for q in sec_queries if hits_in_top_k(q["section_rank"], k)) / ns
        sec_mrr  = sum(
            1.0 / q["section_rank"] if q["section_rank"] is not None else 0.0
            for q in sec_queries
        ) / ns
    else:
        sec_hit1 = sec_hit3 = sec_hitk = sec_mrr = 0.0

    return {
        "chapter_level": {
            "Hit@1":  round(ch_hit1,  4),
            "Hit@3":  round(ch_hit3,  4),
            f"Hit@{k}": round(ch_hitk, 4),
            "MRR":    round(ch_mrr,   4),
        },
        "section_level": {
            "num_queries_with_section": len(sec_queries),
            "Hit@1":  round(sec_hit1, 4),
            "Hit@3":  round(sec_hit3, 4),
            f"Hit@{k}": round(sec_hitk, 4),
            "MRR":    round(sec_mrr,  4),
        },
    }


# ── Main query loop ───────────────────────────────────────────────────────────

def run_manual_queries(
    collection,
    model,
    k: int,
    verbose: bool,
) -> list[dict]:
    """
    Run all MANUAL_QUERIES against the collection.
    Returns a list of per-query result dicts.
    """
    print(f"\n=== Running {len(MANUAL_QUERIES)} manual queries (k={k}) ===\n")
    per_query_results = []

    for i, entry in enumerate(MANUAL_QUERIES, start=1):
        query           = entry["query"]
        exp_chapter     = entry["expected_chapter"]
        exp_section     = entry["expected_section"]
        category        = entry["category"]

        query_vec = encode_query(model, query)
        results   = query_collection(collection, query_vec, k)

        ch_rank  = find_first_chapter_rank(results, exp_chapter)
        sec_rank = (
            find_first_section_rank(results, exp_section)
            if exp_section is not None
            else None
        )

        ch_hit1  = hits_in_top_k(ch_rank, 1)
        ch_hit3  = hits_in_top_k(ch_rank, 3)
        ch_hitk  = hits_in_top_k(ch_rank, k)

        sec_hit1 = hits_in_top_k(sec_rank, 1) if exp_section else None
        sec_hit3 = hits_in_top_k(sec_rank, 3) if exp_section else None
        sec_hitk = hits_in_top_k(sec_rank, k) if exp_section else None

        status = "HIT" if ch_hit1 else ("TOP3" if ch_hit3 else ("TOP5" if ch_hitk else "MISS"))

        print(
            f"  [{i:02d}/{len(MANUAL_QUERIES)}] [{status:<5}]  "
            f"ch_rank={ch_rank}  "
            f"sec_rank={sec_rank if exp_section else 'N/A':<4}  "
            f"| {query[:60]}"
        )

        if verbose:
            for rank, r in enumerate(results, start=1):
                marker = "<-- expected chapter" if r["chapter"] == exp_chapter else ""
                sec_marker = (
                    "<-- expected section"
                    if exp_section and exp_section.upper() in r["section"].upper()
                    else ""
                )
                combined_marker = sec_marker or marker
                print(
                    f"       [{rank}] dist={r['distance']:.4f}"
                    f"  chapter={r['chapter']!r}"
                    f"  section={r['section']!r}"
                    f"  {combined_marker}"
                )
            print()

        per_query_results.append(
            {
                "query":            query,
                "category":         category,
                "expected_chapter": exp_chapter,
                "expected_section": exp_section,
                "chapter_rank":     ch_rank,
                "section_rank":     sec_rank,
                "chapter_hit@1":    ch_hit1,
                "chapter_hit@3":    ch_hit3,
                f"chapter_hit@{k}": ch_hitk,
                "section_hit@1":    sec_hit1,
                "section_hit@3":    sec_hit3,
                f"section_hit@{k}": sec_hitk,
                "top_k_results":    results,
            }
        )

    return per_query_results


# ── Metadata filter tests ─────────────────────────────────────────────────────

def run_filter_queries(
    collection,
    model,
    k: int,
    verbose: bool,
) -> tuple[list[dict], float]:
    """
    Run FILTER_QUERIES with ChromaDB `where` clauses and check that every
    returned result respects the filter.  Returns (filter_results, pass_rate).
    """
    print(f"\n=== Running {len(FILTER_QUERIES)} metadata filter queries (k={k}) ===\n")
    filter_results = []
    passed = 0

    for entry in FILTER_QUERIES:
        query           = entry["query"]
        where           = entry["where"]
        expected_chapter = entry["expected_chapter"]

        query_vec = encode_query(model, query)
        results   = query_collection(collection, query_vec, k, where=where)

        all_match = all(r["chapter"] == expected_chapter for r in results)
        passed   += int(all_match)
        status    = "PASS" if all_match else "FAIL"

        print(
            f"  [{status}]  filter={where}  query={query!r}  "
            f"results_count={len(results)}"
        )

        if verbose:
            for rank, r in enumerate(results, start=1):
                chapter_ok = r["chapter"] == expected_chapter
                flag = "" if chapter_ok else "  <-- FILTER VIOLATED"
                print(
                    f"       [{rank}] dist={r['distance']:.4f}"
                    f"  chapter={r['chapter']!r}"
                    f"  section={r['section']!r}"
                    f"{flag}"
                )
            print()

        filter_results.append(
            {
                "query":            query,
                "where_filter":     where,
                "expected_chapter": expected_chapter,
                "all_results_match_filter": all_match,
                "results":          results,
            }
        )

    pass_rate = passed / len(FILTER_QUERIES) if FILTER_QUERIES else 0.0
    print(f"\n  Filter pass rate: {passed}/{len(FILTER_QUERIES)} = {pass_rate:.1%}\n")
    return filter_results, pass_rate


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(metrics: dict, filter_pass_rate: float, k: int) -> None:
    """Print a human-readable summary table to stdout."""
    ch  = metrics["chapter_level"]
    sec = metrics["section_level"]

    print("\n" + "=" * 60)
    print("  Phase 2.3 — Retrieval Validation Summary")
    print("=" * 60)
    print(f"\n  Chapter-level metrics  (n={len(MANUAL_QUERIES)} queries)")
    print(f"    Hit@1  : {ch['Hit@1']:.1%}")
    print(f"    Hit@3  : {ch['Hit@3']:.1%}")
    print(f"    Hit@{k:<2} : {ch[f'Hit@{k}']:.1%}")
    print(f"    MRR    : {ch['MRR']:.4f}")

    ns = sec["num_queries_with_section"]
    print(f"\n  Section-level metrics  (n={ns} queries with expected_section)")
    print(f"    Hit@1  : {sec['Hit@1']:.1%}")
    print(f"    Hit@3  : {sec['Hit@3']:.1%}")
    print(f"    Hit@{k:<2} : {sec[f'Hit@{k}']:.1%}")
    print(f"    MRR    : {sec['MRR']:.4f}")

    print(f"\n  Metadata filter pass rate: {filter_pass_rate:.1%}")
    print("=" * 60 + "\n")


# ── Output persistence ────────────────────────────────────────────────────────

def save_results(
    per_query_results: list[dict],
    filter_results: list[dict],
    filter_pass_rate: float,
    metrics: dict,
    k: int,
) -> Path:
    """Serialise all results to data/results/retrieval_validation.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "retrieval_validation.json"

    payload = {
        "timestamp":                datetime.now(timezone.utc).isoformat(),
        "model_name":               EMBEDDING_MODEL,
        "collection_name":          COLLECTION_NAME,
        "num_queries":              len(per_query_results),
        "k":                        k,
        "metrics":                  metrics,
        "metadata_filter_pass_rate": round(filter_pass_rate, 4),
        "per_query_results":        per_query_results,
        "filter_query_results":     filter_results,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"  Results saved to {out_path}")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2.3 — Retrieval validation for the tmt_chunks ChromaDB collection.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        metavar="N",
        help="Number of top results to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full ranked result list for every query.",
    )
    args = parser.parse_args()

    k       = args.k
    verbose = args.verbose

    device     = detect_device()
    collection = open_collection(CHROMA_DIR)
    model      = load_model(EMBEDDING_MODEL, device)

    # ── Manual queries ────────────────────────────────────────────────────────
    per_query_results = run_manual_queries(collection, model, k, verbose)

    # ── Metadata filter queries ───────────────────────────────────────────────
    filter_results, filter_pass_rate = run_filter_queries(collection, model, k, verbose)

    # ── Compute and display metrics ───────────────────────────────────────────
    metrics = compute_metrics(per_query_results, k)
    print_summary(metrics, filter_pass_rate, k)

    # ── Persist results ───────────────────────────────────────────────────────
    save_results(per_query_results, filter_results, filter_pass_rate, metrics, k)

    print("Phase 2.3 complete.\n")


if __name__ == "__main__":
    main()
