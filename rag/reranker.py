"""
Phase 3 — Cross-Encoder Reranking Evaluation
Retrieves k=10 candidates from the bi-encoder, reranks them with a
cross-encoder, and compares three configurations side-by-side:

  1. Bi-encoder only (k=3, baseline from Phase 2.3)
  2. Bi-encoder + BAAI/bge-reranker-v2-m3  (retrieve 10 → return 3)
  3. Bi-encoder + ncbi/MedCPT-Cross-Encoder (retrieve 10 → return 3)

Metrics: Hit@1, Hit@3, MRR at chapter level (all 20 queries) and
         section level (11 symptom queries).

Usage:
    python rag/reranker.py
    python rag/reranker.py --skip-comparison
    python rag/reranker.py --verbose
    python rag/reranker.py --retrieve-k 15 --return-k 5
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
    RERANKER_MODEL,
    RERANKER_COMPARISON_MODEL,
    RERANK_TOP_K_RETRIEVE,
    RERANK_TOP_K_RETURN,
    RESULTS_DIR,
)

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "tmt_chunks"

# ── Gold-standard queries ─────────────────────────────────────────────────────
#
# These are IDENTICAL to the MANUAL_QUERIES defined in
# evaluation/retrieval_validation.py (Phase 2.3) for direct comparability.
# Any changes to the query set must be synchronised between both files.

MANUAL_QUERIES: list[dict] = [
    # ── Symptom queries (11) ──────────────────────────────────────────────────
    #
    # accepted_chapters: all chapters that would be a clinically valid result.
    # expected_chapter remains the PRIMARY expected chapter for backward
    # compatibility with Phase 2.3 single-label metrics.
    {
        "query": "patient with persistent dry cough for 3 weeks",
        "expected_chapter": "Common Symptoms",
        "accepted_chapters": ["Common Symptoms", "Pulmonary Disorders", "Ear, Nose, & Throat Disorders"],
        "expected_section": "COUGH",
        "category": "symptom",
    },
    {
        "query": "progressive shortness of breath on exertion",
        "expected_chapter": "Common Symptoms",
        "accepted_chapters": ["Common Symptoms", "Pulmonary Disorders", "Heart Disease"],
        "expected_section": "DYSPNEA",
        "category": "symptom",
    },
    {
        "query": "sharp chest pain radiating to the left arm",
        "expected_chapter": "Common Symptoms",
        "accepted_chapters": ["Common Symptoms", "Heart Disease"],
        "expected_section": "CHEST PAIN",
        "category": "symptom",
    },
    {
        "query": "heart racing and fluttering sensation",
        "expected_chapter": "Common Symptoms",
        "accepted_chapters": ["Common Symptoms", "Heart Disease"],
        "expected_section": "PALPITATIONS",
        "category": "symptom",
    },
    {
        "query": "swollen ankles and legs bilateral",
        "expected_chapter": "Common Symptoms",
        "accepted_chapters": ["Common Symptoms", "Heart Disease", "Blood Vessel & Lymphatic Disorders", "Kidney Disease"],
        "expected_section": "LOWER EXTREMITY EDEMA",
        "category": "symptom",
    },
    {
        "query": "high fever with chills and night sweats",
        "expected_chapter": "Common Symptoms",
        "accepted_chapters": ["Common Symptoms", "Common Problems in Infectious Diseases & Antimicrobial Therapy"],
        "expected_section": "FEVER",
        "category": "symptom",
    },
    {
        "query": "unintentional weight loss of 10 pounds in 2 months",
        "expected_chapter": "Common Symptoms",
        "accepted_chapters": ["Common Symptoms", "Cancer", "Endocrine Disorders", "Nutritional Disorders & Obesity"],
        "expected_section": "INVOLUNTARY WEIGHT LOSS",
        "category": "symptom",
    },
    {
        "query": "extreme fatigue and tiredness for several weeks",
        "expected_chapter": "Common Symptoms",
        "accepted_chapters": ["Common Symptoms", "Endocrine Disorders", "Blood Disorders", "Palliative Care & Pain Management"],
        "expected_section": "FATIGUE",
        "category": "symptom",
    },
    {
        "query": "sudden severe headache worst of my life",
        "expected_chapter": "Common Symptoms",
        "accepted_chapters": ["Common Symptoms", "Nervous System Disorders"],
        "expected_section": "ACUTE HEADACHE",
        "category": "symptom",
    },
    {
        "query": "painful urination with increased frequency",
        "expected_chapter": "Common Symptoms",
        "accepted_chapters": ["Common Symptoms", "Urologic Disorders", "Gynecologic Disorders"],
        "expected_section": "DYSURIA",
        "category": "symptom",
    },
    {
        "query": "coughing up blood streaked sputum",
        "expected_chapter": "Common Symptoms",
        "accepted_chapters": ["Common Symptoms", "Pulmonary Disorders"],
        "expected_section": "HEMOPTYSIS",
        "category": "symptom",
    },
    # ── Condition queries (6) ─────────────────────────────────────────────────
    {
        "query": "management of type 2 diabetes with metformin",
        "expected_chapter": "Diabetes Mellitus & Hypoglycemia",
        "accepted_chapters": ["Diabetes Mellitus & Hypoglycemia"],
        "expected_section": None,
        "category": "condition",
    },
    {
        "query": "treatment of atrial fibrillation",
        "expected_chapter": "Heart Disease",
        "accepted_chapters": ["Heart Disease"],
        "expected_section": None,
        "category": "condition",
    },
    {
        "query": "pneumonia diagnosis and antibiotics",
        "expected_chapter": "Pulmonary Disorders",
        "accepted_chapters": ["Pulmonary Disorders", "Bacterial & Chlamydial Infections"],
        "expected_section": None,
        "category": "condition",
    },
    {
        "query": "acute kidney injury creatinine elevated",
        "expected_chapter": "Kidney Disease",
        "accepted_chapters": ["Kidney Disease"],
        "expected_section": None,
        "category": "condition",
    },
    {
        "query": "rheumatoid arthritis joint inflammation treatment",
        "expected_chapter": "Rheumatologic, Immunologic, & Allergic Disorders",
        "accepted_chapters": ["Rheumatologic, Immunologic, & Allergic Disorders"],
        "expected_section": None,
        "category": "condition",
    },
    {
        "query": "major depressive disorder SSRI treatment",
        "expected_chapter": "Psychiatric Disorders",
        "accepted_chapters": ["Psychiatric Disorders", "Geriatric Disorders"],
        "expected_section": None,
        "category": "condition",
    },
    # ── Cross-chapter / emergency queries (3) ─────────────────────────────────
    {
        "query": "fever in immunocompromised patient neutropenia",
        "expected_chapter": "Common Problems in Infectious Diseases & Antimicrobial Therapy",
        "accepted_chapters": ["Common Problems in Infectious Diseases & Antimicrobial Therapy", "Common Symptoms", "Blood Disorders"],
        "expected_section": None,
        "category": "emergency",
    },
    {
        "query": "acute myocardial infarction emergency management",
        "expected_chapter": "Heart Disease",
        "accepted_chapters": ["Heart Disease"],
        "expected_section": None,
        "category": "emergency",
    },
    {
        "query": "diabetic ketoacidosis DKA treatment protocol",
        "expected_chapter": "Diabetes Mellitus & Hypoglycemia",
        "accepted_chapters": ["Diabetes Mellitus & Hypoglycemia", "Electrolyte & Acid-Base Disorders"],
        "expected_section": None,
        "category": "emergency",
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


# ── Model loading ─────────────────────────────────────────────────────────────

def load_bi_encoder(model_name: str, device: str):
    """Load and return a SentenceTransformer bi-encoder model."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading bi-encoder '{model_name}' on device '{device}' ...")
    model = SentenceTransformer(model_name, device=device)
    print("  Bi-encoder loaded.")
    return model


def load_cross_encoder(model_name: str) -> tuple:
    """
    Load a CrossEncoder model, falling back to CPU if MPS fails.
    Returns (model, device_used).
    """
    from sentence_transformers import CrossEncoder

    device = detect_device()
    if device == "mps":
        try:
            print(f"Loading cross-encoder '{model_name}' on device 'mps' ...")
            model = CrossEncoder(model_name, trust_remote_code=True, device="mps")
            print("  Cross-encoder loaded on MPS.")
            return model, "mps"
        except Exception as e:
            print(f"  MPS failed for cross-encoder ({e}), falling back to CPU ...")

    print(f"Loading cross-encoder '{model_name}' on device 'cpu' ...")
    model = CrossEncoder(model_name, trust_remote_code=True, device="cpu")
    print("  Cross-encoder loaded on CPU.")
    return model, "cpu"


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def open_collection(chroma_dir: Path):
    """Open the persistent ChromaDB client and return the tmt_chunks collection."""
    import chromadb
    from chromadb.config import Settings

    print(f"Opening ChromaDB client at {chroma_dir} ...")
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"  Collection '{COLLECTION_NAME}' has {collection.count():,} chunks.")
    return collection


# ── Core reranking functions (importable by other scripts) ────────────────────

def rerank(query: str, candidates: list[dict], model, top_k: int = 3) -> list[dict]:
    """
    Rerank candidate chunks using a cross-encoder.

    Args:
        query:      the search query
        candidates: list of dicts with at least 'chunk_id' and 'text' keys
                    (plus any metadata like chapter, section, distance)
        model:      a loaded CrossEncoder model
        top_k:      number of results to return after reranking

    Returns:
        The top_k candidates reordered by cross-encoder score,
        each dict augmented with 'rerank_score' (float) and
        'original_rank' (1-based int) fields.
    """
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    scored = []
    for original_rank, (candidate, score) in enumerate(zip(candidates, scores), start=1):
        entry = dict(candidate)
        entry["rerank_score"] = float(score)
        entry["original_rank"] = original_rank
        scored.append(entry)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


def retrieve_and_rerank(
    query: str,
    collection,
    bi_encoder,
    reranker,
    retrieve_k: int = 10,
    return_k: int = 3,
) -> list[dict]:
    """
    Full pipeline: bi-encoder retrieval followed by cross-encoder reranking.

    Args:
        query:       the search query string
        collection:  ChromaDB collection object
        bi_encoder:  loaded SentenceTransformer model
        reranker:    loaded CrossEncoder model
        retrieve_k:  number of candidates to fetch from the bi-encoder
        return_k:    number of results to return after reranking

    Returns:
        Top return_k results as dicts, each with rerank_score and original_rank.
    """
    # Encode query
    vec = bi_encoder.encode(
        query, convert_to_numpy=True, normalize_embeddings=True
    ).tolist()

    # Retrieve from ChromaDB — must include documents so the reranker has text
    raw = collection.query(
        query_embeddings=[vec],
        n_results=retrieve_k,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for chunk_id, doc, meta, dist in zip(
        raw["ids"][0],
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        candidates.append(
            {
                "chunk_id": chunk_id,
                "text":     doc,
                "chapter":  meta.get("chapter", ""),
                "section":  meta.get("section", ""),
                "distance": round(float(dist), 6),
            }
        )

    return rerank(query, candidates, reranker, top_k=return_k)


# ── Bi-encoder-only retrieval (baseline, no documents needed) ─────────────────

def retrieve_only(
    query: str,
    collection,
    bi_encoder,
    k: int = 3,
) -> list[dict]:
    """
    Retrieve top-k results using the bi-encoder only (no reranking).
    Returns list of dicts with chunk_id, chapter, section, distance.
    """
    vec = bi_encoder.encode(
        query, convert_to_numpy=True, normalize_embeddings=True
    ).tolist()

    raw = collection.query(
        query_embeddings=[vec],
        n_results=k,
        include=["metadatas", "distances"],
    )

    results = []
    for chunk_id, meta, dist in zip(
        raw["ids"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append(
            {
                "chunk_id": chunk_id,
                "chapter":  meta.get("chapter", ""),
                "section":  meta.get("section", ""),
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


def find_first_accepted_rank(results: list[dict], accepted_chapters: list[str]) -> int | None:
    """Return the 1-based rank of the first result from any accepted chapter, or None."""
    accepted_set = set(accepted_chapters)
    for rank, r in enumerate(results, start=1):
        if r["chapter"] in accepted_set:
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
    Compute Hit@1, Hit@3, Hit@k, and MRR at three levels:
      - chapter_strict: matches only the single expected_chapter
      - chapter_multilabel: matches any accepted_chapter
      - section_level: matches expected_section (symptom queries only)
    """
    n = len(per_query)

    # Chapter-level STRICT (single expected_chapter)
    ch_hit1 = sum(1 for q in per_query if hits_in_top_k(q["chapter_rank"], 1)) / n
    ch_hit3 = sum(1 for q in per_query if hits_in_top_k(q["chapter_rank"], 3)) / n
    ch_hitk = sum(1 for q in per_query if hits_in_top_k(q["chapter_rank"], k)) / n
    ch_mrr  = sum(
        1.0 / q["chapter_rank"] if q["chapter_rank"] is not None else 0.0
        for q in per_query
    ) / n

    # Chapter-level MULTI-LABEL (any accepted chapter)
    ml_hit1 = sum(1 for q in per_query if hits_in_top_k(q["accepted_rank"], 1)) / n
    ml_hit3 = sum(1 for q in per_query if hits_in_top_k(q["accepted_rank"], 3)) / n
    ml_hitk = sum(1 for q in per_query if hits_in_top_k(q["accepted_rank"], k)) / n
    ml_mrr  = sum(
        1.0 / q["accepted_rank"] if q["accepted_rank"] is not None else 0.0
        for q in per_query
    ) / n

    # Section-level (symptom queries only)
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
        "chapter_strict": {
            "Hit@1":      round(ch_hit1, 4),
            "Hit@3":      round(ch_hit3, 4),
            f"Hit@{k}":   round(ch_hitk, 4),
            "MRR":        round(ch_mrr,  4),
        },
        "chapter_multilabel": {
            "Hit@1":      round(ml_hit1, 4),
            "Hit@3":      round(ml_hit3, 4),
            f"Hit@{k}":   round(ml_hitk, 4),
            "MRR":        round(ml_mrr,  4),
        },
        "section_level": {
            "num_queries_with_section": len(sec_queries),
            "Hit@1":      round(sec_hit1, 4),
            "Hit@3":      round(sec_hit3, 4),
            f"Hit@{k}":   round(sec_hitk, 4),
            "MRR":        round(sec_mrr,  4),
        },
    }


# ── Per-configuration query runner ────────────────────────────────────────────

def run_config_baseline(
    collection,
    bi_encoder,
    return_k: int,
    verbose: bool,
) -> list[dict]:
    """
    Configuration 1: bi-encoder only, retrieve return_k directly.
    Mirrors Phase 2.3 behaviour exactly.
    """
    label = f"Bi-encoder only (k={return_k})"
    print(f"\n{'='*60}")
    print(f"  Config: {label}")
    print(f"{'='*60}\n")

    per_query = []
    for i, entry in enumerate(MANUAL_QUERIES, start=1):
        query       = entry["query"]
        exp_chapter = entry["expected_chapter"]
        exp_section = entry["expected_section"]

        accepted  = entry.get("accepted_chapters", [exp_chapter])
        results   = retrieve_only(query, collection, bi_encoder, k=return_k)
        ch_rank   = find_first_chapter_rank(results, exp_chapter)
        acc_rank  = find_first_accepted_rank(results, accepted)
        sec_rank  = (
            find_first_section_rank(results, exp_section)
            if exp_section is not None
            else None
        )

        status = _status_label(acc_rank, return_k)
        print(
            f"  [{i:02d}/{len(MANUAL_QUERIES)}] [{status:<5}]  "
            f"ch_rank={ch_rank}  acc_rank={acc_rank}  "
            f"sec_rank={sec_rank if exp_section else 'N/A':<4}  "
            f"| {query[:55]}"
        )

        if verbose:
            _print_results(results, exp_chapter, exp_section)

        per_query.append(_build_per_query_record(
            entry, results, ch_rank, acc_rank, sec_rank, return_k,
            reranked=False,
        ))

    return per_query


def run_config_reranked(
    collection,
    bi_encoder,
    reranker,
    reranker_label: str,
    retrieve_k: int,
    return_k: int,
    verbose: bool,
) -> list[dict]:
    """
    Configurations 2 & 3: bi-encoder retrieves retrieve_k, cross-encoder reranks
    to return_k.
    """
    print(f"\n{'='*60}")
    print(f"  Config: Bi-encoder + {reranker_label}")
    print(f"  retrieve_k={retrieve_k}  return_k={return_k}")
    print(f"{'='*60}\n")

    per_query = []
    for i, entry in enumerate(MANUAL_QUERIES, start=1):
        query       = entry["query"]
        exp_chapter = entry["expected_chapter"]
        exp_section = entry["expected_section"]

        accepted  = entry.get("accepted_chapters", [exp_chapter])
        results   = retrieve_and_rerank(
            query, collection, bi_encoder, reranker, retrieve_k, return_k
        )
        ch_rank   = find_first_chapter_rank(results, exp_chapter)
        acc_rank  = find_first_accepted_rank(results, accepted)
        sec_rank  = (
            find_first_section_rank(results, exp_section)
            if exp_section is not None
            else None
        )

        status = _status_label(acc_rank, return_k)

        # Show the original_rank of the best-matching result (if any) so we can
        # see whether reranking moved it up or down.
        best_orig = None
        accepted_set = set(accepted)
        for r in results:
            if r["chapter"] in accepted_set:
                best_orig = r.get("original_rank")
                break

        print(
            f"  [{i:02d}/{len(MANUAL_QUERIES)}] [{status:<5}]  "
            f"acc_rank={acc_rank}  orig_rank={best_orig}  "
            f"sec_rank={sec_rank if exp_section else 'N/A':<4}  "
            f"| {query[:55]}"
        )

        if verbose:
            _print_results_reranked(results, exp_chapter, exp_section)

        per_query.append(_build_per_query_record(
            entry, results, ch_rank, acc_rank, sec_rank, return_k,
            reranked=True,
        ))

    return per_query


# ── Small display helpers ─────────────────────────────────────────────────────

def _status_label(ch_rank: int | None, k: int) -> str:
    if ch_rank is None:
        return "MISS"
    if ch_rank == 1:
        return "HIT"
    if ch_rank <= 3:
        return "TOP3"
    if ch_rank <= k:
        return f"TOP{k}"
    return "MISS"


def _print_results(results: list[dict], exp_chapter: str, exp_section: str | None) -> None:
    for rank, r in enumerate(results, start=1):
        ch_marker  = "  <-- expected chapter" if r["chapter"] == exp_chapter else ""
        sec_marker = (
            "  <-- expected section"
            if exp_section and exp_section.upper() in r["section"].upper()
            else ""
        )
        marker = sec_marker or ch_marker
        print(
            f"       [{rank}] dist={r['distance']:.4f}"
            f"  chapter={r['chapter']!r}"
            f"  section={r['section']!r}"
            f"{marker}"
        )
    print()


def _print_results_reranked(
    results: list[dict], exp_chapter: str, exp_section: str | None
) -> None:
    for rank, r in enumerate(results, start=1):
        ch_marker  = "  <-- expected chapter" if r["chapter"] == exp_chapter else ""
        sec_marker = (
            "  <-- expected section"
            if exp_section and exp_section.upper() in r["section"].upper()
            else ""
        )
        marker = sec_marker or ch_marker
        print(
            f"       [{rank}] score={r.get('rerank_score', 0.0):.4f}"
            f"  orig_rank={r.get('original_rank', '?')}"
            f"  chapter={r['chapter']!r}"
            f"  section={r['section']!r}"
            f"{marker}"
        )
    print()


def _build_per_query_record(
    entry: dict,
    results: list[dict],
    ch_rank: int | None,
    acc_rank: int | None,
    sec_rank: int | None,
    k: int,
    reranked: bool,
) -> dict:
    record = {
        "query":            entry["query"],
        "category":         entry["category"],
        "expected_chapter": entry["expected_chapter"],
        "accepted_chapters": entry.get("accepted_chapters", [entry["expected_chapter"]]),
        "expected_section": entry["expected_section"],
        "chapter_rank":     ch_rank,
        "accepted_rank":    acc_rank,
        "section_rank":     sec_rank,
        "chapter_hit@1":    hits_in_top_k(ch_rank, 1),
        "chapter_hit@3":    hits_in_top_k(ch_rank, 3),
        f"chapter_hit@{k}": hits_in_top_k(ch_rank, k),
        "accepted_hit@1":   hits_in_top_k(acc_rank, 1),
        "accepted_hit@3":   hits_in_top_k(acc_rank, 3),
        f"accepted_hit@{k}": hits_in_top_k(acc_rank, k),
        "section_hit@1":    hits_in_top_k(sec_rank, 1) if entry["expected_section"] else None,
        "section_hit@3":    hits_in_top_k(sec_rank, 3) if entry["expected_section"] else None,
        f"section_hit@{k}": hits_in_top_k(sec_rank, k) if entry["expected_section"] else None,
        "top_results":      results,
    }
    if reranked:
        # Capture the original bi-encoder rank of the top reranked result
        record["original_rank_of_top1"] = results[0].get("original_rank") if results else None
    return record


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(configs: list[dict]) -> None:
    """
    Print a side-by-side comparison table for all evaluated configurations.

    Each entry in `configs` should have:
        label, metrics, per_query (list)
    """
    col_w = 26
    sep   = "-" * (12 + col_w * len(configs))

    print("\n" + "=" * (12 + col_w * len(configs)))
    print("  Phase 3 — Reranking Evaluation: Configuration Comparison")
    print("=" * (12 + col_w * len(configs)))

    # Header row
    header = f"  {'Metric':<20}"
    for cfg in configs:
        header += f"  {cfg['label']:<{col_w - 2}}"
    print(header)
    print(sep)

    # Chapter-level STRICT
    print("  Chapter-strict  (n=20, single expected chapter)")
    for metric in ["Hit@1", "Hit@3", "MRR"]:
        row = f"    {metric:<18}"
        for cfg in configs:
            val = cfg["metrics"]["chapter_strict"].get(metric, 0.0)
            row += f"  {val:.1%}{'':<{col_w - 8}}"
        print(row)

    print(sep)

    # Chapter-level MULTI-LABEL
    print("  Chapter-multilabel  (n=20, any accepted chapter)")
    for metric in ["Hit@1", "Hit@3", "MRR"]:
        row = f"    {metric:<18}"
        for cfg in configs:
            val = cfg["metrics"]["chapter_multilabel"].get(metric, 0.0)
            row += f"  {val:.1%}{'':<{col_w - 8}}"
        print(row)

    print(sep)

    # Section-level metrics
    ns = configs[0]["metrics"]["section_level"]["num_queries_with_section"]
    print(f"  Section-level  (n={ns} symptom queries)")
    for metric in ["Hit@1", "Hit@3", "MRR"]:
        row = f"    {metric:<18}"
        for cfg in configs:
            val = cfg["metrics"]["section_level"].get(metric, 0.0)
            row += f"  {val:.1%}{'':<{col_w - 8}}"
        print(row)

    print("=" * (12 + col_w * len(configs)) + "\n")


# ── Output persistence ────────────────────────────────────────────────────────

def save_results(configs: list[dict], retrieve_k: int, return_k: int) -> Path:
    """Save all configuration metrics and per-query details to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "reranking_comparison.json"

    payload = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "phase":           "3",
        "bi_encoder_model": EMBEDDING_MODEL,
        "retrieve_k":      retrieve_k,
        "return_k":        return_k,
        "num_queries":     len(MANUAL_QUERIES),
        "configurations":  [
            {
                "label":     cfg["label"],
                "model":     cfg.get("reranker_model", "none"),
                "metrics":   cfg["metrics"],
                "per_query": cfg["per_query"],
            }
            for cfg in configs
        ],
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {out_path}")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 — cross-encoder reranking evaluation. "
            "Compares bi-encoder baseline vs bi-encoder + reranker(s)."
        )
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Only run the primary reranker (bge-reranker-v2-m3); skip MedCPT.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full ranked result list for every query.",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=RERANK_TOP_K_RETRIEVE,
        metavar="N",
        help=f"Number of candidates to fetch from bi-encoder (default: {RERANK_TOP_K_RETRIEVE}).",
    )
    parser.add_argument(
        "--return-k",
        type=int,
        default=RERANK_TOP_K_RETURN,
        metavar="N",
        help=f"Number of results to return after reranking (default: {RERANK_TOP_K_RETURN}).",
    )
    args = parser.parse_args()

    retrieve_k = args.retrieve_k
    return_k   = args.return_k
    verbose    = args.verbose

    print(f"\nPhase 3 — Reranking Evaluation")
    print(f"  retrieve_k={retrieve_k}  return_k={return_k}")
    print(f"  skip_comparison={args.skip_comparison}\n")

    # ── Infrastructure ────────────────────────────────────────────────────────
    device     = detect_device()
    collection = open_collection(CHROMA_DIR)
    bi_encoder = load_bi_encoder(EMBEDDING_MODEL, device)

    configs: list[dict] = []

    # ── Config 1: bi-encoder baseline ─────────────────────────────────────────
    per_query_baseline = run_config_baseline(
        collection, bi_encoder, return_k, verbose
    )
    metrics_baseline = compute_metrics(per_query_baseline, return_k)
    configs.append(
        {
            "label":          f"Bi-encoder only (k={return_k})",
            "reranker_model": "none",
            "metrics":        metrics_baseline,
            "per_query":      per_query_baseline,
        }
    )

    # ── Config 2: bi-encoder + bge-reranker-v2-m3 ─────────────────────────────
    bge_model, bge_device = load_cross_encoder(RERANKER_MODEL)
    print(f"  (bge cross-encoder running on {bge_device})")

    per_query_bge = run_config_reranked(
        collection, bi_encoder, bge_model,
        reranker_label=RERANKER_MODEL,
        retrieve_k=retrieve_k,
        return_k=return_k,
        verbose=verbose,
    )
    metrics_bge = compute_metrics(per_query_bge, return_k)
    configs.append(
        {
            "label":          f"+ {RERANKER_MODEL.split('/')[-1]}",
            "reranker_model": RERANKER_MODEL,
            "metrics":        metrics_bge,
            "per_query":      per_query_bge,
        }
    )

    # ── Config 3: bi-encoder + MedCPT-Cross-Encoder (optional) ───────────────
    if not args.skip_comparison:
        medcpt_model, medcpt_device = load_cross_encoder(RERANKER_COMPARISON_MODEL)
        print(f"  (MedCPT cross-encoder running on {medcpt_device})")

        per_query_medcpt = run_config_reranked(
            collection, bi_encoder, medcpt_model,
            reranker_label=RERANKER_COMPARISON_MODEL,
            retrieve_k=retrieve_k,
            return_k=return_k,
            verbose=verbose,
        )
        metrics_medcpt = compute_metrics(per_query_medcpt, return_k)
        configs.append(
            {
                "label":          f"+ {RERANKER_COMPARISON_MODEL.split('/')[-1]}",
                "reranker_model": RERANKER_COMPARISON_MODEL,
                "metrics":        metrics_medcpt,
                "per_query":      per_query_medcpt,
            }
        )

    # ── Comparison table ──────────────────────────────────────────────────────
    print_comparison_table(configs)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(configs, retrieve_k, return_k)

    print("Phase 3 complete.\n")


if __name__ == "__main__":
    main()
