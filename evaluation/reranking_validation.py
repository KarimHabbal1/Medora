"""Phase 3 — Reranking Validation.

Compares Phase 2 baseline retrieval against Phase 3 reranked results on the
same manual gold query set.

Usage:
    python evaluation/reranking_validation.py
    python evaluation/reranking_validation.py --retrieve-k 10 --final-k 3
    python evaluation/reranking_validation.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Project root import setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RERANKER_BACKEND,
    RERANKER_MODEL,
    RERANKER_FALLBACK_MODEL,
    RERANK_FINAL_K,
    RERANK_RETRIEVE_K,
    RESULTS_DIR,
)
from evaluation.retrieval_validation import (  # type: ignore
    MANUAL_QUERIES,
    compute_metrics,
    detect_device,
    encode_query,
    find_first_chapter_rank,
    find_first_section_rank,
    hits_in_top_k,
    load_model,
    open_collection,
)
from medora.rag import ColBERTReranker, RerankerCandidate


COLLECTION_NAME = "tmt_chunks"


def retrieve_candidates(collection, query_vec: list[float], k: int) -> list[dict]:
    """Retrieve top-k candidates with full fields needed for reranking."""
    raw = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    candidates: list[dict] = []
    for chunk_id, doc, meta, dist in zip(
        raw["ids"][0],
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        candidates.append(
            {
                "chunk_id": chunk_id,
                "chapter": meta.get("chapter", ""),
                "section": meta.get("section", ""),
                "text": doc or "",
                "distance": round(float(dist), 6),
            }
        )

    return candidates


def to_eval_result(
    query_entry: dict,
    ranked_results: list[dict],
    final_k: int,
    label: str,
) -> dict:
    """Build standardized per-query structure matching Phase 2 metrics logic."""
    exp_chapter = query_entry["expected_chapter"]
    exp_section = query_entry["expected_section"]

    chapter_rank = find_first_chapter_rank(ranked_results, exp_chapter)
    section_rank = (
        find_first_section_rank(ranked_results, exp_section)
        if exp_section is not None
        else None
    )

    return {
        "query": query_entry["query"],
        "category": query_entry["category"],
        "expected_chapter": exp_chapter,
        "expected_section": exp_section,
        "chapter_rank": chapter_rank,
        "section_rank": section_rank,
        "chapter_hit@1": hits_in_top_k(chapter_rank, 1),
        "chapter_hit@3": hits_in_top_k(chapter_rank, 3),
        f"chapter_hit@{final_k}": hits_in_top_k(chapter_rank, final_k),
        "section_hit@1": hits_in_top_k(section_rank, 1) if exp_section else None,
        "section_hit@3": hits_in_top_k(section_rank, 3) if exp_section else None,
        f"section_hit@{final_k}": hits_in_top_k(section_rank, final_k) if exp_section else None,
        "ranked_results": ranked_results,
        "pipeline": label,
    }


def summarize_delta(baseline_metrics: dict, reranked_metrics: dict, final_k: int) -> dict:
    """Compute baseline vs reranked metric deltas."""
    ch_b = baseline_metrics["chapter_level"]
    ch_r = reranked_metrics["chapter_level"]
    sec_b = baseline_metrics["section_level"]
    sec_r = reranked_metrics["section_level"]

    return {
        "chapter_level": {
            "Hit@1_delta": round(ch_r["Hit@1"] - ch_b["Hit@1"], 4),
            "Hit@3_delta": round(ch_r["Hit@3"] - ch_b["Hit@3"], 4),
            f"Hit@{final_k}_delta": round(ch_r[f"Hit@{final_k}"] - ch_b[f"Hit@{final_k}"], 4),
            "MRR_delta": round(ch_r["MRR"] - ch_b["MRR"], 4),
        },
        "section_level": {
            "Hit@1_delta": round(sec_r["Hit@1"] - sec_b["Hit@1"], 4),
            "Hit@3_delta": round(sec_r["Hit@3"] - sec_b["Hit@3"], 4),
            f"Hit@{final_k}_delta": round(sec_r[f"Hit@{final_k}"] - sec_b[f"Hit@{final_k}"], 4),
            "MRR_delta": round(sec_r["MRR"] - sec_b["MRR"], 4),
        },
    }


def print_summary(
    baseline_metrics: dict,
    reranked_metrics: dict,
    deltas: dict,
    retrieve_k: int,
    final_k: int,
) -> None:
    """Print comparison metrics between baseline and reranked pipeline."""
    print("\n" + "=" * 70)
    print("  Phase 3 — Reranking Validation Summary")
    print("=" * 70)
    print(f"  Candidate retrieval depth: k={retrieve_k}")
    print(f"  Final ranked cutoff:       k={final_k}\n")

    for level in ("chapter_level", "section_level"):
        label = "Chapter" if level == "chapter_level" else "Section"
        b = baseline_metrics[level]
        r = reranked_metrics[level]
        d = deltas[level]

        print(f"  {label}-level")
        if level == "section_level":
            print(f"    Queries with section labels: {b['num_queries_with_section']}")
        print(f"    Baseline Hit@1 : {b['Hit@1']:.1%}")
        print(f"    Reranked Hit@1 : {r['Hit@1']:.1%}  (delta {d['Hit@1_delta']:+.1%})")
        print(f"    Baseline Hit@3 : {b['Hit@3']:.1%}")
        print(f"    Reranked Hit@3 : {r['Hit@3']:.1%}  (delta {d['Hit@3_delta']:+.1%})")
        print(f"    Baseline Hit@{final_k}: {b[f'Hit@{final_k}']:.1%}")
        print(
            f"    Reranked Hit@{final_k}: {r[f'Hit@{final_k}']:.1%}  "
            f"(delta {d[f'Hit@{final_k}_delta']:+.1%})"
        )
        print(f"    Baseline MRR   : {b['MRR']:.4f}")
        print(f"    Reranked MRR   : {r['MRR']:.4f}  (delta {d['MRR_delta']:+.4f})\n")


def save_results(
    baseline_results: list[dict],
    reranked_results: list[dict],
    baseline_metrics: dict,
    reranked_metrics: dict,
    deltas: dict,
    retrieve_k: int,
    final_k: int,
) -> Path:
    """Persist Phase 3 benchmark results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "reranking_validation.json"

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "num_queries": len(MANUAL_QUERIES),
        "retrieve_k": retrieve_k,
        "final_k": final_k,
        "baseline_metrics": baseline_metrics,
        "reranked_metrics": reranked_metrics,
        "metric_deltas": deltas,
        "baseline_per_query": baseline_results,
        "reranked_per_query": reranked_results,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 — Compare baseline retrieval against ColBERT reranked retrieval.",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=RERANK_RETRIEVE_K,
        metavar="N",
        help="Number of initial retrieval candidates from ChromaDB (default: 10).",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=RERANK_FINAL_K,
        metavar="N",
        help="Final top-k list size after reranking (default: 3).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed baseline and reranked top-k lists for each query.",
    )
    args = parser.parse_args()

    retrieve_k = max(1, args.retrieve_k)
    final_k = max(1, min(args.final_k, retrieve_k))

    print(f"Opening collection from {CHROMA_DIR} …")
    try:
        collection = open_collection(CHROMA_DIR)
    except Exception as exc:  # noqa: BLE001
        exc_name = type(exc).__name__
        if exc_name == "NotFoundError":
            print(
                "\nERROR: ChromaDB collection 'tmt_chunks' was not found.\n"
                "Build Phase 2 artifacts first:\n"
                "  1) python embeddings/embed_chunks.py\n"
                "  2) python embeddings/build_vector_store.py\n"
            )
            raise SystemExit(1) from exc
        raise

    device = detect_device()
    retriever_model = load_model(EMBEDDING_MODEL, device)
    try:
        reranker = ColBERTReranker(
            RERANKER_MODEL,
            prefer_backend=RERANKER_BACKEND,
            fallback_model_name=RERANKER_FALLBACK_MODEL,
        )
    except ImportError as exc:
        print(
            "\nERROR: Reranker dependency is missing.\n"
            "Install requirements and retry:\n"
            "  pip install -r requirements.txt\n"
        )
        raise SystemExit(1) from exc

    baseline_results: list[dict] = []
    reranked_results: list[dict] = []

    print(f"\n=== Running {len(MANUAL_QUERIES)} Phase 3 comparison queries ===\n")
    for idx, query_entry in enumerate(MANUAL_QUERIES, start=1):
        query = query_entry["query"]
        query_vec = encode_query(retriever_model, query)
        retrieved = retrieve_candidates(collection, query_vec, retrieve_k)

        baseline_topk = retrieved[:final_k]
        baseline_eval = to_eval_result(query_entry, baseline_topk, final_k, label="baseline")
        baseline_results.append(baseline_eval)

        reranker_input = [
            RerankerCandidate(
                chunk_id=item["chunk_id"],
                chapter=item["chapter"],
                section=item["section"],
                text=item["text"],
                retrieval_distance=item["distance"],
            )
            for item in retrieved
        ]
        reranked = reranker.rerank(query=query, candidates=reranker_input, top_n=final_k)
        reranked_topk = [
            {
                "chunk_id": r.chunk_id,
                "chapter": r.chapter,
                "section": r.section,
                "distance": r.retrieval_distance,
                "rerank_score": round(r.score, 6),
            }
            for r in reranked
        ]

        reranked_eval = to_eval_result(query_entry, reranked_topk, final_k, label="reranked")
        reranked_results.append(reranked_eval)

        baseline_rank = baseline_eval["chapter_rank"]
        reranked_rank = reranked_eval["chapter_rank"]
        print(
            f"  [{idx:02d}/{len(MANUAL_QUERIES)}] "
            f"baseline ch_rank={str(baseline_rank):<4} | reranked ch_rank={str(reranked_rank):<4} "
            f"| {query[:56]}"
        )

        if args.verbose:
            print("      Baseline:")
            for r_idx, item in enumerate(baseline_topk, start=1):
                print(
                    f"        [{r_idx}] chapter={item['chapter']!r} "
                    f"section={item['section']!r} dist={item['distance']:.4f}"
                )
            print("      Reranked:")
            for item in reranked:
                print(
                    f"        [{item.rank}] chapter={item.chapter!r} "
                    f"section={item.section!r} score={item.score:.4f}"
                )
            print()

    baseline_metrics = compute_metrics(baseline_results, final_k)
    reranked_metrics = compute_metrics(reranked_results, final_k)
    deltas = summarize_delta(baseline_metrics, reranked_metrics, final_k)

    print_summary(
        baseline_metrics=baseline_metrics,
        reranked_metrics=reranked_metrics,
        deltas=deltas,
        retrieve_k=retrieve_k,
        final_k=final_k,
    )

    out_path = save_results(
        baseline_results=baseline_results,
        reranked_results=reranked_results,
        baseline_metrics=baseline_metrics,
        reranked_metrics=reranked_metrics,
        deltas=deltas,
        retrieve_k=retrieve_k,
        final_k=final_k,
    )
    print(f"Results saved to {out_path}")
    print("Phase 3 validation complete.")


if __name__ == "__main__":
    main()
