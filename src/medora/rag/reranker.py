"""Phase 3 reranker utilities.

This module wraps a ColBERT reranker backend so Phase 3 logic can reuse a
single, testable interface from scripts, notebooks, and future agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RerankerCandidate:
    """Single retrieval candidate to be reranked."""

    chunk_id: str
    chapter: str
    section: str
    text: str
    retrieval_distance: float | None = None


@dataclass(slots=True)
class RerankerResult:
    """Reranked candidate with rank and score."""

    rank: int
    score: float
    chunk_id: str
    chapter: str
    section: str
    text: str
    retrieval_distance: float | None = None


class ColBERTReranker:
    """Thin wrapper around a ColBERT reranker model via RAGatouille.

    The wrapper normalizes return formats so callers can stay independent from
    backend-specific response details.
    """

    def __init__(
        self,
        model_name: str,
        verbose: int = 0,
        prefer_backend: str = "auto",
        fallback_model_name: str = "BAAI/bge-reranker-v2-m3",
    ) -> None:
        self.model_name = model_name
        self.verbose = verbose
        self._backend = ""
        self._model = None

        mode = prefer_backend.lower().strip()
        if mode not in {"auto", "ragatouille", "cross-encoder"}:
            raise ValueError("prefer_backend must be one of: auto, ragatouille, cross-encoder")

        if mode in {"auto", "ragatouille"}:
            try:
                self._backend = "ragatouille"
                self._model = self._load_ragatouille_model(model_name, verbose)
                return
            except Exception as exc:  # noqa: BLE001
                if mode == "ragatouille":
                    raise RuntimeError(f"Failed to initialize RAGatouille backend: {exc}") from exc

        # Auto mode fallback path
        self._backend = "cross-encoder"
        self._model = self._load_cross_encoder_model(fallback_model_name)

    @staticmethod
    def _load_ragatouille_model(model_name: str, verbose: int):
        try:
            from ragatouille import RAGPretrainedModel
        except ImportError as exc:
            raise ImportError(
                "Phase 3 reranking requires 'ragatouille'. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        try:
            return RAGPretrainedModel.from_pretrained(model_name, verbose=verbose)
        except TypeError:
            # Some versions do not expose the verbose keyword.
            return RAGPretrainedModel.from_pretrained(model_name)

    @staticmethod
    def _load_cross_encoder_model(model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoder backend requires 'sentence-transformers'. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        return CrossEncoder(model_name)

    @staticmethod
    def _normalize_backend_result(item: Any, fallback_rank: int) -> tuple[int, float, int | None]:
        """Extract rank, score, and source index from backend output."""
        rank = fallback_rank
        score = 0.0
        source_idx: int | None = None

        if isinstance(item, dict):
            if isinstance(item.get("rank"), int):
                rank = int(item["rank"])

            if isinstance(item.get("score"), (float, int)):
                score = float(item["score"])

            for key in ("result_index", "document_index", "doc_index"):
                if isinstance(item.get(key), int):
                    source_idx = int(item[key])
                    break

        return rank, score, source_idx

    @staticmethod
    def _candidate_from_text_match(candidates: list[RerankerCandidate], text: str) -> RerankerCandidate | None:
        """Fallback alignment when backend returns passage text without index."""
        for cand in candidates:
            if cand.text == text:
                return cand
        return None

    def rerank(
        self,
        query: str,
        candidates: list[RerankerCandidate],
        top_n: int,
    ) -> list[RerankerResult]:
        """Rerank retrieval candidates and return top_n normalized results."""
        if not candidates:
            return []

        top_n = max(1, min(top_n, len(candidates)))
        docs = [c.text for c in candidates]

        if self._backend == "ragatouille":
            raw = self._model.rerank(query=query, documents=docs, k=top_n)
            if not isinstance(raw, list):
                raise RuntimeError("Unexpected reranker backend response: expected a list.")

            results: list[RerankerResult] = []
            for i, item in enumerate(raw, start=1):
                rank, score, source_idx = self._normalize_backend_result(item, fallback_rank=i)

                candidate: RerankerCandidate | None = None
                if source_idx is not None and 0 <= source_idx < len(candidates):
                    candidate = candidates[source_idx]
                elif isinstance(item, dict) and isinstance(item.get("content"), str):
                    candidate = self._candidate_from_text_match(candidates, item["content"])

                if candidate is None:
                    candidate = candidates[min(i - 1, len(candidates) - 1)]

                results.append(
                    RerankerResult(
                        rank=rank,
                        score=score,
                        chunk_id=candidate.chunk_id,
                        chapter=candidate.chapter,
                        section=candidate.section,
                        text=candidate.text,
                        retrieval_distance=candidate.retrieval_distance,
                    )
                )

            results.sort(key=lambda r: r.rank)
            for new_rank, result in enumerate(results, start=1):
                result.rank = new_rank
            return results[:top_n]

        # Cross-encoder path: score each (query, passage) pair and sort desc.
        pairs = [[query, c.text] for c in candidates]
        scores = self._model.predict(pairs)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: float(x[1]), reverse=True)

        results: list[RerankerResult] = []
        for rank, (idx, score) in enumerate(indexed[:top_n], start=1):
            cand = candidates[idx]
            results.append(
                RerankerResult(
                    rank=rank,
                    score=float(score),
                    chunk_id=cand.chunk_id,
                    chapter=cand.chapter,
                    section=cand.section,
                    text=cand.text,
                    retrieval_distance=cand.retrieval_distance,
                )
            )

        return results
