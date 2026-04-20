# Phase 3: ColBERT Reranking

## Overview

Phase 3 adds a reranking layer on top of Phase 2 retrieval. Instead of feeding the first vector-search hits directly to an LLM, the system now:

1. Retrieves a wider candidate set from ChromaDB (`retrieve_k`, default = 10)
2. Reranks these candidates with a ColBERT-first strategy (`mixedbread-ai/mxbai-colbert-large-v1`) and an automatic CrossEncoder fallback
3. Keeps only the top final set (`final_k`, default = 3)

This improves ordering quality while preserving the high-recall behavior confirmed in Phase 2.3.

Backend strategy:

- Preferred backend: RAGatouille + ColBERT
- Fallback backend: sentence-transformers `CrossEncoder` (default fallback model: `BAAI/bge-reranker-v2-m3`)

The fallback is important for Windows/Python 3.13 environments where RAGatouille dependencies may be unavailable.

## Why Reranking Is Needed

Phase 2.3 established:

- Chapter Hit@1 = 90%
- Chapter Hit@3 = 100%

Interpretation: relevant chunks are already in the candidate set, but not always at rank 1. Reranking focuses on this exact problem: better ordering inside the top-k set.

## Implementation

### New Module

- `src/medora/rag/reranker.py`

This module introduces:

- `RerankerCandidate`: normalized candidate object from Chroma retrieval
- `RerankerResult`: normalized reranked output with score and rank
- `ColBERTReranker`: wrapper around RAGatouille backend

Design goals:

- Keep backend details isolated from evaluation/pipeline scripts
- Normalize output format even if backend response format varies by version
- Fail with explicit dependency instructions if `ragatouille` is missing

### New Evaluator

- `evaluation/reranking_validation.py`

This script compares baseline vs reranked results on the exact same Phase 2.3 query set (`MANUAL_QUERIES`) and reports:

- Chapter-level: Hit@1, Hit@3, Hit@k, MRR
- Section-level (for symptom queries): Hit@1, Hit@3, Hit@k, MRR
- Metric deltas: reranked minus baseline

Output file:

- `data/results/reranking_validation.json`

### Config Additions

`config.py` now includes:

- `RERANK_RETRIEVE_K = 10`
- `RERANK_FINAL_K = 3`

### Dependency Update

`requirements.txt` keeps `ragatouille` optional due to platform conflicts. Reranking remains functional through the CrossEncoder fallback using `sentence-transformers`.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run validation with defaults (retrieve 10, keep 3):

```bash
python evaluation/reranking_validation.py
```

Custom depths:

```bash
python evaluation/reranking_validation.py --retrieve-k 15 --final-k 5
```

Verbose inspection:

```bash
python evaluation/reranking_validation.py --verbose
```

## Expected Outcome

Given Phase 2.3 behavior, expected Phase 3 pattern is:

- Hit@3 remains near 100%
- Hit@1 improves (especially on previously rank-2 clinically valid cases)
- MRR improves modestly

## Known Requirements

Phase 3 execution requires all of the following:

1. Existing Phase 2 artifacts:
   - `data/chroma/` with collection `tmt_chunks`
2. Python dependency:
   - `ragatouille`
3. Model download access at runtime:
   - Preferred: `mixedbread-ai/mxbai-colbert-large-v1`
   - Fallback: `BAAI/bge-reranker-v2-m3`

If one is missing, reranking evaluation will not run until that prerequisite is satisfied.
