#!/usr/bin/env python3
"""
Pre-download and cache HuggingFace models for the Triage Agent.

The Triage Agent loads two large language models on startup:
  - Bi-encoder: sentence-transformers/embeddinggemma-300m-medical (~300MB)
  - Reranker: BAAI/bge-reranker-v2-m3 (~600MB)

These are downloaded from HuggingFace on first run, which can take several minutes.
Run this script ONCE before starting the server to pre-cache them locally.

Usage:
    python download_models.py

The models will be cached in ~/.cache/huggingface/hub/ (default HuggingFace cache directory).
"""

import sys


def main():
    print("Downloading HuggingFace models for Medora Triage Agent...")
    print("This may take a few minutes depending on your internet connection.\n")

    try:
        from sentence_transformers import SentenceTransformer
        print("Downloading bi-encoder (sentence-transformers/embeddinggemma-300m-medical)...")
        print("  Size: ~300MB")
        SentenceTransformer("sentence-transformers/embeddinggemma-300m-medical")
        print("  ✓ Bi-encoder cached successfully.\n")
    except Exception as e:
        print(f"  ✗ Error downloading bi-encoder: {e}")
        sys.exit(1)

    try:
        from sentence_transformers import CrossEncoder
        print("Downloading reranker (BAAI/bge-reranker-v2-m3)...")
        print("  Size: ~600MB")
        CrossEncoder("BAAI/bge-reranker-v2-m3", trust_remote_code=True)
        print("  ✓ Reranker cached successfully.\n")
    except Exception as e:
        print(f"  ✗ Error downloading reranker: {e}")
        sys.exit(1)

    print("=" * 60)
    print("All models cached successfully!")
    print("You can now start the backend server without delays:")
    print("  cd backend && python -m uvicorn app.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
