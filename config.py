"""Medora project configuration — paths, model names, parameters."""

from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
SYMPTOMS_DIR = DATA_DIR / "structured_symptoms"
RESULTS_DIR = DATA_DIR / "results"

# PDF source paths (outside repo — not committed to git)
PDF_DIR = Path("/Users/karim/Desktop/folders/Medora_StartUp")
TMT_PDF = PDF_DIR / "2022, CURRENT Medical Diagnosis and Treatment- Original.pdf"
OXFORD_PDF = PDF_DIR / "8205Oxford Handbook of Clinical Medicine 10th 2017 Edition_SamanSarKo - Copy copy.pdf"

# === Chunking Parameters ===
CHUNK_MIN_WORDS = 50
CHUNK_MAX_WORDS = 500
CHUNK_TARGET_WORDS = 200
BASELINE_CHUNK_WORDS = 200
BASELINE_OVERLAP_WORDS = 50

# === Embedding Parameters ===
EMBEDDING_MODEL = "sentence-transformers/embeddinggemma-300m-medical"
EMBEDDING_DIM = 768
EMBEDDING_BATCH_SIZE = 64
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHROMA_DIR = DATA_DIR / "chroma"

# === Model Names ===
LLM_MODEL = "openbiollm-8b"  # placeholder — decided after Phase 3
RERANKER_MODEL = "mixedbread-ai/mxbai-colbert-large-v1"
RERANKER_BACKEND = "auto"  # auto | ragatouille | cross-encoder
RERANKER_FALLBACK_MODEL = "BAAI/bge-reranker-v2-m3"

# === Phase 3 Retrieval + Reranking Parameters ===
RERANK_RETRIEVE_K = 10
RERANK_FINAL_K = 3
