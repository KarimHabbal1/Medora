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

# === Model Names (will be updated after Phase 2 comparison) ===
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "openbiollm-8b"  # placeholder — decided after Phase 2
RERANKER_MODEL = "mixedbread-ai/mxbai-colbert-large-v1"
