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

# === Reranking Parameters ===
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_COMPARISON_MODEL = "ncbi/MedCPT-Cross-Encoder"
RERANK_TOP_K_RETRIEVE = 10   # retrieve this many from bi-encoder
RERANK_TOP_K_RETURN = 3      # return this many after reranking

# === Model Names ===
LLM_MODEL = "openbiollm-8b"  # placeholder — decided after Phase 3

# === LLM Configuration ===
DEFAULT_LLM_PROVIDER = "openai"  # "openai" or "ollama"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
OLLAMA_BASE_URL = "http://localhost:11434"


def make_llm(model: str = None, provider: str = None, ollama_url: str = None, temperature: float = 0):
    """Create an LLM instance — works with both OpenAI and Ollama.

    Args:
        model: model name/id. If it contains ":" it's treated as Ollama format.
               Examples: "gpt-4o-mini" (OpenAI), "gemma2:27b" (Ollama)
        provider: "openai" or "ollama". If None, auto-detected from model name.
        ollama_url: Ollama server URL (default from config)
        temperature: model temperature
    """
    if model is None:
        model = DEFAULT_LLM_MODEL
    if provider is None:
        provider = DEFAULT_LLM_PROVIDER
    if ollama_url is None:
        ollama_url = OLLAMA_BASE_URL

    # Auto-detect: if model contains ":" it's likely Ollama format (e.g. "gemma2:27b")
    if ":" in model and provider == "openai":
        provider = "ollama"

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, request_timeout=120)
    elif provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=ollama_url, temperature=temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'openai' or 'ollama'.")
