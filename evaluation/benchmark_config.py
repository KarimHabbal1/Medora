"""
Benchmark configuration — models, test sets, and metrics.

Two execution environments:
  LOCAL (your machine): API models (GPT-5.4-mini, GPT-5.4) — needs OPENAI_API_KEY
  EC2 (AWS server): Ollama models (Llama, Gemma, Phi, MedLlama) — needs Ollama running

Usage:
  # Run API models locally
  python evaluation/benchmark.py --profile api
  python evaluation/pipeline_benchmark.py --run --profile api

  # Run local models on EC2
  python evaluation/benchmark.py --profile ollama
  python evaluation/pipeline_benchmark.py --run --profile ollama

  # Run a specific model
  python evaluation/benchmark.py --models gpt-5.4-mini
"""

# === Model Configurations ===

API_MODELS = [
    {
        "name": "gpt-5.4-mini",
        "provider": "openai",
        "model_id": "gpt-5.4-mini",
        "description": "Current frontier mini — best cost/quality",
        "cost_input_per_m": 0.75,
        "cost_output_per_m": 4.50,
        "timeout_seconds": 120,
    },
    {
        "name": "gpt-5.4",
        "provider": "openai",
        "model_id": "gpt-5.4",
        "description": "Best available — ceiling benchmark (expensive)",
        "cost_input_per_m": 2.50,
        "cost_output_per_m": 10.00,  # estimate
        "timeout_seconds": 120,
    },
]

OLLAMA_MODELS = [
    {
        "name": "llama3.1-70b",
        "provider": "ollama",
        "model_id": "llama3.1:70b-instruct-q4_K_M",
        "description": "Strongest open-source, 4-bit quantized",
        "timeout_seconds": 300,
    },
    {
        "name": "gemma2-27b",
        "provider": "ollama",
        "model_id": "gemma2:27b",
        "description": "Best that fits fully in A10G VRAM",
        "timeout_seconds": 300,
    },
    {
        "name": "phi4-14b",
        "provider": "ollama",
        "model_id": "phi4:14b",
        "description": "Strong reasoning for size",
        "timeout_seconds": 300,
    },
    {
        "name": "llama3.1-8b",
        "provider": "ollama",
        "model_id": "llama3.1:8b",
        "description": "Lightweight baseline",
        "timeout_seconds": 180,
    },
    {
        "name": "medllama2-7b",
        "provider": "ollama",
        "model_id": "medllama2:7b",
        "description": "Medical-specific (Llama 2 based) — obsolete, fails structured output",
        "timeout_seconds": 180,
    },
    {
        "name": "deepseek-r1-14b",
        "provider": "ollama",
        "model_id": "deepseek-r1:14b",
        "description": "Reasoning model distilled from 93% USMLE parent — chain-of-thought",
        "timeout_seconds": 600,
    },
    {
        "name": "qwen2.5-32b",
        "provider": "ollama",
        "model_id": "qwen2.5:32b",
        "description": "Largest model fitting A10G — excellent structured output",
        "timeout_seconds": 600,
    },
    {
        "name": "qwen3-32b",
        "provider": "ollama",
        "model_id": "qwen3:32b",
        "description": "Latest Qwen, strong reasoning, fits in A10G VRAM",
        "timeout_seconds": 600,
    },
    {
        "name": "llama3.3-70b",
        "provider": "ollama",
        "model_id": "llama3.3:70b",
        "description": "Newest Llama 70B — CPU offload, slow but strongest open-source",
        "timeout_seconds": 900,
    },
    {
        "name": "aloe-8b",
        "provider": "ollama",
        "model_id": "aloe-8b",
        "description": "Medical fine-tune of Llama 3.1 8B — best 8B medical model",
        "timeout_seconds": 300,
    },
]

ALL_MODELS = API_MODELS + OLLAMA_MODELS

# === Test Set Configurations ===

TEST_SETS = {
    "textbook": {
        "name": "From the Book",
        "description": "Cases generated from TMT textbook chunks — validates RAG pipeline works",
        "file": "data/evaluation/textbook_test_cases.json",
        "default_num_cases": 50,
    },
    "medqa": {
        "name": "Related but External (MedQA USMLE)",
        "description": "USMLE vignettes filtered to textbook-covered topics — tests generalization",
        "file": "data/evaluation/medqa_filtered_cases.json",
        "default_num_cases": 50,
    },
    "medcase": {
        "name": "Outside the Book (MedCaseReasoning)",
        "description": "Published rare case reports — tests handling of knowledge gaps",
        "dataset_id": "zou-lab/MedCaseReasoning",
        "split": "test",
        "default_num_cases": 50,
    },
}

# === Execution Profiles ===

PROFILES = {
    "api": {
        "description": "Run locally with API models (needs OPENAI_API_KEY)",
        "models": API_MODELS,
        "test_sets": ["textbook", "medqa", "medcase"],
        "num_cases": 50,
    },
    "api-ceiling": {
        "description": "Run GPT-5.4 on 10 cases per test set (expensive ceiling benchmark)",
        "models": [m for m in API_MODELS if m["name"] == "gpt-5.4"],
        "test_sets": ["textbook", "medqa", "medcase"],
        "num_cases": 10,
    },
    "ollama": {
        "description": "Run on EC2 with all Ollama models",
        "models": OLLAMA_MODELS,
        "test_sets": ["textbook", "medqa", "medcase"],
        "num_cases": 50,
    },
    "quick": {
        "description": "Quick smoke test — 5 cases, GPT-5.4-mini only",
        "models": [API_MODELS[0]],
        "test_sets": ["textbook"],
        "num_cases": 5,
    },
    "full": {
        "description": "Everything — all models, all test sets, 50 cases each",
        "models": ALL_MODELS,
        "test_sets": ["textbook", "medqa", "medcase"],
        "num_cases": 50,
    },
}

# === Metrics to Collect ===

METRICS = [
    "accuracy",
    "exact_match_rate",
    "semantic_match_rate",
    "partial_match_rate",
    "retrieval_hit_rate",
    "retrieval_precision",
    "mean_latency_s",
    "median_latency_s",
    "p95_latency_s",
    "json_error_rate",
    "retrieval_only_fail_rate",
    "generation_only_fail_rate",
    "total_tokens",
    "cost_usd",  # calculated from token count + model pricing
]

# === Judge Model ===
JUDGE_MODEL = "gpt-5.4-mini"  # cheap, fast, good enough for matching


def get_models_by_profile(profile_name: str) -> list[dict]:
    """Get the model list for a given profile."""
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(PROFILES.keys())}")
    return PROFILES[profile_name]["models"]


def get_models_by_names(names: list[str]) -> list[dict]:
    """Get model configs by name."""
    name_map = {m["name"]: m for m in ALL_MODELS}
    result = []
    for name in names:
        if name not in name_map:
            print(f"WARNING: Unknown model '{name}'. Available: {list(name_map.keys())}")
            continue
        result.append(name_map[name])
    return result


def get_test_set_config(name: str) -> dict:
    """Get test set configuration."""
    if name not in TEST_SETS:
        raise ValueError(f"Unknown test set: {name}. Available: {list(TEST_SETS.keys())}")
    return TEST_SETS[name]
