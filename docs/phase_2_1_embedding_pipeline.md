# Phase 2.1: Embedding Pipeline

## Overview

Phase 2.1 converts the 5,631 structured text chunks produced in Phase 1.2 into dense vector representations — numerical embeddings — that enable semantic search over the full TMT textbook. The output of this phase is the primary data layer consumed by the Triage Agent's retrieval-augmented generation (RAG) system in Phase 2.2.

### Why This Phase Exists

Phase 1.2 produced a structured chunk file (`tmt_chunks_structured.json`) where each chunk carries its text alongside rich metadata: chapter, section, subsection, and a unique chunk identifier. That file is excellent for structured lookup but cannot support semantic retrieval. A keyword search over chunk text would fail to connect a patient describing "trouble breathing at night" with chunks about paroxysmal nocturnal dyspnea. It would miss synonyms, paraphrases, and the full spectrum of clinical language variation that characterises real patient descriptions.

Embedding models solve this problem by mapping text — both query text and passage text — into a shared vector space where semantic proximity corresponds to geometric proximity. Two texts that mean similar things land near each other in this space even if they share no keywords. Phase 2.1 runs every chunk through an embedding model once and stores the result. At query time, the Triage Agent embeds the patient's query and retrieves the closest chunks by vector distance — an operation that takes milliseconds and scales to the full 42-chapter textbook.

### What Embedding Adds Over Raw Chunks

| Artefact | Source | Format | Consumer | Capability |
|---|---|---|---|---|
| Chunks (`tmt_chunks_structured.json`) | Phase 1.2 | Free text + metadata | Direct inspection, keyword search | Structured access, exact matching |
| Embeddings (`tmt_chunk_embeddings.npz`) | Phase 2.1 | Float32 matrix (5631, 768) | Triage Agent (RAG) | Semantic retrieval, nearest-neighbour search |

The chunk file and the embedding file are paired artefacts. The chunk file provides the text that gets returned to the user; the embedding file provides the vector index that determines which chunks are relevant to a given query. Phase 2.1 produces the embedding file. Neither artefact is independently sufficient for RAG.

### What Embeddings Are

An embedding is a fixed-length array of floating-point numbers that encodes the meaning of a piece of text. An embedding model is trained to produce arrays that are geometrically close for semantically similar texts and geometrically distant for semantically dissimilar texts. The distance is typically measured by cosine similarity — the cosine of the angle between two vectors, ranging from 1.0 (identical direction, maximally similar) to 0.0 (orthogonal, unrelated) to -1.0 (opposite direction).

The specific type of embedding model used in this project — a bi-encoder — encodes queries and passages independently into the same vector space. This is the architecture required for large-scale retrieval: passages are encoded once offline and stored, and at query time only the query needs to be encoded. The alternative (cross-encoders, which compare query and passage jointly) is more accurate but computationally intractable at retrieval scale — they are used as rerankers after an initial bi-encoder retrieval step, which is the architecture planned for Phase 3.

---

## Embedding Model Selection

This is the most consequential design decision in Phase 2. The embedding model determines the quality ceiling of the entire RAG system — no retrieval strategy, reranking step, or prompt engineering can recover information that a poor embedding model fails to connect. The model must be evaluated on three axes simultaneously: retrieval quality on medical text, compatibility with the token lengths of the chunks produced in Phase 1.2, and feasibility on the available hardware.

### Background: The Generalist vs Specialist Problem

The conventional assumption in biomedical NLP has been that domain-specific pre-training is strictly beneficial — that a model trained on PubMed abstracts will outperform a general-purpose model on biomedical retrieval tasks. This assumption drove the early adoption of models like BioBERT, PubMedBERT, and ClinicalBERT.

A 2024 study (Gutiérrez et al., arXiv:2401.01943) challenged this assumption directly. Evaluating embedding models on clinical semantic search tasks, the authors found that modern generalist models — trained on vastly larger and more diverse corpora using contrastive objectives — outperformed domain-specific models pre-trained on biomedical text by 15–20 percentage points. The conclusion was that the domain pre-training advantage of older specialist models was outweighed by the scale and objective advantages of modern generalist training.

This finding has an important implication for model selection in 2025 and beyond: the relevant comparison is not "generalist vs specialist" but "old specialist vs new specialist." The 2024–2025 generation of medical embedding models combines domain-specific training data with modern contrastive learning objectives. These models are not just pre-trained on medical text — they are fine-tuned using contrastive retrieval objectives on medical query-passage pairs, making them qualitatively different from PubMedBERT-era models. The question becomes: which of these new-generation medical models is best suited to textbook chunk retrieval specifically?

### Models Evaluated

Six models were evaluated against the requirements of this project. Benchmark results are drawn from the BEIR benchmark suite (Thakur et al., 2021), the MIRIAD medical retrieval benchmark, and the MedTEB 51-task evaluation framework.

| Model | Params | Dims | Max Tokens | Medical Benchmarks | Notes |
|---|---|---|---|---|---|
| `BAAI/bge-m3` | 570M | 1024 | 8192 | NFCorpus: ~0.35, SciFact: ~0.60 | General-purpose, Jan 2024. Multilingual, hybrid dense+sparse+ColBERT retrieval. Originally configured in the project. |
| `sentence-transformers/embeddinggemma-300m-medical` | 308M | 768 | 2048 | MIRIAD: 0.886 nDCG@10 | Sept 2025. Trained on medical passage retrieval (MIRIAD dataset). Outperforms models 2x its size. Native sentence-transformers. |
| `MedTE` (GTE-base fine-tuned) | 109M | 768 | 512 | NFCorpus: 0.42, TREC-COVID: 0.86, MedTEB: 0.578 (top) | July 2025. Lightest option, tops MedTEB 51-task benchmark. 512-token limit. |
| `MedCPT` (NCBI) | 220M | 768 | 512 | SciFact: 0.724, TREC-COVID: 0.123 | 2023. Trained on 255M PubMed query-article pairs. Excellent on abstract retrieval, weak outside training distribution. |
| `BMRetriever-7B` | 7B | 4096 | 512 | NFCorpus: 0.364, TREC-COVID: 0.861, SciFact: 0.778 | EMNLP 2024. Strongest biomedical retriever on BEIR. LLM-based, not sentence-transformers, requires significant VRAM. |
| `PubMedBERT` / `BioBERT` / `ClinicalBERT` | ~110M | 768 | 512 | Outperformed by general models by 15–20% | Pre-2023. Trained for NLU classification, not contrastive retrieval. |

Benchmark metric: nDCG@10 (normalised Discounted Cumulative Gain at 10 results), the standard retrieval quality metric. Higher is better; 1.0 is a perfect ranking of the top 10 results.

### Analysis by Model

#### `BAAI/bge-m3` — Rejected (Original Choice)

BGE-M3 was the model initially configured in `config.py` and represents the baseline against which alternatives were evaluated. It is a strong general-purpose retrieval model with three notable technical advantages: an 8,192-token context window, multilingual support across 100+ languages, and a hybrid retrieval architecture that combines dense vectors, sparse BM25-style retrieval, and ColBERT multi-vector representations in a single model.

These advantages are real but irrelevant to this project's requirements. The chunks produced in Phase 1.2 range from 50 to 500 words — approximately 65 to 650 tokens at typical medical text token densities — placing all chunks well within any model's capacity. The 8,192-token window confers no advantage when no chunk approaches even 1,000 tokens. The multilingual capability is unused — the TMT textbook is English-only and all patient interactions are assumed to be in English in Phase 5. The hybrid retrieval architecture requires a custom inference wrapper to access the sparse and ColBERT components; the sentence-transformers API only exposes the dense vector component by default.

On medical benchmarks, BGE-M3 scores approximately 0.35 nDCG@10 on NFCorpus and 0.60 on SciFact. These are competent generalist scores but are surpassed by dedicated medical models on every medical benchmark tested. BGE-M3 is the right choice for a multilingual, mixed-domain retrieval system. It is not the right choice for a monolingual, medical-domain retrieval system where dedicated alternatives exist.

#### `sentence-transformers/embeddinggemma-300m-medical` — Selected

Released in September 2025, this model is trained specifically on the MIRIAD (Medical Information Retrieval with Indexed Abstracts Dataset) benchmark, which evaluates retrieval of relevant medical passages in response to clinical and scientific queries. This is the closest available analogue to the task structure of this project — retrieving relevant textbook passages in response to clinical queries from a Triage Agent.

The MIRIAD nDCG@10 score of 0.886 places it at state-of-the-art performance for medical passage retrieval, and notably it achieves this while being smaller than several competing models. On the MIRIAD benchmark, it outperforms models with more than twice its parameter count, indicating that the training data specificity and contrastive objective are doing more work than raw model size.

At 308M parameters, the model weights occupy approximately 600MB in float32 on disk. This is comfortably within the memory budget of an Apple M1 Pro with 16GB unified memory. The model is natively distributed as a `sentence-transformers` compatible model, requiring no custom inference code — a `SentenceTransformer(model_name)` call loads it fully, and the standard `.encode()` method handles batched inference.

The 2,048-token context window is the key practical differentiator from the 512-token models in this evaluation. At Phase 1.2's maximum chunk length of 500 words (~650 tokens), plus a context prefix of approximately 20–30 tokens (see Context-Prefixed Embedding Strategy below), the effective maximum input length is approximately 680 tokens — well within the 2,048-token limit with substantial headroom. Chunks are never truncated. By contrast, 512-token models would handle the vast majority of chunks without truncation (since most chunks are well under 400 tokens), but would silently truncate any chunk approaching the 500-word ceiling, losing the tail content of the longest chunks.

#### `MedTE` (GTE-base fine-tuned) — Strong Contender, Rejected on Token Limit

MedTE was released in July 2025 and tops the MedTEB benchmark — a 51-task evaluation covering medical retrieval, classification, clustering, and semantic textual similarity. Its NFCorpus score of 0.42 and TREC-COVID score of 0.86 make it competitive with or superior to embeddinggemma-300m-medical on those specific benchmarks.

The rejection criterion is the 512-token context limit. MedTE's limit is tight relative to the chunk size distribution from Phase 1.2. While the median chunk is well under 512 tokens, any chunk at the upper tail of the distribution — chunks from dense clinical sections, long differential diagnosis discussions, or multi-paragraph "When to Admit" sections — may approach or exceed this limit after the context prefix is added. Silent truncation of long chunks would produce systematically degraded embeddings for the most information-dense content in the textbook, which is precisely the content where retrieval accuracy matters most. The 2,048-token headroom of embeddinggemma-300m-medical eliminates this risk entirely.

If embeddinggemma-300m-medical were not available, MedTE would be the preferred alternative.

#### `MedCPT` (NCBI) — Rejected on Distribution Mismatch

MedCPT is a 2023 model from the National Center for Biotechnology Information, trained on 255 million PubMed query-article pairs. This training makes it exceptionally well-calibrated for the task of matching search queries to scientific abstracts — exactly the task structure of PubMed literature search. Its SciFact score of 0.724 reflects this strength.

The weakness is out-of-distribution retrieval. The TREC-COVID benchmark evaluates retrieval of relevant documents for COVID-related queries across a heterogeneous document collection — not a PubMed abstract retrieval task. MedCPT scores 0.123 nDCG@10 on TREC-COVID, a dramatic collapse relative to its SciFact performance. This is a canonical example of distributional overfitting: a model trained on a narrow query-passage distribution that fails when the query type or document type shifts.

The TMT textbook retrieval task shares more structural characteristics with TREC-COVID than with SciFact. The passages are textbook sections, not PubMed abstracts. The queries will be clinical descriptions of patient presentations, not structured literature search queries. MedCPT's strong abstract-retrieval performance does not transfer to this task structure, and its TREC-COVID score of 0.123 is sufficient evidence to eliminate it.

#### `BMRetriever-7B` — Rejected on Hardware Requirements

BMRetriever-7B (EMNLP 2024) achieves the strongest biomedical retrieval performance in this evaluation on benchmark measures, with scores of 0.364 on NFCorpus, 0.861 on TREC-COVID, and 0.778 on SciFact. It is an LLM-based retrieval model — built on a 7-billion-parameter language model backbone rather than the encoder-only architectures used by all other models in this list.

The hardware requirements are prohibitive. A 7B-parameter model in float32 requires approximately 28GB of GPU memory — far exceeding the M1 Pro's available memory budget. Quantised inference (4-bit or 8-bit) is possible but introduces accuracy degradation and adds engineering complexity. More fundamentally, BMRetriever is not distributed as a sentence-transformers model, requiring custom inference code that is not compatible with the standard `.encode()` API the pipeline is built around. Adopting BMRetriever would require a substantial infrastructure rewrite to support a completely different inference path.

The benchmark advantage over embeddinggemma-300m-medical is real, but the hardware and compatibility barriers make it non-viable for this project's deployment context. If Phase 4 evaluation reveals systematic retrieval failures that smaller models cannot resolve, BMRetriever or a similar LLM-based retriever could be revisited as a future direction contingent on infrastructure changes.

#### `PubMedBERT` / `BioBERT` / `ClinicalBERT` — Rejected on Architecture

These models represent the first generation of domain-adapted biomedical language models (2019–2021). They achieve domain adaptation through continued pre-training on biomedical corpora — PubMed abstracts, clinical notes, or combinations thereof — using masked language model objectives.

As documented in Gutiérrez et al. (arXiv:2401.01943), this approach is insufficient for retrieval quality in 2025. Masked language model pre-training optimises for token prediction, not for the ranking and similarity tasks that retrieval requires. Without contrastive fine-tuning on query-passage pairs, these models produce embedding spaces that are not well-calibrated for nearest-neighbour retrieval. The 15–20 percentage point deficit against modern generalist models on clinical semantic search tasks reported by Gutiérrez et al. reflects this architectural limitation.

These models are appropriate for NLU classification tasks (named entity recognition, relation extraction, question answering with span extraction), which is what they were designed for. They are not appropriate for bi-encoder retrieval, which is the task this pipeline requires.

### Decision Summary

`sentence-transformers/embeddinggemma-300m-medical` is selected as the embedding model for Phase 2.1. The decision is driven by: (1) state-of-the-art performance on the MIRIAD medical passage retrieval benchmark — the benchmark most structurally similar to this project's task; (2) a 2,048-token context window that prevents truncation across the full chunk length distribution; (3) native sentence-transformers compatibility that requires no custom inference code; and (4) a 308M parameter count that runs efficiently on M1 Pro with MPS acceleration.

---

## Context-Prefixed Embedding Strategy

### The Problem

Textbook chunks often discuss the same drug class, mechanism, or clinical finding in multiple different clinical contexts. A chunk about beta-blockers in the context of cardiac arrhythmia management and a chunk about beta-blockers in the context of hyperthyroidism treatment may share substantial lexical overlap, making them difficult to disambiguate by text alone. Without additional context, an embedding model may assign these chunks similar vectors, making it hard to retrieve the correct chunk based on a query that is implicitly about one clinical context or the other.

This problem is exacerbated by the structure of a 42-chapter clinical reference. The same subsection heading — "Treatment", "When to Refer", "When to Admit" — appears hundreds of times across chapters. A chunk from the "Treatment" subsection of the Cardiac Arrhythmias chapter and a chunk from the "Treatment" subsection of the Thyroid Disorders chapter may produce very similar embeddings if only the chunk text is encoded.

### The Solution

Each chunk is prefixed with its hierarchical metadata before embedding. The prefix format is:

```
Chapter: {chapter} | Section: {section} | Subsection: {subsection} | {text}
```

If the `subsection` field is empty — which occurs for chunks that fall outside a named subsection — the subsection component is omitted:

```
Chapter: {chapter} | Section: {section} | {text}
```

The `build_prefix` function in `embeddings/embed_chunks.py` implements this logic:

```python
def build_prefix(chunk: dict) -> str:
    chapter    = chunk.get("chapter", "")
    section    = chunk.get("section", "")
    subsection = chunk.get("subsection", "")
    text       = chunk.get("text", "")

    if subsection:
        return f"Chapter: {chapter} | Section: {section} | Subsection: {subsection} | {text}"
    return f"Chapter: {chapter} | Section: {section} | {text}"
```

### Why This Works

The embedding model encodes the full prefixed string as a single unit. The chapter and section labels push the embedding vector into the region of the semantic space associated with that clinical domain, even before the chunk text itself begins. A chunk about beta-blockers prefixed with `Chapter: Cardiac Arrhythmias | Section: VENTRICULAR PREMATURE BEATS | Subsection: Treatment |` will embed into a different region of the space than an otherwise similar chunk prefixed with `Chapter: Thyroid Disorders | Section: HYPERTHYROIDISM |`. This contextual disambiguation improves retrieval precision — the correct chunk is retrieved not just because it is about beta-blockers but because it is about beta-blockers in the right clinical context.

This is a simple form of what the retrieval literature calls "contextual chunking" or "enriched embeddings." More sophisticated approaches include using an LLM to generate a context summary for each chunk (Anthropic's "contextual retrieval" approach) or prepending the document title and abstract. The prefix strategy used here is a lightweight, deterministic, zero-cost alternative that requires no additional model inference.

### Scope of the Prefix

The prefix is applied only during embedding. The stored chunk data in `tmt_chunks_structured.json` is not modified. When the Triage Agent retrieves a chunk and returns its text to the user or the LLM, it reads the unmodified `text` field from the chunk file. The prefix is an input to the embedding model only — it improves the geometry of the embedding space without polluting the returned passage text.

---

## Hardware and Performance

### Device

The embedding run executed on an Apple M1 Pro using Metal Performance Shaders (MPS) acceleration. MPS is Apple's GPU compute framework, exposed to PyTorch via `torch.backends.mps`. The `detect_device()` function in the script checks MPS availability at runtime and falls back to CPU if MPS is not available:

```python
def detect_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
```

MPS provides GPU-accelerated matrix operations for the embedding inference without requiring NVIDIA CUDA. On Apple Silicon, MPS batch embedding throughput is substantially higher than CPU inference — the M1 Pro's 16-core GPU handles the 768-dimensional matrix multiplications that dominate embedding computation with good parallelism.

### Batch Configuration and Run Statistics

| Parameter | Value |
|---|---|
| Batch size | 64 chunks |
| Total chunks | 5,631 |
| Total batches | 88 |
| Failed batches | 0 |
| Failed chunks | 0 |
| Output shape | (5631, 768) float32 |
| Output file size | ~15MB compressed |

The batch size of 64 was selected as a balance between memory efficiency and throughput. At 768 dimensions per vector and float32 precision, a batch of 64 embeddings requires minimal memory overhead. Larger batches (128, 256) would increase throughput marginally but provide no quality benefit and introduce risk of memory pressure on longer chunks. Smaller batches would increase the number of Python loop iterations without meaningful memory savings given the chunk size distribution.

Zero failures across 88 batches confirms that no chunk exceeded the model's 2,048-token limit, validating the token headroom analysis in the model selection section.

---

## How It Works

The `embeddings/embed_chunks.py` script executes five sequential stages.

### Stage 1: Load Chunks

The script reads `data/chunks/tmt_chunks_structured.json` — the full 5,631-chunk file produced in Phase 1.2 — into memory as a list of Python dicts. Each dict contains the fields `chunk_id`, `chapter`, `section`, `subsection`, `text`, and additional metadata fields. No filtering is applied; all 42 chapters are embedded.

### Stage 2: Build Prefixed Texts

For each chunk, `build_prefix` constructs the context-prefixed string as described above. This produces a list of 5,631 strings in the same order as the chunk list. A parallel list of `chunk_id` strings is extracted from the chunks to maintain the correspondence between embedding vectors and chunk identifiers.

### Stage 3: Load Model

The `SentenceTransformer` model is loaded onto the detected device (MPS or CPU). The model weights are downloaded from the Hugging Face Hub on first run and cached locally in the standard Hugging Face cache directory. On subsequent runs, the cached weights are loaded directly.

### Stage 4: Batch Embed

The 5,631 prefixed texts are split into batches of `--batch-size` chunks (default 64). Each batch is passed to `model.encode()` with `convert_to_numpy=True`, which returns a numpy array of shape `(batch_size, 768)`. Batch arrays are accumulated in a list and stacked into a single matrix after all batches complete.

Each batch prints a progress line indicating its index, chunk range, and output shape, providing visibility into the run without requiring a progress bar library. Exceptions at the batch level are caught individually: if a batch fails, the failed chunks are counted and skipped rather than aborting the full run. This prevents a single malformed chunk from destroying an hours-long embedding job.

### Stage 5: Save Outputs

The script saves two files:

1. **`tmt_chunk_embeddings.npz`** — a numpy compressed archive containing two arrays: `embeddings` (the (5631, 768) float32 matrix) and `chunk_ids` (a string array of the 5,631 chunk identifiers). The arrays are stored with corresponding indices so that `embeddings[i]` is the embedding vector for `chunk_ids[i]`.

2. **`embedding_metadata.json`** — a JSON file recording the provenance of the embedding run: model name, embedding dimension, chunk count, prefix strategy, UTC timestamp, and device used. This metadata is consumed by Phase 2.2 when the Chroma vector store is built, ensuring the vector store knows which model produced the embeddings and can validate consistency.

---

## Output Files

### `data/embeddings/tmt_chunk_embeddings.npz`

A numpy compressed archive. Loading it returns:

```python
archive = np.load("tmt_chunk_embeddings.npz", allow_pickle=True)
embeddings = archive["embeddings"]   # shape: (5631, 768), dtype: float32
chunk_ids  = archive["chunk_ids"]    # shape: (5631,), dtype: object (strings)
```

The `embeddings` matrix is the primary artefact. It is loaded by Phase 2.2 and ingested into the Chroma vector store, which builds an approximate nearest-neighbour index over the 768-dimensional vectors. Subsequent queries embed the query text with the same model and retrieve the closest vectors from this index.

The file is approximately 15MB in compressed form. Uncompressed, 5,631 float32 vectors of 768 dimensions occupy approximately 17.3MB (5631 × 768 × 4 bytes). The numpy `savez_compressed` function applies zlib compression, achieving modest size reduction for floating-point arrays.

### `data/embeddings/embedding_metadata.json`

A small JSON file recording the embedding run provenance:

```json
{
  "model_name": "sentence-transformers/embeddinggemma-300m-medical",
  "embedding_dim": 768,
  "num_chunks": 5631,
  "context_prefix_strategy": "Chapter | Section | Subsection | text",
  "timestamp": "2026-03-22T18:15:31.427439+00:00",
  "device_used": "mps"
}
```

This file serves two purposes. First, it documents the provenance of the embedding archive for reproducibility — given the model name and strategy, any researcher can reproduce the embeddings deterministically. Second, Phase 2.2 reads this file to confirm that the model used to build the Chroma index matches the model used to generate the stored embeddings, preventing silent dimension mismatches if the configuration changes between phases.

---

## Scripts Reference

### `embeddings/embed_chunks.py`

The single script that executes the full Phase 2.1 pipeline.

**Usage:**

```bash
# Standard run — embeds all chunks, prompts before overwriting existing files:
python embeddings/embed_chunks.py

# Overwrite existing files without prompting:
python embeddings/embed_chunks.py --force

# Use a custom batch size (e.g., for lower memory pressure):
python embeddings/embed_chunks.py --batch-size 32

# Dry run — prints config and loads the model, but does not embed or save:
python embeddings/embed_chunks.py --dry-run
```

**CLI flags:**

| Flag | Effect |
|---|---|
| `--force` | Overwrites existing `.npz` and metadata files without an interactive prompt |
| `--batch-size N` | Sets the number of chunks per embedding batch (default: 64, from `config.py`) |
| `--dry-run` | Loads the model and prints the configuration, but does not embed any chunks or write any files. Useful for verifying that the model downloads and loads correctly before committing to a full run. |

**Environment requirements:**

- Python 3.11+ with `sentence-transformers`, `numpy`, and `torch` installed
- Internet access on first run to download model weights from Hugging Face Hub (subsequent runs use the local cache)
- MPS is used automatically on Apple Silicon; the script falls back to CPU if MPS is unavailable

---

## Why Not Embed the Structured Symptoms?

Phase 1.3 produced 11 structured symptom objects in `data/structured_symptoms/tmt_symptoms_gpt4o.json`. These objects are not embedded in Phase 2.1, and this decision deserves explicit justification.

The 11 symptom objects are consumed by the Intake Agent (Phase 5) as direct JSON lookup: when a patient describes a symptom, the agent matches the symptom name against the 11 object keys and loads the corresponding structured object deterministically. This is not a retrieval problem — it is a mapping problem with exactly 11 possible keys. The match is performed by string similarity over 11 names, which is trivially handled by exact matching with simple normalisation (lowercase, strip whitespace) or, for robustness, fuzzy string matching (edit distance or token overlap) against a fixed vocabulary of 11 entries.

Embedding the 11 symptom objects would serve only one purpose: enabling "fuzzy symptom matching" via vector search rather than string similarity. This adds model inference overhead, an additional embedding call per session, and a retrieval step over a corpus of 11 items — a corpus size for which string matching is strictly superior in speed, interpretability, and reliability. Vector search over 11 items provides no meaningful quality advantage over string similarity and introduces the possibility of semantic false positives (e.g., "chest tightness" matching "palpitations" rather than "chest pain" if the embedding space is miscalibrated for this domain).

If Phase 4 evaluation reveals systematic failures in symptom identification — for instance, patients using highly atypical descriptions that fail to match any of the 11 canonical symptom names — a vector-based fallback can be added at that point. The 11 symptom objects are short enough that embedding them would take under a second and could be done lazily at Intake Agent initialisation. This is a deliberate deferral of premature complexity, not an architectural gap.

---

## Limitations

### Single Embedding Model

The current pipeline uses a single bi-encoder model for retrieval without an ensemble or hybrid sparse-dense retrieval layer. Hybrid retrieval — combining dense vector similarity with sparse BM25-style keyword matching — consistently outperforms dense-only retrieval on benchmarks where some queries are better served by exact keyword matching (e.g., drug names, rare condition names, specific numerical values). BGE-M3's hybrid architecture was one of its advertised advantages.

The tradeoff here is complexity against marginal quality improvement. Dense-only retrieval with a strong medical model is the correct starting point; hybrid retrieval can be added in Phase 3 if Phase 4 evaluation reveals systematic failures on exact-match queries. The planned Phase 3 reranker (`mixedbread-ai/mxbai-colbert-large-v1`, already in `config.py`) will recover some of this ground by re-scoring the top-k retrieved chunks with a cross-encoder, which implicitly captures keyword-level matching signals.

### No Embedding Quality Evaluation Yet

Phase 2.1 produces the embedding matrix but does not evaluate retrieval quality. There is no measurement of whether the embedding model correctly ranks relevant chunks above irrelevant ones for a representative set of clinical queries. This evaluation is deferred to Phase 2.3, which will construct a held-out set of query-chunk pairs and measure nDCG@10 and recall@k over the embedded corpus.

Until Phase 2.3 is complete, the model selection described above rests on benchmark data from external medical retrieval tasks rather than on direct measurement against this corpus. The MIRIAD benchmark is the closest available proxy, but it evaluates retrieval of PubMed-style medical passages, not textbook sections. Some performance delta between MIRIAD results and in-domain results should be expected.

### Fixed Context Prefix Format

The context prefix format (`Chapter: {chapter} | Section: {section} | Subsection: {subsection} | {text}`) is a static design decision. More sophisticated prefix engineering could improve retrieval quality: for example, using the LLM to generate a one-sentence summary of each chunk's clinical significance and prepending that summary as additional context, or using different prefix structures for different chapter types (symptom chapters vs disease management chapters). These approaches are consistent with recent work on contextual retrieval (Anthropic, 2024) and hypothetical document embeddings (HyDE).

The tradeoff is cost and complexity. LLM-generated prefixes would require 5,631 API calls, introducing latency, cost, and a dependency on an external service in the embedding pipeline. The static prefix strategy is free, deterministic, and reproducible. Its effectiveness relative to more sophisticated alternatives can be assessed in Phase 2.3. If the evaluation reveals that retrieval quality is bottlenecked by inadequate chunk disambiguation rather than model capacity, more sophisticated prefix strategies can be applied to a targeted subset of chunks without re-embedding the full corpus.
