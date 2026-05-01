# Phase 2.2: Vector Store

## Overview

Phase 2.2 takes the two artefacts produced by Phase 2.1 — the numpy embedding matrix (`tmt_chunk_embeddings.npz`) and the structured chunk file (`tmt_chunks_structured.json`) — and loads them into a persistent ChromaDB vector database. The output is a queryable semantic search index over all 5,631 textbook chunks, stored to disk at `data/chroma/` and consumed by the Triage Agent in Phase 3 and beyond.

### Why This Phase Exists

Phase 2.1 produced a (5631, 768) float32 matrix of embedding vectors. That matrix is sufficient for the simplest form of semantic search — load it into RAM, embed a query, compute cosine distances against all rows, return the closest ones. At 5,631 vectors this is computationally trivial: a numpy matmul over 5,631 rows of 768 floats takes under a millisecond on modern hardware.

What numpy alone cannot do:

- **Persist across processes.** A numpy array in memory is gone when the process exits. Every new agent session would need to re-load the 17MB matrix from disk and hold it in RAM for the duration of the session.
- **Index for approximate nearest-neighbour search.** Numpy brute-force cosine similarity scales as O(n) per query. At 5,631 vectors this is fine; at 100,000 or 1,000,000 vectors it becomes a bottleneck. A vector database builds an ANN index (in ChromaDB's case, HNSW) that amortises query cost logarithmically.
- **Filter by metadata.** Numpy has no concept of the metadata attached to each vector. If the Triage Agent wants to restrict a search to the Infectious Diseases chapter, numpy requires a secondary lookup — load the chunk JSON, find indices matching the chapter, slice the embedding matrix, run search — with all the custom bookkeeping that entails. A vector database stores metadata alongside each vector and applies filters at query time in a single operation.
- **Store documents.** Numpy stores numbers. A vector database can store the original text alongside the vector, so a query returns the passage text directly without a secondary lookup against the chunk JSON.
- **Provide a query API.** A vector database exposes a `query()` interface that accepts a vector (or, with a model integration, a string), applies optional metadata filters, and returns ranked results with distances. Building the equivalent from numpy requires writing and maintaining that code permanently.

ChromaDB provides all of these capabilities. Phase 2.2 is the bridge that converts the raw numpy artefact from Phase 2.1 into a production-ready retrieval layer.

### What ChromaDB Is

ChromaDB is an open-source vector database written in Python. It stores vectors, associated document strings, and metadata dictionaries in a persistent on-disk format backed by SQLite (for metadata and document storage) and an HNSW index (for approximate nearest-neighbour vector search). It runs in-process — there is no server daemon to start, no Docker container to manage — and it exposes a pure-Python API.

A ChromaDB "collection" is the unit of storage: a named group of (id, vector, document, metadata) tuples. Collections support upsert (insert or update), delete, and query operations. The query operation takes one or more query vectors, an optional metadata filter expressed as a dict, and a `n_results` integer, and returns the closest matching entries by the collection's configured distance metric.

For Phase 2.2, one collection named `tmt_chunks` is created containing all 5,631 chunks.

---

## Vector Store Selection

The choice of vector store is a foundational infrastructure decision. Getting it wrong means either rebuilding the retrieval layer later or accepting persistent performance or maintainability costs. Five categories of option were evaluated.

### ChromaDB — Selected

ChromaDB is open-source, Python-native, and runs in-process. No server process, no Docker container, no network dependency. A `PersistentClient` call points at a directory on disk; subsequent calls in the same or any future Python process see the same data. Metadata filtering is built into the query API. Document text is stored alongside vectors. The library is installable with `pip install chromadb` and has no system-level dependencies beyond Python itself.

The limitations are real but irrelevant at this scale: ChromaDB is not designed for billion-vector corpora, does not support distributed deployment, and is not the right choice if the retrieval layer needs to serve thousands of concurrent queries per second in a cloud production environment. None of those constraints apply here. A corpus of 5,631 vectors is trivially small. The system runs locally on a single machine. A thesis-scale project should not introduce operational complexity it does not need.

### FAISS (Facebook AI Similarity Search) — Rejected

FAISS is the most widely used library for large-scale approximate nearest-neighbour search. It is extremely fast, highly optimised for GPU acceleration, and supports a broad range of index types (flat, IVF, PQ, HNSW, and composites thereof) that offer different quality/speed/memory tradeoffs. For corpora in the tens of millions of vectors where raw retrieval throughput dominates, FAISS is the right tool.

The problem is that FAISS is a pure vector search library, not a database. It stores vectors and returns indices into those vectors. It has no concept of metadata, no document storage, no persistence API (the caller is responsible for serialising the index to disk with `faiss.write_index` and reloading it), and no query interface that returns text. Using FAISS for this project would require building a parallel metadata store (a dict or SQLite database, keyed by vector index), writing custom serialisation and deserialisation logic, and maintaining the correspondence between FAISS index positions and chunk metadata manually. This is a non-trivial amount of infrastructure code for capabilities that ChromaDB provides out of the box.

The performance advantage of FAISS over ChromaDB is real at scale and irrelevant at 5,631 vectors. Both would complete a nearest-neighbour query in under a millisecond. The engineering overhead of FAISS's missing infrastructure layer is not justified.

### Pinecone — Rejected

Pinecone is a fully managed cloud vector database. It handles scaling, replication, and infrastructure; the caller interacts with it through an HTTP API. Its query performance and metadata filtering capabilities are excellent.

Three rejection criteria apply simultaneously. First, Pinecone requires an internet connection for every query — the Triage Agent cannot function offline, and any network disruption disrupts the retrieval layer. Second, Pinecone requires an API key and incurs cost at scale, introducing an external financial dependency into a thesis project. Third, and most importantly, a thesis project that depends on a managed external service is not fully reproducible: a future researcher who wants to run the system cannot do so without a Pinecone account and the same API key. Local reproducibility is a non-negotiable property of thesis-grade research software.

### Weaviate, Qdrant, Milvus — Rejected

These are full-featured, production-grade vector databases with sophisticated filtering, multi-tenancy, and horizontal scaling capabilities. They are the right choices for enterprise deployments serving millions of documents and thousands of concurrent users. They are all substantially overkill for 5,631 vectors.

More practically, all three require running a separate server process — typically via Docker — before any code can interact with them. That is a significant operational overhead for a local development and research environment. It means every session requires a running Docker container, the database server must be configured and kept alive, and any collaborator or examiner who wants to run the system must have Docker installed and must know to start the container first. ChromaDB's in-process model eliminates this entire class of operational friction.

### Simple Numpy Cosine Similarity — Rejected

Direct brute-force cosine similarity over the numpy embedding matrix would work at this corpus size. The computation is fast, the implementation is trivial (a matrix multiplication followed by an argsort), and no additional library is required.

The problems arise immediately when the retrieval layer is integrated into the Triage Agent. There is no metadata filtering without custom code. There is no document text without a secondary lookup against the chunk JSON. There is no persistence model — the matrix must be loaded into RAM at agent startup and held there. There is no query API — the agent must invoke the numpy computation directly rather than through a stable database interface. As the corpus grows (more textbooks, updated editions), managing a growing numpy matrix with associated metadata becomes progressively harder to maintain correctly.

The numpy approach is appropriate for a quick offline analysis of embedding quality. It is not appropriate as the retrieval backend for an agent that will be developed, tested, and extended across multiple phases. Using it would mean reimplementing incrementally, in custom code, exactly what ChromaDB provides.

### Decision Summary

| Option | Persistence | Metadata Filtering | Document Storage | In-Process | Cost/Deps | Verdict |
|---|---|---|---|---|---|---|
| **ChromaDB** | Yes (SQLite) | Yes (native) | Yes | Yes | Free, pip | **Selected** |
| FAISS | Manual | No (DIY) | No (DIY) | Yes | Free, pip | Rejected — missing infrastructure |
| Pinecone | Yes (cloud) | Yes | Yes | No (HTTP) | API key, paid | Rejected — not reproducible |
| Weaviate/Qdrant/Milvus | Yes (server) | Yes | Yes | No (Docker) | Free, complex | Rejected — operational overhead |
| Numpy brute-force | No (manual) | No (DIY) | No (DIY) | Yes | None | Rejected — no database features |

ChromaDB is selected because it provides the full database feature set — persistence, metadata filtering, document storage, query API — with zero operational overhead, zero cost, and zero external dependencies. At 5,631 vectors, no scale optimisation that any alternative provides is relevant.

---

## Collection Structure Decision

With ChromaDB chosen, the next design question is how to structure the data within it. Three options were considered.

### Option A: Single Collection — Selected

All 5,631 chunks are stored in a single collection named `tmt_chunks`. Metadata fields (`chapter`, `section`, `subsection`, and others) are attached to every record. When the Triage Agent needs chapter-scoped retrieval — for instance, to search only within the Cardiology chapters for a cardiac presentation — it passes a `where` filter to the query:

```python
results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    where={"chapter": "Heart Disease"},
)
```

Cross-chapter search — the primary retrieval mode for the Triage Agent — requires no filter at all. A single `collection.query()` call searches all 5,631 vectors.

Code complexity is minimal: one collection to create, one to query, one to maintain. If the corpus grows (a second textbook, updated chapters), entries are added to the same collection with no structural change.

### Option B: Multiple Collections (One Per Chapter)

The TMT textbook has 42 chapters. One collection per chapter would mean 42 ChromaDB collections, each containing approximately 134 chunks on average. Chapter-scoped queries become trivial — query the right collection. But cross-chapter queries, which are the Triage Agent's primary use case, require querying all 42 collections, collecting 42 result sets, merging them, and re-ranking by distance. This is a materially more complex query path with no performance benefit: ChromaDB handles millions of vectors in a single collection. Partitioning 5,631 vectors across 42 collections does not improve query speed; it only adds merge complexity.

There is also a maintenance burden. Adding a new chapter requires creating a new collection. Renaming a chapter (if chapter metadata changes in a revised source) requires migrating documents between collections. The single-collection approach with metadata filtering handles all of these scenarios without structural change.

### Option C: Two Collections (Chapter 2 Symptoms vs. the Rest)

The project has two major agent roles: the Intake Agent (Phase 5) handles symptom collection from the patient using the structured symptom data from Phase 1.3, and the Triage Agent (Phase 3) performs retrieval-augmented reasoning over the full textbook. One might reasonably split the vector store to mirror this architectural boundary — a symptom collection and a general collection.

The problem is that this split does not serve either agent. The Intake Agent does not use vector search for symptoms. It loads the 11 structured symptom objects from `data/structured_symptoms/tmt_symptoms_gpt4o.json` directly and matches patient descriptions against them using string similarity over 11 known keys. There is no retrieval query that benefits from a dedicated symptom collection. The Triage Agent needs the full textbook, and the Chapter 2 (symptom chapter) content is legitimately part of the textbook it retrieves from — a patient's presenting symptom may be directly relevant to a chunk from the symptom chapter.

The two-collection split would add code complexity (two collections to build, two to query depending on context, a routing decision at query time) with no benefit to either consumer.

### Decision Summary

Option A — a single collection with metadata filtering — is the correct choice. At 5,631 vectors, no partitioning strategy improves performance. Metadata filtering provides all the scoping power of separate collections without the code complexity, maintenance overhead, or query merge logic. The Triage Agent's primary retrieval mode is cross-chapter search, which requires no structural partitioning.

---

## Distance Metric Decision

ChromaDB collections require a distance metric at creation time, specified via the `hnsw:space` metadata key. Three options are available.

### Cosine Similarity — Selected

Cosine similarity measures the angle between two vectors in the embedding space, independent of their magnitudes. A cosine distance of 0 means the vectors point in exactly the same direction (semantically identical); a cosine distance of 2 means they point in opposite directions (semantically opposite). For embeddings produced by bi-encoder models, this is the geometrically correct measure of semantic similarity.

The critical reason cosine similarity is the correct choice here is that `sentence-transformers/embeddinggemma-300m-medical` — the model selected in Phase 2.1 — was trained using a contrastive objective that optimises cosine similarity between related (query, passage) pairs. The model's weights, training data, and evaluation benchmarks (including the MIRIAD nDCG@10 score of 0.886 used in model selection) are all defined with respect to cosine similarity. Every benchmark comparison in Phase 2.1 was measured using cosine similarity. Using any other distance metric in the vector store would be measuring something the model was not trained or evaluated to optimise, producing retrieval results that are geometrically inconsistent with the model's learned representation.

### Euclidean Distance (L2)

Euclidean distance measures the straight-line distance between two points in embedding space. Unlike cosine similarity, it is sensitive to vector magnitude — two vectors pointing in the same direction but at different distances from the origin will have non-zero L2 distance even though their semantic content is identical by cosine measure.

Sentence-transformer models do not guarantee that their output vectors have uniform magnitude. The magnitude of an embedding vector carries no meaningful information about the semantic content of the encoded text — it is a side-effect of the model's architecture and the specific input. Using L2 distance conflates this uninformative magnitude variation with semantic distance, introducing noise into the ranking. L2 would be appropriate only if the embeddings were explicitly L2-normalised before storage, in which case cosine similarity and L2 distance produce equivalent rankings anyway (since all vectors lie on the unit sphere). The cosine setting handles this directly.

### Inner Product (Dot Product)

Dot product (inner product) is the product of vector magnitudes and the cosine of the angle between them. It is cosine similarity scaled by the magnitudes of both vectors. Dot product retrieval is used in some production retrieval systems (notably DPR and some ColBERT variants) where the model is explicitly trained to use magnitude as a relevance signal — the model learns to emit larger-magnitude vectors for more important passages.

`embeddinggemma-300m-medical` was not trained with a dot-product objective and makes no guarantee that embedding magnitude encodes passage importance. Using inner product here would conflate magnitude variation (which is uninformative for this model) with similarity, producing noisy rankings.

### Decision Summary

Cosine similarity is not a genuine tradeoff for this model. It is the only metric that is geometrically consistent with how the model was trained. The collection is created with `metadata={"hnsw:space": "cosine"}`.

---

## Persistence Decision

ChromaDB supports two client modes: persistent (data written to disk) and ephemeral (data held in memory).

### Persistent — Selected

`chromadb.PersistentClient(path=str(chroma_dir))` writes the collection to disk at `data/chroma/`. SQLite stores the document texts and metadata records. HNSW index files store the vector index. All data survives process restarts; the next session opens the same client, calls `get_or_create_collection`, and has immediate access to all 5,631 vectors without re-embedding or re-indexing.

The practical consequence is that Phase 2.2 needs to be run once. Every downstream phase — the Triage Agent in Phase 3, evaluation scripts in Phase 2.3, any future RAG experimentation — loads the persistent store in read-only mode without triggering any rebuild.

### Ephemeral (In-Memory)

`chromadb.EphemeralClient()` stores data in RAM only. The store is destroyed when the process exits. For Phase 2.2's use case, this would mean re-embedding 5,631 chunks — approximately 10 minutes of inference on M1 Pro — every time the Triage Agent starts a new session. There is no operational scenario where this is preferable to persistence.

Ephemeral mode is appropriate for unit tests where a clean, isolated store is needed for each test run. Phase 2.2's unit tests could use an ephemeral client with a small subset of chunks rather than the full persistent store. This is a testing-layer consideration, not a deployment decision.

### Storage Location and Git Policy

The persistent store is written to `data/chroma/`, which is added to `.gitignore`. The ChromaDB files are generated artefacts — anyone with the chunk JSON and embedding `.npz` can reproduce the store by running `python embeddings/build_vector_store.py`. Committing the generated store would add 20–30MB of binary files to the repository, bloating history and providing no reproducibility benefit over committing the script and data files that produce it.

Note: an earlier experimental store exists at `chroma_symptoms/` in the project root. That directory is from pre-Phase-2.2 experimentation and is not the canonical store. The canonical persistent store is `data/chroma/` only.

---

## How It Works

The `embeddings/build_vector_store.py` script executes six sequential stages.

### Stage 1: Load Chunks

```python
chunks = load_chunks(CHUNKS_FILE)
```

The script reads `data/chunks/tmt_chunks_structured.json` into memory as a list of 5,631 Python dicts. This is the same chunk file produced in Phase 1.2 and consumed (for embedding) by Phase 2.1. The full list is loaded without filtering — all 42 chapters are included.

### Stage 2: Load Embeddings

```python
embeddings, chunk_ids_npz = load_embeddings(NPZ_FILE)
```

The `.npz` archive from Phase 2.1 is loaded with `np.load(..., allow_pickle=True)`. The `embeddings` array (shape: (5631, 768), dtype: float32) and the `chunk_ids` string array are extracted. The embedding vectors are cast to float32 explicitly — ChromaDB's upsert API expects lists of Python floats, and ensuring float32 precision avoids any silent upcast to float64 that would double memory usage during the `.tolist()` conversion.

### Stage 3: Verify Alignment

```python
verify_alignment(chunk_ids_npz, chunks)
```

The script confirms that the `.npz` file and the chunk JSON contain exactly the same count of entries. A count mismatch indicates that the embedding run and the chunk file are out of sync — for instance, if chunks were added or removed from the JSON after the embedding run completed. A mismatch causes an immediate `sys.exit(1)` rather than silently proceeding with misaligned data, which would produce a vector store where embedding[i] does not correspond to chunk[i].

At the time of writing, both sources contain 5,631 entries and alignment verification passes.

### Stage 4: Deduplicate IDs

```python
unique_ids = deduplicate_ids(chunk_ids_npz)
```

ChromaDB requires that every document in a collection have a unique string ID. The chunk IDs in the `.npz` archive (which mirror the `chunk_id` fields in the JSON) are not guaranteed to be globally unique — and in practice, 4 duplicates exist, inherited from Phase 1.2 chunking edge cases.

The `deduplicate_ids` function passes through IDs that appear only once. For IDs that appear more than once, the second occurrence is suffixed `_dup2`, the third `_dup3`, and so on. The first occurrence retains the original ID. The 4 duplicate chunk IDs found in the current corpus are:

```
ch02_chest_pain_020
ch02_dyspnea_012
ch02_edema_006
ch02_headache_009
```

Each appears twice, producing 4 deduplicated suffixed IDs. The final ID list contains 5,631 unique entries.

This deduplication is a workaround for a data quality issue in Phase 1.2, not a permanent solution. The correct fix is to eliminate the duplicate IDs at the chunker level so that every chunk has a genuinely unique identifier from the point of creation. That fix is deferred to Phase 1.2 maintenance; the suffix approach ensures Phase 2.2 is unblocked.

### Stage 5: Build Collection and Upsert

```python
collection = get_or_create_collection(CHROMA_DIR, force=args.force)
upsert_batches(collection, unique_ids, embeddings, chunks)
```

A `chromadb.PersistentClient` is opened at `data/chroma/`. The `tmt_chunks` collection is created (or retrieved if it already exists) with `metadata={"hnsw:space": "cosine"}`. Anonymous telemetry is disabled via `Settings(anonymized_telemetry=False)`.

All 5,631 records are upserted in batches of 500. ChromaDB's internal HTTP-style request handling has a practical upper bound on batch size; 500 is the recommended maximum per upsert operation. At 5,631 total records, 12 batches are required (11 full batches of 500 and one final batch of 131).

Each upsert call passes four parallel lists:

| List | Content |
|---|---|
| `ids` | Unique chunk ID strings |
| `embeddings` | List of 768-float lists |
| `documents` | Raw chunk text strings (the `text` field) |
| `metadatas` | Metadata dicts (7 fields per chunk, see below) |

Using `collection.upsert()` rather than `collection.add()` means the operation is idempotent: running the script a second time without `--force` will update existing records rather than raising a duplicate-ID error. This is intentional — if the chunk text or metadata changes but the IDs remain stable, a re-run updates the store correctly without requiring a full rebuild.

### Stage 6: Count Verification

```python
verify_count(collection, expected_total)
```

After all batches complete, `collection.count()` is called and compared against the expected total of 5,631. A mismatch is reported as a WARNING with the delta — it does not abort, since a partial store may still be queryable, but it flags that something went wrong during the upsert. In the reference run, `collection.count()` returned 5,631, matching the expected total exactly.

### Stage 7: Sample Queries (Optional)

If `--skip-verify` is not set, the script loads the embedding model and runs 3 sample queries against the newly built store. This serves as an end-to-end sanity check: it confirms that the HNSW index is queryable, that cosine distances are in the expected range, and that the retrieved chunks correspond to the correct clinical domain. Results are documented in the Sample Query Results section below.

---

## Metadata Schema

Each chunk stored in the collection carries a metadata dictionary with 7 fields. ChromaDB metadata values must be one of four scalar types: `str`, `int`, `float`, or `bool`. No nested dicts, no lists, no None values.

| Field | Type | Purpose |
|---|---|---|
| `chapter` | `str` | Full chapter name (e.g., `"Heart Disease"`). Enables chapter-scoped queries via `where={"chapter": "Heart Disease"}`. |
| `section` | `str` | Section heading within the chapter (e.g., `"CHEST PAIN"`). Enables section-level filtering for high-precision retrieval. |
| `subsection` | `str` | Subsection heading (e.g., `"When to Admit"`). Empty string for chunks outside a named subsection. |
| `page_range` | `str` | Page range from the source PDF, stored as a string. Stored as string because the raw value may be a list `[42, 43]` or an int `42`, neither of which ChromaDB accepts as metadata. The `build_metadata_record` function converts all variants to string via `str(page_range)`. |
| `word_count` | `int` | Number of words in the chunk text. Can be used to filter out very short or very long chunks from retrieval, or to weight results by information density. |
| `chunk_type` | `str` | Classification of the chunk's structural role in the source text (e.g., `"body"`, `"header"`). Enables filtering by chunk type if needed. |
| `source` | `str` | Source textbook identifier. Currently `"tmt"` for all chunks, since only the TMT textbook has been chunked. Will disambiguate when additional textbooks are added. |

The chunk's full text is stored as the ChromaDB "document" — the `documents` argument to `collection.upsert()`. This means query results include the passage text directly in the `results["documents"]` list, without requiring a secondary lookup against the chunk JSON. The stored text is the raw `chunk["text"]` value — the unmodified text without the context prefix used during embedding. The context prefix is an input to the embedding model only; the stored and returned text is always the clean passage text that will be shown to the Triage Agent and ultimately to the user.

---

## Sample Query Results

Three sample queries were run against the built collection immediately after upsert to validate retrieval behaviour. The queries represent the three primary retrieval scenarios the Triage Agent will encounter: a multi-system presentation, a chronic disease management query, and an infectious disease query. The embedding model used for query encoding is the same model used during Phase 2.1 (`sentence-transformers/embeddinggemma-300m-medical`), ensuring consistency between passage and query embedding spaces.

ChromaDB with `hnsw:space=cosine` returns distances where 0 = identical direction (maximum similarity) and 2 = opposite direction (maximum dissimilarity). In practice, for semantically related but non-identical texts, distances typically fall in the 0.3–0.7 range. Distances below 0.5 indicate strong semantic similarity; distances above 1.0 indicate little semantic relationship.

### Query 1: "chest pain with shortness of breath"

| Rank | Section | Subsection | Distance |
|---|---|---|---|
| 1 | CHEST PAIN | When to Admit | 0.4025 |
| 2 | Heart Disease / Differential Diagnosis | — | 0.4099 |
| 3 | DYSPNEA | Symptoms | 0.4595 |

The top result (distance 0.4025) is from the chest pain section of the symptom chapter — a direct lexical and semantic match. The second result (0.4099) is from the Heart Disease chapter's differential diagnosis section, representing the correct cross-chapter link: chest pain is a cardinal presentation in the differential diagnosis of cardiac disease. The third result (0.4595) retrieves a dyspnea chunk — shortness of breath is the second symptom in the query, and the model correctly connects it to the dyspnea section. All three results are from the expected clinical domain (cardiopulmonary), and the cross-chapter retrieval (symptom chapter + disease chapter) demonstrates that the single-collection design enables the cross-chapter search the Triage Agent requires.

### Query 2: "diabetes management insulin"

| Rank | Section | Subsection | Distance |
|---|---|---|---|
| 1 | Diabetes Mellitus | Patient Education & Self-Management | 0.4772 |
| 2 | Preoperative Evaluation | Management of Endocrine Diseases | 0.5341 |
| 3 | Diabetes Mellitus | General | 0.5403 |

Results 1 and 3 are from the Diabetes Mellitus chapter — the correct primary source for diabetes management and insulin content. Result 2 (distance 0.5341) is a cross-chapter hit from the Preoperative Evaluation chapter's section on managing endocrine diseases in perioperative patients, which is a clinically valid secondary source: insulin management protocols differ in the preoperative setting and this content is genuinely relevant to the query. The distances are slightly higher than in Query 1 (0.47–0.54 vs 0.40–0.46), reflecting that "diabetes management insulin" is a more general query than "chest pain with shortness of breath" and the correct chunks are more diffuse in the embedding space.

### Query 3: "fever in immunocompromised patient"

| Rank | Section | Subsection | Distance |
|---|---|---|---|
| 1 | Infectious Diseases / Immunocompromised Patient | When to Admit | 0.3438 |
| 2 | Infectious Diseases / Fever of Unknown Origin | Clinical Findings | 0.3761 |
| 3 | Infectious Diseases / Fever of Unknown Origin | Duration of Fever | 0.3839 |

This query produces the lowest distances of the three tests (0.34–0.38), indicating the strongest semantic match. The top result is from a subsection explicitly titled "Immunocompromised Patient" within the Infectious Diseases chapter — the context prefix strategy from Phase 2.1 has encoded both the chapter context and the subsection context into the embedding, allowing this chunk to rank first despite competing with many other fever-related chunks in the corpus. Results 2 and 3 are both from the Fever of Unknown Origin section of the same chapter — clinically appropriate secondary results, since FUO workup is a central concern in immunocompromised patients. All three results are from the correct chapter.

The pattern across all three queries confirms that: (1) the context-prefixed embeddings from Phase 2.1 are correctly retrieving domain-relevant content; (2) cross-chapter retrieval works as expected (Query 1 spans symptom and disease chapters); (3) the cosine distance values are interpretable and within the expected semantic similarity range.

---

## Output Files

### `data/chroma/`

The ChromaDB persistent store directory. After a successful Phase 2.2 run, this directory contains several files managed internally by ChromaDB:

- SQLite database file(s) — store chunk text and metadata records
- HNSW index files — store the approximate nearest-neighbour vector index for the 768-dimensional embedding space

The total on-disk size is approximately 20–30MB, depending on the ChromaDB version's internal storage format. This is larger than the 15MB compressed `.npz` file from Phase 2.1 because ChromaDB stores the embedding vectors in uncompressed form alongside the SQLite metadata and HNSW index structures.

This directory is added to `.gitignore` and is not committed to the repository. It is a generated artefact. Any collaborator or examiner can reproduce it by running:

```bash
python embeddings/build_vector_store.py
```

provided that `data/chunks/tmt_chunks_structured.json` and `data/embeddings/tmt_chunk_embeddings.npz` are present (both of which are committed, or can be regenerated from earlier phases).

---

## Scripts Reference

### `embeddings/build_vector_store.py`

The single script that executes the full Phase 2.2 pipeline.

**Usage:**

```bash
# Standard run — builds the store (or updates it if already present):
python embeddings/build_vector_store.py

# Delete the existing collection and rebuild from scratch:
python embeddings/build_vector_store.py --force

# Build the store but skip the sample query verification:
python embeddings/build_vector_store.py --skip-verify

# Dry run — load data and print config, do not build the store:
python embeddings/build_vector_store.py --dry-run
```

**CLI flags:**

| Flag | Effect |
|---|---|
| `--force` | Deletes the existing `tmt_chunks` collection before upserting. Without this flag, the script calls `get_or_create_collection` and upserts into the existing collection, updating any records whose IDs match. Use `--force` when the embedding model or chunk file has changed and a clean rebuild is required. |
| `--skip-verify` | Skips the sample query stage after upsert. The embedding model is not loaded, saving approximately 30 seconds of model load time. Useful in CI or batch rebuild contexts where end-to-end query verification is handled separately. |
| `--dry-run` | Loads the chunk JSON and embedding `.npz`, verifies alignment, prints the full configuration (paths, model, batch size, collection name), and exits without opening a ChromaDB client or upserting anything. Useful for confirming that the input data files are present and correctly aligned before committing to a full rebuild. |

**Configuration values from `config.py`:**

| Parameter | Value |
|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/embeddinggemma-300m-medical` |
| `EMBEDDING_DIM` | 768 |
| `CHROMA_DIR` | `data/chroma/` |
| `CHUNKS_DIR` | `data/chunks/` |
| `EMBEDDINGS_DIR` | `data/embeddings/` |

**Environment requirements:**

- Python 3.11+ with `chromadb`, `numpy`, `sentence-transformers`, and `torch` installed
- `data/chunks/tmt_chunks_structured.json` present (Phase 1.2 output)
- `data/embeddings/tmt_chunk_embeddings.npz` present (Phase 2.1 output)
- Internet access on first run of the sample query stage (to download the embedding model if not cached); the store-building stage itself requires no network access

---

## Limitations

### HNSW Approximate Nearest-Neighbour Tradeoff

ChromaDB's HNSW index is an approximate nearest-neighbour (ANN) algorithm. HNSW builds a hierarchical graph structure over the embedding vectors and traverses it during query time to find candidates without exhaustively comparing the query against every stored vector. This approximation trades a small amount of recall (some true nearest neighbours may be missed) for a large reduction in query latency at scale.

At 5,631 vectors, this tradeoff is irrelevant. Brute-force exact search over 5,631 768-dimensional vectors takes under a millisecond. HNSW provides no meaningful latency benefit at this corpus size. The ANN recall deficit — however small — is a pure cost with no compensating benefit. The HNSW index is used because it is ChromaDB's only index type; there is no option to use exact search instead. In practice, HNSW recall at this corpus size is effectively 100% — the graph traversal finds the true nearest neighbours without approximation errors because the graph is small enough that the algorithm degrades to near-exact behaviour.

If the corpus grows to hundreds of thousands of vectors (multiple textbook editions, extended clinical references), the HNSW approximation begins to matter. The quality/speed tradeoff should be revisited if the corpus scale increases by an order of magnitude.

### No Reranking Layer

The vector store returns the top-k chunks by cosine distance. This first-stage retrieval is optimised for recall — returning a set of candidates that contains the relevant chunk — rather than precision. The ranking within the top-k candidates is determined entirely by the bi-encoder's cosine similarity, which is a coarse relevance signal compared to a cross-encoder that reads both the query and the full passage jointly.

Phase 3 will add a cross-encoder reranking layer using `mixedbread-ai/mxbai-colbert-large-v1` (already specified in `config.py`). The reranker will take the top-k candidates from the vector store and re-score them by reading query and passage together, substantially improving ranking precision within the candidate set. The vector store's role is to reduce the candidate space from 5,631 to a manageable top-k (typically 10–20); the reranker's role is to find the best 1–3 from that reduced set.

### Duplicate Chunk IDs Inherited from Phase 1.2

The 4 duplicate chunk IDs handled by the `deduplicate_ids` function are a data quality issue originating in the Phase 1.2 chunker. The chunker assigns IDs by combining a chapter slug, a section slug, and a sequential counter. In cases where the same section heading appears multiple times within a chapter (possibly due to PDF parsing inconsistencies or repeated section headers in the source), the counter resets, producing two chunks with identical IDs.

The `_dup2` suffix workaround ensures Phase 2.2 is unblocked, but it means two distinct chunks share a conceptually identical base ID, and only one of them has the "canonical" form of that ID. If a system component constructs a chunk ID from chapter/section metadata and uses it to look up a specific chunk, it may find the wrong one. The correct fix is in Phase 1.2: the chunker should detect and resolve ID collisions at the point of generation so that every chunk emerges with a globally unique identifier. That fix is deferred to a Phase 1.2 maintenance pass; the current workaround is acceptable for Phase 2.2 and 2.3.

### No Embedding Freshness Validation

The script loads chunk JSON and embedding `.npz` independently and verifies only that their counts match. It does not verify that the chunk at index i in the JSON corresponds to the chunk with the ID at index i in the `.npz`. If the chunk JSON were to be reordered after the embedding run (for instance, by a sort operation applied to the JSON file), the count would still match but every embedding would be assigned to the wrong chunk. The alignment verification catches count mismatches but not ordering mismatches.

A more robust check would compute a checksum of the chunk ID list from both sources and confirm they are identical in order, not just in count. This is a low-probability failure mode — nothing in the pipeline reorders the chunk JSON after Phase 1.2 produces it — but it is worth noting as a correctness assumption that is currently unenforced.
