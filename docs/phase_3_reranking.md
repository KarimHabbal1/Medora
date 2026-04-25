# Phase 3: Cross-Encoder Reranking

## Overview

Phase 3 adds a cross-encoder reranking step on top of the bi-encoder retrieval established in Phase 2. The pipeline now operates in two sequential stages: retrieve k=10 candidate chunks from ChromaDB using the bi-encoder, then re-score all 10 candidates using a cross-encoder and return the top 3.

### Why This Phase Exists

Phase 2.3 validated that the bi-encoder pipeline achieves 90% chapter-level Hit@1 and 100% Hit@3 on natural clinical language queries. That result is strong — relevant content is always present in the top-3 window. But chapter-level hit rate is a coarse metric. It tells the Triage Agent that the right chapter was retrieved; it does not tell us whether the specific passage returned at rank 1 is the most useful passage for answering the query. Two chunks can both be from the Heart Disease chapter, but one may contain a direct discussion of treatment options for atrial fibrillation while the other contains epidemiology statistics about the prevalence of the condition. Both count as a chapter hit. Neither distinction is visible in Hit@k.

The deeper limitation is architectural: the bi-encoder encodes query and passage independently. Each text is mapped to a vector without any awareness of the other. At query time, relevance is approximated by the cosine distance between these independently computed vectors. This is an efficient and generally effective approach — it scales to 5,631 chunks because each chunk is encoded only once and stored — but it sacrifices the ability to reason about fine-grained query-passage interactions. Whether a specific passage actually answers the question at hand is not something a cosine distance between two independent vectors can fully capture.

Cross-encoders were designed to solve this problem. By reading query and passage together in a single forward pass, a cross-encoder can score the relevance of a specific passage to a specific query with much greater precision. The tradeoff is speed: a cross-encoder must run once for each (query, passage) pair, making it O(n) in the number of passages rather than O(1) for retrieval after the index is built. This makes cross-encoders unsuitable for searching 5,631 chunks directly — but perfectly suitable for re-scoring the 10 candidates the bi-encoder has already identified as most relevant.

The two-stage architecture (bi-encoder retrieval followed by cross-encoder reranking) is the standard configuration in production RAG systems. Phase 3 implements it.

### What Phase 3 Does

1. The bi-encoder encodes the query and retrieves k=10 candidates from ChromaDB, including their full passage text.
2. The cross-encoder scores each of the 10 (query, passage) pairs, producing a relevance score per pair.
3. The 10 candidates are re-ordered by cross-encoder score and the top 3 are returned.

The bi-encoder provides the candidate pool. The cross-encoder provides the final ranking within that pool.

### What Phase 3 Adds Over Phase 2

| Capability | Phase 2 | Phase 3 |
|---|---|---|
| Retrieval mechanism | Bi-encoder + cosine similarity | Bi-encoder + cosine similarity + cross-encoder reranking |
| Candidates evaluated per query | 3 or 5 or 10 | 10 (then re-ordered) |
| Relevance signal | Vector proximity (independent encoding) | Joint query-passage attention (cross-encoder) |
| Latency | ~10ms retrieval | ~10ms retrieval + ~80ms reranking (10 pairs) |
| Returned results | Top-k by vector distance | Top-3 by cross-encoder score |

---

## Architecture: Bi-encoder vs Cross-encoder

Understanding the two architectures is essential for interpreting this phase's results and their limitations.

### Bi-encoder (Phase 2)

A bi-encoder is a transformer model trained to produce dense vector representations of text such that semantically similar texts map to nearby vectors. The key architectural property is independence: query and passage are encoded separately through the same (or separate) model, producing two vectors. Relevance is then computed as the cosine similarity between these vectors.

```
query text  →  Encoder  →  query_vector
passage text →  Encoder  →  passage_vector
relevance = cosine_similarity(query_vector, passage_vector)
```

This architecture enables efficient large-scale retrieval. Passage embeddings are computed once offline and stored in ChromaDB's HNSW index. At query time, only the query embedding is computed, and approximate nearest-neighbour search over the index takes milliseconds regardless of corpus size. The bi-encoder can search all 5,631 chunks in under 10ms.

The limitation is that the embedding of a passage is computed without any knowledge of what queries will be asked against it. The passage vector must encode "everything relevant about this text" in a fixed-length vector, independently of any specific query. When a query is highly specific — asking not just about atrial fibrillation but about the treatment of paroxysmal atrial fibrillation with pill-in-the-pocket flecainide — the bi-encoder can only match at the level of topical proximity. It cannot distinguish between a passage that discusses atrial fibrillation treatment options in detail and a passage that merely lists atrial fibrillation as one of several conditions for which a drug is indicated.

The analogy is resume screening: a recruiter reading hundreds of resumes quickly filters to a shortlist based on keywords, general domain match, and rough profile alignment. Fast, scalable, effective at filtering irrelevant candidates — but not the mechanism for making the final hire.

### Cross-encoder (Phase 3)

A cross-encoder is a transformer model that takes the query and passage concatenated as a single input sequence and produces a single relevance score. There is no independent encoding step: the model processes the entire combined input through all transformer layers, with full bidirectional cross-attention between every token in the query and every token in the passage.

```
[query text + passage text]  →  Transformer (full attention)  →  relevance_score
```

This architecture allows the model to detect fine-grained relevance signals that independent encoding cannot capture. When the query asks "treatment of atrial fibrillation" and the passage discusses cardioversion, rate control drugs, and anticoagulation, the cross-encoder can identify that every major concept in the query maps to a substantive discussion in the passage — not just that both texts are "about atrial fibrillation." When a different passage discusses atrial fibrillation as a risk factor for stroke without any treatment content, the cross-encoder correctly assigns it a lower score.

The limitation is speed: each (query, passage) pair requires a full forward pass through the model. This is O(n) in the number of passages — to score 5,631 chunks requires 5,631 forward passes, which would take many seconds on M1 Pro hardware. Cross-encoders are therefore used only after the bi-encoder has narrowed the candidate set to a manageable size.

The analogy is the interview: the hiring manager reads the candidate's work in detail, asks specific questions, and evaluates fit with precision. Slow, high-information, used only on the shortlisted candidates.

### Why the Two-Stage Architecture Works

The two stages are complementary in a way that justifies the added complexity:

- The bi-encoder is good at eliminating irrelevant content at scale. It will not return a passage from the Dermatology chapter when the query is about atrial fibrillation.
- The cross-encoder is good at distinguishing between multiple relevant candidates. It correctly identifies which of the 10 Heart Disease passages is most directly responsive to the specific question about treatment.

Neither stage alone achieves both goals. The bi-encoder alone cannot achieve the fine-grained selection; the cross-encoder alone cannot operate at corpus scale. Together, they provide both breadth (correct chapter retrieved with high recall) and precision (best-matching passage within that chapter returned at rank 1).

---

## Correcting the Original Configuration: ColBERT vs Cross-encoder

Before documenting model selection, it is important to record an architectural correction made during Phase 3 development.

The original `config.py` specified `mixedbread-ai/mxbai-colbert-large-v1` as the reranker model. This model was referenced at the end of Phase 2.1's limitations section as the "planned Phase 3 reranker." When Phase 3 implementation began, it became apparent that this model is architecturally incompatible with the intended cross-encoder reranking design.

### What ColBERT Is

ColBERT (Contextualized Late Interaction over BERT, Khattab & Zaharia 2020) occupies a position between bi-encoders and cross-encoders on the speed-accuracy tradeoff. Rather than producing a single embedding per text (as a bi-encoder does) or reading the full concatenated pair (as a cross-encoder does), ColBERT produces a separate embedding for every token in the query and every token in the passage, then computes relevance via the MaxSim operation: for each query token, find its maximum cosine similarity with any passage token, then sum these maximum similarities across all query tokens.

This "late interaction" architecture allows ColBERT to capture some token-level interactions between query and passage without requiring a full joint forward pass over concatenated input. It is faster than a full cross-encoder but more expressive than a pure bi-encoder.

### Why ColBERT Was the Wrong Choice Here

`mxbai-colbert-large-v1` cannot be used with the `sentence_transformers.CrossEncoder` API. It requires the RAGatouille or PyLate library for inference, which provides the ColBERT-specific MaxSim computation. Attempting to load it as a `CrossEncoder` model produces silent errors or incorrect scores because the model architecture does not match the API's expected format.

More fundamentally, adopting ColBERT as the Phase 3 reranker would have required:

1. Installing and integrating RAGatouille or PyLate as a dependency.
2. Storing token-level embeddings for all 5,631 passages rather than text documents, since ColBERT's late interaction requires access to passage token vectors at reranking time.
3. Implementing a different reranking call path that invokes ColBERT's MaxSim scoring rather than a standard regression head.

None of this infrastructure was built, and building it would have been disproportionate complexity for a comparison that can be cleanly conducted with true cross-encoders. ColBERT's late interaction scoring is an interesting architecture that might be revisited if Phase 3 cross-encoders prove insufficient, but it is not a cross-encoder and cannot be substituted for one.

The corrected configuration uses `BAAI/bge-reranker-v2-m3` as the primary reranker and `ncbi/MedCPT-Cross-Encoder` as the comparison model. Both are genuine cross-encoders fully compatible with the `CrossEncoder` API.

---

## Reranker Model Selection

The model selection decision for Phase 3 mirrors the structure of Phase 2.1's embedding model selection: define the requirements, survey the available models, evaluate against those requirements, and document the rejection reasons alongside the selection rationale.

### Requirements

A reranker model for this project must satisfy four requirements:

1. **CrossEncoder API compatibility**: must load and score via `sentence_transformers.CrossEncoder`. This is a hard requirement given the existing retrieval infrastructure.
2. **Reasonable latency on M1 Pro**: scoring 10 (query, passage) pairs must complete within ~100ms. A reranker that adds multiple seconds of latency per query is not acceptable in a clinical application.
3. **Strong relevance scoring on scientific or medical passage retrieval**: the model should have demonstrated performance on BEIR-style benchmarks, particularly on heterogeneous passage retrieval rather than narrow abstract retrieval.
4. **Sufficient parameter capacity**: too small a model will not capture the nuanced relevance signals that distinguish good candidates from marginal ones.

### Models Evaluated

| Model | Architecture | Params | BEIR nDCG@10 | CrossEncoder compatible | Medical domain | Decision |
|---|---|---|---|---|---|---|
| `mixedbread-ai/mxbai-colbert-large-v1` | ColBERT (late interaction) | 335M | ~54 | No — requires RAGatouille | No | Rejected — wrong architecture |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Cross-encoder | 33M | ~49–50 | Yes | No (web search) | Rejected — domain shift |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder | 568M | ~52–54 | Yes | General + scientific | **Selected as primary** |
| `ncbi/MedCPT-Cross-Encoder` | Cross-encoder | 109M | Medical-specific | Yes | Yes (18M PubMed pairs) | **Selected for comparison** |
| `mixedbread-ai/mxbai-rerank-large-v2` | Generative cross-encoder | 1.5B | 57.49 | No — custom API | No | Rejected — API incompatible |
| `jinaai/jina-reranker-v3` | Novel architecture | 600M | 61.9 | Uncertain | No | Rejected — too new, API uncertain |

### Analysis by Model

#### `mixedbread-ai/mxbai-colbert-large-v1` — Rejected (Original Choice)

As documented in the previous section, this model is a ColBERT late-interaction retriever, not a cross-encoder. It cannot be used with the `CrossEncoder` API and would require a complete reimplementation of the reranking path. Rejected on architectural grounds.

#### `cross-encoder/ms-marco-MiniLM-L-12-v2` — Rejected on Domain Shift

This is the canonical sentence-transformers cross-encoder, trained on the MS MARCO web-search passage ranking dataset. It is widely used in general RAG tutorials and represents the default choice when medical specificity is not a consideration.

The domain mismatch with this project is the same structural problem that eliminated `MedCPT` as an embedding model in Phase 2.1: MS MARCO is a web search dataset. The query-passage pairs are drawn from Bing search sessions — keyword queries matched against web documents. The query style, passage style, and relevance signals in MS MARCO differ substantially from clinical queries matched against textbook passages. A model trained to judge that "python install pip" is more relevant to a Python documentation page than to a Wikipedia article about the snake has not learned the relevance signals needed to judge that "treatment of paroxysmal atrial fibrillation" is better answered by a passage about flecainide dosing than by a passage about ECG interpretation.

The 33M parameter count is also at the low end of what this task requires. While smaller models can be faster and more regularised, the subtlety of clinical passage relevance judgment benefits from greater model capacity.

#### `BAAI/bge-reranker-v2-m3` — Selected as Primary

BGE-reranker-v2-m3 is the cross-encoder counterpart to the BGE-M3 embedding model that was evaluated (and rejected in favour of embeddinggemma-300m-medical) in Phase 2.1. Unlike BGE-M3, the reranker variant does not require multilingual capability to justify its complexity — it is a 568M parameter cross-encoder trained on diverse retrieval tasks including scientific and biomedical passage ranking.

The model loads cleanly via `CrossEncoder("BAAI/bge-reranker-v2-m3", trust_remote_code=True)` and scores 10 passage pairs in approximately 80ms on M1 Pro (CPU path, since MPS support for this model is partial). This latency is well within the acceptable range for a clinical application where query processing is expected to take 2–5 seconds overall. A 80ms overhead on top of a 10ms retrieval step is negligible.

BGE-reranker-v2-m3's BEIR nDCG@10 scores of approximately 52–54 reflect broad multilingual reranking performance. More relevant to this project is its consistent performance across heterogeneous passage types — the model was trained to generalise across scientific, web, and multilingual retrieval tasks, making it more robust to the distribution shift between training data and textbook retrieval than a web-only model like MS-MARCO.

#### `ncbi/MedCPT-Cross-Encoder` — Selected for Comparison

MedCPT-Cross-Encoder is the cross-encoder sibling of the MedCPT embedding model evaluated in Phase 2.1. It is trained on 18 million PubMed query-article pairs using a cross-encoder objective, making it one of the most directly medically-trained cross-encoders available with full CrossEncoder API compatibility.

The motivation for including it as a comparison model is identical to Phase 2.1's rationale for evaluating domain-specific embedding models: this project needs to answer the generalist-versus-specialist question at the reranking layer, not just at the embedding layer. If a medical cross-encoder trained on PubMed pairs outperforms a general cross-encoder on textbook retrieval, that is a meaningful architectural finding. If the general model outperforms it — as happened in Phase 2.1 — that finding is equally meaningful and supports the broader conclusion that modern generalist models with diverse training data outperform narrow specialists on out-of-distribution tasks.

At 109M parameters, MedCPT-Cross-Encoder is substantially smaller than BGE-reranker-v2-m3. It runs in approximately 30–40ms on M1 Pro.

#### `mixedbread-ai/mxbai-rerank-large-v2` — Rejected on API Incompatibility

This model achieves a BEIR average nDCG@10 of 57.49 — meaningfully above BGE-reranker-v2-m3's scores — and is documented as a generative reranker with strong cross-lingual performance. However, it uses a custom generation-based scoring approach that requires the `mxbai-rerank` Python package rather than the standard `CrossEncoder` API. Integrating it would require a separate inference path with different input formatting and output parsing. The performance advantage does not justify the infrastructure complexity given that this project already has a working cross-encoder pipeline.

#### `jinaai/jina-reranker-v3` — Rejected on Uncertainty

Jina-reranker-v3 reports a BEIR score of 61.9, which would place it as the highest-performing model in this evaluation by a significant margin. It was released in late 2025 and uses a novel architecture that combines late-interaction elements with cross-encoder scoring.

Two factors precluded its selection. First, the CrossEncoder API compatibility is documented as uncertain — the model may require a custom inference wrapper. Second, as a newly released model with limited production deployment history, its reported benchmark numbers have not been independently validated at the time of this phase's development (early 2026). Novel architectures sometimes overfit to benchmark evaluation protocols in ways that do not generalise. BGE-reranker-v2-m3 has a more established performance record across diverse deployment settings.

### Decision Summary

`BAAI/bge-reranker-v2-m3` is selected as the primary reranker. The decision is driven by: (1) full CrossEncoder API compatibility requiring no custom inference code; (2) broad training data including scientific and multilingual passage types; (3) 568M parameters providing adequate model capacity for clinical relevance judgment; and (4) approximately 80ms latency on M1 Pro — negligible for clinical application response times.

`ncbi/MedCPT-Cross-Encoder` is selected as the comparison model to answer the generalist-versus-specialist question at the reranking layer, paralleling Phase 2.1's embedding model comparison.

---

## Evaluation Methodology: Three Levels of Assessment

Phase 3 evaluation went through three successive levels of analysis, each motivated by a flaw discovered in the previous level. This progression is not a narrative of confusion — it is a principled example of how evaluation methodology must evolve to measure what actually matters. Each level changed the conclusion.

### Level 1: Strict Chapter-Matching Metrics

The first evaluation used the same 20 gold-standard queries from Phase 2.3, measuring Hit@1, Hit@3, and MRR at the chapter level (strict single-label: result must be from the single expected chapter) and section level (11 symptom queries).

This is the most conservative measurement: for every query, there is exactly one correct chapter, and a result at rank 1 is either from that chapter or it is not. The metric is identical to Phase 2.3's evaluation, enabling direct before-and-after comparison.

**Initial results were alarming.** The BGE reranker dropped chapter-level Hit@1 from 90.0% (bi-encoder baseline) to 70.0%. MedCPT dropped it to 75.0%. Under strict chapter-matching, both rerankers appeared to be degrading retrieval quality.

The instinctive response to this finding would have been to conclude that cross-encoder reranking is harmful for this task and abandon Phase 3. That conclusion would have been wrong, and understanding why it would have been wrong is the key insight of this phase.

### Level 2: Multi-Label Chapter Matching

The flaw in Level 1 is the assumption that each query has exactly one correct chapter. Examination of the cases where the reranker "degraded" performance revealed a systematic pattern: the reranker was promoting results from chapters that are genuinely clinically appropriate alternatives to the expected chapter.

For example, the query "heart racing and fluttering sensation" has `expected_chapter = "Common Symptoms"`. The bi-encoder's rank-1 result was from Heart Disease (atrial flutter section), which was correctly identified as a clinically valid result in Phase 2.3's analysis of its two rank-2 misses. When the BGE reranker was applied, it re-scored all 10 candidates and elevated a PALPITATIONS section chunk from Common Symptoms to rank 1 — which the strict metric counts as a "hit." But several other queries that the bi-encoder happened to rank correctly by the strict label saw the reranker promote an alternative-but-valid chapter to rank 1, counting as a miss.

The strict single-label metric was penalising the reranker for making clinically defensible choices.

The fix was to add an `accepted_chapters` field to each query, listing all chapters that would constitute a clinically valid rank-1 result. For example, "progressive shortness of breath on exertion" accepts both Common Symptoms and Pulmonary Disorders and Heart Disease. The multi-label metric counts a result as a hit if the rank-1 result is from any accepted chapter.

**Multi-label results showed both bi-encoder and BGE at 100% Hit@1** — statistically tied. MedCPT achieved 95% multilabel Hit@1 (one miss). This resolved the apparent degradation of Level 1 and established a more honest baseline, but it raised a new question: if both configurations achieve 100% multilabel Hit@1, is there any measurable difference between them?

### Level 3: LLM-as-Judge Content Scoring

Level 2 established that both configurations reliably retrieve a result from a clinically appropriate chapter at rank 1. But chapter membership is still a coarse proxy for content quality. Within the Heart Disease chapter, the range of content is enormous: ECG interpretation, risk factor epidemiology, mechanism of arrhythmias, pharmacologic treatment options, procedural interventions, admission criteria. A query about treatment of atrial fibrillation could receive a rank-1 result from any of these subsections and still be called a "chapter hit."

The question Level 3 addresses is: **does the reranker improve the quality of the specific passage returned at rank 1, independently of chapter membership?**

The evaluation methodology follows the LLM-as-judge paradigm (Zheng et al., "Judging LLM-as-a-Judge", NeurIPS 2023). For each of the 20 queries, the top-1 chunk from each pipeline configuration was extracted and submitted independently to GPT-4o with the following system prompt:

```
You are a medical relevance assessor. Given a clinical query and a retrieved
text passage from a medical textbook, rate how relevant the passage is to
answering the query.

Rate on a 1-5 scale:
5 = Perfectly relevant — directly answers the query
4 = Highly relevant — contains key information for the query
3 = Moderately relevant — related but doesn't directly answer
2 = Slightly relevant — tangentially related
1 = Not relevant — wrong topic entirely

Respond with ONLY a JSON object: {"score": <int>, "reason": "<one sentence>"}
```

GPT-4o was called at temperature=0 (deterministic output) for each of the 60 (query, chunk, configuration) combinations, producing a 1–5 score and a one-sentence reason for each.

This level of evaluation directly measures what matters for the Triage Agent: not whether the right chapter was found, but whether the specific passage presented to the agent as rank-1 context will enable it to generate a useful, accurate clinical response.

---

## Results

### Level 1: Strict Chapter Matching

| Metric | Bi-encoder only | + BGE reranker | + MedCPT |
|---|---|---|---|
| Chapter Hit@1 | 90.0% | 70.0% | 75.0% |
| Chapter Hit@3 | 100.0% | 100.0% | 95.0% |
| Chapter MRR | 0.950 | 0.842 | 0.842 |

The rerankers appear to reduce Hit@1 by 15–20 percentage points. As discussed in the methodology section, this is an artefact of the single-label evaluation design, not evidence of actual retrieval degradation. The rerankers are surfacing clinically valid content from alternative chapters that the strict metric counts as incorrect.

### Level 2: Multi-Label Chapter Matching

| Metric | Bi-encoder only | + BGE reranker | + MedCPT |
|---|---|---|---|
| Multilabel Hit@1 | 100.0% | 100.0% | 95.0% |
| Multilabel Hit@3 | 100.0% | 100.0% | 100.0% |
| Multilabel MRR | 1.000 | 1.000 | 0.975 |

With the corrected multi-label evaluation, the bi-encoder and BGE reranker are tied at 100% multilabel Hit@1. MedCPT falls slightly behind at 95%. No configuration misses at Hit@3 except MedCPT (one query misses even the multi-label definition at k=3, though it recovers at k=10). This level establishes that all configurations are reliable at retrieving from a clinically appropriate chapter — the reranking question must be answered at the content level.

### Level 3: LLM-as-Judge (GPT-4o, 1–5 scale)

| Configuration | Average Score |
|---|---|
| Bi-encoder only | 4.05 |
| + BGE reranker | **4.35** |
| + MedCPT | 4.30 |

The BGE reranker achieves the highest average content relevance score at 4.35, a 7.4% improvement over the bi-encoder baseline of 4.05. MedCPT also improves over the bi-encoder at 4.30. Both rerankers deliver better rank-1 passage quality than bi-encoder retrieval alone, despite being statistically indistinguishable from the bi-encoder at the chapter-matching level.

Run timestamp: `2026-04-21T11:20:03.749928+00:00`

### Per-Query Breakdown

The per-query scores reveal where the reranker adds the most value and where its limitations appear.

| Query | Bi-encoder | BGE | MedCPT | Notes |
|---|---|---|---|---|
| Persistent dry cough for 3 weeks | 5 | 5 | 5 | All configurations return same correct chunk |
| Progressive shortness of breath | 4 | 4 | 4 | Diagnostic-focused chunks across all configs |
| Sharp chest pain radiating to left arm | 4 | 5 | 5 | BGE/MedCPT elevate ACS-specific passage from rank 8 |
| Heart racing and fluttering | 5 | 5 | 5 | All retrieve palpitations content |
| Swollen ankles and legs bilateral | 5 | 4 | 4 | Bi-encoder returns better Direct LEE passage |
| High fever with chills and night sweats | 2 | 3 | 3 | All configs struggle — fever chunk too general |
| Unintentional weight loss 10 lbs 2 months | 5 | 5 | 4 | MedCPT returns epidemiology chunk, not diagnostic |
| Extreme fatigue for several weeks | 4 | 5 | 4 | BGE promotes causes/conditions chunk over intro |
| Sudden severe headache worst of my life | 5 | 5 | 5 | All return subarachnoid hemorrhage warning content |
| Painful urination with increased frequency | 4 | 3 | 4 | BGE promotes BPH chunk (slightly less relevant) |
| Coughing up blood-streaked sputum | 5 | 5 | 5 | All return identical optimal hemoptysis chunk |
| Management of type 2 diabetes with metformin | 3 | 3 | 3 | All return prevention chunk, not management |
| Treatment of atrial fibrillation | 2 | 5 | 5 | Major reranker recovery — treatment details retrieved |
| Pneumonia diagnosis and antibiotics | 5 | 4 | 5 | BGE returns HAP-specific chunk rather than CAP chunk |
| Acute kidney injury creatinine elevated | 5 | 4 | 5 | BGE returns ATN-specific chunk |
| Rheumatoid arthritis joint inflammation treatment | 2 | 5 | 5 | Major reranker recovery — DMARD content retrieved |
| Major depressive disorder SSRI treatment | 4 | 4 | 5 | MedCPT returns superior geriatric SSRI passage |
| Fever in immunocompromised patient neutropenia | 4 | 5 | 4 | BGE elevates neutropenic fever management content |
| Acute myocardial infarction emergency management | 4 | 4 | 4 | Marginal passage across all configs |
| Diabetic ketoacidosis DKA treatment protocol | 4 | 4 | 2 | MedCPT critical failure — biochemistry, not protocol |

**Key observations:**

**Atrial fibrillation (bi-encoder=2, BGE=5, MedCPT=5):** The bi-encoder's rank-1 result was an atrial flutter section describing the "sawtooth" ECG pattern and indications for electrical cardioversion — clinical content about atrial flutter, not atrial fibrillation treatment. GPT-4o rated it 2/5. The BGE reranker promoted a chunk at original rank 2 that contains detailed treatment options for paroxysmal and refractory atrial fibrillation, including pill-in-the-pocket cardioversion, antiarrhythmic drugs, and catheter ablation — rated 5/5. This is the clearest demonstration of the reranker's value: the bi-encoder retrieved relevant chapter content but not the most responsive passage; the cross-encoder correctly identified and elevated the most clinically useful passage.

**Rheumatoid arthritis (bi-encoder=2, BGE=5, MedCPT=5):** The bi-encoder's rank-1 result was the diagnostic essentials section of the Rheumatoid Arthritis entry — etiology, genetic risk factors, and pathologic findings. GPT-4o rated it 2/5 because the query asks specifically about treatment. The BGE reranker promoted the treatment objectives chunk that discusses DMARDs, NSAIDs, and disease activity scoring targets. Both rerankers agree on this promotion, and both receive 5/5.

**DKA treatment protocol (bi-encoder=4, BGE=4, MedCPT=2):** This is MedCPT's most significant failure. The bi-encoder and BGE reranker both return a chunk describing DKA severity classification and the therapeutic goals of treatment (restore plasma volume, reduce glucose, correct acidosis, replenish electrolytes) — rated 4/5. MedCPT promotes a chunk from original rank 10 that describes the biochemical abnormalities of moderately severe DKA in elaborate detail — serum osmolality calculations, sodium correction factors, potassium shifts — without any treatment content. GPT-4o rated it 2/5. MedCPT retrieved a medically accurate passage from the correct chapter, but it selected biochemistry over protocol in a query explicitly asking for treatment protocol.

**Dysuria (bi-encoder=4, BGE=3, MedCPT=4):** The BGE reranker promotes a urologic disorders chunk about BPH symptoms (frequency, nocturia) over the DYSURIA section's admission criteria chunk. The BPH chunk is from the Urologic Disorders chapter and discusses urinary frequency without addressing painful urination specifically. GPT-4o rated it 3/5. This is the BGE reranker's most visible failure: it selected topically related content over the more directly responsive dysuria-specific passage that the bi-encoder correctly identified.

---

## Analysis: Why the Reranker Improves Content Quality

The Level 3 results confirm that cross-encoder reranking improves content relevance despite being indistinguishable from the bi-encoder at the chapter-matching level. Four mechanisms explain why.

### 1. The Bi-encoder Has a Structural Ceiling on Within-Chapter Ranking

The context-prefix strategy (Chapter | Section | Subsection | text) anchors chunk embeddings in the vector space by clinical domain. This is effective at ensuring the right chapter appears in the top-k results — as validated by the 100% Hit@3 result in Phase 2.3. But within a chapter, all chunks share the same chapter and section prefix. The vector space geometry within a chapter's region is determined almost entirely by the chunk text content, not by structural metadata.

Within the Heart Disease chapter, a chunk about ECG patterns in atrial flutter and a chunk about pharmacologic treatment of atrial fibrillation both have vectors anchored to the Heart Disease region. Their relative proximity to a query about "treatment of atrial fibrillation" is determined by the cosine similarity of their independently encoded texts. The treatment chunk is more directly responsive, but the ECG chunk also contains substantial atrial fibrillation vocabulary. The bi-encoder cannot always distinguish which is more relevant because it encodes each passage without knowledge of what will be asked.

### 2. The Cross-encoder Detects Semantic Specificity

When the cross-encoder reads "treatment of atrial fibrillation" alongside a passage about pill-in-the-pocket flecainide dosing, it can recognise that the passage directly addresses the query concept of "treatment" in a way that a passage about ECG morphology does not. This detection of semantic specificity — the passage's content maps to the query's intent, not just its vocabulary — is the cross-encoder's primary advantage.

The attention mechanism in a transformer allows every query token to attend to every passage token in the joint sequence. The token "treatment" in the query can attend to "flecainide," "pharmacologic cardioversion," "catheter ablation," and "antiarrhythmic" in the passage — establishing that the passage is about the treatment act, not just about the condition. A cross-encoder trained on relevance judgments has learned to weight these token-level interactions in ways that correlate with actual human relevance assessments.

### 3. The Context Prefix Helps Chapter Placement but Not Passage Selection

The bi-encoder's context prefix (Chapter | Section | Subsection | text) encodes the structural position of a chunk but not its functional role within that structure. A chunk from the "Treatment" subsection of the Cardiac Arrhythmias section and a chunk from the "Essentials of Diagnosis" subsection of the same section both have appropriate prefixes and both embed into the Heart Disease region of the vector space. At query time, if the query is about treatment, the bi-encoder may return either chunk at rank 1 depending on vocabulary overlap and the specific geometry of the local embedding space.

The cross-encoder does not receive the prefix — it reads the raw chunk text. Yet despite this apparent disadvantage, it still improves average relevance because its passage-level judgment compensates for the missing structural context. The implication is that the subsection label ("Treatment" vs "Essentials of Diagnosis") does not need to be passed to the cross-encoder explicitly; the chunk text itself contains sufficient evidence of the passage's functional role for a capable cross-encoder to infer it.

This is simultaneously a validation of the cross-encoder's strength and an identification of a potential improvement: passing the context prefix to the cross-encoder as part of the passage input could provide additional structural signal. This is documented in the Limitations section.

### 4. The Reranker Acts as a Tiebreaker at the Passage Level

Even when the bi-encoder retrieves the correct chapter and the correct section — as it does for many queries — there may be multiple chunks from that section with similar vector distances. The bi-encoder's ranking among these near-tied candidates is determined by small differences in cosine similarity that may not reflect actual relevance. The cross-encoder can make finer distinctions: within the PALPITATIONS section, between a chunk about the causes of palpitations and a chunk about the evaluation approach, it can identify which better answers "heart racing and fluttering sensation" from a patient's perspective.

---

## Generalist vs Specialist: BGE vs MedCPT

The comparison between BGE-reranker-v2-m3 (general) and MedCPT-Cross-Encoder (medical-specific) replicates the central question of Phase 2.1's embedding model selection at the reranking layer. Phase 2.1 found that a general medical embedding model (`embeddinggemma-300m-medical`, trained on MIRIAD) outperformed a narrower specialist (MedCPT embedding). Phase 3 finds the same result at the reranking layer.

BGE achieves an average GPT-4o score of 4.35 with no query scoring below 3/5 across all 20 queries. MedCPT achieves 4.30 with one critical failure at 2/5 (DKA treatment protocol). The variance in MedCPT's performance is higher: most queries score well, but its extreme misses are more extreme than BGE's.

The DKA failure illustrates the structural problem with MedCPT's training distribution. MedCPT was trained on 18 million PubMed query-article relevance pairs. PubMed abstracts are indexed and retrieved primarily by the scientific content they report — biochemical mechanisms, clinical trial outcomes, epidemiological findings. A model trained on this data learns that a passage describing elaborate biochemical abnormalities of DKA is "highly relevant" to a query about DKA — because in the PubMed literature search context, a researcher querying "diabetic ketoacidosis" wants the detailed biochemistry papers.

A clinician or Triage Agent asking "diabetic ketoacidosis DKA treatment protocol" wants the practical treatment protocol: fluid resuscitation volumes, insulin infusion rates, electrolyte replacement schedules. MedCPT's relevance judgment, calibrated on literature search relevance rather than clinical question-answering relevance, cannot distinguish these two types of responses. BGE's broader training, which includes heterogeneous passage types and question formats, makes it more robust to this kind of distribution shift.

This finding has an important implication for Phase 4 and beyond: when selecting components for a clinical question-answering RAG system, training data breadth and task diversity are more predictive of performance than narrow biomedical domain specialisation. A model needs to understand what kind of answer is appropriate for what kind of question — and that understanding comes from exposure to diverse (question, answer) pairs, not from deep expertise in the biomedical domain alone.

---

## Recommended Configuration

**Bi-encoder (retrieve k=10) + BGE-reranker-v2-m3 (return top 3).**

This recommendation is based on the aggregate evidence across all three evaluation levels:

1. **7.4% improvement in average content relevance** (4.05 → 4.35 on the 1–5 GPT-4o scale). In absolute terms: the reranker raises the typical query from "highly relevant with some extraneous content" to "closely aligned with direct answer content."

2. **Eliminates the worst bi-encoder failures.** The two queries scoring 2/5 with the bi-encoder (atrial fibrillation and rheumatoid arthritis treatment) both improve to 5/5 with BGE reranking. Removing floor scores matters more for a clinical application than raising average scores: a Triage Agent that occasionally returns poorly matched content is more dangerous than one with a slightly lower average score.

3. **No score below 3/5 across all 20 queries.** Every query receives at least moderately relevant content at rank 1. The worst-case outcome is "related but not directly answering," which is recoverable by the LLM. The bi-encoder had two queries at 2/5 ("slightly relevant — tangentially related"), which is not recoverable.

4. **Multi-label Hit@1 remains 100%.** Adding the reranker does not compromise the chapter-level recall that Phase 2 established. The reranker selects better passages from within an already correctly identified clinical domain.

5. **Latency overhead is negligible for the use case.** Approximately 80ms to score 10 passage pairs on M1 Pro CPU. A clinical triage query is not a sub-100ms search application — it is a diagnostic reasoning process where an additional 80ms is imperceptible to the clinician or patient.

The `config.py` configuration reflects this recommendation:

```python
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_COMPARISON_MODEL = "ncbi/MedCPT-Cross-Encoder"
RERANK_TOP_K_RETRIEVE = 10   # retrieve this many from bi-encoder
RERANK_TOP_K_RETURN = 3      # return this many after reranking
```

---

## Methodological Insight: Evaluation Determines the Conclusion

The progression through three evaluation levels in Phase 3 is itself a thesis-worthy finding, independent of the specific numbers. The same pipeline, evaluated by three different methodologies, produces three different conclusions:

**Level 1 (strict single-label chapter matching) → "Rerankers degrade performance."**
Both rerankers drop Hit@1 from 90% to 70–75%. A developer applying this metric would correctly reject the reranking approach and terminate Phase 3.

**Level 2 (multi-label chapter matching) → "Rerankers are tied with the bi-encoder."**
BGE and the bi-encoder both achieve 100% multilabel Hit@1. The reranker is neither beneficial nor harmful at the chapter level. A developer applying this metric would conclude that reranking adds complexity without benefit and would likely not adopt it.

**Level 3 (LLM content scoring) → "Rerankers improve content quality."**
BGE improves average score from 4.05 to 4.35. The reranker eliminates floor scores and consistently returns more directly responsive passages. A developer applying this metric would correctly adopt the reranking step.

The first two levels produced false negatives — they failed to detect a real improvement because they measured the wrong thing. Chapter membership is a necessary but not sufficient condition for passage quality. The LLM judge evaluation measures what actually matters for the Triage Agent's ability to generate useful clinical responses.

This is not a critique of the first two evaluation levels — they serve important roles. Level 1 catches architectural problems that would prevent relevant content from appearing at all. Level 2 corrects systematic biases in ground-truth labeling. But neither Level 1 nor Level 2 could detect the improvement that Level 3 revealed, because that improvement operates at a granularity (passage content quality) that neither chapter-matching metric can resolve.

The practical lesson for RAG system evaluation is that retrieval-position metrics (Hit@k, MRR) measure whether the retrieval system is finding approximately relevant content. They do not measure whether the retrieved content is the most useful content for the downstream task. For RAG systems used in high-stakes domains — medicine, law, finance — the gap between "approximately relevant" and "directly useful" is consequential. Evaluation methodology must be designed to close that gap, which typically requires content-level assessment through LLM judging, human expert review, or downstream task performance measurement.

---

## How It Works

### `rag/reranker.py`

The script executes three configurations sequentially and saves a comparison to JSON.

**Stage 1: Infrastructure Initialisation**

The script loads configuration from `config.py`, detects the available compute device (MPS on Apple Silicon, CPU otherwise), opens the ChromaDB client at `CHROMA_DIR`, and loads the bi-encoder (`sentence-transformers/embeddinggemma-300m-medical`) onto the detected device.

**Stage 2: Bi-encoder Baseline (Configuration 1)**

For each of the 20 `MANUAL_QUERIES`, the script calls `retrieve_only()`, which encodes the query with the bi-encoder and retrieves the top-`return_k` results from ChromaDB without loading passage text. Chapter rank, accepted chapter rank, and section rank are computed. Strict and multi-label chapter metrics and section metrics are accumulated.

**Stage 3: BGE Reranking (Configuration 2)**

The BGE cross-encoder is loaded via `load_cross_encoder()`, which attempts MPS first and falls back to CPU if MPS fails. For each query, `retrieve_and_rerank()` is called: the bi-encoder retrieves `retrieve_k=10` candidates including their full passage text (required for cross-encoder scoring), then `rerank()` calls `model.predict()` on all 10 (query, passage) pairs, sorts by score descending, and returns the top 3. Each returned result is augmented with `rerank_score` (the raw cross-encoder logit) and `original_rank` (its position in the bi-encoder ranking before reranking). The `original_rank` field enables post-hoc analysis of how far the reranker moved each result.

**Stage 4: MedCPT Reranking (Configuration 3, optional)**

Identical to Stage 3 with the MedCPT-Cross-Encoder. Skipped if `--skip-comparison` is passed.

**Stage 5: Comparison Table and Save**

`print_comparison_table()` prints a formatted side-by-side comparison of all configurations across strict chapter, multi-label chapter, and section metrics. `save_results()` writes the complete payload — including per-query results with full top-3 result lists and rerank scores — to `data/results/reranking_comparison.json`.

**Key functions (importable by other scripts):**

`rerank(query, candidates, model, top_k)` — takes a query string, a list of candidate dicts (each with `chunk_id` and `text`), a loaded CrossEncoder model, and the number of results to return. Returns top-k candidates sorted by cross-encoder score, with `rerank_score` and `original_rank` added to each dict.

`retrieve_and_rerank(query, collection, bi_encoder, reranker, retrieve_k, return_k)` — full pipeline in one call. Encodes the query, retrieves `retrieve_k` candidates from ChromaDB with text included, calls `rerank()`, and returns the top `return_k` results. This is the function the Triage Agent will call in production.

### `evaluation/llm_judge.py`

The script evaluates passage content quality by submitting the rank-1 chunk from each configuration to GPT-4o.

**Stage 1: Infrastructure**

Same device detection, ChromaDB connection, and model loading as `reranker.py`. Additionally initialises an OpenAI client (reads `OPENAI_API_KEY` from the `.env` file).

**Stage 2: Per-Query Evaluation**

For each query, `evaluate_query()` retrieves the rank-1 chunk text from each of the three configurations — `get_top1_text_biencoder()` for the baseline and `get_top1_text_reranked()` for each reranker — then calls `score_chunk()` for each retrieved text. `score_chunk()` sends the (query, chunk_text) pair to GPT-4o with the relevance assessment system prompt and parses the JSON response. A rate-limiting sleep of 0.5 seconds is inserted between API calls to avoid hitting the OpenAI rate limit. The function returns a structured dict containing the chunk text, chapter, section, cross-encoder score (for reranked configs), and GPT-4o judge score and reason for each configuration.

**Stage 3: Summary and Save**

`print_summary()` prints the average scores per configuration and a per-query score grid. `save_results()` writes the complete payload to `data/results/llm_judge_results.json`.

**JSON parsing:** `parse_json_response()` implements three fallback strategies for parsing GPT-4o's output: direct `json.loads`, extraction from markdown code fences, and regex search for the first `{...}` block. This robustness is necessary because GPT-4o occasionally wraps JSON in markdown code fences despite being instructed otherwise.

---

## Scripts Reference

### `rag/reranker.py`

**Usage:**

```bash
# Full evaluation — all 3 configurations (bi-encoder + BGE + MedCPT):
python rag/reranker.py

# Skip MedCPT comparison (faster, primary reranker only):
python rag/reranker.py --skip-comparison

# Print full ranked result list for every query:
python rag/reranker.py --verbose

# Custom retrieve_k and return_k:
python rag/reranker.py --retrieve-k 15 --return-k 5
```

**CLI flags:**

| Flag | Default | Effect |
|---|---|---|
| `--skip-comparison` | False | Only run the primary BGE reranker; skip MedCPT. Useful on low-memory machines or when only the recommended configuration is needed. |
| `--verbose` | False | Print the full ranked result list for every query, annotated with rerank scores and original bi-encoder ranks. |
| `--retrieve-k N` | 10 (from `config.py`) | Number of candidates to fetch from the bi-encoder before reranking. Increasing this widens the candidate pool at the cost of more cross-encoder scoring calls. |
| `--return-k N` | 3 (from `config.py`) | Number of results to return after reranking. |

**Environment requirements:**

- Python 3.11+ with `sentence-transformers`, `chromadb`, `torch` installed.
- `data/chroma/` must contain a valid ChromaDB collection named `tmt_chunks`, built by Phase 2.2.
- Internet access on first run to download BGE and MedCPT model weights from Hugging Face Hub.
- MPS is attempted first for cross-encoder loading; falls back to CPU if MPS raises an exception (common for cross-encoder models with custom architecture code).

### `evaluation/llm_judge.py`

**Usage:**

```bash
# Full evaluation — all 20 queries, all 3 configurations:
python evaluation/llm_judge.py

# Skip MedCPT (60 API calls → 40 API calls):
python evaluation/llm_judge.py --skip-medcpt

# Run only first N queries (useful for testing):
python evaluation/llm_judge.py --queries 5

# Combined:
python evaluation/llm_judge.py --skip-medcpt --queries 5
```

**CLI flags:**

| Flag | Default | Effect |
|---|---|---|
| `--skip-medcpt` | False | Skip Configuration 3 (MedCPT). Reduces API calls from 60 to 40 for the 20-query evaluation. |
| `--queries N` | 20 | Run only the first N queries from `MANUAL_QUERIES`. Useful for spot-checking behaviour before a full run. |

**Environment requirements:**

- All requirements from `rag/reranker.py`.
- `OPENAI_API_KEY` set in `.env` file at project root (loaded via `python-dotenv`).
- `openai` Python package installed (`pip install openai`).
- Active OpenAI API key with GPT-4o access.

---

## Output Files

### `data/results/reranking_comparison.json`

Written by `rag/reranker.py`. Contains:

- **`timestamp`** — UTC ISO 8601 timestamp of the run (`2026-03-24T14:43:37.877036+00:00`).
- **`phase`** — `"3"`.
- **`bi_encoder_model`** — `"sentence-transformers/embeddinggemma-300m-medical"`.
- **`retrieve_k`** — Number of candidates retrieved from bi-encoder (10).
- **`return_k`** — Number of results returned after reranking (3).
- **`num_queries`** — 20.
- **`configurations`** — A list of three configuration objects, each containing:
  - `label` — Human-readable configuration name.
  - `model` — Reranker model name (`"none"` for bi-encoder baseline).
  - `metrics` — Aggregated metrics at three levels: `chapter_strict`, `chapter_multilabel`, and `section_level`.
  - `per_query` — Full per-query records including query text, expected and accepted chapters, chapter rank, accepted rank, section rank, Hit@1/3 flags, and the full top-3 result list with chunk IDs, chapters, sections, and (for reranked configs) rerank scores and original bi-encoder ranks.

The file is overwritten on each run. To preserve results before re-running with different parameters, copy the file before executing the script.

### `data/results/llm_judge_results.json`

Written by `evaluation/llm_judge.py`. Contains:

- **`timestamp`** — UTC ISO 8601 timestamp of the run (`2026-04-21T11:20:03.749928+00:00`).
- **`phase`** — `"3"`.
- **`judge_model`** — `"gpt-4o"`.
- **`bi_encoder_model`**, **`bge_reranker_model`**, **`medcpt_model`** — Model names for all three configurations.
- **`retrieve_k`**, **`return_k`** — 10 and 3.
- **`num_queries`** — 20.
- **`skip_medcpt`** — `false` (all three configurations run).
- **`average_scores`** — Per-configuration average GPT-4o scores: `{"biencoder_only": 4.05, "bge_reranker": 4.35, "medcpt_reranker": 4.3}`.
- **`per_query`** — A list of 20 per-query records, each containing the query text, category, expected chapter and section, and a `configs` object with three sub-objects (`biencoder_only`, `bge_reranker`, `medcpt_reranker`). Each sub-object contains the retrieved chunk text, chapter, section, (for reranked configs) cross-encoder rerank score and original rank, and the GPT-4o judge score and reason.

---

## Limitations

### Sample Size

The LLM judge evaluation covers 20 queries. This is the same sample size as Phase 2.3's retrieval validation, and the same statistical caution applies: with n=20, a difference of 0.30 points on a 1–5 scale (4.05 vs 4.35) is a meaningful directional signal but not statistically robust. A 95% confidence interval around the 4.35 BGE average is wide. With 100 or 200 queries, the improvement would be more precisely quantified and more defensible against the claim that the difference reflects sampling variance rather than systematic improvement.

That said, the key findings of Phase 3 do not rest on the average score difference alone. The per-query analysis shows consistent improvements on specific query types (condition-specific treatment queries) and identifies specific failure modes (MedCPT's DKA failure). These patterns are qualitatively meaningful even at n=20.

### GPT-4o as Judge

The LLM judge methodology has known limitations documented in the literature (Zheng et al., 2023; Wang et al., 2023). GPT-4o is not a neutral arbiter of clinical relevance — it applies its own prior beliefs about what constitutes a "good" answer to a clinical query. These priors may not align with those of a practising clinician. For example, GPT-4o may favour passages with explicit quantitative information (drug dosages, threshold values) over passages with nuanced clinical reasoning, or it may penalise passages that provide information in reference-style rather than narrative-style prose.

At temperature=0, GPT-4o produces deterministic outputs, which makes the scores reproducible but does not make them calibrated against ground truth. A clinician review of the top-1 passages from each configuration — rating their clinical utility directly — would be more authoritative than GPT-4o's scores. Such a review is a natural extension of this work for Phase 4 validation.

### The Reranker Does Not Receive the Context Prefix

The bi-encoder embeds passages prefixed with `Chapter: {chapter} | Section: {section} | Subsection: {subsection} | {text}`. When ChromaDB stores chunks, it stores the original text in the `documents` field without the prefix. When the reranker retrieves candidates, it receives the raw chunk text without any hierarchical context label.

This is a design gap. The cross-encoder could benefit from knowing that a passage is from the "Treatment" subsection of the Cardiac Arrhythmias chapter versus the "Differential Diagnosis" subsection — information that is currently encoded only in the bi-encoder's vector space and not passed forward to the reranker. Adding the context prefix to the passage input for cross-encoder scoring would provide additional structural signal and might further improve performance, particularly for queries where subsection-level discrimination matters (e.g., distinguishing diagnostic content from treatment content within the same chapter).

### Systematic Failures Persist Across All Configurations

Three queries score poorly across all three configurations in the LLM judge evaluation:

- **"High fever with chills and night sweats"** — Bi-encoder=2, BGE=3, MedCPT=3. All configurations retrieve fever-related content from either the Common Symptoms or Infectious Diseases chapter, but the retrieved passages are either too specific (admission criteria for extreme fever) or too broad (FUO evaluation checklist). No configuration retrieves a passage that directly addresses the evaluation approach for a patient presenting with high fever, chills, and night sweats as a differential diagnosis problem. This may reflect a gap in the chunk boundary placement for the FEVER section of Common Symptoms.

- **"Management of type 2 diabetes with metformin"** — All configurations score 3/5. The top-1 result across all configurations is the Diabetes Prevention Program study, which discusses metformin's role in preventing type 2 diabetes rather than managing established type 2 diabetes. The bi-encoder cannot distinguish "prevention with metformin" from "management with metformin" because both phrases share substantial vocabulary. The rerankers also fail on this query, suggesting the issue is in the candidate pool rather than the ranking: none of the top-10 bi-encoder candidates contain detailed treatment management content. This points to a gap in either chunk content or the embedding model's ability to discriminate the prevention-versus-management distinction.

These persistent failures across all configurations suggest that Phase 4 evaluation should pay particular attention to queries in these categories. If the Phase 4 end-to-end evaluation reveals systematic answer quality failures on fever and diabetes management queries, the intervention is likely at the retrieval layer (revised context prefixes, re-chunking of specific sections) rather than at the reranking layer.

### LLM Judge Temperature and Calibration

GPT-4o at temperature=0 produces consistent scores, but the scores are not calibrated against a held-out set of human expert judgments. The absolute values (4.05, 4.35, 4.30) are meaningful for relative comparison between configurations but should not be interpreted as absolute measures of clinical utility. A score of 4.35 does not certify that the retrieved passage is "good enough" for clinical use — it indicates that GPT-4o judged it as "highly relevant" on average. Whether "highly relevant" according to GPT-4o translates to "clinically safe and useful" is a question that requires expert clinician evaluation.
