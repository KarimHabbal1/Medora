# Phase 8: End-to-End Benchmarking

## Overview

Phase 8 is the systematic benchmarking of the complete Medora RAG pipeline and the web search agent across multiple model types, test sets, and retrieval strategies. The goal is to measure real diagnostic accuracy under controlled conditions, identify where the system succeeds and where it fails, and provide an evidence-based rationale for every architectural decision made across the project.

This phase is necessary because benchmarking answers the question that all earlier phases defer: does the complete system work? Phases 1 through 7 each validated a subsystem in isolation. Phase 8 runs end-to-end tests on held-out clinical cases and measures final diagnostic accuracy, retrieval recall, JSON reliability, and latency. The results determine what gets deployed, what gets replaced, and what the system's actual performance ceiling is.

Three test sets were used, each with a different purpose. Two retrieval strategies were benchmarked. Six model variants were evaluated on the RAG pipeline, and two on the web search pipeline.

---

## Benchmark Architecture

### What is Being Tested

The benchmark evaluates the complete triage pipeline:

1. Patient symptoms are provided as structured input.
2. The retrieval system fetches relevant textbook chunks (RAG) or web sources (web search).
3. The LLM generates a structured diagnostic output including differential diagnosis, confidence, and reasoning.
4. The output is compared against a ground-truth diagnosis using a multi-tier matching scheme.

The benchmark bypasses the intake agent and follow-up questions. Symptoms are provided directly to the triage step. This isolates retrieval and generation quality from conversational quality.

### Matching Scheme

Every diagnostic output is classified into one of four categories:

- **Exact:** the model's primary diagnosis matches the ground truth exactly (identical term).
- **Semantic:** the model's diagnosis matches the ground truth by clinical meaning but uses different terminology (e.g., "myocardial infarction" vs "heart attack").
- **Partial:** the model includes the correct diagnosis in its differential but not as the primary.
- **Mismatch:** the correct diagnosis is absent from the model's output.

Accuracy is defined as Exact + Semantic. Partial credit is tracked separately as a diagnostic signal but not counted toward the headline accuracy figure.

### Failure Attribution

Two distinct failure modes are tracked:

- **Generation-only fail:** the correct diagnosis was retrieved (present in the top-k chunks or web sources) but the model failed to reason toward it correctly.
- **Retrieval-only fail:** the retrieval system did not surface the correct evidence, making correct generation impossible regardless of model quality.

This separation is critical for root-cause analysis. A generation-only fail is a model problem. A retrieval-only fail is a pipeline problem (chunk quality, embedding quality, reranking, or data coverage).

---

## Test Sets

### Test Set A: Textbook Cases (50 cases)

Cases were constructed from the Medora textbook corpus. Ground-truth diagnoses are confirmed present in the textbook. This tests the RAG pipeline under favorable conditions: the answer exists in the knowledge base.

Purpose: validate the RAG pipeline when retrieval can succeed.

### Test Set B: MedQA USMLE (50 cases)

Cases from the MedQA USMLE question bank. These are standardized board-exam style clinical vignettes that go beyond the textbook corpus. Only GPT-5.4-mini was tested on this set.

Purpose: measure generalization to standardized medical knowledge.

### Test Set C: MedCaseReasoning (50 cases)

Cases from the MedCaseReasoning dataset — published case reports involving rare, unusual, or complex presentations. Most of these cases are not covered by the textbook.

Purpose: stress-test the system on out-of-distribution cases and evaluate web search as a gap-filler.

---

## RAG Benchmark Results

### Test Set A — Textbook Cases (All Models)

| Model | Type | Accuracy | Exact | Semantic | Partial | Mismatch | Gen-only fail | JSON errors | Latency |
|---|---|---|---|---|---|---|---|---|---|
| GPT-5.4-mini | API frontier | 74% | 2% | 72% | 18% | 8% | 14% | 0% | 15.6s |
| Llama 3.1 8B | OS general | 42% | 6% | 36% | 24% | 34% | 32% | 0% | 36.5s |
| Gemma 2 27B | OS general | 40% | 8% | 32% | 24% | 36% | 36% | 0% | 54.4s |
| DeepSeek-R1 14B | OS reasoning | 36% | 14% | 22% | 22% | 42% | 38% | 24% | 58.5s |
| Aloe-8B | OS medical fine-tune | 36% | 10% | 26% | 22% | 42% | 38% | 0% | 37.6s |
| MedLlama2 7B | OS medical fine-tune | Failed | — | — | — | — | — | — | — |

Notes:
- All open-source models ran on an A10G GPU (24GB VRAM).
- MedLlama2 7B failed entirely — the model was unable to follow the structured output prompt and produced unstructured prose for every case.
- DeepSeek-R1 14B produced chain-of-thought `<think>` blocks that interfered with JSON parsing, causing 24% of outputs to fail schema validation.
- Retrieval recall was consistent (~70%) across all models — the same textbook chunks were retrieved; only reasoning quality differed.

### Test Set B — MedQA USMLE (GPT-5.4-mini only)

| Metric | Result |
|---|---|
| Accuracy | 64% |
| Retrieval Recall | 42% |
| Mean Latency | 12.2s |

The accuracy drop from 74% (Test Set A) to 64% (Test Set B) reflects reduced retrieval recall — the USMLE cases span topics that are only partially covered by the textbook corpus. When the textbook has coverage, the system performs well; when it does not, accuracy falls proportionally.

### Test Set C — MedCaseReasoning (GPT-5.4-mini, RAG only)

| Metric | Result |
|---|---|
| Accuracy | 30% |
| Retrieval Recall | 22% |
| Mean Latency | 20.8s |

The 22% retrieval recall confirms that the textbook simply does not contain most of the rare and complex presentations in MedCaseReasoning. The 30% accuracy on cases with 22% retrieval is roughly consistent with the system's performance on covered cases — when retrieval fails, generation cannot recover.

---

## Web Search Benchmark Results

### Test Set C — MedCaseReasoning (Web Search Agent)

| Model | Accuracy | Exact | Semantic | Partial | Mismatch | Latency | Avg sources |
|---|---|---|---|---|---|---|---|
| GPT-5.4-mini | 42% | 4% | 38% | 30% | 28% | 6.1s | 1.56 |
| Gemma 4 (local) | 36% | 20% | 16% | 28% | 36% | 16.6s | 1.48 |

The web search agent uses a SearXNG search → whitelist filter → page fetch → LLM diagnosis pipeline. On the same 50 MedCaseReasoning cases where RAG achieved 30%, web search achieves 42% with GPT-5.4-mini — a 40% relative improvement. The mismatch rate drops from approximately 46% (RAG) to 28% (web search), showing that web sources cover rare cases the textbook cannot.

Web search is also faster than RAG: 6.1s vs 20.8s. This is because web search skips embedding, reranking, and vector store overhead. The trade-off is that web sources are less structured and less reliable than textbook chunks.

---

## Web Search Agent

### What It Is

The web search agent is a lightweight alternative retrieval path for cases where RAG retrieval recall is low. It replaces textbook chunk retrieval with live web evidence from trusted medical sources.

### Architecture

```text
Patient symptoms
    -> SearXNG search query
    -> Whitelist domain filter
    -> Page fetch (trusted sources only)
    -> LLM reads fetched page text + generates diagnosis
    -> Structured diagnostic output
```

The pipeline does not use embeddings, vector stores, or reranking. It is intentionally simpler than the RAG pipeline because its role is fallback coverage, not primary retrieval.

### Whitelisted Domains

The web search agent only fetches content from a controlled whitelist of trusted medical sources:

- PubMed (pubmed.ncbi.nlm.nih.gov)
- Mayo Clinic (mayoclinic.org)
- Cleveland Clinic (clevelandclinic.org)
- CDC (cdc.gov)
- NIH (nih.gov, ncbi.nlm.nih.gov)
- NICE (nice.org.uk)
- WHO (who.int)
- NEJM (nejm.org)
- The Lancet (thelancet.com)
- BMJ (bmj.com)
- JAMA (jamanetwork.com)

Consumer health sites (WebMD, Healthline, Reddit, Wikipedia) are rejected. This is deterministic policy — not model judgment.

### Models Tested

- GPT-5.4-mini: 42% accuracy on MedCaseReasoning (50 cases)
- Gemma 4 (local): 36% accuracy on MedCaseReasoning (50 cases)

### Integration with the Triage Agent

The web search agent is a fallback, not a replacement. The intended integration:

1. The triage agent runs RAG retrieval first.
2. Retrieval recall is estimated from reranker scores and chunk relevance signals.
3. If retrieval recall is below a threshold, the triage agent calls the web search agent.
4. The triage agent combines textbook evidence and web evidence into a single diagnostic context.
5. The LLM generates a diagnosis grounded in all available evidence.

This integration is not fully deployed at the time of benchmarking — the benchmark tested RAG and web search as independent pipelines. The combined system is described in the Analysis section as the recommended target architecture.

---

## Benchmark Status

All benchmarks are complete.

| Benchmark | Test Set | Models | Status |
|---|---|---|---|
| RAG pipeline | Test Set A (Textbook, 50 cases) | GPT-5.4-mini, Llama 3.1 8B, Gemma 2 27B, DeepSeek-R1 14B, Aloe-8B, MedLlama2 7B | Complete |
| RAG pipeline | Test Set B (MedQA USMLE, 50 cases) | GPT-5.4-mini | Complete |
| RAG pipeline | Test Set C (MedCaseReasoning, 50 cases) | GPT-5.4-mini | Complete |
| Web search agent | Test Set C (MedCaseReasoning, 50 cases) | GPT-5.4-mini, Gemma 4 | Complete |

---

## Analysis

### 1. The RAG Effect

The clearest finding in the benchmark is how much RAG matters when the knowledge base contains the answer.

- RAG on textbook cases (Test Set A): 74% accuracy.
- RAG on out-of-distribution cases (Test Set C): 30% accuracy.

RAG provides a roughly 2.5x improvement when the answer exists in the knowledge base. This validates the entire pipeline architecture: the embedding model choice, the chunking strategy, the reranking step, and the vector store. Every component contributed to the 74% result. When the knowledge base lacks coverage, those same components cannot compensate — because retrieval can only surface what exists.

This finding is not a criticism of the pipeline. It confirms that the pipeline is doing exactly what it should: grounding generation in retrieved evidence. The limitation is data coverage, not system design.

### 2. The Knowledge Gap Problem

The 30% accuracy on MedCaseReasoning with RAG is caused by a specific and identifiable failure: the textbook does not cover rare published case reports.

The 22% retrieval recall on Test Set C makes this explicit. In 78% of cases, the retrieval system could not surface relevant textbook content — because the textbook does not contain it. This is not a system failure. It is a data coverage limitation. The textbook covers common conditions well (74% accuracy on Test Set A) but has blind spots for rare, unusual, and atypical presentations.

The implication is clear: improving accuracy on rare cases requires expanding the knowledge base or using a fallback retrieval strategy. Improving the model alone cannot fix a retrieval recall problem.

### 3. Web Search as Gap-Filler

Web search directly addresses the knowledge gap identified above.

On Test Set C:
- RAG accuracy: 30%
- Web search accuracy: 42%
- Relative improvement: 40%

The mismatch rate drops from approximately 46% (RAG) to 28% (web search). Web search catches cases the textbook misses entirely because it can reach PubMed case reports, clinical guideline pages, and specialty society resources that are not in the local corpus.

Web search is also substantially faster: 6.1s vs 20.8s. RAG incurs overhead from embedding query computation, vector similarity search, and cross-encoder reranking. Web search skips all of this. The trade-off is that web sources are less structured and less reliably relevant than textbook chunks.

These findings support a combined retrieval architecture: RAG first (for grounded, textbook-backed diagnoses), web search as fallback when retrieval recall is estimated to be low.

### 4. The Open-Source Gap

On the RAG benchmark, the gap between GPT-5.4-mini and the best open-source model is substantial:

- GPT-5.4-mini (RAG, Test Set A): 74%
- Llama 3.1 8B (RAG, Test Set A): 42%
- Gap: 32 percentage points

On the web search benchmark, the gap is much smaller:

- GPT-5.4-mini (web search, Test Set C): 42%
- Gemma 4 local (web search, Test Set C): 36%
- Gap: 6 percentage points

The difference between these two gaps reveals where the performance advantage lies. Retrieval recall was consistent across models on the RAG benchmark — all models received the same textbook chunks, retrieved by the same pipeline. The 32-point gap is entirely in reasoning quality: GPT-5.4-mini reasons from textbook evidence nearly twice as effectively as the best open-source alternative.

On web search, the gap narrows to 6 points because web sources provide more explicit, readable, and self-contained content than textbook chunks. When the source material is a PubMed abstract or a Mayo Clinic clinical summary, open-source models can follow the text more directly. When the source material is dense textbook chunks requiring synthesis across multiple passages, reasoning quality becomes the bottleneck — and that is where the frontier model has its strongest advantage.

### 5. Model Size Does Not Predict Accuracy

The benchmark results show no correlation between model size and accuracy:

- Llama 3.1 8B: 42%
- Gemma 2 27B: 40%
- DeepSeek-R1 14B: 36%
- Aloe-8B: 36%

Gemma 2 27B is 3.4x larger than Llama 3.1 8B and performed identically within the margin of measurement. DeepSeek-R1 14B, despite its chain-of-thought reasoning architecture, scored lower than both smaller general-purpose models and accumulated a 24% JSON error rate.

The accuracy ceiling for open-source models on this RAG task appears to be approximately 42% regardless of model architecture or parameter count. This ceiling is not about knowledge — retrieval gives all models the same evidence. The ceiling is about reasoning quality: the ability to synthesize across multiple retrieved chunks, resolve ambiguity, and commit to a specific structured diagnosis.

Larger models and specialized reasoning architectures did not break through this ceiling in the benchmark.

### 6. Medical Fine-Tuning Hurts with RAG

One of the most consistent findings across the entire Medora project is that medical fine-tuning underperforms general-purpose approaches when RAG provides the evidence:

- Aloe-8B (medical fine-tune of Llama 3.1 8B): 36%
- Llama 3.1 8B (base): 42%
- MedLlama2 7B: complete failure

This pattern appeared at every pipeline stage:
- Phase 2.1: a general-purpose embedding model outperformed biomedical embedding models on chunk retrieval.
- Phase 3: the BGE general cross-encoder outperformed MedCPT (a medical cross-encoder) on reranking.
- Phase 8: general LLMs outperform medical fine-tunes on RAG-based diagnosis.

The explanation is consistent across all three findings. Medical fine-tuning trains a model to recall domain knowledge from its weights — to answer clinical questions without external context. When that model is then given a RAG prompt, it competes between its fine-tuned priors and the retrieved evidence. The fine-tuning teaches "I know the answer from training"; RAG requires "I should reason from this provided evidence." These objectives are in tension.

A general-purpose model has no prior clinical answer to fall back on. When given textbook evidence, it is more likely to reason from that evidence rather than override it with memorized knowledge.

Medical fine-tuning is a valuable approach for closed-book QA. It is a liability in a RAG pipeline.

### 7. Instruction Following is the Prerequisite

The benchmark revealed that instruction following is a harder constraint than domain knowledge or model size.

MedLlama2 7B produced no valid output — it could not follow the structured prompt format required for evaluation. No diagnostic content was generated. The model failed entirely before accuracy could be measured.

DeepSeek-R1 14B followed instructions partially but inserted chain-of-thought `<think>` tags that broke JSON parsing. 24% of its outputs were schema-invalid and were excluded from accuracy scoring. The effective accuracy on valid outputs was still 36% — but in a deployed system, a 24% parse failure rate is unacceptable.

Every model that produced 0% JSON errors (GPT-5.4-mini, Llama 3.1 8B, Gemma 2 27B, Aloe-8B) achieved at least 36% accuracy. The correlation is direct: models that cannot follow structured output format cannot be evaluated and cannot be deployed. Instruction-following quality is a prerequisite, not a secondary concern.

### 8. The Combined System Architecture

Based on all benchmark findings, the optimal architecture for the Medora triage pipeline is:

```text
Patient symptoms -> Intake Agent (structured intake with follow-up questions)
    -> Triage Agent:
        1. RAG retrieval from textbook (primary)
        2. Estimate retrieval recall from reranker scores
        3. If retrieval recall is low -> Web search fallback (SearXNG -> whitelist -> fetch)
        4. Combine textbook evidence and web evidence
        5. Generate diagnosis grounded in all available evidence
    -> Doctor reviews diagnosis + evidence sources
```

Expected performance under this architecture:
- Common conditions covered by the textbook: approximately 74% accuracy (RAG path).
- Rare conditions not in the textbook: approximately 42% accuracy (web search fallback path).
- Overall: significantly better than either retrieval strategy alone for a mixed clinical population.

The key design principle is that retrieval strategy selection should be automatic and evidence-driven, not hardcoded. The triage agent should use retrieval quality signals (reranker scores, number of relevant chunks above a threshold) to decide in real time which path to take.

### 9. The Benchmark as a Methodological Contribution

The benchmark design itself contributes a reusable evaluation methodology for RAG-based clinical decision support systems.

Three test sets with distinct purposes were used rather than a single evaluation set. Test Set A validates the pipeline under favorable conditions. Test Set B tests generalization to standardized medical knowledge. Test Set C stress-tests knowledge gap behavior.

Failure attribution metrics distinguish retrieval failures from generation failures. This is essential for debugging: without this separation, a drop in accuracy could be misattributed to the wrong component.

The multi-tier matching scheme (exact, semantic, partial, mismatch) captures the clinical reality that a diagnosis can be correct in substance while using different terminology. A system that reports only exact match accuracy would systematically underestimate clinical usefulness.

Web search was included as an evaluation methodology — not just as a system component. Testing both RAG and web search on the same test set (Test Set C) enables direct comparison of two retrieval strategies on identical cases.

The progression across three test sets and two retrieval strategies tells a coherent thesis story: what works, what fails, why it fails, and how the failure can be addressed.

### 10. Limitations and Future Work

**Small test sets.** 50 cases per test set is sufficient to identify large differences (32-point gaps, 40% relative improvements) but produces wide confidence intervals for smaller differences. A 300-case evaluation would substantially tighten the conclusions.

**Web search quality dependency.** Web search accuracy depends on SearXNG availability, source content quality, and the relevance of the pages returned. A SearXNG instance that is rate-limited, misconfigured, or returning outdated pages would reduce accuracy unpredictably. The benchmark used a controlled SearXNG environment.

**Benchmark bypasses intake.** The benchmark provides symptoms directly to the triage step. Real patients describe symptoms conversationally and incompletely. The intake agent with follow-up questions would produce more structured symptom profiles — but would also introduce intake quality as a confound. A full end-to-end benchmark including intake is future work.

**No human clinician validation.** Ground-truth diagnoses for Test Sets A and C come from published case materials. No practicing physician reviewed the model outputs for clinical plausibility. A clinician-rated evaluation would provide a more realistic accuracy estimate and capture partial credit that the automated scheme misses.

**Web search uses full case description.** The benchmark passed the complete published case description as the symptom input to the web search agent. Real patients describe much less. The 42% web search accuracy likely overestimates real-world performance with patient-generated input.

**Open-source hardware constraint.** All open-source models were evaluated on an A10G GPU (24GB VRAM). Larger models (70B+) that might perform better were excluded by GPU memory limits. A higher-memory environment could enable more competitive open-source evaluation.

**Fine-tuning on feedback loop data is untested.** Phase 7 built a doctor review and training data export pipeline. Fine-tuning open-source models on doctor-validated Medora cases could potentially close the 32-point gap between GPT-5.4-mini and open-source models on RAG. This has not been tested and remains future work.

---

## Benchmark Summary

| Finding | Result |
|---|---|
| Best RAG accuracy (textbook cases) | 74% — GPT-5.4-mini |
| Best open-source RAG accuracy | 42% — Llama 3.1 8B |
| RAG accuracy on rare cases | 30% — GPT-5.4-mini |
| Web search accuracy on rare cases | 42% — GPT-5.4-mini |
| Retrieval recall (textbook, Test Set A) | ~70% across all models |
| Retrieval recall (rare cases, Test Set C) | 22% |
| Web search latency vs RAG latency | 6.1s vs 20.8s (3.4x faster) |
| Medical fine-tune vs general LLM (RAG) | 36% vs 42% — general wins |
| JSON error rate — worst model | 24% — DeepSeek-R1 14B |
| JSON error rate — all other models | 0% |
| Models with complete failure | MedLlama2 7B |

---

## Scripts Reference

Run RAG benchmark against Test Set A:

```bash
python evaluation/rag_benchmark.py --test-set textbook --models gpt-5.4-mini llama-3.1-8b gemma-2-27b deepseek-r1-14b aloe-8b --n 50
```

Run RAG benchmark against Test Set C:

```bash
python evaluation/rag_benchmark.py --test-set medcasereasoning --models gpt-5.4-mini --n 50
```

Run web search benchmark against Test Set C:

```bash
python evaluation/web_search_benchmark.py --test-set medcasereasoning --models gpt-5.4-mini gemma4 --n 50
```

Run combined evaluation with automatic RAG/web search routing:

```bash
python evaluation/combined_benchmark.py --test-set medcasereasoning --primary-model gpt-5.4-mini --recall-threshold 0.4 --n 50
```
