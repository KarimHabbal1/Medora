# Phase 6: Web Evidence System

## Overview

Phase 6 adds a privacy-safe web evidence agent to Medora. The agent is implemented as a standalone package under `agents/web_evidence/` and is designed to supplement the local textbook-based RAG pipeline with current public medical evidence. It accepts a clinical question and optional patient context, removes patient-identifying information, builds a de-identified web search query, searches trusted medical sources, fetches and cleans source pages, ranks the sources by trust and recency, extracts medical claims, runs a council-style review, and returns a structured JSON result that can later be included in a doctor-facing pre-visit report.

This phase is necessary because Medora's earlier phases operate over a fixed local textbook corpus. That corpus is valuable because it is stable, auditable, and privacy-preserving, but some clinical information changes after publication. Antibiotic guidelines, vaccination schedules, infectious disease recommendations, drug safety warnings, and screening guidance can all change over time. A clinical decision-support system should not rely only on static textbook content when the question explicitly asks for current guideline-level evidence.

The central challenge of Phase 6 is that current evidence retrieval must not compromise patient privacy. Medora is designed for hospital deployment. Patient data must not be sent to hosted LLM APIs, and patient-identifying information must not be included in web search queries. Phase 6 therefore uses a controlled architecture: deterministic privacy filtering and trusted-source validation are always performed before evidence synthesis, while local open-source LLMs are optional and restricted to hospital-controlled infrastructure.

The final implementation supports deterministic-only operation, local LLM-assisted operation, and experimental comparison across multiple approaches. The latest real local-model experiment using `llama3.2:latest` showed that the current measured best approach is `deterministic_only`. The longer-term architectural target remains a controlled hybrid approach, but only after prompt and model improvements reduce safety warnings and improve claim extraction quality.

---

## Why This Phase Exists

The earlier Medora pipeline already provides a strong local medical knowledge base:

- Phase 1.1 extracts clean text from medical PDFs.
- Phase 1.2 chunks the extracted text while preserving document structure.
- Phase 1.3 converts common symptoms into structured JSON.
- Phase 2.1 embeds the textbook chunks.
- Phase 2.2 stores the embeddings in ChromaDB.
- Phase 2.3 validates retrieval quality.
- Phase 3 improves ranking using cross-encoder reranking.

That pipeline answers the question: "What does the local medical corpus say about this clinical issue?" Phase 6 answers a different question: "Is there current public medical evidence that should update or supplement the local corpus?"

This distinction matters. A textbook may correctly explain the pathophysiology and general management of pneumonia, but the latest antibiotic recommendations may depend on updated resistance patterns or current guideline statements. A static corpus cannot guarantee that those recommendations are current. Phase 6 adds a controlled web evidence layer so Medora can retrieve current evidence without abandoning privacy or traceability.

### What This Phase Adds

Phase 6 adds four capabilities:

1. **Privacy-safe web querying:** patient-identifying information is removed before search.
2. **Trusted-source filtering:** retrieved evidence is ranked using a medical source-tier policy.
3. **Council review:** evidence is checked by source, recency, conflict, safety, and final-review roles.
4. **Approach comparison:** deterministic and local LLM-assisted approaches are evaluated experimentally.

The goal is not to let the web replace clinical judgment. The goal is to produce a cautious, source-backed, physician-reviewable evidence summary.

---

## Role in the Medora System

The web evidence agent is not a patient-facing chatbot. It does not provide a diagnosis and does not give direct treatment instructions. It is a backend evidence tool that can support a doctor-facing pre-visit report.

In the intended Medora workflow:

1. A patient provides symptoms before a visit.
2. The conversational intake and diagnostic agents produce a preliminary structured report.
3. If a clinical question requires current evidence, the web evidence agent is called.
4. The web evidence agent returns current, source-backed evidence with limitations and council decisions.
5. The physician reviews the complete output before using it clinically.

This separation is deliberate. It keeps the system within the role of clinical decision support, not autonomous medical decision-making.

---

## Implementation Summary

The Phase 6 implementation added the following package:

```text
agents/web_evidence/
|-- __init__.py
|-- schemas.py
|-- config.py
|-- pii_sanitizer.py
|-- query_builder.py
|-- search_client.py
|-- page_fetcher.py
|-- source_ranker.py
|-- council.py
|-- synthesizer.py
|-- prompts.py
|-- agent.py
|-- cli.py
|-- llm/
|   |-- __init__.py
|   |-- base.py
|   |-- ollama_client.py
|   |-- hf_client.py
|   `-- mock_llm.py
`-- experiments/
    |-- __init__.py
    |-- approaches.py
    `-- runner.py
```

It also added:

- `evaluation/web_evidence_validation.py`
- `docs/phase_6_web_evidence_council.md`
- `data/results/web_evidence_approach_comparison.csv`
- `data/results/web_evidence_llama32_approach_comparison.csv`
- `data/results/web_evidence_mock20_approach_comparison.csv`
- `data/results/web_evidence_searxng_smoke_test.json`

The completed earlier phases were preserved. Phase 6 does not rewrite PDF extraction, chunking, embedding, ChromaDB indexing, retrieval validation, or reranking.

---

## End-to-End Pipeline

The Phase 6 pipeline is:

```text
User input
  -> privacy sanitizer
  -> query builder
  -> search client
  -> page fetcher
  -> source ranker
  -> claim extraction
  -> council review
  -> synthesis
  -> final JSON
```

The following running example is used throughout this section:

```json
{
  "clinical_question": "What are the latest guideline recommendations for community-acquired pneumonia antibiotics in adults?",
  "patient_context": {
    "name": "John Smith",
    "age": 65,
    "sex": "male",
    "phone": "123-456-7890",
    "symptoms": ["fever", "productive cough", "shortness of breath"],
    "red_flags": ["hypoxia"],
    "comorbidities": ["diabetes"]
  },
  "reason_for_web": "Guidelines may have changed after textbook publication."
}
```

This example intentionally contains patient identifiers so the privacy behavior is visible.

### Step 1: User Input

The input is represented by `WebEvidenceRequest` in `schemas.py`. The request contains the clinical question, optional patient context, and the reason web evidence is needed.

This step exists because the web evidence agent needs a clean input contract. A medical system cannot rely on loosely structured strings alone. The agent must know which field is the clinical question, which field is patient context, and which field explains why web evidence is being requested.

Without a schema, downstream components could accidentally treat raw patient context as search text. That would create a privacy risk and make the system difficult to test.

### Step 2: Privacy Sanitizer

The sanitizer is implemented in `pii_sanitizer.py`. It removes or avoids patient-identifying information before anything is sent to search.

For the running example, the sanitizer removes:

```text
name: John Smith
phone: 123-456-7890
exact age: 65
```

It keeps:

```text
age_range: older adult
sex: male
symptoms: fever, productive cough, shortness of breath
red_flags: hypoxia
comorbidities: diabetes
```

The sanitized context becomes:

```json
{
  "age_range": "older adult",
  "sex": "male",
  "symptoms": ["fever", "productive cough", "shortness of breath"],
  "red_flags": ["hypoxia"],
  "comorbidities": ["diabetes"]
}
```

This step is essential in a medical system because patient data must not leave the controlled environment unnecessarily. Without this step, a search query could include a name, phone number, hospital ID, or exact date of birth.

### Step 3: Query Builder

The query builder is implemented in `query_builder.py`. It combines the sanitized clinical question with allowed medical context and trusted-source hints.

For the running example, the query becomes:

```text
latest guideline recommendations community acquired pneumonia antibiotics adults older adult male fever productive cough shortness of breath hypoxia diabetes guideline recommendation updated WHO CDC NICE NIH PubMed
```

The exact wording may vary slightly depending on normalization, but the important property is that the query contains clinical concepts, not patient identifiers.

This step exists because raw clinical questions are not always optimal search queries. The system adds guideline-related terms and trusted-source hints to bias retrieval toward high-quality medical evidence. Without query construction, search results may be broad, consumer-facing, outdated, or unreliable.

### Step 4: Search

Search is implemented in `search_client.py`. The production-oriented backend is `SearxNGSearchClient`, which uses a self-hosted SearxNG instance. The testing backend is `StaticSearchClient`, which returns deterministic mock results.

In mock mode, the running example retrieves sources such as:

| Source | Domain | Tier | Type |
|---|---|---:|---|
| NICE pneumonia in adults guideline | `nice.org.uk` | 1 | guideline |
| CDC clinical guidance for respiratory infections | `cdc.gov` | 1 | guideline |
| FDA drug safety communications | `fda.gov` | 1 | government |
| CDC adult immunization schedule | `cdc.gov` | 1 | guideline |
| IDSA community-acquired pneumonia guideline | `idsociety.org` | 2 | guideline |

This step is needed because Phase 6 must retrieve current public evidence. Without search, the system would be limited to the static textbook corpus. However, search is also risky because public search can return low-quality or consumer-facing pages. That is why search is followed by source ranking and council review.

### Step 5: Page Fetching

Page fetching is implemented in `page_fetcher.py`. It fetches source pages, extracts titles, extracts text from headings and paragraphs, attempts to identify published or updated dates, and limits maximum text length.

This step exists because search snippets are not enough for evidence extraction. A snippet may omit key caveats, dates, or context. The system needs page text to extract supported claims.

Without page fetching, the agent might summarize search-result snippets rather than real source content. That would be too weak for a medical evidence system.

### Step 6: Source Ranking

Source ranking is implemented in `source_ranker.py`. It assigns each source:

- a domain
- a tier
- a source type
- a date if available
- a reliability score

The running example ranks Tier 1 government and guideline sources above Tier 2 professional society sources, while rejecting or down-ranking consumer websites and forums.

This step is necessary because web search does not guarantee quality. A search engine may return a high-ranking consumer article before a guideline. In a medical system, source quality must be explicit and auditable.

Without source ranking, the final report could include weak sources such as blogs, forums, or consumer health summaries.

### Step 7: Claim Extraction

Claim extraction is implemented in `council.py` through `MedicalClaimExtractor`. It can run deterministically or with a local LLM.

Example extracted claims from the mock pneumonia run include:

```json
[
  {
    "claim": "NICE guideline recommendations describe diagnosis, severity assessment, antimicrobial prescribing, and management of pneumonia in adults.",
    "supporting_sources": ["https://www.nice.org.uk/guidance/ng138"],
    "confidence": "high"
  },
  {
    "claim": "Guideline recommendations for community-acquired pneumonia in adults should be interpreted with clinical judgment.",
    "supporting_sources": ["https://www.idsociety.org/practice-guideline/community-acquired-pneumonia-cap-in-adults/"],
    "confidence": "moderate"
  }
]
```

This step exists because the final summary should be based on explicit claims, not unstructured page text. Without claim extraction, the system would have no intermediate evidence object to inspect, score, or validate.

### Step 8: Council Review

Council review is implemented in `council.py`. The council includes:

- `SourceValidator`
- `RecencyChecker`
- `MedicalClaimExtractor`
- `ConflictChecker`
- `SafetyCritic`
- `FinalReviewer`

For the running example, the council typically accepts source quality because Tier 1 sources are present. It accepts with caution overall because deterministic conflict detection is limited and because the output is intended for physician review.

This step is needed because evidence retrieval alone is not enough. A medical system needs checks for source quality, recency, conflict, and safety. Without council review, the system could present unsupported or unsafe conclusions.

### Step 9: Synthesis

Synthesis is implemented in `synthesizer.py`. It creates a cautious doctor-facing summary from claims, sources, conflicts, and limitations.

The synthesis must not:

- give a final diagnosis
- give direct patient instructions
- invent claims
- omit uncertainty
- omit physician-review language

Example summary:

```text
For physician review, guideline-level sources suggest preliminary evidence for community-acquired pneumonia management in adults. NICE and IDSA guideline sources discuss diagnosis, severity assessment, antimicrobial prescribing, and management decisions. This is not a final diagnosis or direct patient treatment instruction; it requires clinician judgment.
```

This step exists because doctors need a readable summary, not only raw JSON. Without synthesis, the result would be technically structured but harder to use in a report.

### Step 10: Final JSON

The final output is a `WebEvidenceResult`:

```json
{
  "clinical_question": "What are the latest guideline recommendations for community-acquired pneumonia antibiotics in adults?",
  "sanitized_query": "latest guideline recommendations community acquired pneumonia antibiotics adults older adult male fever productive cough shortness of breath hypoxia diabetes guideline recommendation updated WHO CDC NICE NIH PubMed",
  "summary": "For physician review...",
  "key_findings": [],
  "conflicts": [],
  "limitations": [],
  "sources": [],
  "council_review": {},
  "final_decision": "accept_with_caution",
  "privacy_report": {
    "removed_identifiers": ["exact_age", "name", "phone"],
    "kept_medical_context": ["age_range", "sex", "symptoms", "red_flags", "comorbidities"],
    "deidentified": true
  },
  "execution_metadata": {}
}
```

This final object is suitable for storage, inspection, evaluation, and later inclusion in a doctor-facing report.

---

## Simplified Web Search Agent (`agents/web_search.py`)

### Why a Simplified Version

The council architecture provides comprehensive evidence validation but adds complexity that slows down the pipeline. For the benchmarking phase and initial deployment, a simplified agent was built that strips the pipeline to its core: search → fetch → LLM diagnosis.

The council remains available for future use when evidence validation becomes critical (e.g., production deployment where false claims could cause harm). The simplified agent prioritizes speed and simplicity for the thesis evaluation.

### Architecture

The simplified agent has 4 steps:
1. **Build search query** — takes patient symptoms, adds "diagnosis symptoms" prefix
2. **Search SearXNG** — calls the metasearch engine, filters to whitelisted domains only
3. **Fetch pages** — extracts text from top 5 results using BeautifulSoup, truncates to 2000 chars each
4. **LLM diagnosis** — sends symptoms + all fetched content to an LLM with a structured diagnosis prompt

### Whitelisted Sources (Strict)

Only results from these domains are used — everything else is dropped:
- PubMed, PMC, NCBI, NIH
- Mayo Clinic, Cleveland Clinic, Hopkins Medicine
- MedlinePlus, Merck Manuals
- CDC, WHO, NICE
- BMJ, NEJM, The Lancet
- Medscape

No Wikipedia, no WebMD, no HealthLine, no social media.

### LLM Output Format

The LLM returns structured JSON:
```json
{
    "primary_diagnosis": "most likely condition",
    "confidence": "high/moderate/low",
    "evidence_summary": "2-3 sentences connecting symptoms to diagnosis",
    "key_findings": [{"claim": "...", "source": "...", "url": "..."}],
    "differential_diagnoses": ["other possible conditions"]
}
```

### Model Support

Uses `config.make_llm()` — works with both OpenAI and Ollama:
```bash
# With GPT-5.4-mini
python agents/web_search.py --symptoms "painful rash, fever, joint pain" --model gpt-5.4-mini --provider openai

# With local Ollama model
python agents/web_search.py --symptoms "painful rash, fever, joint pain" --model gemma4:latest
```

### Benchmark Results

Tested on MedCaseReasoning Test Set C (50 cases — conditions outside the textbook):

| Model | Accuracy | Latency | Avg Sources |
|---|---|---|---|
| GPT-5.4-mini | 42% | 6.1s | 1.56 |
| Gemma 4 (local) | 36% | 16.6s | 1.48 |

Compared to RAG-only on the same test set (30%), web search improves accuracy by 40% relative.

### Integration with Triage Agent

The web search agent is triggered when the Triage Agent's RAG retrieval recall is low:
```
Triage Agent → RAG retrieval → low recall detected → web search fallback → combined evidence → diagnosis
```

This is not yet implemented as an automated trigger — currently the web search agent is called standalone. Phase 9 (future work) would integrate it as an automatic fallback.

---

## Privacy Design

Privacy is the most important design constraint in Phase 6. Medora is a hospital-deployable system. Patient information must not be exposed to external LLM APIs, public search engines, logs, or uncontrolled services.

### Why Patient Data Must Never Leave the System

Clinical input can contain identifiers even when the user does not intend to provide them. A patient may write:

```text
Patient name is John Smith, 65 years old, phone 123-456-7890, has fever and cough.
```

If this raw text is sent to a web search engine or hosted LLM API, the system has leaked identifiable health information. This is unacceptable for a privacy-preserving clinical decision-support system.

Phase 6 therefore treats de-identification as a required preprocessing step, not an optional feature.

### What Counts as PII in This Context

The sanitizer treats the following as identifying or potentially identifying:

- names
- phone numbers
- email addresses
- street addresses
- hospital IDs
- patient IDs
- national IDs
- exact dates of birth
- exact dates when used as identifying context
- precise location
- unknown free-text fields

Medical privacy is broader than simply removing names. A combination of exact age, location, rare disease, and date may also identify someone. The sanitizer therefore removes unknown free-text fields and reduces exact age to a range.

### Regex-Based Detection

`pii_sanitizer.py` uses deterministic regex rules for obvious identifiers:

- email patterns
- phone-number patterns
- date-of-birth patterns
- exact date patterns
- hospital or national ID patterns
- address patterns
- explicit name patterns such as "my name is" or "patient name is"

This approach is predictable and testable. A local LLM is not used for privacy filtering because privacy should not depend on model judgment.

### Allow-List Context Policy

The sanitizer uses an allow-list for patient context. Only specific medical fields are allowed through:

- age range
- sex
- symptoms
- red flags
- comorbidities
- pregnancy status
- immunocompromised status
- medications

All unknown fields are excluded. This is safer than a block-list approach. A block-list tries to predict every possible identifier; an allow-list only keeps fields known to be clinically useful and relatively safe.

### Age Range Instead of Exact Age

Exact age can increase identifiability. The system converts exact age into broad ranges:

- infant
- child
- adolescent
- adult
- middle-aged adult
- older adult

For example:

```text
65 years old -> older adult
```

This preserves clinical usefulness while reducing identifiability.

### Free Text Removal

Free-text fields are high risk because they may contain anything: names, locations, IDs, clinician notes, or family details. The sanitizer therefore excludes unknown free-text fields by default.

### Example

Input:

```text
Patient name is John Smith, 65 years old, phone 123-456-7890, fever, cough, hypoxia.
```

Safe search context:

```text
older adult fever cough hypoxia
```

This is safer because it preserves the medical question while removing direct identifiers.

### LLM Privacy Boundary

The LLM never sees raw patient identifiers. Local LLM calls receive only:

- sanitized clinical question
- fetched public source text
- extracted claims
- ranked source metadata

The system fetches web evidence first, then optionally passes source text to a local LLM. The LLM does not receive raw patient identity and does not perform web search.

---

## Design Alternatives Considered

The Phase 6 architecture was not chosen randomly. Several alternatives were considered and then evaluated against privacy, safety, controllability, latency, and evidence quality.

### Alternative A: Fully LLM-Based System

A fully LLM-based design would send the clinical question to a powerful hosted or local model and ask it to search, reason, evaluate sources, and generate a final summary.

This option was considered because it is simple from a user perspective. A modern GPT-style system can often produce fluent answers and may appear to handle search, synthesis, and reasoning in one step.

It was rejected for four reasons.

First, hosted LLM APIs violate the privacy goal. Patient data could leave the hospital-controlled environment.

Second, a general LLM may hallucinate sources or claims. In medicine, an unsupported claim is not just a quality problem; it is a safety problem.

Third, source trust becomes difficult to audit. If the model decides which sources are reliable, the policy becomes hidden inside model behavior.

Fourth, the output may be difficult to reproduce. The same prompt may produce different summaries, especially if generation settings or model versions change.

For these reasons, Phase 6 does not use a fully LLM-based design.

### Alternative B: Deterministic-Only System

A deterministic-only system uses fixed code for every step:

- regex privacy filtering
- deterministic query construction
- deterministic source ranking
- deterministic claim extraction
- deterministic council checks
- deterministic summary generation

This approach is safe, fast, reproducible, and easy to audit. It performed best in the latest `llama3.2:latest` comparison because it produced more claims, triggered no safety warnings, and ran in milliseconds in mock mode.

Its weakness is semantic depth. Keyword-based claim extraction may miss nuanced guideline statements, exceptions, or conflicts. A deterministic summary is also less flexible and less natural than a good local LLM summary.

This approach is currently the best measured option, but it may not remain best after stronger local models and better prompts are evaluated.

### Alternative C: Full LLM Council

The full LLM council uses a local LLM for claim extraction, conflict checking, safety criticism, final review, and synthesis.

This approach was considered because local LLMs can understand language more flexibly than deterministic rules. They may detect subtle conflicts or summarize complex guideline text better than keyword logic.

However, the latest `llama3.2:latest` run showed the downside. `full_llm_council` had much higher latency and scored below deterministic-only. It averaged about 73.9 seconds per query in the 5-question local-model comparison, compared with 0.0009 seconds for deterministic-only in mock search mode. It also produced fewer claims and triggered safety warnings.

The risk is that too many safety-critical decisions become model-dependent. More LLM does not automatically mean a better medical system.

### Alternative D: Hybrid System

The hybrid design keeps deterministic control over privacy, source trust, recency, and safety, while using a local LLM only for language-understanding tasks such as claim extraction and synthesis.

This design was proposed because it matches the strengths of each method:

- deterministic code is best for policy, privacy, and auditability
- local LLMs are best for interpreting source text and producing readable summaries

Theoretically, this is the best long-term architecture. In the current `llama3.2:latest` run, however, `hybrid_recommended` scored lower than deterministic-only because local LLM claim extraction produced fewer claims and the synthesis triggered safety warnings. This does not invalidate the hybrid architecture; it shows that the current model and prompts are not yet strong enough to outperform the deterministic baseline.

The conclusion is staged:

- use deterministic-only as the current measured baseline
- use LLM synthesis only when readable output is worth the latency cost
- continue improving the hybrid approach as the future production target

---

## Search Backend Options

Search is intentionally pluggable. Different backends are appropriate at different stages.

| Option | Privacy | Setup | Use |
|---|---|---|---|
| SearxNG | Best option when self-hosted; hospital controls the search frontend | Requires deployment and configuration of `SEARXNG_BASE_URL` | Final production search backend |
| DuckDuckGo-style public search | Lower privacy control because it depends on a public third-party service | Easy for development exploration | Useful for early prototyping only with sanitized queries |
| Mock search | No external privacy risk; no network calls | No setup required | Testing, validation, CI, reproducible experiments |

### Why DuckDuckGo Was Useful Initially

During early development, a simple public search option such as DuckDuckGo is useful because it requires little setup and can quickly show whether query construction retrieves reasonable medical pages. However, it is not ideal for final deployment because the hospital does not control the search backend.

The committed Phase 6 implementation does not hard-code DuckDuckGo as the production backend. The architecture supports search-provider replacement through `BaseSearchClient`, and the production direction is self-hosted SearxNG.

### Why SearxNG Is the Production Choice

SearxNG can be hosted by the hospital. That means the institution can control search configuration, logging, network policy, and privacy behavior. It also avoids depending on Google or Bing API keys.

### Why Mock Search Is Required

Mock search is required because experiments must be reproducible. Without mock mode, search results may change from day to day, making it difficult to compare approaches fairly. Mock mode also allows validation without internet access.

---

## Local LLM Architecture

The local LLM architecture is deliberately separated from search.

The LLM does not fetch web pages. The system fetches web pages through `search_client.py` and `page_fetcher.py`. The LLM only processes text that Medora provides to it after privacy filtering, source retrieval, and source ranking.

This separation is important:

- search remains auditable
- source trust remains deterministic
- the LLM cannot choose arbitrary web evidence
- local model failures can be isolated and handled

### Ollama

`OllamaLocalLLM` is the preferred local model backend because it is simple to deploy on a hospital-controlled server. It communicates with a local Ollama HTTP server and can run models such as:

- `llama3.2:latest`
- future candidate: `llama3.1:8b`

Ollama is preferred because it avoids heavy Python model-loading complexity inside the Medora process. The model server is separate from the application.

### Hugging Face

`HuggingFaceLocalLLM` is optional. It is useful when the hospital wants direct local Transformers integration. It uses `local_files_only=True`, meaning models must already be available locally.

This backend is optional because it may require heavy dependencies and more deployment complexity.

### Mock LLM

`MockLocalLLM` is required for deterministic testing. It returns valid JSON without network calls or model weights. It is not a clinical model; it is a testing tool.

### Hospital Deployment Layout

A hospital-controlled deployment would contain:

```text
Hospital server
|-- Medora backend
|-- Ollama local model server
|-- SearxNG self-hosted search instance
|-- local ChromaDB vector store
`-- local logs and result storage
```

No patient identifiers need to leave this environment.

---

## Trusted Source Policy

The trusted source policy is implemented in `config.py` and applied in `source_ranker.py`.

Tier 1 sources:

- `who.int`
- `cdc.gov`
- `nice.org.uk`
- `nih.gov`
- `ncbi.nlm.nih.gov`
- `pubmed.ncbi.nlm.nih.gov`
- `fda.gov`
- `ema.europa.eu`

Tier 2 sources:

- `nejm.org`
- `thelancet.com`
- `bmj.com`
- `jamanetwork.com`
- `cochranelibrary.com`
- `acc.org`
- `heart.org`
- `diabetes.org`
- `idsociety.org`

Tier 3 sources:

- `mayoclinic.org`
- `clevelandclinic.org`
- `hopkinsmedicine.org`
- `msdmanuals.com`

Rejected by default:

- `reddit.com`
- `quora.com`
- `wikipedia.org`
- `healthline.com`
- `webmd.com`
- `verywellhealth.com`
- `medicalnewstoday.com`
- blogs
- forums

Source trust is deterministic because it is a policy decision. A language model should not decide whether a domain is acceptable for clinical evidence.

---

## Council Architecture

The council is implemented in `council.py`.

### SourceValidator

Checks whether final sources come from acceptable tiers. It rejects results with no Tier 1 or Tier 2 support.

### RecencyChecker

Checks whether sources are recent, old, or missing dates. Undated or old sources are accepted with caution.

### MedicalClaimExtractor

Extracts evidence claims from source text. It can run deterministically or with a local LLM.

### ConflictChecker

Checks whether claims contradict each other. Deterministic logic catches simple conflicts; local LLM mode can attempt a deeper review.

### SafetyCritic

Rejects unsafe patient-directed language such as starting, stopping, or dosing medications. It also checks for physician-review framing.

### FinalReviewer

Combines council decisions into:

- `accept`
- `accept_with_caution`
- `reject`

The final reviewer can be deterministic or local LLM-assisted depending on the selected approach.

---

## Failure Cases and Safety Behavior

The agent is designed to fail safely. It should not crash, fabricate evidence, or produce unsupported medical claims.

### Missing `SEARXNG_BASE_URL`

If SearxNG is not configured, `SearxNGSearchClient` returns a structured error. The final output records:

```json
"limitations": ["missing_searxng_base_url"]
```

The result is rejected because no reliable evidence was retrieved.

This occurred in the SearxNG smoke test:

```json
{
  "final_decision": "reject",
  "limitations": ["missing_searxng_base_url"]
}
```

### Page Fetch Failure

If a page cannot be fetched, the fetcher returns a `FetchedSource` with an error field. The agent records the limitation and may fall back to the search snippet if appropriate.

This prevents a broken website from crashing the pipeline.

### LLM JSON Failure

If a local LLM returns invalid JSON:

1. the system retries once with stricter JSON-only instructions
2. if retry fails, deterministic fallback is used
3. the fallback is recorded in `execution_metadata`

This prevents arbitrary prose from becoming a council decision.

### Low-Quality Sources

If no reliable Tier 1 or Tier 2 sources are present, `SourceValidator` rejects the evidence. The final result is not accepted simply because some web pages were found.

### Conflicting Evidence

If conflicts are detected, the result is accepted with caution or rejected depending on severity. The final summary must mention conflicts or limitations rather than hiding them.

---

## Code Mapping

This section maps each major concept to the implementation files.

| Concept | File | Responsibility |
|---|---|---|
| Schemas and JSON contract | `agents/web_evidence/schemas.py` | Defines request, source, claim, council, privacy, and result dataclasses |
| Source policy and constants | `agents/web_evidence/config.py` | Defines trusted tiers, rejected domains, scoring constants, and fetch limits |
| Privacy filtering | `agents/web_evidence/pii_sanitizer.py` | Removes identifiers, converts exact age to age range, keeps only allowed medical context |
| Query construction | `agents/web_evidence/query_builder.py` | Builds a privacy-safe search query from sanitized clinical terms |
| Web search | `agents/web_evidence/search_client.py` | Implements SearxNG and static mock search clients |
| Page parsing | `agents/web_evidence/page_fetcher.py` | Fetches URLs, extracts page text, titles, dates, and handles fetch errors |
| Trust scoring | `agents/web_evidence/source_ranker.py` | Assigns tiers, source types, recency scoring, and reliability scores |
| Council logic | `agents/web_evidence/council.py` | Implements source validation, recency checking, claim extraction, conflict checking, safety criticism, and final review |
| Synthesis | `agents/web_evidence/synthesizer.py` | Creates cautious doctor-facing summaries, with optional local LLM synthesis |
| Prompt templates | `agents/web_evidence/prompts.py` | Stores local LLM prompts for claim extraction, conflict checking, safety, final review, and synthesis |
| Orchestration | `agents/web_evidence/agent.py` | Runs the full pipeline from request to final result |
| CLI | `agents/web_evidence/cli.py` | Exposes command-line usage for individual questions |
| Local LLM interface | `agents/web_evidence/llm/base.py` | Defines model-agnostic local LLM interface and JSON parsing helpers |
| Ollama backend | `agents/web_evidence/llm/ollama_client.py` | Calls local Ollama HTTP API |
| Hugging Face backend | `agents/web_evidence/llm/hf_client.py` | Optional local Transformers backend |
| Mock model | `agents/web_evidence/llm/mock_llm.py` | Deterministic fake LLM for tests and mock evaluation |
| Approach definitions | `agents/web_evidence/experiments/approaches.py` | Defines five compared approaches |
| Experiment runner | `agents/web_evidence/experiments/runner.py` | Runs approach comparisons and computes metrics |
| Evaluation CLI | `evaluation/web_evidence_validation.py` | Command-line entry point for validation and comparison runs |

This mapping makes the architecture traceable. A reviewer can inspect each design decision in the corresponding file.

---

## Approaches Compared

Five approaches were implemented and evaluated.

### Approach A: `deterministic_only`

Everything is deterministic:

- deterministic search
- deterministic ranking
- deterministic claim extraction
- deterministic council
- deterministic synthesis
- no local LLM

Strengths:

- fastest
- easiest to audit
- reproducible
- no model dependency
- currently best measured with `llama3.2:latest`

Weakness:

- may miss nuanced claims because extraction is keyword-based

### Approach B: `llm_synthesis_only`

This approach keeps evidence extraction and council review deterministic, but uses a local LLM for final synthesis.

Strength:

- improves summary flexibility without changing source validation

Weakness:

- adds latency
- does not improve evidence extraction

In the `llama3.2:latest` run, it tied with deterministic-only on score but was much slower.

### Approach C: `llm_claims_and_synthesis`

This approach uses a local LLM for claim extraction and synthesis.

Strength:

- could improve semantic claim extraction with a stronger model

Weakness:

- with `llama3.2:latest`, it extracted fewer claims and triggered a safety warning

### Approach D: `full_llm_council`

This approach uses a local LLM for claim extraction, conflict checking, safety criticism, final review, and synthesis.

Strength:

- most flexible

Weaknesses:

- high latency
- more model dependence
- more opportunities for JSON failure
- harder to audit

This approach is not recommended for current deployment.

### Approach E: `hybrid_recommended`

This approach keeps privacy, source validation, recency, and safety deterministic, while using a local LLM for claim extraction and synthesis.

Strength:

- best long-term architectural balance

Weakness in current run:

- with `llama3.2:latest`, safety score dropped to `0.82`, and average claims dropped to `3.8`

This remains the future target, but it is not the current measured best.

---

## Experiments and Results

### Experiment 1: Five-Approach Local Ollama Comparison

Command:

```bash
python evaluation/web_evidence_validation.py --mock --provider ollama --model llama3.2:latest --approaches deterministic_only llm_synthesis_only llm_claims_and_synthesis full_llm_council hybrid_recommended --max-questions 5
```

Purpose:

- compare deterministic vs local LLM-assisted approaches
- isolate LLM behavior by using mock search
- measure latency, safety, evidence completeness, and JSON reliability

Model:

```text
llama3.2:latest
```

Results:

| Approach | Queries | Caution | Valid JSON | Privacy Pass | Avg Sources | Tier 1 | Tier 2 | Avg Claims | Safety Warnings | Avg Latency | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `deterministic_only` | 5 | 5 | 100% | 100% | 5.0 | 80% | 20% | 8.0 | 0 | 0.0009 s | 0.988 |
| `llm_synthesis_only` | 5 | 5 | 100% | 100% | 5.0 | 80% | 20% | 8.0 | 0 | 25.3132 s | 0.988 |
| `full_llm_council` | 5 | 3 | 100% | 100% | 5.0 | 80% | 20% | 3.8 | 1 | 73.9265 s | 0.9574 |
| `llm_claims_and_synthesis` | 5 | 5 | 100% | 100% | 5.0 | 80% | 20% | 3.6 | 1 | 73.8594 s | 0.9492 |
| `hybrid_recommended` | 5 | 5 | 100% | 100% | 5.0 | 80% | 20% | 3.8 | 2 | 49.5789 s | 0.9349 |

Interpretation:

- `deterministic_only` won because it had perfect safety, maximum claim count, and near-zero latency.
- `llm_synthesis_only` tied in score because it kept deterministic extraction and safety, but it added about 25 seconds of latency.
- `hybrid_recommended` lost because its safety score dropped to `0.82` and it extracted fewer claims.
- `full_llm_council` is risky because it adds many LLM calls, high latency, and model-dependent review decisions without improving score.

The main finding is:

```text
More LLM does not mean a better medical system.
```

In this run, adding more LLM roles increased latency and reduced safety/evidence scores. The safest measured architecture was deterministic.

### Experiment 2: Full 20-Question Mock Evaluation

Command:

```bash
python evaluation/web_evidence_validation.py --mock --provider mock --approaches deterministic_only llm_synthesis_only llm_claims_and_synthesis full_llm_council hybrid_recommended
```

Purpose:

- validate the experiment framework over the full 20-question set
- confirm JSON output, privacy behavior, source filtering, and metric computation
- run without internet or model dependencies

Results:

| Approach | Queries | Caution | Valid JSON | Privacy Pass | Avg Sources | Tier 1 | Tier 2 | Avg Claims | Safety Warnings | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hybrid_recommended` | 20 | 20 | 100% | 100% | 5.0 | 80% | 20% | 8.0 | 0 | 0.988 |
| `deterministic_only` | 20 | 20 | 100% | 100% | 5.0 | 80% | 20% | 8.0 | 0 | 0.988 |
| `llm_claims_and_synthesis` | 20 | 20 | 100% | 100% | 5.0 | 80% | 20% | 8.0 | 0 | 0.988 |
| `llm_synthesis_only` | 20 | 20 | 100% | 100% | 5.0 | 80% | 20% | 8.0 | 0 | 0.988 |
| `full_llm_council` | 20 | 20 | 100% | 100% | 5.0 | 80% | 20% | 8.0 | 0 | 0.988 |

Interpretation:

All approaches tied because the run used deterministic mock evidence and `MockLocalLLM`. This does not prove clinical superiority. It proves that the framework works across all 20 validation questions.

### Experiment 3: SearxNG Smoke Test

Command:

```bash
python -m agents.web_evidence.cli --question "latest CDC adult pneumococcal vaccine recommendations" --approach deterministic_only --json-output data/results/web_evidence_searxng_smoke_test.json
```

Result:

```json
{
  "final_decision": "reject",
  "limitations": ["missing_searxng_base_url"]
}
```

Interpretation:

The environment did not have `SEARXNG_BASE_URL` configured, so real web retrieval did not run. This is still a useful safety test. The system did not hallucinate sources or continue unsafely. It returned structured JSON, recorded the limitation, and rejected the result.

### Future Model: `llama3.1:8b`

`llama3.1:8b` remains a planned local model candidate. It should be evaluated because it may behave differently from `llama3.2:latest` in JSON following, claim coverage, latency, and safety. The current results should not be generalized to all local LLMs.

---

## Final Design Decision

The final decision is staged because the theoretical architecture and the measured current behavior are not identical.

### Current Best: `deterministic_only`

The current measured best approach is `deterministic_only`.

Reasons:

- highest overall score in the `llama3.2:latest` run
- tied with `llm_synthesis_only` but much faster
- no safety warnings
- maximum extracted claims
- no local model dependency
- easiest to audit

This is the safest current deployment option.

### Optional: `llm_synthesis_only`

`llm_synthesis_only` is an optional mode if a more natural doctor-facing summary is required.

Reasons:

- tied with deterministic-only on overall score
- kept deterministic claim extraction and safety checks
- did not reduce evidence completeness

Tradeoff:

- added about 25 seconds latency per query in the local `llama3.2:latest` run

### Future Target: `hybrid_recommended`

`hybrid_recommended` remains the future target architecture.

Reasons:

- preserves deterministic privacy, source trust, recency, and safety
- uses local LLM only where semantic understanding is useful
- aligns with the long-term goal of better claim extraction and synthesis

Current limitation:

- with `llama3.2:latest`, it scored lower because of safety warnings and reduced claim count

The practical conclusion is:

```text
Deploy deterministic_only now.
Use llm_synthesis_only only if summary readability justifies latency.
Continue improving hybrid_recommended as the future production architecture.
```

---

## What We Learned

Phase 6 produced several important engineering findings.

First, privacy must be deterministic. It is not acceptable to ask a model to decide what is safe to send outside the system.

Second, source trust must be deterministic. Medical source policy should be explicit, inspectable, and reproducible.

Third, local LLM integration works. Ollama `llama3.2:latest` returned valid JSON and did not require fallback in the recorded comparison.

Fourth, local LLMs are not automatically better. In the current run, LLM-heavy approaches were slower and scored lower.

Fifth, a comparison framework is necessary. Without running all approaches, it would have been easy to assume that `hybrid_recommended` was best. The experiment showed that the current measured best is actually deterministic-only.

---

## Limitations

The evaluation is not a clinical gold-standard study. It is an engineering evaluation.

Current limitations:

- the local `llama3.2:latest` run used mock search, so it did not test live web retrieval
- the full 20-question run used `MockLocalLLM`, so it did not test real model behavior
- the SearxNG smoke test did not retrieve evidence because `SEARXNG_BASE_URL` was missing
- local Hugging Face model behavior was not evaluated
- no clinician-labeled quality scores were used
- prompt quality likely affected LLM-assisted approaches
- the deterministic conflict checker only catches simple contradiction patterns

These limitations are acceptable for Phase 6 because the goal was to build the architecture, validate privacy and safety behavior, and compare approaches experimentally. Clinical validation remains future work.

---

## Future Work

Future work should focus on making the web evidence agent clinically stronger and more realistic.

1. **Run real SearxNG evaluation.** Configure `SEARXNG_BASE_URL` and repeat the one-question smoke test with a live self-hosted SearxNG backend.
2. **Evaluate `llama3.1:8b`.** Compare it against `llama3.2:latest` on JSON validity, latency, safety warnings, and claim coverage.
3. **Run the full 20-question local LLM evaluation.** The 5-question local run showed useful differences, but the full set is needed for stronger conclusions.
4. **Improve prompts.** The hybrid approach needs better claim extraction and synthesis prompts to reduce safety warnings and increase supported claim coverage.
5. **Add clinician validation.** Doctors should rate claim correctness, source usefulness, and summary usefulness.
6. **Add quote spans.** Every claim should include the exact supporting text span from the source.
7. **Cache source pages.** Cached evidence pages would make experiments reproducible even when websites change.
8. **Improve conflict detection.** The current deterministic conflict checker is simple and should be expanded.
9. **Add specialty-specific source policies.** Cardiology, infectious disease, oncology, and primary care may require different trusted source lists.
10. **Integrate doctor feedback.** Physician review outcomes should later feed into Medora's feedback and improvement loop.

---

## Scripts Reference

Run deterministic CLI in mock mode:

```bash
python -m agents.web_evidence.cli --question "latest guideline for community acquired pneumonia antibiotics in adults" --approach deterministic_only --mock
```

Run hybrid mode with mock local LLM:

```bash
python -m agents.web_evidence.cli --question "updated diabetes metformin contraindications" --approach hybrid_recommended --provider mock --mock
```

Run hybrid mode with local Ollama:

```bash
python -m agents.web_evidence.cli --question "latest guideline for community acquired pneumonia antibiotics in adults" --approach hybrid_recommended --provider ollama --model llama3.2:latest
```

Run the local Ollama comparison:

```bash
python evaluation/web_evidence_validation.py --mock --provider ollama --model llama3.2:latest --approaches deterministic_only llm_synthesis_only llm_claims_and_synthesis full_llm_council hybrid_recommended --max-questions 5
```

Run the full 20-question mock evaluation:

```bash
python evaluation/web_evidence_validation.py --mock --provider mock --approaches deterministic_only llm_synthesis_only llm_claims_and_synthesis full_llm_council hybrid_recommended
```

Run the SearxNG smoke test:

```bash
python -m agents.web_evidence.cli --question "latest CDC adult pneumococcal vaccine recommendations" --approach deterministic_only --json-output data/results/web_evidence_searxng_smoke_test.json
```

Use a self-hosted SearxNG instance:

```bash
set SEARXNG_BASE_URL=http://localhost:8080
python -m agents.web_evidence.cli --question "latest guideline for community acquired pneumonia antibiotics in adults" --approach deterministic_only
```

