# Phase 1.2: Chunking Strategy

**Project:** Medora — AI-powered medical Q&A system
**Phase:** 1.2 — Chunking Strategy
**Input:** `data/chunks/tmt_raw_sections.json` (7,942 sections from Phase 1.1)
**Output:** `data/chunks/tmt_chunks_structured.json` and `data/chunks/tmt_chunks_baseline.json`
**Status:** Complete
**Date:** March 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Why Chunks Matter for RAG](#why-chunks-matter-for-rag)
3. [Structure-Aware Chunking](#structure-aware-chunking)
   - [Step 1: Drop Empty Sections](#step-1-drop-empty-sections)
   - [Step 2: Short Section Merging](#step-2-short-section-merging)
   - [Step 3: Chunk Emission](#step-3-chunk-emission)
   - [Step 4: Junk Filtering](#step-4-junk-filtering)
   - [Step 5: Post-Merge of Short Chunks](#step-5-post-merge-of-short-chunks)
   - [Long Section Splitting](#long-section-splitting)
   - [Essentials of Diagnosis Boxes](#essentials-of-diagnosis-boxes)
   - [Chunk ID Format](#chunk-id-format)
4. [Baseline Fixed-Size Chunking](#baseline-fixed-size-chunking)
5. [Final Statistics](#final-statistics)
6. [Edge Cases Handled](#edge-cases-handled)
7. [Scripts Reference](#scripts-reference)
8. [Config Parameters](#config-parameters)

---

## Overview

Phase 1.2 takes the 7,942 structured sections produced by Phase 1.1 (PDF extraction) and converts them into chunks suitable for embedding and retrieval.

The raw sections from Phase 1.1 are not ready to be embedded directly. The core problem is size distribution: the sections vary enormously in length, from a single sentence to several thousand words. An embedding model that receives a 3,000-word section and a 15-word section produces vectors with very different informational density. When both vectors sit in the same retrieval index, the system's ability to match a user query to the right passage degrades, because the similarity scores are not comparable across such different input lengths.

Two distinct chunking strategies are implemented:

1. **Structure-aware chunking** — uses the book's section hierarchy to make intelligent splitting and merging decisions. This is the primary strategy intended for production use.
2. **Baseline fixed-size chunking** — a simple sliding-window approach that ignores structure. This exists purely as a comparison baseline for Phase 2 evaluation.

Both strategies share the same input (the raw sections JSON) and their outputs are written in parallel. The evaluation in Phase 2 will compare retrieval quality between them, providing a quantitative justification for the complexity of the structure-aware approach.

---

## Why Chunks Matter for RAG

### The relationship between chunk size and embedding quality

Embedding models compress a variable-length text passage into a fixed-size vector (for example, 1024 floating-point numbers). Every piece of information in the input passage must be encoded into that same fixed-size space. When a passage is very long, the embedding must represent many distinct ideas simultaneously, and individual ideas become diluted — the vector is pulled in many directions at once and ends up as a weak representative of any one of them.

When a passage is very short, the opposite problem arises. A 10-word sentence contains almost no context about what it is discussing. The embedding for "Management is usually conservative" has no way to convey that this refers to the management of a specific condition in a specific clinical context, because that context was not part of the input.

### The practical consequence for retrieval

When a user asks "What is the first-line treatment for community-acquired pneumonia in an outpatient?", the retrieval system computes the embedding of that question and searches for the nearest vectors in the index. If the relevant chunk is a 2,000-word section covering all aspects of pneumonia management, the embedding of that section points toward a general "pneumonia management" region of the vector space, not specifically toward the outpatient treatment subcomponent. The query, which is specific, may not find this chunk as its nearest neighbour.

Conversely, if the relevant passage was split into 50-word fragments, each fragment loses the surrounding sentences that give it meaning, and its embedding may be too weak and context-free to match the query reliably.

### The target range

Based on standard RAG practice and the nature of medical text, the target range for Medora's chunks is 100 to 500 words per chunk, with a soft target of around 200 words. At this size, a chunk typically contains one coherent clinical concept with sufficient surrounding context — for example, the treatment approach for a specific condition, the interpretation of a specific diagnostic test, or the clinical features of a specific disease presentation. This is also the size range where modern embedding models such as BGE-M3 are known to perform well.

---

## Structure-Aware Chunking

The structure-aware chunker operates in five sequential steps. The input to the chunker is the list of raw section dicts loaded from `tmt_raw_sections.json`. The output is a list of chunk dicts written to `tmt_chunks_structured.json`.

### Step 1: Drop Empty Sections

Before any merging or splitting, sections that carry no usable content are removed. A section is considered empty if it has zero words of main text AND no essentials box text. These occur at chapter transition boundaries and at the end of certain subsections where the parser flushed a section record that happened to capture only whitespace.

```
active_sections = [
    s for s in raw_sections
    if not (s["word_count"] == 0 and not s.get("essentials_text", "").strip())
]
```

A section that has zero words in its `text` field but does have a populated `essentials_text` field is retained, because the essentials box is itself a meaningful chunk that will be emitted separately in Step 3.

In practice, this step removes a small number of records — the exact count is printed at runtime. The remaining sections are referred to as `active_sections`.

### Step 2: Short Section Merging

Many raw sections are very short — a subsection heading followed by a single sentence, or a brief note at the end of a topic. These short sections have word counts below `CHUNK_MIN_WORDS` (50 words). Embedding a 20-word sentence in isolation would produce a weak, context-poor vector.

The merging strategy is a single forward pass through `active_sections`. A short section is held in a `pending` variable rather than immediately emitted. The algorithm then examines the next section in the list:

- If the next section shares the same `(chapter, section)` group as the pending section, the pending text is prepended to the next section's text, and the page ranges are unioned. The two sections become one combined section.
- If the next section is in a different `(chapter, section)` group, the pending section cannot be merged across a section boundary. It is flushed as-is and tagged as `kept_short`.

The `(chapter, section)` boundary constraint is important. Without it, a short section at the end of one clinical topic might be incorrectly merged with a section from an entirely different topic, producing a chunk that mixes unrelated content. By restricting merging to within the same section, the clinical coherence of each chunk is preserved.

The page range of the merged section is computed as:

```
merged_page_range = [
    min(pending["page_range"][0], current["page_range"][0]),
    max(pending["page_range"][1], current["page_range"][1]),
]
```

After the forward pass, any remaining pending section (the last section in the book, or the last section in a group with no valid merge target) is flushed as `kept_short`.

### Step 3: Chunk Emission

After short-section merging, the algorithm iterates through `merged_sections` and emits chunks. Three distinct cases arise for each section:

**3a — Essentials box chunks:**
If a section has `has_essentials_box == True` and a non-empty `essentials_text` field, an essentials chunk is emitted first, before any main-text chunk from this section. The essentials chunk is given `subsection = "Essentials of Diagnosis"` and receives its own chunk ID. See the [Essentials of Diagnosis Boxes](#essentials-of-diagnosis-boxes) section for the rationale.

**3b — Long section splitting:**
If the main text word count exceeds `CHUNK_MAX_WORDS` (500 words), the section is split into multiple sub-chunks using `split_long_section()`. Each sub-chunk gets a separate chunk ID with an incrementing sequence number. See the [Long Section Splitting](#long-section-splitting) section for the full algorithm.

**3c — Normal sections:**
If the main text is between `CHUNK_MIN_WORDS` and `CHUNK_MAX_WORDS` words (inclusive), it is emitted as a single chunk without modification.

Sections that were tagged `kept_short` (below 50 words with no merge target) also fall through to case 3c and are emitted as-is. They are retained because short clinical statements — such as "All patients with orbital cellulitis must be referred emergently" — carry genuine medical information despite their brevity, and filtering them by word count alone would lose real content.

### Step 4: Junk Filtering

After all chunks are emitted from the merging and splitting steps, some chunks are artifacts of PDF extraction rather than real medical content. These are identified and removed by a set of content-based rules.

The junk filter removes a chunk if any of the following conditions are true:

| Criterion | Rationale |
|---|---|
| Word count < 5 | Too short to carry any meaning; almost certainly a stray header or page number |
| Contains "CMDT 2022" | This is a running page header that appears at the top of many pages in the PDF; the parser occasionally captures it as body text |
| Text matches `^[\d\s.]+$` | Pure digits and whitespace — a page number captured in isolation |
| Chapter is "Index" | The book index is an alphabetical list of terms and page numbers; it has no clinical value for retrieval |

Critically, the filter is based on content rather than word count alone. This distinction matters: a chunk of 8 words that is a genuine clinical instruction should be kept; a chunk of 8 words that is a page header artifact should be removed. The "CMDT 2022" substring check and the pure-digits regex handle artifact detection without collateral damage to legitimate short content.

The `is_junk_chunk()` function is applied as a list comprehension filter over all emitted chunks, and the count of removed chunks is printed.

### Step 5: Post-Merge of Short Chunks

Even after the earlier merging pass, some chunks may end up below `CHUNK_MIN_WORDS` (50 words) after junk filtering. This can happen, for example, when a `kept_short` section was not merged earlier because no valid forward neighbour existed in the same group.

A second pass over the chunk list addresses this. For each chunk whose word count is below 50, the algorithm checks whether the immediately preceding chunk in the output list shares the same `(chapter, section)`. If so, the short chunk's text is appended to the previous chunk, the word counts are updated, and the page ranges are merged. The short chunk is consumed into the preceding chunk.

If the short chunk is the first chunk from its `(chapter, section)` group (i.e. no preceding chunk in the same section), it is kept as-is rather than merged across a section boundary.

This pass operates in-place on the accumulating `merged_chunks` list:

```python
for chunk in chunks:
    if chunk["word_count"] < CHUNK_MIN_WORDS and merged_chunks:
        prev = merged_chunks[-1]
        same_section = (
            prev["chapter"] == chunk["chapter"]
            and prev["section"] == chunk["section"]
        )
        if same_section:
            # Append short chunk text to the previous chunk
            ...
            merged_chunks[-1] = prev
        else:
            merged_chunks.append(chunk)
    else:
        merged_chunks.append(chunk)
```

After this step, the final list of structured chunks is complete.

---

### Long Section Splitting

Long sections (over 500 words) are split by `split_long_section()`, which uses a two-pass strategy. The goal is to produce sub-chunks that are each close to `CHUNK_TARGET_WORDS` (200 words) without exceeding `CHUNK_MAX_WORDS` (500 words).

#### Pass 1: Paragraph Boundaries

The section text is split on double newlines (`\n{2,}`), which separate paragraphs in the extracted TMT text. Each paragraph is treated as a segment. Empty segments are discarded.

These paragraph segments are then fed into `merge_to_target()`, a greedy packing algorithm that assembles them into sub-chunks.

#### The `merge_to_target` Algorithm

`merge_to_target(segments, target_words, max_words)` walks through the list of segments and accumulates them into a running buffer:

1. If a segment's word count is zero, skip it.
2. If a single segment exceeds `max_words` on its own, flush any current buffer first, then force-split the segment by word boundaries (see below).
3. If adding the next segment to the current buffer would cause the buffer to exceed `max_words`, flush the buffer as a sub-chunk and start a new buffer.
4. Otherwise, append the segment to the buffer.
5. After the final segment, flush any remaining buffer content.

The result is a list of sub-chunk strings, each within the `max_words` ceiling, packed as fully as possible given the paragraph boundaries.

#### Pass 2: Sentence Boundaries

If any sub-chunk produced by the paragraph pass is still over `CHUNK_MAX_WORDS` — meaning a single paragraph is very long with no double-newline breaks — that sub-chunk is fed back through `merge_to_target()` using sentence-level segments instead of paragraph-level segments.

Sentences are split by `split_into_sentences()`, which uses the pattern `(?<=[.!?])\s+(?=[A-Z])` — a period, exclamation mark, or question mark followed by whitespace and a capital letter. This pattern correctly identifies most sentence boundaries in English medical prose without splitting on abbreviations like "Dr." or unit expressions like "10 mg i.v." (which typically do not have a capital letter immediately after the period).

The sentence segments are packed by `merge_to_target()` with the same `max_words` ceiling.

#### `force_split_by_words`: Last Resort

If a segment is too large even at the sentence level — for example, a dense enumeration like the book index that contains no punctuation patterns the sentence splitter can act on — `force_split_by_words(text, max_words)` is used as a last resort. It simply chops the word list into consecutive slices of at most `max_words` words each, with no regard for sentence boundaries. This ensures the `CHUNK_MAX_WORDS` ceiling is never violated, even in degenerate cases.

```python
def force_split_by_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    chunks = []
    for start in range(0, len(words), max_words):
        part = " ".join(words[start : start + max_words])
        if part:
            chunks.append(part)
    return chunks
```

This function is called inside `merge_to_target()` whenever a single segment exceeds `max_words`, so it is never called at the top level by user code.

---

### Essentials of Diagnosis Boxes

The TMT textbook opens most disease-specific chapters with a boxed section titled "Essentials of Diagnosis". This box lists the key diagnostic criteria for the condition in a bulleted format — for example, the classic symptoms, the confirmatory laboratory findings, and the distinguishing clinical features.

During Phase 1.1 PDF extraction, the Essentials box was detected by its distinct font (a slightly larger, separate font from the body text) and captured into a dedicated `essentials_text` field on the section record, alongside the main `text` field. The `has_essentials_box` flag on the section record indicates whether an essentials box is present.

In Phase 1.2, each section with a populated essentials box emits a separate chunk with `subsection = "Essentials of Diagnosis"`. This chunk is emitted before the main-text chunk for the same section, so it appears first in the output list.

The rationale for treating essentials boxes as their own chunks:

1. **High retrieval value.** Diagnostic criteria boxes are exactly what a clinician asks about when entering a diagnostic query ("What are the criteria for diagnosing X?"). If the essentials text is mixed into the main section text, its signal is diluted by the surrounding prose.
2. **Structural coherence.** The essentials box is a distinct, self-contained element in the book. Treating it as a separate chunk preserves the editorial intent of the textbook authors.
3. **Consistent format.** Every essentials box is a bulleted list with a consistent structure. Its embedding will occupy a distinctive region of the vector space, making it easier for the retrieval system to surface it for diagnostic queries.

---

### Chunk ID Format

Every structured chunk is assigned a deterministic, hierarchical ID of the form:

```
tmt::<chapter_slug>::<section_slug>::<subsection_slug>::<N>
```

Examples:
```
tmt::heart_disease::heart_failure::clinical_findings::1
tmt::pulmonary_disorders::pneumonia::essentials_of_diagnosis::1
tmt::pulmonary_disorders::pneumonia::main::2
```

The `<N>` suffix is a sequence number that starts at 1 within each `(chapter, section, subsection)` group. When a long section is split into multiple sub-chunks, they share the same prefix and differ only in `N`. When a section has no subsection, the subsection slug defaults to `"main"`.

The slug for each component is produced by `slugify(text, max_len=40)`:

1. Strip leading bullet characters (`»`, `º`, `▸`, `•`, `-`, `–`, `—`) that appear in TMT subsection names.
2. Lowercase the text.
3. Replace whitespace with underscores.
4. Remove all characters that are not alphanumeric or underscore.
5. Collapse consecutive underscores into one.
6. Truncate to `max_len` characters (default 40).

Examples of slug conversion:

| Input | Slug |
|---|---|
| `"Heart Failure"` | `heart_failure` |
| `"» Clinical Findings"` | `clinical_findings` |
| `"Essentials of Diagnosis"` | `essentials_of_diagnosis` |
| `"Community-Acquired Pneumonia"` | `community_acquired_pneumonia` |

These IDs are deterministic: running the chunker twice on the same input always produces the same IDs. This is important for reproducibility and for any downstream system that stores or references chunks by ID.

---

## Baseline Fixed-Size Chunking

The baseline chunker is a deliberately simple implementation whose only purpose is to provide a comparison point for Phase 2 evaluation. It does not use any structural information from the book.

### How it works

**Step 1 — Concatenate all text.** The text from all raw sections is concatenated into one long string. A separator `\n\n[chapter / section]\n` is inserted between sections for readability and to make the boundary visible in the text. The character offset of each section's text within this concatenated string is recorded in a `section_spans` list.

**Step 2 — Split into a word list.** The full concatenated string is split by whitespace into a flat list of words. The total word count of this corpus is printed.

**Step 3 — Slide a fixed window.** A window of `BASELINE_CHUNK_WORDS` (200) words is slid across the word list with a stride of `BASELINE_CHUNK_WORDS - BASELINE_OVERLAP_WORDS` (200 - 50 = 150 words). Each window position produces one chunk.

The stride is less than the window size, which means consecutive chunks overlap by `BASELINE_OVERLAP_WORDS` (50) words. This overlap is important: without it, a sentence that happens to straddle a window boundary would be split in two, with each half in a different chunk and neither half being coherent on its own.

**Step 4 — Assign metadata.** Each chunk is assigned chapter and section metadata from whichever section contains the chunk's starting word position. The mapping from word index to section is estimated proportionally:

```
estimated_char = int(word_idx / total_words * len(full_text))
```

This is an approximation. The character position is proportionally mapped from the word index position in the full word list, then matched against the `section_spans` list by a linear scan. For the purpose of baseline metadata (chapter name, section name), this approximation is sufficient — the exact character-level precision of the structured chunker is not required here.

**Chunk IDs** for baseline chunks follow the format `tmt_baseline::<NNNN>` where `NNNN` is a zero-padded sequential integer (e.g., `tmt_baseline::0001`).

### Why overlap matters

Without overlap, a 200-word window cuts at an arbitrary word boundary. The sentence that was in progress at the cut point is split: its first part is the last sentence (incomplete) of chunk N, and its continuation is the first words (mid-sentence) of chunk N+1. Neither chunk contains a coherent sentence at its boundary.

With a 50-word overlap, each chunk's last 50 words are repeated as the first 50 words of the next chunk. A sentence that straddles the strict boundary appears in full in at least one of the two chunks.

The trade-off is that the baseline produces more total chunks (8,216 vs 4,557 for the structured approach), and many words appear in two chunks rather than one. This inflates the index size and increases redundancy, but ensures that no sentence is permanently fragmented.

---

## Final Statistics

### Structured chunks

| Metric | Value |
|---|---|
| Total chunks | 4,557 |
| Mean word count | 189 words |
| Median word count | 148 words |
| Minimum word count | 5 words |
| Maximum word count | 500 words |

### Baseline chunks

| Metric | Value |
|---|---|
| Total chunks | 8,216 |
| Mean word count | ~200 words |
| Median word count | ~200 words |
| Minimum word count | <200 (final window) |
| Maximum word count | 200 words |

### Comparison

| Property | Structured | Baseline |
|---|---|---|
| Total chunks | 4,557 | 8,216 |
| Mean words per chunk | 189 | ~200 |
| Median words per chunk | 148 | ~200 |
| Respects section boundaries | Yes | No |
| Essentials boxes separated | Yes | No |
| Chapter/section metadata | Full hierarchy | Approximate |
| Overlap between chunks | No | 50 words |

The structured approach produces roughly half as many chunks as the baseline, because it merges short sections rather than emitting them as undersized windows, and it uses the natural section boundaries of the book as splitting points. The baseline's chunk count is higher because the 150-word stride means every 200-word region of the corpus is covered by at least one chunk (with overlapping coverage at boundaries).

The quality comparison between these two approaches is the subject of Phase 2 evaluation.

---

## Edge Cases Handled

### Index chapter excluded

The TMT textbook ends with an alphabetical Index chapter — a dense list of medical terms paired with page numbers. This content has no clinical value for retrieval: a user querying about a diagnosis or treatment will not benefit from finding an index entry in response. The entire Index chapter is excluded during the junk-filtering step (Step 4) by checking `chunk["chapter"] in {"Index"}`.

Without this exclusion, the index entries would produce thousands of short, meaningless chunks that would pollute the retrieval index with noise.

### Junk artifacts filtered by content

Junk filtering uses content-based rules rather than a word-count threshold. This is a deliberate design choice: filtering by word count alone would remove short but clinically important statements. The rules target specific artifact patterns (running headers containing "CMDT 2022", pure-digit page numbers, fragments under 5 words) without touching legitimate short content.

### Short chunks merged to preserve information

Two separate merging passes handle short content:

- **Forward merge (Step 2):** Applied before chunk emission, merges short raw sections into their successor within the same `(chapter, section)` group.
- **Post-merge (Step 5):** Applied after chunk emission and junk filtering, appends short residual chunks to the preceding chunk in the same section.

Together these passes ensure that short fragments of clinical text are embedded with enough surrounding context to produce useful vectors, while maintaining section-level boundaries so that unrelated topics are not mixed.

### Oversized chunks force-split at word boundaries

The `force_split_by_words()` fallback ensures that the `CHUNK_MAX_WORDS` ceiling is never violated, even for input that has no paragraph breaks and no recognizable sentence boundaries. This fallback is rarely triggered on normal medical prose, but it is essential for robustness against unexpected input formats.

---

## Scripts Reference

### `data_processing/chunker.py`

The main chunking script. Reads the raw sections JSON, runs both the structure-aware and baseline chunking strategies, writes both output files, and prints a quality report.

**Usage:**
```bash
python data_processing/chunker.py
```

No command-line arguments are required. All paths and parameters are read from `config.py`.

**Input:**
```
data/chunks/tmt_raw_sections.json
```

**Outputs:**
```
data/chunks/tmt_chunks_structured.json
data/chunks/tmt_chunks_baseline.json
```

**Runtime output (example):**
```
============================================================
Medora — TMT Structure-Aware Chunker
============================================================

Loading raw sections from:
  data/chunks/tmt_raw_sections.json
Loaded 7,942 raw sections

--- STRUCTURED CHUNKING ---
[structured] Starting structure-aware chunking …
[structured] Dropped 12 empty sections — 7,930 remain
[structured] After short-section merging: 6,841 sections
[structured] Dropped 187 junk chunks (page numbers, headers, <5 words) — 4,612 remain
[structured] Merged 55 short chunks (<50 words) into previous chunk (same section) — 4,557 final chunks
[structured] Total structured chunks produced: 4,557

--- BASELINE CHUNKING ---
[baseline] Concatenating all section text …
[baseline] Total words in concatenated corpus: 1,423,817
[baseline] Sliding window: size=200, overlap=50, stride=150
[baseline] Total baseline chunks produced: 8,216
```

**Chunk schema (structured):**
```json
{
  "chunk_id":   "tmt::pulmonary_disorders::pneumonia::clinical_findings::1",
  "source":     "TMT_2022",
  "chapter":    "Pulmonary Disorders",
  "section":    "Pneumonia",
  "subsection": "Clinical Findings",
  "text":       "Fever, chills, and pleuritic chest pain are common presenting symptoms...",
  "page_range": [512, 515],
  "word_count": 193,
  "chunk_type": "structured"
}
```

**Chunk schema (baseline):**
```json
{
  "chunk_id":   "tmt_baseline::0042",
  "source":     "TMT_2022",
  "chapter":    "Pulmonary Disorders",
  "section":    "Pneumonia",
  "text":       "Fever chills and pleuritic chest pain are common presenting symptoms...",
  "word_count": 200,
  "chunk_type": "baseline"
}
```

Note that baseline chunks do not have `subsection` or `page_range` fields, because the sliding-window approach has no mechanism to recover subsection boundaries or exact page positions.

---

## Config Parameters

All chunking parameters are defined in `config.py` at the project root. Changing a parameter and re-running `chunker.py` will regenerate both output files.

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_MIN_WORDS` | `50` | Sections with fewer than this many words are treated as "short" and are candidates for forward merging. Post-emission chunks below this threshold are also candidates for backward post-merging. |
| `CHUNK_MAX_WORDS` | `500` | Sections with more than this many words are split into sub-chunks. The `merge_to_target` algorithm uses this as a hard ceiling that no sub-chunk may exceed. |
| `CHUNK_TARGET_WORDS` | `200` | The soft target size when splitting long sections. Passed to `merge_to_target()` as the `target_words` argument. Currently used for documentation purposes; the hard ceiling is what drives the actual split points. |
| `BASELINE_CHUNK_WORDS` | `200` | The fixed window size for the baseline chunker, in words. |
| `BASELINE_OVERLAP_WORDS` | `50` | The number of words of overlap between consecutive baseline chunks. The stride is `BASELINE_CHUNK_WORDS - BASELINE_OVERLAP_WORDS`. |

### How to adjust and re-run

1. Open `config.py` and modify the relevant parameter.
2. Run `python data_processing/chunker.py` from the project root.
3. Both output files in `data/chunks/` will be overwritten with the new chunks.
4. Check the quality report printed to stdout to verify the distribution is as expected.

**Example: tighten the size range.** To target a narrower chunk size band (e.g., 75–300 words with a 150-word target), set:
```python
CHUNK_MIN_WORDS    = 75
CHUNK_MAX_WORDS    = 300
CHUNK_TARGET_WORDS = 150
```

**Example: wider baseline window.** To experiment with a 300-word baseline window and 75-word overlap:
```python
BASELINE_CHUNK_WORDS   = 300
BASELINE_OVERLAP_WORDS = 75
```

Increasing `CHUNK_MAX_WORDS` will reduce the number of long-section splits, producing fewer, longer structured chunks. Decreasing `CHUNK_MIN_WORDS` will cause fewer sections to be treated as short, reducing the number of forward merges. The optimal values for retrieval quality will be determined by the Phase 2 evaluation results.

---

*End of Phase 1.2 documentation.*
