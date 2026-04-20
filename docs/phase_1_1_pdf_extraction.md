# Phase 1.1: Smart PDF Extraction (TMT)

**Project:** Medora — AI-powered medical Q&A system
**Phase:** 1.1 — Smart PDF Extraction
**Source text:** Current Medical Diagnosis and Treatment (CMDT), a.k.a. "TMT", 2022 edition
**Status:** Complete
**Date:** March 2026

---

## Table of Contents

1. [Overview](#overview)
2. [The Problem: Garbled Characters (PUA)](#the-problem-garbled-characters-pua)
3. [The Solution: Dictionary-Based Character Mapping](#the-solution-dictionary-based-character-mapping)
4. [LLM-Assisted Resolution (Final 27 Entries)](#llm-assisted-resolution-final-27-entries)
5. [Font-Based Text Classification](#font-based-text-classification)
6. [Two-Column Layout Handling](#two-column-layout-handling)
7. [The State Machine (Structure Building)](#the-state-machine-structure-building)
8. [Output Format](#output-format)
9. [Scripts Reference](#scripts-reference)
10. [Verification](#verification)
11. [Limitations and Notes](#limitations-and-notes)

---

## Overview

### What this phase does and why it was needed

Phase 1.1 extracts the full text content of the TMT medical textbook (a 1,874-page PDF) and transforms it into a structured JSON dataset that can be used as the knowledge base for Medora's retrieval-augmented generation (RAG) pipeline.

The goal is not simply to dump raw text out of the PDF. A medical question-answering system needs to know *what* a piece of text is — is it a chapter introduction, a diagnostic criteria box, a reference list entry, or body text? Without this structure, all text looks identical to the retrieval system, which leads to poor chunk quality and irrelevant answers. The extraction phase exists to recover that structure from the PDF before it is lost.

The TMT textbook is organized into approximately 50 chapters, each containing multiple named sections (e.g., "Essentials of Diagnosis", "General Considerations", "Clinical Findings", "Treatment"). Each section is the natural unit of medical knowledge — it is coherent, self-contained, and the correct granularity for retrieval. The output of this phase is a list of these sections as structured JSON objects.

### The old approach and its failures

The initial naive approach used PyMuPDF's `get_text("text")` method, which returns a flat string for each page with newlines at line breaks. Section headings were detected by applying regular expressions to guess which lines looked like titles — for example, checking whether a line was short, title-cased, and followed by body text.

This approach had two fundamental problems:

**First**, heading detection via regex is fragile. Medical headings do not follow a single consistent pattern. "Essentials of Diagnosis" is clearly a heading, but so is "When to Refer", "When to Admit", "Prognosis", "PNEUMONIA", and hundreds of others. A regex that catches all of them without false positives is extremely difficult to write, and any that was written broke on edge cases. The result was that headings were missed or body sentences were misclassified as headings.

**Second**, and more severely, many characters in the extracted text were garbled — they appeared as private-use Unicode symbols (for example, ``) instead of letters like `fi`, `fl`, or other ligatures and special characters. This meant the raw text was full of unreadable noise that no language model could interpret correctly. The regex approach offered no way to fix this.

### The new approach

The solution is to use PyMuPDF's `get_text("dict")` method instead of `get_text("text")`. This method returns a structured dictionary for every page, breaking text down into blocks, lines, and individual spans. Each span includes not just the text characters, but also metadata about the font: its name, its size in points, and the bounding box (position on the page) of that span.

This font metadata is the key insight. In the TMT PDF, the publisher used different fonts for different types of content — headings are in a different font than body text, which is in a different font than the Essentials of Diagnosis boxes, which are different from references. By reading the font name and size of each span, the parser can classify text far more reliably than any regex could.

The full pipeline for Phase 1.1 is:

1. Explore the PDF structure to identify which fonts are used for which content types.
2. Build a character mapping to fix garbled (PUA) characters, font by font.
3. Use GPT-4o to resolve the small number of entries that the automated method cannot fix.
4. Run the main parser, which uses font analysis to classify text and a state machine to build the section hierarchy.
5. Verify the output is clean and contains no remaining garbled characters.

---

## The Problem: Garbled Characters (PUA)

### What PUA characters are and why they appear in PDFs

PUA stands for **Private Use Area** — a range of Unicode code points (U+E000 to U+F8FF, and extensions) that are not assigned to any standard character. They exist so that software vendors can define their own custom characters for internal use. In a well-formed document, PUA characters would never appear in user-visible text.

However, they appear frequently when extracting text from PDFs, and the reason is specific to how PDFs store text internally.

### How PDFs store text

A PDF does not store text as readable Unicode strings the way a plain text file does. Instead, it stores a sequence of **glyph IDs** — integers that index into the font's glyph table. The font's glyph table contains the actual shapes (visual forms) of characters, and a separate data structure called a **ToUnicode** mapping table tells the PDF viewer which Unicode character each glyph ID corresponds to.

When a PDF is displayed on screen, the viewer looks up each glyph ID in the ToUnicode table to get the Unicode character, and then renders the glyph shape at the right position. The text appears correctly because the viewer uses the ToUnicode table.

When a text extraction tool like PyMuPDF reads the PDF, it also uses the ToUnicode table to convert glyph IDs to Unicode characters. The problem arises when the publisher created the PDF with a font that has an **incorrect or incomplete ToUnicode table**. This is common with commercial publishing workflows that embed custom fonts with remapped glyph tables — a technique sometimes used for copy-protection or for handling ligatures and special characters.

In the TMT PDF, certain glyphs — particularly ligatures like `fi`, `fl`, `ffi`, and occasionally individual letters — are mapped in the ToUnicode table to PUA code points rather than to the correct Unicode characters. When PyMuPDF reads these glyphs, it faithfully returns whatever the ToUnicode table says, which is the PUA code point. The result is garbled text.

### Why different fonts produce different garbled characters for the same hex code

This is a critical subtlety. The TMT PDF uses multiple embedded fonts, and each font has its own ToUnicode table. The same PUA code point (for example, U+F001) may be mapped to the `fi` ligature in one font, but to the letter `f` in another font, and to something else entirely in a third font.

This means that a single global search-and-replace — "replace all U+F001 with `fi`" — would be incorrect and would corrupt text from fonts where U+F001 means something different. The mapping must be built and applied **per font**. A character mapping that is valid for font `f1` is not valid for font `f27`.

### The scale of the problem

An initial scan of the raw extracted text across all 1,874 pages found **16,345 garbled entries** containing PUA characters. These appeared throughout the text, not just in one section. Common patterns included:

- `ﬁ` appearing where `fi` was expected (e.g., "deﬁnition" instead of "definition")
- `ﬂ` appearing where `fl` was expected (e.g., "inﬂammation" instead of "inflammation")
- Single garbled characters where individual letters had been remapped

Without fixing these, the extracted text would be full of noise, breaking tokenization and embedding quality downstream.

---

## The Solution: Dictionary-Based Character Mapping

### The core algorithm

The automated PUA mapping algorithm works on a word-by-word basis. For each word in the extracted text that contains a PUA character, the algorithm attempts to recover the intended word by trying every possible letter substitution.

The process for a single garbled word is:

1. Identify which characters in the word are PUA (garbled).
2. For each garbled character, try replacing it with each letter `a` through `z` (and also common digraphs like `fi`, `fl`, `ff`, `ffi`, `ffl` for ligature candidates).
3. For each candidate substitution, check whether the resulting word exists in a standard English dictionary.
4. If exactly one substitution produces a valid dictionary word, record the mapping: `(font_name, pua_code_point) -> correct_character`.
5. If zero or multiple substitutions are valid, the word is ambiguous and is flagged for manual review.

This works well because English spelling is highly constrained. Given a word like `deﬁnition` where one character is garbled, there are only 26 + 5 = 31 candidates to try, and only one of them (`fi`) produces a word that exists in the dictionary (`definition`). The probability of a false match is very low for medical text, because medical terms tend to be long and their spelling is well-defined.

### Why this works deterministically

The key insight is that medical English — while full of technical vocabulary — is still constrained by the same character patterns as general English. Words like "inflammation", "hypertension", "fibrosis", and "lymphocyte" all contain the `fi` and `fl` ligatures in predictable positions. Because the English dictionary covers most of the base words used in medicine (the Latin and Greek roots are shared with standard English vocabulary), the dictionary lookup approach resolves the majority of PUA characters correctly.

For the cases where the word is a rare medical term not in the standard dictionary (such as "Hydatidiform"), the automated method fails and the entry is flagged. These are handled separately by the LLM step described below.

### Per-font mapping: why it is necessary

Because the same PUA code point can mean different things in different fonts (as explained above), the mapping is stored as a dictionary keyed by `(font_name, code_point)` pairs, not just by `code_point` alone.

During the mapping build phase, the algorithm groups all garbled words by the font name of the span they appear in. Within each font group, it builds the substitution table independently. This ensures that if font `f1` uses U+F001 for `fi` but font `f27` uses U+F001 for `f`, the two mappings do not interfere with each other.

### The iterative process

Building the character map was an iterative process, not a single pass. This is because the algorithm can only resolve a garbled word if exactly one character is garbled. Words with two or more garbled characters are ambiguous — there are too many combinations to check exhaustively.

**First pass:** The algorithm runs on all 16,345 garbled entries. It successfully resolves words that have exactly one garbled character. This produces **114 confirmed character mappings**. After applying these 114 mappings back to the text, some previously ambiguous words (which had two garbled characters) now have only one garbled character remaining — because one of their garbled characters has been resolved and corrected.

**Second pass:** The algorithm runs again on the remaining flagged entries. With the corrected text from the first pass, more words are now resolvable. This produces an additional batch of mappings, bringing the total to **218 mappings**. After applying these, the remaining garbled count drops from 16,345 to 65.

**Final manual fixes:** A small number of mappings were resolved by direct inspection of the remaining 65 entries — cases where the pattern was visually obvious (for example, a standalone garbled character that clearly corresponded to a hyphen or a dash in context). These were added directly to the mapping table.

After all three rounds, the garbled entry count was: **16,345 → 65 → 27 → 0**.

The final 27 that the dictionary method could not resolve were sent to GPT-4o.

---

## LLM-Assisted Resolution (Final 27 Entries)

### Why the dictionary method fails for rare medical terms with multiple garbled chars

The 27 remaining entries all shared a common characteristic: they were rare or highly specialized medical terms that (a) were not in the standard English dictionary, and (b) contained more than one garbled character, making exhaustive substitution impractical.

For example, a term like "Hydatidiform mole" — a type of gestational trophoblastic disease — was represented in the raw extracted text as something like `hydaidifoxm moly`. The word `hydatidiform` is not in the standard English dictionary, so even if the algorithm tried all single-character substitutions, it would find no valid dictionary word and would be unable to resolve the entry. With two or three garbled characters in one word, the search space becomes too large.

### How GPT-4o was used

For these 27 entries, the garbled text was sent to GPT-4o along with contextual information: the chapter name, the surrounding sentences, and an explanation that the text is from a medical textbook and contains garbled ligature characters that need to be corrected.

The prompt was structured to ask GPT-4o to return the corrected version of each garbled phrase. Because GPT-4o has extensive medical knowledge, it can recognize what a garbled string is likely trying to say even when multiple characters are wrong. It effectively acts as a very powerful spell-checker with medical domain knowledge.

The model's output was a corrected text string for each of the 27 entries. These corrections were reviewed manually to confirm they were plausible, and then applied as patches to the extracted JSON.

### Examples

| Garbled text | Corrected text |
|---|---|
| `hydaidifoxm moly` | `Hydatidiform mole` |
| `ﬁbﬁnolysis` | `fibrinolysis` |
| `ﬂuoroquinoﬂone` | `fluoroquinolone` |

(Note: the exact garbled strings in your output may differ from the examples above, which are illustrative.)

### Cost

The 27 entries were sent in two API calls to GPT-4o. The total token usage was small (the entries are short phrases, not paragraphs), making the cost negligible — well under $0.10. The LLM step is used only as a last resort for the residual edge cases that the deterministic method cannot handle.

---

## Font-Based Text Classification

### How get_text("dict") works

When `page.get_text("dict")` is called on a PyMuPDF page object, it returns a Python dictionary with the following hierarchy:

- **Page** contains a list of **blocks**
- Each **block** (if it is a text block, not an image block) contains a list of **lines**
- Each **line** contains a list of **spans**
- Each **span** contains: the text string (`text`), the font name (`font`), the font size in points (`size`), the font flags (bold, italic, etc.), and the bounding box (`bbox`) — a tuple of (x0, y0, x1, y1) in page coordinates

This is fundamentally richer than `get_text("text")`, which throws away all of this metadata and returns only the concatenated characters. With the span-level metadata, the parser can make decisions based on how text looks on the page, not just what the characters say.

### The classification rules

Through exploration of the TMT PDF (using `tmt_explore.py`), the following font patterns were identified:

| Font size | Font name pattern | Classification | Action |
|---|---|---|---|
| 10 – 11 pt | Heading font family | Section heading | Start a new section |
| 11.5 pt | `f27` (or equivalent) | Essentials of Diagnosis box | Capture separately as `essentials_text` |
| 9 pt | Body font family | Body text | Append to current section |
| 8 pt | Any | References / bibliography | Skip entirely |
| 6 pt or less | Any | Superscript (footnote numbers) | Skip entirely |

These thresholds were determined empirically by examining the font metadata of known headings and body paragraphs in several chapters. The thresholds are specific to the TMT 2022 edition — a different edition or a different publisher's PDF may use different font sizes.

The Essentials of Diagnosis box is a special recurring feature of the TMT textbook. At the start of each disease chapter, there is a boxed section that lists the key diagnostic criteria in a bulleted format, set in a slightly larger, distinct font. Capturing this separately is important because it is high-value clinical information that deserves its own metadata field.

### Why font size is more reliable than regex for heading detection

A regex approach must infer structure from the characters themselves — for example, assuming that a short line in title case is a heading. But many body sentences are also short, and many headings are long. The font approach does not look at what the characters say at all — it looks at the rendering properties of the text. The publisher has already encoded the structure in the font choices; the parser simply reads that encoding. This makes font-based classification nearly immune to the edge cases that break regex-based approaches.

---

## Two-Column Layout Handling

### How the TMT book uses two columns per page

The body text of the TMT textbook is laid out in two columns per page. The left and right columns are essentially independent streams of text — the left column runs from top to bottom, then the right column runs from top to bottom. A naive extraction that processes blocks in their internal PDF order may interleave lines from the left column with lines from the right column, producing scrambled text.

### How blocks are sorted by x-position relative to page midpoint

The parser handles this by sorting blocks before processing them. For each page, the midpoint of the page width is computed. Each block's horizontal center is compared to this midpoint. Blocks whose center is to the left of the midpoint are assigned to the left column; blocks whose center is to the right are assigned to the right column.

Within each column, blocks are sorted by their vertical position (the y-coordinate of their top edge), top to bottom. The parser then processes all left-column blocks in order, followed by all right-column blocks in order.

### Reading order: left column top-to-bottom, then right column

This reconstruction of reading order is essential for producing coherent text. Without it, a section that spans the bottom of the left column and the top of the right column would be split and its text interleaved with the right column's content above it. The resulting text would be nonsensical.

The column-sorting logic is applied at the page level, before the block iteration loop. This keeps the logic centralized and easy to adjust if a different textbook uses a different layout (for example, a single-column layout would require no sorting at all, or a three-column layout would require a different partitioning approach).

---

## The State Machine (Structure Building)

### How the parser walks through pages maintaining current section state

The parser uses a simple state machine to build the section hierarchy as it reads through the document page by page, block by block, span by span.

The state machine tracks the following state variables at any given point during parsing:

- `current_chapter_num` — the index of the chapter currently being parsed
- `current_chapter_title` — the title string of the current chapter
- `current_section` — the title of the current section (e.g., "General Considerations")
- `current_subsection` — the title of the current subsection, if any
- `current_text_buffer` — a list of text strings accumulated for the current section
- `current_page_start` — the page number where the current section began
- `essentials_buffer` — text accumulated for the Essentials of Diagnosis box, if one is open

### Chapter boundaries from doc.get_toc()

PyMuPDF provides a `doc.get_toc()` method that returns the table of contents of the PDF as a list of `(level, title, page_number)` tuples. The TMT 2022 edition has a well-formed table of contents with entries for all 50 chapters.

The parser uses this TOC to know when a new chapter begins. When the current page number crosses a chapter boundary (as recorded in the TOC), the parser closes the current chapter and opens a new one. This is more reliable than trying to detect chapter boundaries from the text content itself, because chapter title pages can have varied formatting.

The 50 chapters cover the major organ systems and disease categories of internal medicine: cardiology, pulmonology, gastroenterology, infectious disease, oncology, and so on.

### Section detection from font analysis

Within a chapter, section boundaries are detected purely by font analysis. When a span is classified as a heading (based on the font size and font name rules described above), the parser treats it as the start of a new section.

The parser distinguishes between top-level sections (e.g., "Essentials of Diagnosis", "General Considerations", "Clinical Findings") and subsections (e.g., "Symptoms and Signs", "Laboratory Findings") based on font size differences within the heading category. Larger heading fonts indicate a top-level section; slightly smaller heading fonts indicate a subsection.

### The "flush" mechanism: when a new heading is found, save the current section

When the parser encounters a new heading, it must first save (flush) the section that was being accumulated. The flush operation:

1. Joins all strings in `current_text_buffer` into a single text string.
2. Creates a section record (a Python dictionary) with all metadata fields filled in.
3. Appends the section record to the output list.
4. Resets `current_text_buffer` to an empty list and updates the state variables for the new section.

This flush-on-heading-detection pattern is a standard technique for extracting structured content from sequentially read documents. It ensures that every section is complete before it is saved, and that no text is attributed to the wrong section.

The same flush operation happens at chapter boundaries and at the end of the document.

---

## Output Format

### JSON structure

The output of the main parser is a single JSON file containing a list of section objects. Each section object has the following fields:

```json
{
  "source": "TMT_2022",
  "chapter": 12,
  "chapter_title": "Pulmonary Disorders",
  "section": "Pneumonia",
  "subsection": "Community-Acquired Pneumonia",
  "text": "Community-acquired pneumonia (CAP) is defined as an acute infection of the pulmonary parenchyma in a patient who has acquired the infection in the community...",
  "page_range": [512, 518],
  "word_count": 1423,
  "has_essentials_box": true,
  "essentials_text": "Fever, chills, and cough with purulent sputum. Chest pain, shortness of breath. Rales and signs of consolidation on chest examination..."
}
```

### Field descriptions

| Field | Type | Description |
|---|---|---|
| `source` | string | Always `"TMT_2022"` — identifies the source document |
| `chapter` | integer | Chapter number (1 to 50) |
| `chapter_title` | string | Full title of the chapter |
| `section` | string | Title of the section within the chapter |
| `subsection` | string or null | Title of the subsection, if this text belongs to one; null otherwise |
| `text` | string | Full cleaned text content of the section or subsection |
| `page_range` | list of two integers | `[start_page, end_page]` — the page numbers (1-indexed) where this section appears |
| `word_count` | integer | Number of whitespace-delimited words in `text` |
| `has_essentials_box` | boolean | True if this section is accompanied by an Essentials of Diagnosis box |
| `essentials_text` | string or null | The text of the Essentials of Diagnosis box, if present; null otherwise |

### Output statistics

The final output file contains:

- **7,942 section records**
- **43 chapters** (chapters without extractable text sections are excluded)
- **8,655,430 characters** of clean text
- **0 PUA characters** remaining

---

## Scripts Reference

All scripts are located in the `data_processing/` directory of the Medora repository. They should be run from the repository root, in the order listed below. Each script has a `--help` flag that describes its arguments.

---

### `data_processing/tmt_explore.py`

**Purpose:** Exploratory analysis of the PDF structure. This script was used at the beginning of Phase 1.1 to understand how the TMT PDF is organized — which fonts are used, what sizes they appear at, and how many blocks and spans are on a typical page.

**What it does:**
- Opens the PDF with PyMuPDF
- Iterates over a sample of pages (e.g., pages 50–100)
- For each span on each page, prints the font name, font size, and the first 60 characters of text
- Produces a summary of all unique (font name, font size) combinations seen, sorted by frequency

**How to run:**
```bash
python data_processing/tmt_explore.py --pdf path/to/tmt_2022.pdf --pages 50 100
```

**When to use it:** If you need to re-examine font patterns, or if you are adapting the parser for a different edition of the textbook. Run this first to understand the new edition's font structure before adjusting the classification rules.

---

### `data_processing/build_pua_mapping.py`

**Purpose:** Builds the per-font PUA character mapping using the automated dictionary-based algorithm described above.

**What it does:**
- Reads the raw extracted text (output of an initial unfiltered extraction pass)
- Scans for all spans containing PUA characters
- Groups garbled words by font name
- For each garbled word, tries all single-character substitutions (a-z and common ligatures)
- Checks each candidate against an English word list
- Records unambiguous mappings in a JSON file
- Reports the count of resolved, ambiguous, and unresolved entries

**How to run:**
```bash
python data_processing/build_pua_mapping.py \
    --raw-text path/to/raw_extracted.json \
    --wordlist path/to/english_words.txt \
    --output data_processing/pua_mapping.json
```

Run this script twice (two passes) to take advantage of the cascading resolution effect described in the algorithm section. After the first run, apply the mapping to the raw text and feed the result back in as input to the second run.

---

### `data_processing/analyze_flagged_pua.py`

**Purpose:** Analyzes the entries that remain garbled after the automated mapping passes, to determine which ones can be fixed manually and which need LLM assistance.

**What it does:**
- Reads the partially corrected text
- Scans for remaining PUA characters
- Groups the flagged entries and prints them with surrounding context
- Reports how many entries remain per font and per PUA code point
- Flags entries that have more than one garbled character (candidates for LLM resolution)

**How to run:**
```bash
python data_processing/analyze_flagged_pua.py \
    --text path/to/partially_corrected.json \
    --mapping data_processing/pua_mapping.json
```

**Output:** A human-readable report of remaining garbled entries, printed to stdout. Review this report to decide which entries to send to GPT-4o.

---

### `data_processing/tmt_parser.py`

**Purpose:** The main parser. Reads the PDF, applies the PUA character mapping, classifies text by font, handles two-column layout, runs the state machine, and writes the final structured JSON output.

**What it does:**
- Loads the PUA mapping from `pua_mapping.json`
- Opens the PDF with PyMuPDF and reads the table of contents
- Iterates over all pages, sorting blocks by column position
- For each span, applies the PUA mapping to clean the text
- Classifies each span as a heading, Essentials box, body text, reference, or superscript
- Maintains the state machine to track the current chapter and section
- Flushes completed sections to the output list
- Writes the output as a JSON file

**How to run (full book):**
```bash
python data_processing/tmt_parser.py \
    --pdf path/to/tmt_2022.pdf \
    --mapping data_processing/pua_mapping.json \
    --output data/tmt_sections.json
```

**How to run (Chapter 2 only, for testing):**
```bash
python data_processing/tmt_parser.py \
    --pdf path/to/tmt_2022.pdf \
    --mapping data_processing/pua_mapping.json \
    --output data/tmt_sections_ch2_test.json \
    --test
```

The `--test` flag restricts the parser to Chapter 2 only, which allows rapid iteration when adjusting classification rules or debugging the state machine. Chapter 2 is a good test chapter because it contains all the major content types: headings, subsections, body text, and an Essentials of Diagnosis box.

---

### `data_processing/resolve_pua_with_llm.py`

**Purpose:** Sends the remaining garbled entries (after automated mapping) to GPT-4o for correction.

**What it does:**
- Reads the list of flagged entries from `analyze_flagged_pua.py`
- Constructs a prompt for each entry including: the garbled text, the chapter and section context, and an instruction to correct the medical terminology
- Calls the OpenAI API (GPT-4o model)
- Saves the API responses to a corrections file

**How to run:**
```bash
python data_processing/resolve_pua_with_llm.py \
    --flagged data_processing/flagged_entries.json \
    --output data_processing/llm_corrections.json
```

**Requirements:** An OpenAI API key must be set in the `OPENAI_API_KEY` environment variable.

**Note:** Review `llm_corrections.json` manually before applying corrections. The LLM output is generally accurate for medical terms, but it should be spot-checked to ensure no corrections introduced new errors.

---

### `data_processing/apply_pua_corrections.py`

**Purpose:** Patches the LLM corrections into the parser's JSON output, replacing the garbled strings with the corrected versions.

**What it does:**
- Reads the main parser output (`tmt_sections.json`)
- Reads the LLM corrections file (`llm_corrections.json`)
- For each correction, finds the matching section in the output and performs a string replacement
- Writes the patched output to a new file

**How to run:**
```bash
python data_processing/apply_pua_corrections.py \
    --sections data/tmt_sections.json \
    --corrections data_processing/llm_corrections.json \
    --output data/tmt_sections_clean.json
```

The output file `tmt_sections_clean.json` is the final, fully corrected dataset that is passed to Phase 1.2 (chunking and embedding).

---

## Verification

### How to verify the output is clean

After all PUA corrections have been applied, the output file should be scanned to confirm that no PUA characters remain. This is done by iterating over every character in every text field of every section record and checking whether its Unicode code point falls in the PUA range (U+E000 to U+F8FF, and the supplementary ranges U+F0000 to U+10FFFF).

The verification script is embedded as a function in `tmt_parser.py` and can also be run as a standalone check:

```python
import json
import sys

PUA_RANGES = [(0xE000, 0xF8FF), (0xF0000, 0xFFFFF), (0x100000, 0x10FFFF)]

def is_pua(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in PUA_RANGES)

with open("data/tmt_sections_clean.json") as f:
    sections = json.load(f)

total_chars = 0
pua_found = 0
for section in sections:
    for field in ["text", "essentials_text"]:
        if section.get(field):
            for ch in section[field]:
                total_chars += 1
                if is_pua(ch):
                    pua_found += 1
                    print(f"PUA U+{ord(ch):04X} in section: {section['chapter_title']} / {section['section']}")

print(f"\nTotal characters scanned: {total_chars:,}")
print(f"PUA characters found: {pua_found}")
```

### The verification result

Running this scan on the final output file produced:

```
Total characters scanned: 8,655,430
PUA characters found: 0
```

This confirms that the full text of the TMT 2022 textbook — across all 7,942 sections and 43 chapters — has been extracted with no remaining garbled characters. The output is ready for downstream processing.

---

## Limitations and Notes

### Tables with visual elements

Some pages in the TMT textbook contain tables that include not just text but also ruled lines, shading, or embedded images (for example, diagnostic algorithm flowcharts or grading tables with colored cells). PyMuPDF extracts the text content of these tables correctly, but it cannot reconstruct the table structure — the rows, columns, and cell boundaries are lost. The extracted text for these elements appears as a flat sequence of cell values, which may be difficult to interpret without the original table layout.

For the purposes of Medora's RAG pipeline, this is acceptable. The text content of tables (e.g., drug dosages, diagnostic criteria, laboratory reference ranges) is still valuable for retrieval, even without the grid structure. However, if a future phase requires structured table data (for example, to answer questions about specific drug dosages in tabular form), a dedicated table extraction step would be needed.

### Math formulas

Some chapters (particularly those involving pharmacology or laboratory interpretation) contain mathematical formulas — for example, equations for creatinine clearance or osmolality. PyMuPDF extracts the characters of these formulas as text, but subscripts, superscripts, and special mathematical symbols may not be reconstructed with correct ordering or notation. The formula may appear as a sequence of characters that is technically present but not formatted correctly.

The impact on Medora is limited because the system is designed for clinical text Q&A, not formula evaluation. The surrounding prose explanation of any formula is typically more useful for retrieval than the formula itself.

### Parser is specific to the TMT 2022 edition

The font size thresholds, font name patterns, and column layout assumptions in the parser were determined empirically for the TMT 2022 edition. A different edition of the same book — or a different publisher's textbook — may use different fonts, different sizes, or a different page layout. If the parser is ever applied to a different source document, the exploration step (`tmt_explore.py`) must be repeated to identify the new edition's font patterns, and the classification rules in `tmt_parser.py` must be updated accordingly.

The PUA character mapping is also edition-specific. Different PDF generation workflows produce different PUA mappings, so `build_pua_mapping.py` must be re-run from scratch for a different source PDF.

### Blank appendix pages

Pages 1861 and 1862 of the TMT 2022 PDF are blank (they are the final pages of the appendix section). The parser handles these gracefully — blank pages produce no blocks or spans when processed by PyMuPDF, so the state machine simply advances past them without any output. No special handling is required, but it is noted here to explain why these pages produce no section records.

### No section records from front matter

The front matter of the PDF (table of contents, preface, author list, copyright page) is not included in the output. The parser begins extracting at Chapter 1 using the page number provided by the table of contents. Front matter pages before Chapter 1 are skipped. This is intentional — front matter does not contain clinical content and would add noise to the RAG retrieval results.

---

*End of Phase 1.1 documentation.*
