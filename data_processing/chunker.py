"""
data_processing/chunker.py

Structure-aware chunker for TMT (Current Medical Diagnosis and Treatment) raw sections.

Reads:  data/chunks/tmt_raw_sections.json   (7,942 sections from the parser)

Writes:
  data/chunks/tmt_chunks_structured.json   — structure-aware chunks
  data/chunks/tmt_chunks_baseline.json     — fixed-size baseline chunks

Run with:
  python data_processing/chunker.py
"""

import json
import re
import sys
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow importing config from the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHUNKS_DIR,
    CHUNK_MIN_WORDS,   # 50  — sections shorter than this are "short"
    CHUNK_MAX_WORDS,   # 500 — sections longer than this are "long"
    CHUNK_TARGET_WORDS,  # 200 — target size when splitting long sections
    BASELINE_CHUNK_WORDS,  # 200 — fixed window size for baseline
    BASELINE_OVERLAP_WORDS,  # 50  — overlap between baseline windows
)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
RAW_SECTIONS_FILE = CHUNKS_DIR / "tmt_raw_sections.json"
STRUCTURED_OUTPUT = CHUNKS_DIR / "tmt_chunks_structured.json"
BASELINE_OUTPUT   = CHUNKS_DIR / "tmt_chunks_baseline.json"


# ===========================================================================
# Utility helpers
# ===========================================================================

def slugify(text: str, max_len: int = 40) -> str:
    """
    Convert a human-readable string into a URL/ID-safe slug.

    Steps:
      1. Lowercase
      2. Replace whitespace with underscores
      3. Strip leading bullet characters (», º) that appear in TMT subsection names
      4. Remove every character that is not alphanumeric or underscore
      5. Collapse multiple underscores into one
      6. Truncate to max_len characters

    Examples:
      "Heart Failure"          → "heart_failure"
      "» Clinical Findings"    → "clinical_findings"
      "Essentials of Diagnosis"→ "essentials_of_diagnosis"
    """
    text = text.strip()
    # Remove TMT-specific bullet prefixes before lowercasing
    text = re.sub(r"^[»º▸•\-–—]+\s*", "", text)
    text = text.lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w]", "", text)          # keep only [a-z0-9_]
    text = re.sub(r"_+", "_", text)            # collapse consecutive underscores
    text = text.strip("_")
    return text[:max_len] if text else "unknown"


def word_count(text: str) -> int:
    """Return the number of whitespace-delimited words in text."""
    return len(text.split()) if text.strip() else 0


def build_chunk_id(chapter: str, section: str, subsection: str, n: int) -> str:
    """
    Build a deterministic chunk ID from hierarchy components plus a sequence number.

    Format: tmt::<chapter_slug>::<section_slug>::<subsection_slug>::<N>

    The sequence number N starts at 1 within each (chapter, section, subsection) group.
    When subsection is empty, its slug defaults to "main".
    """
    ch = slugify(chapter) or "unknown_chapter"
    se = slugify(section)  or "unknown_section"
    su = slugify(subsection) if subsection.strip() else "main"
    return f"tmt::{ch}::{se}::{su}::{n}"


# ===========================================================================
# Text splitting helpers
# ===========================================================================

def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraph-sized blocks by double newlines.

    Paragraphs that are still empty after stripping are discarded.
    Returns a list of non-empty paragraph strings.
    """
    # Try splitting on double (or more) newlines first
    blocks = re.split(r"\n{2,}", text)
    paragraphs = [b.strip() for b in blocks if b.strip()]
    return paragraphs


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into individual sentences using a simple heuristic:
    split on a period (or ! or ?) followed by whitespace and a capital letter.

    This handles the pattern ". A" but avoids splitting on abbreviations like
    "Dr. Smith" in most cases (since they rarely have a capital after space+period
    in the same way full stops do in running prose).

    Returns a list of non-empty sentence strings.
    """
    # Pattern: end of sentence punctuation (.!?) followed by whitespace + capital
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]


def force_split_by_words(text: str, max_words: int) -> list[str]:
    """
    Last-resort hard split: chop text into word-boundary chunks of at most max_words
    words each.

    This is used when a segment has no paragraph breaks and no sentence boundaries
    (e.g. the book index, which is a dense list of terms and page numbers with no
    punctuation patterns that the sentence splitter can act on).  Without this
    fallback those segments would be emitted as-is, violating the CHUNK_MAX_WORDS
    ceiling.

    # FIX: over-500 fallback — force word-boundary split when all other strategies fail
    """
    words = text.split()
    chunks: list[str] = []
    for start in range(0, len(words), max_words):
        part = " ".join(words[start : start + max_words])
        if part:
            chunks.append(part)
    return chunks


def merge_to_target(segments: list[str], target_words: int, max_words: int) -> list[str]:
    """
    Greedily merge a list of text segments into sub-chunks that each stay
    within max_words, aiming for roughly target_words each.

    Algorithm:
      - Walk through segments accumulating words.
      - When adding the next segment would exceed max_words, flush the current
        accumulation as a sub-chunk and start a new one.
      - Any segment that individually exceeds max_words is force-split by word
        boundaries rather than emitted as-is, so the max_words ceiling is always
        respected.

    Args:
        segments:    Ordered list of text strings (paragraphs or sentences).
        target_words: Soft target size. Unused directly but kept for future tuning.
        max_words:   Hard ceiling — never exceed this per sub-chunk.

    Returns:
        List of merged text strings.
    """
    chunks: list[str] = []
    current_parts: list[str] = []
    current_wc = 0

    for seg in segments:
        seg_wc = word_count(seg)

        if seg_wc == 0:
            # Skip blank segments
            continue

        if seg_wc > max_words:
            # FIX: over-500 — a single segment is too large to emit whole.
            # Flush any accumulated content, then force-split the oversized
            # segment by word boundaries so we never breach max_words.
            if current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = []
                current_wc = 0
            chunks.extend(force_split_by_words(seg, max_words))
            continue

        if current_wc + seg_wc > max_words and current_parts:
            # Adding this segment would overflow — flush first
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_wc = 0

        current_parts.append(seg)
        current_wc += seg_wc

    # Flush any remaining content
    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def split_long_section(text: str) -> list[str]:
    """
    Split a section whose word count exceeds CHUNK_MAX_WORDS into multiple
    sub-chunks, each ideally between CHUNK_TARGET_WORDS and CHUNK_MAX_WORDS words.

    Strategy (two-pass):
      Pass 1 — split on paragraph boundaries (double newlines).
               Merge adjacent paragraphs until a sub-chunk would exceed CHUNK_MAX_WORDS.
      Pass 2 — for any sub-chunk that is STILL over CHUNK_MAX_WORDS (i.e. a single
               very long paragraph), split it further at sentence boundaries.

    Returns:
        Ordered list of sub-chunk text strings.
    """
    # Pass 1: paragraph-level split + greedy merge
    paragraphs = split_into_paragraphs(text)
    if len(paragraphs) <= 1:
        # No paragraph breaks — treat the whole text as one block for sentence splitting
        paragraphs = [text]

    paragraph_chunks = merge_to_target(paragraphs, CHUNK_TARGET_WORDS, CHUNK_MAX_WORDS)

    # Pass 2: any oversized paragraph chunk is broken at sentence boundaries
    final_chunks: list[str] = []
    for chunk in paragraph_chunks:
        if word_count(chunk) > CHUNK_MAX_WORDS:
            sentences = split_into_sentences(chunk)
            sentence_chunks = merge_to_target(sentences, CHUNK_TARGET_WORDS, CHUNK_MAX_WORDS)
            final_chunks.extend(sentence_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks


# ===========================================================================
# Structured chunking
# ===========================================================================

def make_structured_chunk(
    chunk_id: str,
    section: dict,
    text: str,
    subsection_override: str | None = None,
) -> dict:
    """
    Build a single structured chunk dict from a raw section and a text block.

    Args:
        chunk_id:            Pre-computed ID string.
        section:             The raw section dict from tmt_raw_sections.json.
        text:                The actual text for this chunk (may differ from section["text"]).
        subsection_override: If provided, use this as the subsection name instead of
                             the section's own subsection field.

    Returns:
        A dict matching the structured chunk schema.
    """
    sub = subsection_override if subsection_override is not None else section["subsection"]
    return {
        "chunk_id":   chunk_id,
        "source":     section["source"],
        "chapter":    section["chapter"],
        "section":    section["section"],
        "subsection": sub,
        "text":       text.strip(),
        "page_range": section["page_range"],
        "word_count": word_count(text),
        "chunk_type": "structured",
    }


def build_structured_chunks(raw_sections: list[dict]) -> tuple[list[dict], dict]:
    """
    Convert raw sections into structure-aware chunks following these rules:

    1. Drop sections where word_count == 0 AND essentials_text is empty.
    2. If a section has has_essentials_box == True, emit the essentials_text as a
       separate chunk with subsection = "Essentials of Diagnosis".
    3. Short sections (< CHUNK_MIN_WORDS words of main text): merge with the NEXT
       section that shares the same chapter AND section name. If no valid neighbour
       exists, keep the short section as-is.
    4. Long sections (> CHUNK_MAX_WORDS words): split using split_long_section().
    5. Normal sections (CHUNK_MIN_WORDS–CHUNK_MAX_WORDS words): emit as-is.

    The function uses a sequence-number counter keyed on (chapter, section, subsection)
    to generate the ::N suffix in chunk IDs.

    Returns:
        (chunks, stats) where stats is a dict with counts of each chunk_type outcome.
    """
    print("[structured] Starting structure-aware chunking …")

    # -----------------------------------------------------------------------
    # Step 1 — Filter out fully empty sections
    # -----------------------------------------------------------------------
    active_sections = [
        s for s in raw_sections
        if not (s["word_count"] == 0 and not s.get("essentials_text", "").strip())
    ]
    dropped = len(raw_sections) - len(active_sections)
    print(f"[structured] Dropped {dropped} empty sections — {len(active_sections)} remain")

    # -----------------------------------------------------------------------
    # Step 2 — Handle short-section merging
    #
    # We do a single forward pass.  When we encounter a short section, we try
    # to append it to a "pending" accumulator.  When we encounter a normal or
    # long section (or a section boundary), we flush the accumulator.
    #
    # A section boundary is defined as a change in (chapter, section) tuple.
    # -----------------------------------------------------------------------
    merged_sections: list[dict] = []   # sections after short-merge
    stats = {"essentials": 0, "merged": 0, "split": 0, "normal": 0, "kept_short": 0}

    pending: dict | None = None   # a short section waiting to be merged forward

    for i, sec in enumerate(active_sections):
        # Determine how to group sections for merge eligibility
        group_key = (sec["chapter"], sec["section"])
        main_text = sec["text"].strip()
        main_wc   = word_count(main_text)

        if pending is not None:
            pending_key = (pending["chapter"], pending["section"])

            if group_key == pending_key:
                # Same chapter+section — merge pending into current section by
                # prepending its text to the current section's text
                merged_text = pending["text"].strip() + "\n\n" + main_text
                sec = dict(sec)   # shallow copy so we don't mutate the original
                sec["text"] = merged_text
                sec["word_count"] = word_count(merged_text)
                sec["page_range"] = [
                    min(pending["page_range"][0], sec["page_range"][0]),
                    max(pending["page_range"][1], sec["page_range"][1]),
                ]
                stats["merged"] += 1
                pending = None
            else:
                # Different section — flush pending as-is (keep_short)
                merged_sections.append(pending)
                stats["kept_short"] += 1
                pending = None

        # Now decide what to do with `sec`
        if main_wc < CHUNK_MIN_WORDS and main_wc > 0:
            # Short — defer, try to merge with next section
            pending = sec
        else:
            merged_sections.append(sec)

    # Flush any leftover pending short section
    if pending is not None:
        merged_sections.append(pending)
        stats["kept_short"] += 1

    print(f"[structured] After short-section merging: {len(merged_sections)} sections")

    # -----------------------------------------------------------------------
    # Step 3 — Emit chunks from merged_sections
    #
    # We track sequence numbers per (chapter, section, subsection) so that
    # split sub-chunks and essentials chunks share a coherent N counter.
    # -----------------------------------------------------------------------
    chunks: list[dict] = []
    seq_counters: dict[tuple, int] = {}   # (ch, se, su) → next N

    def next_seq(ch: str, se: str, su: str) -> int:
        """Increment and return the next sequence number for this hierarchy triple."""
        key = (ch, se, su)
        seq_counters[key] = seq_counters.get(key, 0) + 1
        return seq_counters[key]

    for sec in merged_sections:
        ch  = sec["chapter"]
        se  = sec["section"]
        su  = sec["subsection"]
        main_text = sec["text"].strip()
        main_wc   = word_count(main_text)

        # -----------------------------------------------------------------------
        # 3a — Essentials box chunk (emitted BEFORE the main text chunk)
        # -----------------------------------------------------------------------
        if sec.get("has_essentials_box") and sec.get("essentials_text", "").strip():
            ess_sub = "Essentials of Diagnosis"
            n   = next_seq(ch, se, ess_sub)
            cid = build_chunk_id(ch, se, ess_sub, n)
            chunks.append(make_structured_chunk(cid, sec, sec["essentials_text"], ess_sub))
            stats["essentials"] += 1

        # -----------------------------------------------------------------------
        # 3b — Main text: skip if empty (section might have been essentials-only)
        # -----------------------------------------------------------------------
        if not main_text:
            continue

        if main_wc > CHUNK_MAX_WORDS:
            # --- Long section: split into sub-chunks ---
            sub_texts = split_long_section(main_text)
            for sub_text in sub_texts:
                if not sub_text.strip():
                    continue
                n   = next_seq(ch, se, su)
                cid = build_chunk_id(ch, se, su, n)
                chunks.append(make_structured_chunk(cid, sec, sub_text))
            stats["split"] += 1
        else:
            # --- Normal (or kept-short) section: emit as-is ---
            n   = next_seq(ch, se, su)
            cid = build_chunk_id(ch, se, su, n)
            chunks.append(make_structured_chunk(cid, sec, main_text))
            stats["normal"] += 1

    # -----------------------------------------------------------------------
    # Step 4 — Drop junk chunks (artifacts from PDF extraction)
    #
    # After splitting and merging, some chunks are artifacts rather than
    # real medical content. We filter by content, not just word count,
    # to avoid dropping short-but-important clinical statements like
    # "All patients with orbital cellulitis must be referred emergently."
    #
    # Drop criteria (must meet ANY):
    #   - Under 5 words (too short to carry any meaning)
    #   - Contains "CMDT 2022" (running page header artifact)
    #   - Text is purely digits/whitespace (page numbers)
    #   - Chapter is "Index" (alphabetical term+page list, no clinical value)
    #
    # Keep: everything else, even if short — a 10-word referral
    # instruction is medically critical.
    # -----------------------------------------------------------------------
    # Chapters that contain no clinical content — just reference material
    _NON_CLINICAL_CHAPTERS = {"Index"}

    def is_junk_chunk(chunk: dict) -> bool:
        """Check if a chunk is a PDF artifact rather than real content."""
        text = chunk["text"].strip()
        wc = chunk["word_count"]

        # Non-clinical chapters (Index = alphabetical term list with page numbers)
        if chunk["chapter"] in _NON_CLINICAL_CHAPTERS:
            return True

        # Too short to carry meaning
        if wc < 5:
            return True

        # Running page header artifact
        if "CMDT 2022" in text:
            return True

        # Pure digits / page numbers
        if re.match(r"^[\d\s.]+$", text):
            return True

        return False

    before_filter = len(chunks)
    chunks = [c for c in chunks if not is_junk_chunk(c)]
    dropped_junk = before_filter - len(chunks)
    print(f"[structured] Dropped {dropped_junk} junk chunks (page numbers, headers, <5 words) — {len(chunks)} remain")

    print(f"[structured] Total structured chunks produced: {len(chunks)}")
    return chunks, stats


# ===========================================================================
# Baseline chunking
# ===========================================================================

def build_baseline_chunks(raw_sections: list[dict]) -> list[dict]:
    """
    Build fixed-size baseline chunks by concatenating all section text in order,
    then sliding a window of BASELINE_CHUNK_WORDS words with BASELINE_OVERLAP_WORDS
    overlap.

    Each chunk is assigned metadata from whichever section contains the majority
    of its starting character position in the full concatenated text.

    Returns:
        List of baseline chunk dicts.
    """
    print("[baseline] Concatenating all section text …")

    # -----------------------------------------------------------------------
    # Step 1 — Build one long string with section separators.
    #           Record the character span each section occupies so we can map
    #           a chunk's start position back to its originating section.
    # -----------------------------------------------------------------------
    parts: list[str] = []
    section_spans: list[dict] = []   # [{start, end, chapter, section}, …]
    char_cursor = 0

    for sec in raw_sections:
        text = sec.get("text", "").strip()
        if not text:
            continue

        # Add a section separator that includes the hierarchy for readability
        separator = f"\n\n[{sec['chapter']} / {sec['section']}]\n"
        combined  = separator + text
        parts.append(combined)

        span_start = char_cursor + len(separator)   # point to actual content start
        span_end   = char_cursor + len(combined)

        section_spans.append({
            "char_start": span_start,
            "char_end":   span_end,
            "chapter":    sec["chapter"],
            "section":    sec["section"],
        })
        char_cursor += len(combined)

    full_text = "".join(parts)
    all_words  = full_text.split()
    total_words = len(all_words)

    print(f"[baseline] Total words in concatenated corpus: {total_words:,}")

    # -----------------------------------------------------------------------
    # Step 2 — Build a character-offset index for each word so we can map
    #           a word-position back to a character position and thus to a section.
    # -----------------------------------------------------------------------
    # Rather than storing every word's char offset (memory-heavy), we use the
    # simpler approach: when we need the section for chunk starting at word i,
    # we estimate the char offset as proportional to word count.
    # More precisely: re-join the slice and walk spans linearly.

    def get_section_for_word_index(word_idx: int) -> tuple[str, str]:
        """
        Return (chapter, section) for the section that contains word_idx.
        Uses a proportional estimate: char_pos ≈ word_idx / total_words * len(full_text).
        For baseline metadata this approximation is sufficient.
        """
        estimated_char = int(word_idx / total_words * len(full_text)) if total_words else 0
        for span in section_spans:
            if span["char_start"] <= estimated_char < span["char_end"]:
                return span["chapter"], span["section"]
        # Fallback: last section
        if section_spans:
            return section_spans[-1]["chapter"], section_spans[-1]["section"]
        return "Unknown", "Unknown"

    # -----------------------------------------------------------------------
    # Step 3 — Slide a fixed window across all_words
    # -----------------------------------------------------------------------
    stride  = BASELINE_CHUNK_WORDS - BASELINE_OVERLAP_WORDS   # step size = 150
    chunks: list[dict] = []
    chunk_idx = 0

    print(f"[baseline] Sliding window: size={BASELINE_CHUNK_WORDS}, overlap={BASELINE_OVERLAP_WORDS}, stride={stride}")

    word_pos = 0
    while word_pos < total_words:
        window = all_words[word_pos: word_pos + BASELINE_CHUNK_WORDS]
        if not window:
            break

        chunk_text = " ".join(window)
        chapter, section = get_section_for_word_index(word_pos)
        chunk_idx += 1

        chunks.append({
            "chunk_id":   f"tmt_baseline::{chunk_idx:04d}",
            "source":     "TMT_2022",
            "chapter":    chapter,
            "section":    section,
            "text":       chunk_text,
            "word_count": len(window),
            "chunk_type": "baseline",
        })

        word_pos += stride

    print(f"[baseline] Total baseline chunks produced: {len(chunks)}")
    return chunks


# ===========================================================================
# Quality report
# ===========================================================================

def word_count_buckets(word_counts: list[int]) -> dict:
    """
    Bin word counts into descriptive size buckets and return a dict with
    bucket labels as keys and counts as values.

    Buckets (matching the parser report format):
      <50, 50-100, 100-200, 200-300, 300-500, 500-1000, >1000
    """
    buckets = {
        "<50":      0,
        "50-100":   0,
        "100-200":  0,
        "200-300":  0,
        "300-500":  0,
        "500-1000": 0,
        ">1000":    0,
    }
    for wc in word_counts:
        if wc < 50:
            buckets["<50"] += 1
        elif wc < 100:
            buckets["50-100"] += 1
        elif wc < 200:
            buckets["100-200"] += 1
        elif wc < 300:
            buckets["200-300"] += 1
        elif wc <= 500:
            buckets["300-500"] += 1
        elif wc <= 1000:
            buckets["500-1000"] += 1
        else:
            buckets[">1000"] += 1
    return buckets


def print_quality_report(
    structured_chunks: list[dict],
    structured_stats: dict,
    baseline_chunks: list[dict],
) -> None:
    """
    Print a human-readable quality report for both chunking methods to stdout.

    For each method reports:
      - Total chunks
      - Word count distribution (min, max, mean, median)
      - Word count bucket histogram
      - (Structured only) chunk-type breakdown
    """
    separator = "=" * 60

    # -----------------------------------------------------------------------
    # Structured report
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("QUALITY REPORT — Structured Chunks")
    print(separator)

    s_wcs = [c["word_count"] for c in structured_chunks]

    print(f"Total chunks : {len(structured_chunks):,}")
    if s_wcs:
        print(f"Word count   : min={min(s_wcs)}, max={max(s_wcs)}, "
              f"mean={statistics.mean(s_wcs):.1f}, median={statistics.median(s_wcs):.1f}")

    print("\nWord count distribution:")
    for label, count in word_count_buckets(s_wcs).items():
        bar = "#" * (count // max(1, len(s_wcs) // 40))   # rough bar scaling
        print(f"  {label:>10}  {count:>6}  {bar}")

    print("\nChunk type breakdown:")
    print(f"  essentials  : {structured_stats['essentials']:>6}")
    print(f"  merged      : {structured_stats['merged']:>6}  (short sections merged forward)")
    print(f"  kept_short  : {structured_stats['kept_short']:>6}  (short sections with no valid merge target)")
    print(f"  split       : {structured_stats['split']:>6}  (long sections split into sub-chunks)")
    print(f"  normal      : {structured_stats['normal']:>6}  (50–500 word sections kept as-is)")

    # -----------------------------------------------------------------------
    # Baseline report
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("QUALITY REPORT — Baseline Fixed-Size Chunks")
    print(separator)

    b_wcs = [c["word_count"] for c in baseline_chunks]

    print(f"Total chunks : {len(baseline_chunks):,}")
    if b_wcs:
        print(f"Word count   : min={min(b_wcs)}, max={max(b_wcs)}, "
              f"mean={statistics.mean(b_wcs):.1f}, median={statistics.median(b_wcs):.1f}")

    print("\nWord count distribution:")
    for label, count in word_count_buckets(b_wcs).items():
        bar = "#" * (count // max(1, len(b_wcs) // 40))
        print(f"  {label:>10}  {count:>6}  {bar}")

    print(f"\n{separator}\n")


# ===========================================================================
# Main entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Medora — TMT Structure-Aware Chunker")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Load raw sections
    # -----------------------------------------------------------------------
    print(f"\nLoading raw sections from:\n  {RAW_SECTIONS_FILE}")
    with open(RAW_SECTIONS_FILE, "r", encoding="utf-8") as fh:
        raw_sections: list[dict] = json.load(fh)
    print(f"Loaded {len(raw_sections):,} raw sections\n")

    # -----------------------------------------------------------------------
    # Structured chunking
    # -----------------------------------------------------------------------
    print("--- STRUCTURED CHUNKING ---")
    structured_chunks, structured_stats = build_structured_chunks(raw_sections)

    print(f"\nWriting structured chunks to:\n  {STRUCTURED_OUTPUT}")
    with open(STRUCTURED_OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(structured_chunks, fh, indent=2, ensure_ascii=False)
    print(f"Written: {len(structured_chunks):,} chunks")

    # -----------------------------------------------------------------------
    # Baseline chunking
    # -----------------------------------------------------------------------
    print("\n--- BASELINE CHUNKING ---")
    baseline_chunks = build_baseline_chunks(raw_sections)

    print(f"\nWriting baseline chunks to:\n  {BASELINE_OUTPUT}")
    with open(BASELINE_OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(baseline_chunks, fh, indent=2, ensure_ascii=False)
    print(f"Written: {len(baseline_chunks):,} chunks")

    # -----------------------------------------------------------------------
    # Quality report
    # -----------------------------------------------------------------------
    print_quality_report(structured_chunks, structured_stats, baseline_chunks)

    print("Done.")
