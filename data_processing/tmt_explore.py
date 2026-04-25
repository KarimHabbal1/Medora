"""
TMT PDF Structure Explorer
Analyzes the structure of the Current Medical Diagnosis and Treatment PDF
for building a parser.
"""

import fitz  # PyMuPDF
import collections
import unicodedata

PDF_PATH = "/Users/karim/Desktop/folders/Medora_StartUp/2022, CURRENT Medical Diagnosis and Treatment- Original.pdf"

def separator(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ── 1. TABLE OF CONTENTS ─────────────────────────────────────────────────────
def analyse_toc(doc):
    separator("1. TABLE OF CONTENTS")
    toc = doc.get_toc()
    print(f"Total TOC entries: {len(toc)}")
    print()

    if not toc:
        print("No embedded TOC found.")
        return

    print(f"Format: [level, title, page_number]")
    print(f"Showing first 30 entries:\n")
    for i, entry in enumerate(toc[:30]):
        level, title, page = entry[0], entry[1], entry[2]
        indent = "  " * (level - 1)
        print(f"  [{i+1:3d}] L{level}  p{page:>5}  {indent}{title}")

    # Summarise level distribution
    level_counts = collections.Counter(e[0] for e in toc)
    print(f"\nLevel distribution:")
    for lvl in sorted(level_counts):
        print(f"  Level {lvl}: {level_counts[lvl]} entries")


# ── 2. FONT ANALYSIS (pages 50-55) ───────────────────────────────────────────
def analyse_fonts(doc):
    separator("2. FONT ANALYSIS  (pages 50–55, 0-indexed 49–54)")

    # (font_name, size_rounded, bold_flag) -> list of sample snippets
    combos = collections.defaultdict(list)

    for page_num in range(49, 55):          # 0-indexed
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:      # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_name = span.get("font", "unknown")
                    size      = round(span.get("size", 0), 1)
                    flags     = span.get("flags", 0)
                    bold      = bool(flags & 2**4)   # bit 4 = bold
                    text      = span.get("text", "").strip()
                    if not text:
                        continue
                    key = (font_name, size, bold)
                    if len(combos[key]) < 3:          # keep up to 3 samples
                        combos[key].append(text[:80])

    print(f"Unique (font_name, size, bold) combinations found: {len(combos)}\n")
    # Sort by size descending for readability
    for key in sorted(combos.keys(), key=lambda k: -k[1]):
        fname, fsize, fbold = key
        samples = combos[key]
        bold_str = "BOLD" if fbold else "    "
        print(f"  {bold_str}  size={fsize:5.1f}  font={fname}")
        for s in samples:
            print(f"            sample: {repr(s)}")
        print()


# ── 3. PAGE STRUCTURE SAMPLE ─────────────────────────────────────────────────
def analyse_page_structure(doc, page_num=50):   # 0-indexed → PDF page 51
    separator(f"3. PAGE STRUCTURE SAMPLE  (0-indexed page {page_num})")

    page = doc[page_num]
    data = page.get_text("dict")

    blocks = data["blocks"]
    text_blocks = [b for b in blocks if b.get("type") == 0]
    image_blocks = [b for b in blocks if b.get("type") == 1]

    print(f"Page size (w × h): {data['width']:.1f} × {data['height']:.1f} pts")
    print(f"Total blocks : {len(blocks)}")
    print(f"  Text blocks : {len(text_blocks)}")
    print(f"  Image blocks: {len(image_blocks)}")
    print()

    for bi, block in enumerate(text_blocks[:8]):   # show first 8 text blocks
        lines = block.get("lines", [])
        total_spans = sum(len(l.get("spans", [])) for l in lines)
        bbox = block.get("bbox", ())
        # Collect all text in block
        block_text = " ".join(
            span.get("text", "")
            for line in lines
            for span in line.get("spans", [])
        ).strip()

        print(f"  Block {bi+1}:  lines={len(lines)}  spans={total_spans}  "
              f"bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})")
        print(f"    Text preview: {repr(block_text[:120])}")

        # Show span-level detail for first line
        if lines:
            first_line = lines[0]
            for si, span in enumerate(first_line.get("spans", [])[:4]):
                print(f"      span[{si}]: font={span.get('font')!r}  "
                      f"size={span.get('size', 0):.1f}  "
                      f"flags={span.get('flags', 0)}  "
                      f"text={repr(span.get('text', '')[:60])}")
        print()


# ── 4. UNICODE / PRIVATE USE AREA CHECK ──────────────────────────────────────
def analyse_unicode(doc):
    separator("4. UNICODE ISSUES  (pages 50–100, 0-indexed 49–99)")

    pua_chars = collections.Counter()  # char -> count
    pua_in_context = []                # (page, char, context snippet)

    for page_num in range(49, min(100, len(doc))):
        page = doc[page_num]
        text = page.get_text("text")
        for i, ch in enumerate(text):
            code = ord(ch)
            if 0xE000 <= code <= 0xF8FF:
                pua_chars[ch] += 1
                if len(pua_in_context) < 20:
                    snippet = text[max(0, i-15):i+15].replace("\n", " ")
                    pua_in_context.append((page_num + 1, ch, snippet))

    if not pua_chars:
        print("No Private Use Area characters found in pages 50–100.")
    else:
        total_hits = sum(pua_chars.values())
        print(f"Total PUA characters found: {total_hits}")
        print(f"Unique PUA characters     : {len(pua_chars)}\n")
        print("Top occurrences:")
        for ch, count in pua_chars.most_common(20):
            code = ord(ch)
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = "UNNAMED"
            print(f"  U+{code:04X}  ({name})  count={count}")

        print(f"\nFirst up-to-20 in-context examples:")
        for pdf_page, ch, ctx in pua_in_context:
            print(f"  Page {pdf_page:4d}  U+{ord(ch):04X}  ...{repr(ctx)}...")


# ── 5. FRONT MATTER — where does medical content start? ───────────────────────
def analyse_front_matter(doc):
    separator("5. FRONT MATTER  (scanning first 20 pages)")

    medical_keywords = [
        "diagnosis", "treatment", "patient", "clinical", "disease",
        "symptoms", "therapy", "prognosis", "etiology", "pathology",
        "chapter", "cardiovascular", "pulmonary", "gastrointestinal"
    ]

    for page_num in range(min(40, len(doc))):
        page = doc[page_num]
        text = page.get_text("text").lower()
        hits = [kw for kw in medical_keywords if kw in text]
        word_count = len(text.split())

        # Get a short preview
        preview = page.get_text("text").strip().replace("\n", " ")[:120]

        print(f"  Page {page_num+1:3d} (0-idx {page_num:3d}):  "
              f"words={word_count:5d}  kw_hits={len(hits):2d}  "
              f"preview: {repr(preview)}")

    # Also cross-check with TOC
    print()
    toc = doc.get_toc()
    if toc:
        # Find the first TOC entry with level 1
        for entry in toc:
            if entry[0] == 1:
                print(f"First level-1 TOC entry: page={entry[2]}  title={entry[1]!r}")
                break


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Opening PDF: {PDF_PATH}")
    doc = fitz.open(PDF_PATH)
    print(f"Total pages: {len(doc)}")
    print(f"PyMuPDF version: {fitz.version}")

    analyse_toc(doc)
    analyse_fonts(doc)
    analyse_page_structure(doc)
    analyse_unicode(doc)
    analyse_front_matter(doc)

    doc.close()
    print("\n" + "=" * 70)
    print("  Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
