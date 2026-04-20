"""
apply_pua_corrections.py

Applies LLM-verified PUA (Private Use Area) font corrections to the raw
TMT sections JSON. For each correction in pua_llm_corrections.json, it
finds every section whose page_range overlaps the correction's page number
and whose text (or essentials_text) contains the garbled original string,
then replaces it with the corrected string.

Usage:
    python data_processing/apply_pua_corrections.py
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make config.py importable — it lives one level up (the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402  (import after sys.path manipulation)

# ---------------------------------------------------------------------------
# File paths (derived from config so they stay consistent with the project)
# ---------------------------------------------------------------------------
SECTIONS_FILE = config.CHUNKS_DIR / "tmt_raw_sections.json"
CORRECTIONS_FILE = config.DATA_DIR / "pua_llm_corrections.json"


def load_json(path: Path) -> list:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: list) -> None:
    """Save data to a JSON file with consistent formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def page_in_range(page: int, page_range: list) -> bool:
    """Return True if `page` falls within [page_range[0], page_range[1]]."""
    return page_range[0] <= page <= page_range[1]


def apply_corrections(sections: list, corrections: list) -> tuple[list, int, int]:
    """
    Apply every correction to the matching sections.

    A section is a match when:
      1. Its page_range overlaps the correction's page number.
      2. The correction's original (garbled) string appears in its
         `text` or `essentials_text` field.

    Returns:
        (updated_sections, sections_updated_count, corrections_applied_count)
    """
    sections_updated = set()   # track indices of changed sections
    corrections_applied = 0    # total number of (correction × section) replacements

    for i, correction in enumerate(corrections):
        page = correction["page"]
        original = correction["original"]
        corrected = correction["corrected"]

        matched_any = False  # did this correction touch at least one section?

        for idx, section in enumerate(sections):
            # Skip sections that don't cover the correction's page
            if not page_in_range(page, section.get("page_range", [0, 0])):
                continue

            changed = False  # did we modify this section in this pass?

            # --- Replace in main text ---
            if original in section.get("text", ""):
                section["text"] = section["text"].replace(original, corrected)
                changed = True

            # --- Replace in essentials_text (if present and non-empty) ---
            essentials = section.get("essentials_text", "")
            if essentials and original in essentials:
                section["essentials_text"] = essentials.replace(original, corrected)
                changed = True

            if changed:
                sections_updated.add(idx)
                corrections_applied += 1
                matched_any = True

        if matched_any:
            print(
                f"  [OK]  page {page:>5} | '{original}' → '{corrected}'"
            )
        else:
            print(
                f"  [??]  page {page:>5} | '{original}' — no matching section found"
            )

    return sections, len(sections_updated), corrections_applied


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading sections from:   {SECTIONS_FILE}")
    sections = load_json(SECTIONS_FILE)
    print(f"  {len(sections):,} sections loaded.")

    print(f"\nLoading corrections from: {CORRECTIONS_FILE}")
    corrections = load_json(CORRECTIONS_FILE)
    print(f"  {len(corrections)} corrections loaded.\n")

    # ------------------------------------------------------------------
    # 2. Apply corrections
    # ------------------------------------------------------------------
    print("Applying corrections...")
    sections, n_sections_updated, n_corrections_applied = apply_corrections(
        sections, corrections
    )

    # ------------------------------------------------------------------
    # 3. Save updated sections back to the same file
    # ------------------------------------------------------------------
    print(f"\nSaving updated sections to: {SECTIONS_FILE}")
    save_json(SECTIONS_FILE, sections)
    print("  Done.")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total corrections in file : {len(corrections)}")
    print(f"  Corrections applied       : {n_corrections_applied}")
    print(f"  Sections updated          : {n_sections_updated}")
    print("=" * 60)
