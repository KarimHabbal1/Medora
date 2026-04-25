"""
resolve_pua_with_llm.py

Reads flagged PUA (Private Use Area) words from flagged_pua_words.json,
filters to the two heavily-garbled fonts (f145 — obstetrics, f90 — drug tables),
groups them by font, sends each group to GPT-4o for context-aware correction,
and writes the results to pua_llm_corrections.json.

Run from the repo root:
    python data_processing/resolve_pua_with_llm.py
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FLAGGED_JSON = REPO_ROOT / "data" / "flagged_pua_words.json"
OUTPUT_JSON  = REPO_ROOT / "data" / "pua_llm_corrections.json"
DOTENV_PATH  = Path("/Users/karim/Desktop/folders/Medora_StartUp/data_processing/.env")

# Fonts we want to resolve — the others are mostly symbols/math/hyphens
# that were already handled by character-mapping or are not recoverable.
TARGET_FONTS = {"f145", "f90"}

# GPT model to use
MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Load environment & initialise OpenAI client
# ---------------------------------------------------------------------------
def load_client() -> OpenAI:
    """Load OPENAI_API_KEY from the .env file and return an OpenAI client."""
    if not DOTENV_PATH.exists():
        sys.exit(f"[ERROR] .env file not found at: {DOTENV_PATH}")

    load_dotenv(dotenv_path=DOTENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        sys.exit("[ERROR] OPENAI_API_KEY is not set in the .env file.")

    print(f"[INFO] Loaded API key from {DOTENV_PATH}")
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(font: str, entries: list[dict]) -> str:
    """
    Build the prompt sent to GPT-4o for a single font group.

    Each entry is numbered so the model can reply with matching numbered
    corrections.  We include chapter, section, page, and surrounding
    context to give as much signal as possible.
    """
    # Use chapter/section from the first entry as the group header;
    # individual context lines carry per-entry detail.
    first = entries[0]
    chapter = first.get("chapter", "Unknown")
    section = first.get("section", "Unknown")

    lines = [
        "The following medical text was extracted from a PDF but has garbled characters "
        "(shown as Unicode placeholders such as U+E01A, U+E016, etc.).",
        "Based on the medical context, chapter, section, and surrounding text provided, "
        "determine the correct text for each entry.",
        "",
        f"Chapter: {chapter}",
        f"Section: {section}",
        f"Font ID: {font}",
        "",
        "Garbled entries:",
    ]

    for i, entry in enumerate(entries, start=1):
        word    = entry.get("word", "")
        context = entry.get("context", "")
        page    = entry.get("page", "?")
        residuals = ", ".join(entry.get("residual_chars", []))

        lines.append(
            f'{i}. "{word}"  '
            f'(page {page}, residual PUA chars: {residuals}, '
            f'surrounding context: "{context}")'
        )

    lines += [
        "",
        "For each entry, respond ONLY with the corrected text in the same numbered format.",
        "Do not include explanations — just the corrected text on each line, e.g.:",
        "1. Hydatidiform mole",
        "2. Choriocarcinoma",
        "...",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPT-4o call
# ---------------------------------------------------------------------------
def call_gpt(client: OpenAI, prompt: str, font: str) -> str:
    """Send the prompt to GPT-4o and return the raw response text."""
    print(f"[INFO] Sending prompt to GPT-4o for font {font} "
          f"({prompt.count(chr(10))} lines)…")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical text expert helping to reconstruct garbled words "
                    "from a digitised medical textbook (CMDT 2022). "
                    "Reply only with the numbered corrected terms — no prose."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,   # deterministic — we want the single most-likely correction
    )

    reply = response.choices[0].message.content.strip()
    print(f"[INFO] Received response ({len(reply)} chars).")
    return reply


# ---------------------------------------------------------------------------
# Parse numbered response
# ---------------------------------------------------------------------------
def parse_numbered_response(response_text: str, expected_count: int) -> list[str]:
    """
    Extract the corrected terms from GPT-4o's numbered response.

    Accepts lines like:
        1. Hydatidiform mole
        2. Choriocarcinoma
    Returns a list of corrected strings indexed 0 … expected_count-1.
    Falls back to the raw line text if the numbering is unexpected.
    """
    corrections: dict[int, str] = {}

    for line in response_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match leading number followed by a period or parenthesis
        match = re.match(r'^(\d+)[.)]\s*(.*)', line)
        if match:
            idx = int(match.group(1))
            corrections[idx] = match.group(2).strip()

    # Build ordered list; fall back to empty string if a number is missing
    result = []
    for i in range(1, expected_count + 1):
        result.append(corrections.get(i, ""))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # -- Load flagged words --------------------------------------------------
    if not FLAGGED_JSON.exists():
        sys.exit(f"[ERROR] Input file not found: {FLAGGED_JSON}")

    with open(FLAGGED_JSON, encoding="utf-8") as fh:
        all_entries: list[dict] = json.load(fh)

    print(f"[INFO] Loaded {len(all_entries)} total flagged entries.")

    # -- Filter to target fonts ----------------------------------------------
    target_entries = [e for e in all_entries if e.get("font") in TARGET_FONTS]
    print(f"[INFO] {len(target_entries)} entries match target fonts: {TARGET_FONTS}")

    if not target_entries:
        sys.exit("[WARN] No entries found for target fonts. Nothing to do.")

    # -- Group by font -------------------------------------------------------
    groups: dict[str, list[dict]] = {}
    for entry in target_entries:
        font = entry["font"]
        groups.setdefault(font, []).append(entry)

    for font, entries in groups.items():
        print(f"[INFO] Font {font}: {len(entries)} entries")

    # -- Initialise OpenAI client --------------------------------------------
    client = load_client()

    # -- Process each font group ---------------------------------------------
    all_corrections: list[dict] = []

    for font, entries in groups.items():
        prompt = build_prompt(font, entries)
        raw_response = call_gpt(client, prompt, font)

        corrected_texts = parse_numbered_response(raw_response, len(entries))

        for entry, corrected in zip(entries, corrected_texts):
            record = {
                "font":      font,
                "original":  entry.get("word", ""),
                "corrected": corrected,
                "chapter":   entry.get("chapter", ""),
                "section":   entry.get("section", ""),
                "page":      entry.get("page"),
                "context":   entry.get("context", ""),
                "residual_chars": entry.get("residual_chars", []),
            }
            all_corrections.append(record)
            print(f'  [{font} p.{entry.get("page")}] '
                  f'"{entry.get("word")}"  →  "{corrected}"')

    # -- Write output --------------------------------------------------------
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(all_corrections, fh, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Saved {len(all_corrections)} corrections → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
