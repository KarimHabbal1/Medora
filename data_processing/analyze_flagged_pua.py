"""
analyze_flagged_pua.py
======================
Analyze flagged_pua_words.json to identify missing PUA → letter mappings
that should be added to FONT_PUA_MAP in tmt_parser.py.

Approach
--------
1.  Group entries by font, count occurrences, list PUA chars used.
2.  For the top-10 fonts by count, show 10 example words.
3.  Deep-dive on U+E03F (14,472 occurrences).
4.  For every (font, pua_char) pair, apply a dictionary-based heuristic
    to propose the most likely letter substitution.
5.  Output a Python dict ready to be merged into FONT_PUA_MAP.

Dictionary heuristic
--------------------
For every position in a word where a PUA char appears we try replacing it
with each of the 26 ASCII letters (a–z) and check whether the resulting
word (or a word fragment it is part of) is in the English / medical word
list.  The letter that produces the most dictionary hits across all examples
is chosen.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = Path("/Users/karim/Desktop/folders/Medora_StartUp/Medora/data")
FLAGGED    = DATA_DIR / "flagged_pua_words.json"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 72)
print("Loading flagged_pua_words.json …")
with FLAGGED.open() as fh:
    records: list[dict[str, Any]] = json.load(fh)
print(f"  {len(records):,} records loaded.\n")

# ---------------------------------------------------------------------------
# Helper: extract raw PUA characters from a word string
# ---------------------------------------------------------------------------
def pua_chars_in(text: str) -> list[str]:
    """Return list of unique PUA codepoints (as 'U+XXXX') found in text."""
    seen = []
    for ch in text:
        cp = ord(ch)
        if 0xE000 <= cp <= 0xF8FF:
            key = f"U+{cp:04X}"
            if key not in seen:
                seen.append(key)
    return seen


def raw_pua_chars_in(text: str) -> list[str]:
    """Return raw PUA char objects found in text."""
    return [ch for ch in text if 0xE000 <= ord(ch) <= 0xF8FF]

# ---------------------------------------------------------------------------
# Step 1 – Group by font
# ---------------------------------------------------------------------------
print("=" * 72)
print("STEP 1 — Group by font")
print("=" * 72)

# font -> list of records
by_font: dict[str, list[dict]] = defaultdict(list)
for rec in records:
    by_font[rec["font"]].append(rec)

# font -> Counter of PUA codepoints
font_pua_counts: dict[str, Counter] = {}
for font, recs in by_font.items():
    cnt: Counter = Counter()
    for rec in recs:
        for cp in rec.get("residual_chars", []):
            cnt[cp] += 1
    font_pua_counts[font] = cnt

# Sort fonts by total record count descending
sorted_fonts = sorted(by_font.keys(), key=lambda f: len(by_font[f]), reverse=True)

print(f"\n{'Font':<10} {'#Records':>10}  PUA chars used")
print("-" * 72)
for font in sorted_fonts:
    recs  = by_font[font]
    chars = ", ".join(
        f"{cp}({n})" for cp, n in font_pua_counts[font].most_common()
    )
    print(f"{font:<10} {len(recs):>10}  {chars}")

# ---------------------------------------------------------------------------
# Step 2 – Top-10 fonts: 10 example words
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 2 — Top-10 fonts: 10 example words each")
print("=" * 72)

top10_fonts = sorted_fonts[:10]
for font in top10_fonts:
    recs = by_font[font]
    print(f"\n--- Font: {font}  ({len(recs)} records) ---")
    seen_words: set[str] = set()
    shown = 0
    for rec in recs:
        w = rec["word"]
        if w in seen_words:
            continue
        seen_words.add(w)
        puas = ", ".join(rec.get("residual_chars", []))
        ctx  = rec.get("context", "")
        print(f"  [{puas}] {w!r}")
        if ctx:
            # Trim context to 80 chars around [HERE]
            snip = ctx[:120] if len(ctx) > 120 else ctx
            print(f"       ctx: {snip!r}")
        shown += 1
        if shown >= 10:
            break

# ---------------------------------------------------------------------------
# Step 3 – Deep-dive on U+E03F
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 3 — Deep-dive on U+E03F")
print("=" * 72)

target_cp = "U+E03F"
e03f_records = [r for r in records if target_cp in r.get("residual_chars", [])]
print(f"\n  Total records containing {target_cp}: {len(e03f_records):,}")

fonts_with_e03f: Counter = Counter()
for rec in e03f_records:
    fonts_with_e03f[rec["font"]] += 1
print("\n  Fonts using U+E03F:")
for font, cnt in fonts_with_e03f.most_common():
    print(f"    {font}: {cnt:,}")

print(f"\n  20 example words containing U+E03F:")
seen_w: set[str] = set()
shown = 0
for rec in e03f_records:
    w = rec["word"]
    if w in seen_w:
        continue
    seen_w.add(w)
    other = [c for c in rec.get("residual_chars", []) if c != target_cp]
    extra = f"  (also: {', '.join(other)})" if other else ""
    ctx   = rec.get("context", "")
    snip  = ctx[:130] if len(ctx) > 130 else ctx
    print(f"  font={rec['font']} | {w!r}{extra}")
    if snip:
        print(f"    ctx: {snip!r}")
    shown += 1
    if shown >= 20:
        break

# ---------------------------------------------------------------------------
# Step 4 – Dictionary-based mapping inference
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 4 — Dictionary-based mapping inference for all (font, pua) pairs")
print("=" * 72)

# ------------------------------------------------------------------
# Build a lightweight word set.  We prefer /usr/share/dict/words if
# available, otherwise fall back to a small medical word list embedded
# in the script, plus common English words.
# ------------------------------------------------------------------
WORDLIST_PATH = Path("/usr/share/dict/words")
word_set: set[str] = set()
if WORDLIST_PATH.exists():
    with WORDLIST_PATH.open() as fh:
        for line in fh:
            w = line.strip().lower()
            if len(w) >= 3:
                word_set.add(w)
    print(f"\n  Loaded {len(word_set):,} words from {WORDLIST_PATH}")
else:
    # Minimal fallback
    COMMON = (
        "the and are for that this with have from they been were said "
        "which their there when what about would could should after "
        "more also some into than then its two has him his how man "
        "our out use may each she may than many most such well even "
        "she him only most case risk pain dose time patient patients "
        "treatment disease therapy clinical management diagnosis "
        "symptoms signs function pressure blood cardiac pulmonary "
        "renal hepatic chronic acute inflammatory infection immune "
        "antibiotics analgesics opioid receptor agonist antagonist "
        "palliative cancer tumor surgery radiation chemotherapy "
        "mortality morbidity prognosis oncology hematology "
    )
    for tok in COMMON.split():
        word_set.add(tok.lower())
    print(f"\n  Using {len(word_set)} fallback words (no /usr/share/dict/words found)")

# Common English letter frequency order for tie-breaking
LETTER_FREQ = "etaoinshrdlcumwfgypbvkjxqz"

def score_substitution(word: str, pua_char: str, candidate: str) -> int:
    """
    Replace pua_char with candidate in word, then count how many
    space-split tokens (or the whole word) are in the word set.
    """
    replaced = word.replace(pua_char, candidate)
    # Strip punctuation from edges of each token
    tokens = [t.strip(string.punctuation) for t in replaced.split()]
    score = 0
    for tok in tokens:
        tok_lower = tok.lower()
        if tok_lower in word_set:
            score += 1
        # Also try token fragments (for hyphenated words etc.)
        for part in re.split(r"[-/]", tok_lower):
            if len(part) >= 3 and part in word_set:
                score += 1
    return score


# For each (font, pua_char) pair collect sample words
pair_examples: dict[tuple[str, str], list[str]] = defaultdict(list)
for rec in records:
    font = rec["font"]
    word = rec["word"]
    for cp in rec.get("residual_chars", []):
        key = (font, cp)
        if len(pair_examples[key]) < 60:          # keep up to 60 samples
            pair_examples[key].append(word)

# Score every letter for every pair
pair_scores: dict[tuple[str, str], dict[str, int]] = {}
for (font, cp_str), words in pair_examples.items():
    # Get raw PUA char from cp string
    cp_int  = int(cp_str[2:], 16)
    pua_raw = chr(cp_int)

    letter_scores: Counter = Counter()
    for w in words:
        if pua_raw not in w:
            continue
        for letter in string.ascii_lowercase:
            letter_scores[letter] += score_substitution(w, pua_raw, letter)
    pair_scores[(font, cp_str)] = dict(letter_scores)

# Sort pairs by total confidence signal (total score)
def best_letter(scores: dict[str, int]) -> tuple[str, int]:
    if not scores:
        return ("?", 0)
    best = max(scores, key=lambda l: (scores[l], LETTER_FREQ.index(l) if l in LETTER_FREQ else 99))
    return (best, scores[best])

print(f"\n{'Font':<10} {'PUA':>8}  {'Best':>6}  {'Score':>6}  Scores (top 5)")
print("-" * 72)

# Collect results per font for the final dict
proposed: dict[str, dict[str, str]] = defaultdict(dict)

for (font, cp_str) in sorted(pair_scores.keys(), key=lambda k: (k[0], k[1])):
    scores = pair_scores[(font, cp_str)]
    letter, score = best_letter(scores)
    top5 = ", ".join(
        f"{l}={v}" for l, v in sorted(scores.items(), key=lambda x: -x[1])[:5]
    )
    print(f"{font:<10} {cp_str:>8}  {letter!r:>6}  {score:>6}  {top5}")
    if letter != "?" and score > 0:
        cp_int  = int(cp_str[2:], 16)
        pua_raw = chr(cp_int)
        proposed[font][pua_raw] = letter

# ---------------------------------------------------------------------------
# Step 5 – Output Python dict for merging into FONT_PUA_MAP
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 5 — Proposed additions to FONT_PUA_MAP")
print("=" * 72)
print()
print("# ----------------------------------------------------------------")
print("# PROPOSED NEW / UPDATED ENTRIES — merge into tmt_parser.py")
print("# (Only entries NOT already in FONT_PUA_MAP are shown.)")
print("# ----------------------------------------------------------------")

# Existing mappings from tmt_parser.py (hard-coded for comparison)
EXISTING: dict[str, dict[str, str]] = {
    "f14":  {"\ue02c": "t"},
    "f15":  {"\ue030": "f"},
    "f19":  {"\ue005": "-", "\ue006": "[", "\ue007": "]"},
    "f2":   {"\ue034": "c", "\ue035": "c", "\ue040": "o", "\ue041": "o"},
    "f21":  {"\ue025": "a", "\ue029": "e", "\ue02c": "h", "\ue033": "p", "\ue035": "r", "\ue037": "t"},
    "f25":  {"\ue038": "c", "\ue044": "o"},
    "f3":   {"\ue035": "a", "\ue041": "m"},
    "f30":  {"\ue021": "a", "\ue025": "e", "\ue028": "h", "\ue02f": "p", "\ue031": "r", "\ue033": "t"},
    "f32":  {"\ue026": "a", "\ue02a": "e", "\ue035": "p", "\ue036": "r", "\ue038": "t"},
    "f36":  {"\ue03c": "-", "\ue04b": ""},
    "f37":  {"\ue030": "t", "\ue039": "a", "\ue045": "h", "\ue048": "p"},
    "f40":  {"\ue01a": "a", "\ue01e": "e", "\ue021": "h", "\ue029": "p", "\ue02a": "r", "\ue02c": "t", "\ue043": "m", "\ue046": "p"},
    "f42":  {"\ue032": "p"},
    "f44":  {"\ue03c": "c", "\ue042": "i", "\ue045": "l", "\ue048": "o"},
    "f45":  {"\ue006": "(", "\ue007": ")", "\ue009": "-", "\ue023": "a", "\ue027": "e", "\ue02a": "h", "\ue031": "p", "\ue032": "r", "\ue034": "t", "\ue03a": "a", "\ue03c": "c", "\ue046": "m", "\ue048": "o", "\ue049": "\u2013"},
    "f6":   {"\ue02e": "h", "\ue029": "a", "\ue02f": "h", "\ue030": "h", "\ue037": "p"},
    "f69":  {"\ue044": "o"},
    "f72":  {"\ue006": "(", "\ue007": ")", "\ue034": "h", "\ue03d": "p"},
    "f73":  {"\ue005": "(", "\ue006": ")", "\ue008": "-"},
    "f89":  {"\ue045": "o"},
    "f90":  {"\ue020": "2", "\ue021": "a", "\ue025": "e", "\ue028": "h", "\ue030": "r", "\ue032": "t"},
    "f97":  {"\ue039": "p"},
    "f99":  {"\ue023": "a", "\ue027": "e", "\ue032": "p", "\ue033": "r", "\ue035": "t"},
    "f103": {"\ue043": "o"},
    "f104": {"\ue034": "c", "\ue040": "o"},
    "f111": {"\ue03b": "f"},
    "f134": {"\ue036": "h", "\ue03d": "p"},
    "f139": {"\ue020": "a", "\ue024": "e", "\ue02f": "p", "\ue030": "r", "\ue032": "t"},
    "f140": {"\ue02a": "t"},
    "f141": {"\ue01e": "a", "\ue021": "e", "\ue02e": "t", "\ue025": "a", "\ue036": "t"},
    "f155": {"\ue011": "4", "\ue039": "c", "\ue03b": "e", "\ue045": "o"},
    "f163": {"\ue038": "r"},
    "f164": {"\ue03b": "e"},
    "f174": {"\ue020": "a", "\ue031": "t"},
}

new_entries: dict[str, dict[str, str]] = defaultdict(dict)
for font, char_map in proposed.items():
    existing_for_font = EXISTING.get(font, {})
    for pua_raw, letter in char_map.items():
        if pua_raw not in existing_for_font:
            new_entries[font][pua_raw] = letter

# Print as a Python dict literal
print("PROPOSED_PUA_MAP_ADDITIONS = {")
for font in sorted(new_entries.keys()):
    char_map = new_entries[font]
    if not char_map:
        continue
    print(f'    # {font}')
    print(f'    "{font}": {{')
    for pua_raw, letter in sorted(char_map.items(), key=lambda x: ord(x[0])):
        cp_str = f"U+{ord(pua_raw):04X}"
        print(f'        "\\u{ord(pua_raw):04x}": "{letter}",  # {cp_str}')
    print("    },")
print("}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total_new_pairs = sum(len(v) for v in new_entries.values())
total_existing  = sum(len(v) for v in EXISTING.values())
print(f"\n  Existing FONT_PUA_MAP entries  : {total_existing}")
print(f"  Newly proposed entries         : {total_new_pairs}")
print(f"  Unique new fonts               : {len(new_entries)}")

# Print a note about U+E03F
print("\n--- U+E03F summary ---")
e03f_fonts = {f for (f, cp) in pair_scores if cp == "U+E03F"}
for fnt in sorted(e03f_fonts):
    sc = pair_scores.get((fnt, "U+E03F"), {})
    l, v = best_letter(sc)
    top3 = ", ".join(f"{ll}={vv}" for ll, vv in sorted(sc.items(), key=lambda x: -x[1])[:3])
    print(f"  {fnt}: best={l!r} (score={v})  top3=[{top3}]")

print("\nDone.")
