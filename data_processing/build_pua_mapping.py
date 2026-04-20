"""
build_pua_mapping.py
────────────────────
Automatically maps Private Use Area (PUA) characters in the TMT PDF to their
most likely ASCII equivalents by frequency analysis against the system
dictionary.

Strategy
--------
1. Extract text from pages 20-200 (0-indexed 19-199) using page.get_text("text").
2. Collect every word that contains at least one PUA character (U+E000–U+F8FF).
3. For each unique PUA character, try substituting it with every letter a-z
   plus common punctuation and count how many resulting words appear in the
   system dictionary.  The candidate with the most matches wins.
4. For words with MULTIPLE PUA characters, iterate: solve the most-constrained
   (highest-frequency) character first, substitute it everywhere, then repeat
   until all PUA chars are resolved or no progress can be made.
5. Report uncertain mappings (low match count, ties).
6. Print the final PUA_MAPPING dict.

Dependencies: PyMuPDF (fitz), standard library only.
"""

import fitz  # PyMuPDF
import re
import collections
import unicodedata

# ── Configuration ──────────────────────────────────────────────────────────────
PDF_PATH   = "/Users/karim/Desktop/folders/Medora_StartUp/2022, CURRENT Medical Diagnosis and Treatment- Original.pdf"
DICT_PATH  = "/usr/share/dict/words"
PAGE_START = 19   # 0-indexed → PDF page 20
PAGE_END   = 200  # exclusive  → PDF page 200
MAX_EXAMPLES = 200

CANDIDATES = list("abcdefghijklmnopqrstuvwxyz") + ["-", "'", "."]
PUA_RE     = re.compile(r"[\uE000-\uF8FF]")

# ── 1. Load system dictionary ──────────────────────────────────────────────────
def load_dictionary(path: str) -> set:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        words = {w.strip().lower() for w in fh if w.strip()}
    print(f"[dict] Loaded {len(words):,} words from {path}")
    return words


# ── 2. Extract text and collect PUA word examples ─────────────────────────────
def collect_pua_words(pdf_path: str, page_start: int, page_end: int):
    """
    Returns:
        pua_char_freq  : Counter  { pua_char -> total occurrences in running text }
        pua_word_bank  : dict     { pua_char -> list of (word, page_num) }  (≤MAX_EXAMPLES each)
        all_pua_words  : list of words that contain any PUA character
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    end = min(page_end, total_pages)
    print(f"[pdf] Opened '{pdf_path}' — {total_pages} pages total")
    print(f"[pdf] Scanning pages {page_start+1}–{end} (0-indexed {page_start}–{end-1})")

    pua_char_freq  = collections.Counter()
    pua_word_bank  = collections.defaultdict(list)  # char -> [(word, page)]

    for page_num in range(page_start, end):
        page = doc[page_num]
        text = page.get_text("text")

        # Count raw PUA character frequency in running text
        for ch in text:
            if 0xE000 <= ord(ch) <= 0xF8FF:
                pua_char_freq[ch] += 1

        # Tokenise into words (allow PUA chars inside words)
        words = re.findall(r"[A-Za-z\uE000-\uF8FF][A-Za-z\uE000-\uF8FF'\-\.]*", text)
        for word in words:
            if PUA_RE.search(word):
                for ch in set(word):
                    if 0xE000 <= ord(ch) <= 0xF8FF:
                        if len(pua_word_bank[ch]) < MAX_EXAMPLES:
                            pua_word_bank[ch].append((word.lower(), page_num + 1))

    doc.close()

    unique_pua = sorted(pua_char_freq.keys(), key=lambda c: -pua_char_freq[c])
    print(f"\n[scan] Found {len(unique_pua)} unique PUA characters across all text")
    print(f"[scan] Total PUA char occurrences: {sum(pua_char_freq.values()):,}\n")

    for ch in unique_pua:
        code = ord(ch)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "UNNAMED"
        print(f"  U+{code:04X}  {name:30s}  freq={pua_char_freq[ch]:6d}  "
              f"word_examples={len(pua_word_bank[ch])}")

    return pua_char_freq, pua_word_bank


# ── 3. Score a single PUA char substitution ────────────────────────────────────
def score_substitution(pua_char: str, replacement: str,
                        word_bank: list, dictionary: set,
                        known_mapping: dict) -> tuple[int, list]:
    """
    For every (word, page) in word_bank:
      - first apply known_mapping to resolve already-solved PUA chars
      - then replace pua_char with replacement
      - if the resulting word is pure ASCII and in the dictionary, count it
    Returns (match_count, list_of_matched_words).
    """
    matched = []
    for word, page in word_bank:
        # Apply already-known substitutions
        resolved = word
        for k, v in known_mapping.items():
            resolved = resolved.replace(k, v)
        # Apply the candidate substitution
        resolved = resolved.replace(pua_char, replacement)
        # Only score if no PUA chars remain (clean word)
        if not PUA_RE.search(resolved) and resolved.lower() in dictionary:
            matched.append(resolved.lower())
    return len(matched), matched


# ── 4. Solve one PUA character ─────────────────────────────────────────────────
def solve_one(pua_char: str, word_bank: list,
              dictionary: set, known_mapping: dict) -> dict:
    """
    Try every candidate replacement, return a result dict with:
        winner, win_score, runner_up, runner_score, examples, uncertain
    """
    scores = {}
    examples_map = {}
    for cand in CANDIDATES:
        score, matched = score_substitution(
            pua_char, cand, word_bank, dictionary, known_mapping
        )
        scores[cand] = score
        examples_map[cand] = matched

    best_score = max(scores.values())
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    winner, win_score    = ranked[0]
    runner_up, run_score = ranked[1] if len(ranked) > 1 else ("?", 0)

    # Uncertainty flags
    uncertain = False
    reason    = ""
    if win_score == 0:
        uncertain = True
        reason = "NO DICTIONARY MATCHES"
    elif run_score > 0 and run_score / win_score >= 0.85:
        uncertain = True
        reason = f"TIE/CLOSE: '{winner}'({win_score}) vs '{runner_up}'({run_score})"

    return {
        "char":        pua_char,
        "winner":      winner,
        "win_score":   win_score,
        "runner_up":   runner_up,
        "run_score":   run_score,
        "examples":    examples_map[winner][:5],
        "uncertain":   uncertain,
        "reason":      reason,
        "all_scores":  ranked[:5],
    }


# ── 5. Iterative multi-PUA solver ──────────────────────────────────────────────
def build_mapping(pua_char_freq: collections.Counter,
                  pua_word_bank: dict,
                  dictionary: set) -> tuple[dict, list]:
    """
    Iteratively solves the easiest (highest-freq) PUA characters first,
    using solved chars to help disambiguate the remaining ones.

    Returns:
        mapping   : { pua_char -> replacement_letter }
        uncertain : list of result dicts for uncertain characters
    """
    mapping   = {}
    uncertain = []
    results   = {}

    # Order by frequency (most frequent = most evidence = easiest to solve)
    order = sorted(pua_char_freq.keys(), key=lambda c: -pua_char_freq[c])

    max_passes = len(order) + 1  # safety cap
    remaining  = list(order)
    last_size  = -1

    for pass_num in range(1, max_passes + 1):
        if not remaining:
            break
        if len(remaining) == last_size:
            # No progress last pass — give up on the rest
            print(f"\n[solver] No progress on pass {pass_num}; "
                  f"{len(remaining)} characters unresolvable.")
            for ch in remaining:
                r = solve_one(ch, pua_word_bank[ch], dictionary, mapping)
                r["uncertain"] = True
                if not r["reason"]:
                    r["reason"] = "STUCK IN MULTI-PUA CLUSTER"
                uncertain.append(r)
                results[ch] = r
            break

        last_size = len(remaining)
        newly_solved = []

        print(f"\n[solver] Pass {pass_num} — {len(remaining)} characters to solve")

        for ch in remaining:
            r = solve_one(ch, pua_word_bank[ch], dictionary, mapping)
            results[ch] = r

            if not r["uncertain"]:
                mapping[ch] = r["winner"]
                newly_solved.append(ch)
                print(f"  SOLVED  U+{ord(ch):04X} -> '{r['winner']}'  "
                      f"score={r['win_score']}  "
                      f"examples={r['examples'][:3]}")
            else:
                print(f"  PENDING U+{ord(ch):04X}  {r['reason']}")

        for ch in newly_solved:
            remaining.remove(ch)

    # Second chance: re-try uncertain ones now that mapping may be richer
    if uncertain:
        still_uncertain = []
        print(f"\n[solver] Re-trying {len(uncertain)} uncertain characters "
              f"with enriched mapping …")
        for r in uncertain:
            ch = r["char"]
            r2 = solve_one(ch, pua_word_bank[ch], dictionary, mapping)
            if not r2["uncertain"]:
                mapping[ch] = r2["winner"]
                print(f"  RESCUED U+{ord(ch):04X} -> '{r2['winner']}'  "
                      f"score={r2['win_score']}")
            else:
                still_uncertain.append(r2)
                print(f"  STILL UNCERTAIN U+{ord(ch):04X}  {r2['reason']}")
                # Still assign the best guess
                if r2["win_score"] > 0:
                    mapping[ch] = r2["winner"]
        uncertain = still_uncertain

    return mapping, uncertain, results


# ── 6. Report ──────────────────────────────────────────────────────────────────
def report(mapping: dict, uncertain: list, results: dict,
           pua_char_freq: collections.Counter):
    sep = "─" * 70

    print("\n" + "=" * 70)
    print("  FINAL PUA MAPPING REPORT")
    print("=" * 70)

    # Full table
    print(f"\n{'Codepoint':<12} {'Freq':>6}  {'->':>2}  {'Score':>6}  "
          f"{'Runner-up':>10}  Example words")
    print(sep)

    for ch in sorted(mapping.keys(), key=lambda c: ord(c)):
        r     = results.get(ch, {})
        freq  = pua_char_freq[ch]
        win   = mapping[ch]
        score = r.get("win_score", "?")
        ru    = r.get("runner_up", "?")
        rsc   = r.get("run_score", "?")
        exs   = ", ".join(r.get("examples", [])[:5])
        flag  = "  *** UNCERTAIN ***" if ch in [u["char"] for u in uncertain] else ""
        print(f"  U+{ord(ch):04X}  {freq:6d}  ->  '{win}'  "
              f"score={score:>5}  runner='{ru}'({rsc})  {exs}{flag}")

    # Uncertain section
    if uncertain:
        print(f"\n{'='*70}")
        print(f"  UNCERTAIN MAPPINGS  ({len(uncertain)} characters)")
        print(f"{'='*70}")
        for r in uncertain:
            ch = r["char"]
            print(f"\n  U+{ord(ch):04X}  freq={pua_char_freq[ch]}")
            print(f"    Reason   : {r['reason']}")
            print(f"    Top candidates:")
            for cand, sc in r["all_scores"][:5]:
                print(f"      '{cand}' -> {sc} dict matches")
            if r["examples"]:
                print(f"    Examples : {r['examples'][:5]}")
    else:
        print("\n  No uncertain mappings — all characters resolved confidently.")

    # Python dict output
    print("\n" + "=" * 70)
    print("  PYTHON DICTIONARY")
    print("=" * 70)
    print("\nPUA_MAPPING = {")
    for ch in sorted(mapping.keys(), key=lambda c: ord(c)):
        esc = f"\\u{ord(ch):04x}"
        print(f'    "{esc}": "{mapping[ch]}",')
    print("}")


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  PUA Character Mapper for TMT PDF")
    print("=" * 70 + "\n")

    dictionary      = load_dictionary(DICT_PATH)
    pua_char_freq, pua_word_bank = collect_pua_words(PDF_PATH, PAGE_START, PAGE_END)

    if not pua_char_freq:
        print("\nNo PUA characters found in the specified page range. Exiting.")
        return

    mapping, uncertain, results = build_mapping(pua_char_freq, pua_word_bank, dictionary)
    report(mapping, uncertain, results, pua_char_freq)


if __name__ == "__main__":
    main()
