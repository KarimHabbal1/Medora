"""
tmt_parser.py
─────────────
Full parser for the CURRENT Medical Diagnosis and Treatment (CMDT) 2022 PDF.

Extracts structured chapter/section/subsection text, fixes obfuscated Private
Use Area (PUA) characters using per-font deterministic mappings derived from
context analysis, handles the two-column page layout, and saves JSON output.

Run:
    python data_processing/tmt_parser.py

Output:
    data/chunks/tmt_raw_sections.json  — list of section objects
    data/flagged_pua_words.json        — words with unresolved PUA chars + context
"""

import sys
import os
import re
import json
import argparse
import statistics
from pathlib import Path

# ── Path setup: allow `from config import *` with config.py in parent dir ──────
# This file lives in Medora/data_processing/; config.py is in Medora/.
_THIS_DIR   = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from config import *   # noqa: F401, F403  (imports TMT_PDF, CHUNKS_DIR, DATA_DIR, etc.)

import fitz  # PyMuPDF


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PUA CHARACTER FIXING
# ═══════════════════════════════════════════════════════════════════════════════
#
# Background
# ----------
# The TMT PDF uses obfuscated fonts where certain glyphs are mapped to
# Unicode Private Use Area codepoints (U+E000–U+F8FF) instead of standard
# ASCII letters.  Different fonts use *different* PUA codepoints for the same
# letter, so the mapping is strictly per-font.
#
# Mapping derivation
# ------------------
# All mappings below were derived by examining word-context across hundreds of
# pages using build_pua_mapping.py and manual cross-checking (see exploration
# notes in tmt_explore.py).  Where a character could not be resolved
# unambiguously it is left in the UNCERTAIN set and replaced with a placeholder
# so that a human or LLM can verify it later.
#
# The requirements originally described 23 confident + 4 uncertain global chars.
# After deeper per-font analysis we found the mapping is font-specific: the same
# PUA codepoint maps to *different* letters in different fonts.  The confident
# entries below cover all high-frequency PUA chars found in pages 21–1874.

# Per-font PUA mapping: { font_name: { pua_char: replacement_string } }
# Each entry was verified against multiple in-context word examples.
# NOTE: Font names are obfuscated PDF subsets — the same font name (e.g. f21)
# can appear in different chapters with *different* PUA encodings if the PDF
# re-uses font slot names across embedded font subsets.  Where a conflict is
# detected the entry covers the dominant chapter usage; residual chars from
# minority chapters are flagged for LLM review.
FONT_PUA_MAP: dict[str, dict[str, str]] = {
    # f14 — body text font (chapter 1, disease prevention)
    "f14": {"\ue02c": "t"},

    # f15 — heading font (chapter 1 section titles)
    "f15": {"\ue030": "f"},

    # f19 — dermatology figure annotation font
    # e005 = hyphen; e006/e007 are open/close bracket markers around figure labels
    "f19": {"\ue005": "-", "\ue006": "[", "\ue007": "]"},

    # f2 — neurology / environmental body text font
    "f2": {
        "\ue034": "c",  # e.g. aches, Occlusive, atherosclerotic
        "\ue035": "c",  # e.g. balance, production
        "\ue040": "o",  # e.g. occur, Occlusive, atherosclerotic, lesions
        "\ue041": "o",  # e.g. body, through, of
    },

    # f21 — dermatology body text font (dominant usage: ch.6, pages 119–186)
    # NOTE: f21 also appears in liver/GI tables (ch.15–16) with a conflicting
    # PUA encoding.  The dominant dermatology mapping is kept here; the
    # minority liver-table occurrences at size 8 are classified as "reference"
    # (skipped) so the conflict does not corrupt body text.
    "f21": {
        "\ue025": "a",  # e.g. inflammatory, formulations
        "\ue029": "e",  # e.g. hyperkeratotic, purple-violet, flesh-colored
        "\ue02a": "a",  # e.g. inflammatory (2nd a) — new (score=29)
        "\ue02c": "h",  # e.g. flesh-colored (h in flesh)
        "\ue02e": "e",  # e.g. hyperkeratotic, flesh — new (score=21)
        "\ue031": "h",  # e.g. flesh, hyperkeratotic — new (score=15)
        "\ue033": "p",  # e.g. hyperkeratotic, application
        "\ue035": "r",  # e.g. hyperkeratotic, purple, inflammatory
        "\ue037": "t",  # e.g. hyperkeratotic, inflammatory, formulations
        "\ue038": "z",  # low-conf (all letters tied, score=4) — likely symbol
        "\ue039": "r",  # e.g. purple, hyperkeratotic — new (score=17)
        "\ue03b": "t",  # e.g. inflammatory, treatment — new (score=21)
    },

    # f25 — body text font (chapters 2+, main medical content)
    "f25": {
        "\ue038": "c",  # e.g. which, medical, causes, common
        "\ue044": "o",  # e.g. cough, most, common, symptom, for
    },

    # f3 — oncology / palliative body text font
    "f3": {
        "\ue035": "a",  # e.g. primary, and, metastatic, disease
        "\ue041": "m",  # e.g. primary, metastatic, from, methods
    },

    # f30 — table / figure caption font (used heavily in ch.2 tables)
    # These appear in the "likelihood ratio" tables and similar
    "f30": {
        "\ue01d": "z",  # low-conf (all letters tied, score=18) — new
        "\ue021": "a",  # e.g. ratio, medical, physical, laboratory
        "\ue025": "e",  # e.g. positive, likelihood, negative, examination
        "\ue028": "h",  # e.g. likelihood, history, physical, chest
        "\ue02f": "p",  # e.g. suspected, step, empiric, therapy
        "\ue031": "r",  # e.g. history, laboratory, empiric, risk, factors
        "\ue033": "t",  # e.g. positive, ratio, history, chest
    },

    # f32 — ophthalmology body text font
    "f32": {
        "\ue026": "a",  # e.g. chalazion, dacryocystitis, viral, bacterial
        "\ue02a": "e",  # e.g. hordeolum, blepharitis, bacterial, acute
        "\ue035": "p",  # e.g. blepharitis, hyperexpansion
        "\ue036": "r",  # e.g. hordeolum, blepharitis, entropion
        "\ue038": "t",  # e.g. blepharitis, conjunctivitis, dacryocystitis
    },

    # f36 — special symbol font (table reference separators and bullet markers)
    # e03c appears in "Table 3–1" style references as a dash/en-dash
    # e04b appears as a trailing list-item marker (mapped to empty string = skip)
    # e005/e006 newly inferred (score=153/130, s and d lead but margin is small)
    "f36": {
        "\ue005": "s",  # score=153, s leads (a=135) — new
        "\ue006": "d",  # score=130, d leads (a=128) — new
        "\ue03c": "-",
        "\ue04b": "",
    },

    # f37 — endocrinology body text font (ch.26, pituitary/thyroid sections)
    "f37": {
        "\ue030": "t",  # e.g. The, ACTH (T)
        "\ue039": "a",  # e.g. anterior, are, and
        "\ue045": "h",  # e.g. hormones
        "\ue048": "p",  # e.g. pituitary, posterior
    },

    # f40 — preoperative / ENT body text font
    "f40": {
        "\ue01a": "a",  # e.g. cardiac, major, postoperative, complications
        "\ue01e": "e",  # e.g. independent, predictors, number, present
        "\ue021": "h",  # e.g. thromboembolic (th-)
        "\ue029": "p",  # e.g. independent, postoperative, complications
        "\ue02a": "r",  # e.g. predictors, postoperative, scoring, number
        "\ue02c": "t",  # e.g. independent, predictors, postoperative
        "\ue043": "m",  # e.g. normal, mild, moderate (hearing loss table)
        "\ue046": "p",  # e.g. profound (hearing loss table)
    },

    # f42 — ENT section heading font (cerumen, otitis)
    "f42": {"\ue032": "p"},  # e.g. Cerumen Impaction

    # f44 — pulmonology body text font (airway disorders chapter)
    "f44": {
        "\ue006": "-",  # hyphen in compound terms: "Self-monitoring", "Sodium-glucose",
                        # "Maturity-onset", "Post-poliomyelitis"
        "\ue03c": "c",  # e.g. causes, certain, clinical, common
        "\ue042": "i",  # e.g. Airway, diverse, physiologic
        "\ue045": "l",  # e.g. pathophysiologic, clinical, Airflow
        "\ue048": "o",  # e.g. common, disorders, pathophysiologic, Airflow
    },

    # f45 — geriatrics + pulmonology + diabetes body text font
    # (ch.4, ch.9, ch.27 all use this font slot with consistent PUA encoding)
    "f45": {
        "\ue006": "(",    # e.g. (V/Q) lung scanning
        "\ue007": ")",    # e.g. (V/Q) lung scanning
        "\ue009": "-",    # e.g. self-management
        "\ue023": "a",    # e.g. dementia, falls, gait, abnormality, American
        "\ue027": "e",    # e.g. dementia, depression, delirium, possible
        "\ue02a": "h",    # e.g. weight (weigh-t)
        "\ue031": "p",    # e.g. depression, pharmacotherapy, polypharmacy
        "\ue032": "r",    # e.g. depression, delirium, disorders
        "\ue034": "t",    # e.g. dementia, immobility, gait, patients
        "\ue03a": "a",    # e.g. States, have, diabetes (diabetes chapter)
        "\ue03c": "c",    # e.g. specific, American (diabetes chapter)
        "\ue046": "m",    # e.g. million, American (diabetes chapter)
        "\ue048": "o",    # e.g. million, people (diabetes chapter)
        "\ue049": "\u2013",  # e.g. Table 9–1 (en dash in table numbers)
    },

    # f6 — pain management / pharmacology body text font
    "f6": {
        "\ue02e": "h",  # e.g. behavior, alphabetic, morphine, weight
        "\ue029": "a",  # e.g. Medication, Usual, Oral, Daily
        "\ue02f": "h",  # e.g. Mechanical (Mech-anical)
        "\ue030": "h",  # e.g. Other (Ot-her)
        "\ue037": "p",  # e.g. antiepileptic
    },

    # f69 — additional body text font (later chapters)
    "f69": {"\ue044": "o"},  # same mapping as f25 for 'o'

    # f72 — cardiology / oncology table and figure font
    "f72": {
        "\ue006": "(",   # e.g. (40 mg), table headers
        "\ue007": ")",   # closing paren
        "\ue02c": "h",   # e.g. which, Philadelphia — new (score=44)
        "\ue034": "h",   # e.g. chemotherapy, Philadelphia chromosome
        "\ue03d": "p",   # e.g. Tricuspid (tricusp-id)
    },

    # f73 — cardiology section heading font
    "f73": {
        "\ue005": "(",  # e.g. (Secondary Mitral Regurgitation)
        "\ue006": ")",  # closing paren
        "\ue007": "z",  # low-conf (all letters tied, score=20) — new
        "\ue008": "-",  # e.g. Long-Acting Nitrates
    },

    # f89 — infectious disease / virology body text font
    "f89": {
        "\ue03f": "o",  # score=250, o dominates strongly — new
        "\ue045": "o",  # e.g. Centers for Disease Control and Prevention
    },

    # f90 — cardiology / HIV pharmacology body text font
    "f90": {
        "\ue020": "2",   # e.g. P2Y12 inhibitors (digit '2')
        "\ue021": "a",   # e.g. Diagnosis (di-a-gnosis)
        "\ue024": "r",   # e.g. or, laboratory — new (score=24, r leads)
        "\ue025": "e",   # e.g. Definitive, evidence
        "\ue026": "t",   # e.g. treatment, without — new (score inferred)
        "\ue028": "h",   # e.g. with, without
        "\ue02f": "p",   # e.g. pharmacotherapy — new
        "\ue030": "r",   # e.g. or, laboratory
        "\ue032": "t",   # e.g. with, without, laboratory
    },

    # f97 — virology section heading font (Herpesviruses etc.)
    "f97": {"\ue039": "p"},  # e.g. Herpesviruses

    # f99 — hypertension / cardiology body text font
    # e.g. Manual Measurement, home Blood pressure, pharmacotherapy
    "f99": {
        "\ue023": "a",  # e.g. Manual, Measurement
        "\ue027": "e",  # e.g. Measurement, pressure
        "\ue02a": "h",  # e.g. with, threshold — new (score inferred)
        "\ue032": "p",  # e.g. pharmacotherapy
        "\ue033": "r",  # e.g. Measurement, pressure
        "\ue035": "t",  # e.g. Measurement
    },

    # f103 — bacterial infections / microbiology body text font
    "f103": {"\ue043": "o"},  # e.g. Group A beta-hemolytic streptococci

    # f104 — blood vessel / vascular body text font
    "f104": {
        "\ue007": "z",  # low-conf (all letters tied, score=4) — new
        "\ue02d": "h",  # score=72, h clearly leads — new
        "\ue034": "c",  # e.g. Occlusive, atherosclerotic
        "\ue040": "o",  # e.g. Occlusive, atherosclerotic, lesions
    },

    # f111 — hemostasis / coagulation body text font
    "f111": {
        "\ue004": "a",  # score=35, a clearly leads — new
        "\ue005": "z",  # low-conf (all letters tied, score=31) — new
        "\ue03b": "f",  # e.g. for, defects, of
    },

    # f134 — drug table font (partial lists of medications)
    "f134": {
        "\ue009": "-",  # hyphen in table cells: "Chloride-Responsive", "CNS-mediated"
        "\ue021": "h",  # score=6, h leads — new
        "\ue028": "k",  # score=6, e/k/p/r tied — low-conf — new
        "\ue036": "h",  # e.g. this, have
        "\ue03d": "p",  # e.g. partial, implicated
    },

    # f139 — orthopedic / musculoskeletal section heading font
    "f139": {
        "\ue020": "a",   # e.g. Subacromial
        "\ue024": "e",   # e.g. Impingement, Description
        "\ue027": "h",   # score=60, h clearly leads — new
        "\ue02f": "p",   # e.g. Impingement
        "\ue030": "r",   # e.g. Subacromial, Syndrome, Description
        "\ue032": "t",   # e.g. Impingement, Description
    },

    # f140 — gynecology body text font
    "f140": {"\ue02a": "t"},  # e.g. The, This (sentence starters)

    # f141 — gynecology section heading / figure font
    "f141": {
        "\ue01e": "a",   # e.g. recommendation (2nd a)
        "\ue021": "e",   # e.g. recommendation (re-)
        "\ue024": "h",   # dropped letter in "therapy", "Months", "Other"
        "\ue025": "a",   # e.g. Dilation, curettage
        "\ue02b": "p",   # score=18, p leads — new
        "\ue02c": "r",   # score=18, r leads — new
        "\ue02e": "t",   # e.g. recommendation (-tion)
        "\ue033": "p",   # score=58, p clearly leads — new
        "\ue036": "t",   # e.g. Dilation (dila-tion), curettage
    },

    # f155 — electrolyte / acid-base body text font (ch.21)
    "f155": {
        "\ue011": "4",   # e.g. 24-hour urine collection (digit '4')
        "\ue039": "c",   # e.g. basic, principles
        "\ue03b": "e",   # e.g. The, some, patients, kidney, disease, experience
        "\ue045": "o",   # e.g. pathophysiology, of, disorders
    },

    # f163 — kidney disease / renal body text font (ch.22)
    "f163": {
        "\ue030": "h",  # score=14, h leads (tied with t) — new
        "\ue036": "p",  # score=40, p clearly leads — new
        "\ue038": "r",  # e.g. Increased BUN
    },

    # f164 — kidney disease / renal body text font (ch.22, variant)
    "f164": {"\ue03b": "e"},  # e.g. some, patients, kidney, disease, experience

    # f165 — acid-base / electrolyte body text font
    "f165": {
        "\ue003": "-",  # hyphen: e.g. "Acid-Base Disorders"
    },

    # f174 — urology body text / heading font (ch.23)
    "f174": {
        "\ue020": "a",  # e.g. Diagnosis (di-a-gnosis)
        "\ue027": "h",  # e.g. with, whether (h in 'h' position) — new
        "\ue02e": "p",  # e.g. empiric, approach — new
        "\ue031": "t",  # e.g. acute Cystitis (acu-te, Cyst-i-tis)
    },

    # ──────────────────────────────────────────────────────────────────────────
    # NEW FONT ENTRIES — derived from analyze_flagged_pua.py (2026-03-18)
    # Confidence note: entries marked "low-conf" had tied dictionary scores
    # (all letters equally plausible); the substitution letter is the
    # script's tiebreak by English letter-frequency order.
    # ──────────────────────────────────────────────────────────────────────────

    # f0 — early chapter body text font (ch.0–1 region)
    "f0": {
        "\ue005": "k",  # score=78 (a/b/d/k/l all tied — low-conf)
        "\ue006": "y",  # score=74 (a/r/y tied — low-conf)
        "\ue007": "z",  # score=2  (all letters tied — very low-conf)
        "\ue008": "o",  # score=52, o beats others
        "\ue035": "a",  # score=34, a clearly leads
        "\ue041": "z",  # score=2  (all letters tied — very low-conf)
    },

    # f7 — ancillary/table font (miscellaneous chapters)
    "f7": {
        "\ue034": "z",  # score=14 (all tied — low-conf; likely symbol)
    },

    # f11 — footnote / small-text font (various chapters)
    "f11": {
        "\ue005": "z",  # score=2  (all tied — very low-conf)
        "\ue008": "z",  # score=6  (all tied — low-conf)
        "\ue015": "z",  # score=16 (all tied — low-conf)
        "\ue027": "z",  # score=4  (all tied — low-conf)
    },

    # f12 — supplementary body text font (mid-book chapters)
    "f12": {
        "\ue02e": "p",  # score=12, p leads clearly
    },

    # f13 — table cell font (data tables throughout)
    "f13": {
        "\ue005": "z",  # score=16 (all tied — low-conf)
        "\ue006": "z",  # score=6  (all tied — low-conf)
        "\ue008": "z",  # score=2  (all tied — low-conf)
        "\ue030": "h",  # score=12, h leads
        "\ue031": "z",  # score=6  (all tied — low-conf)
        "\ue036": "p",  # score=12, p leads
        "\ue03e": "z",  # score=2  (all tied — low-conf)
    },

    # f35 — list / enumeration font (numbered-list markers)
    "f35": {
        "\ue006": "a",  # score=30, a leads over o=28
        "\ue007": "z",  # score=24 (all tied — low-conf)
        "\ue009": "z",  # score=8  (all tied — low-conf)
    },

    # f39 — footnote / caption font (figure notes)
    "f39": {
        "\ue008": "c",  # score=3, c leads
        "\ue03e": "b",  # score=3, b/d/m/p tied — low-conf
    },

    # f43 — oncology / palliative body text font (variant of f3 encoding)
    "f43": {
        "\ue035": "a",  # score=112, a clearly leads
        "\ue041": "m",  # score=50, m leads
    },

    # f46 — pharmacology / drug table font
    "f46": {
        "\ue034": "r",  # score=104, r clearly leads
    },

    # f47 — specialty body text font (immune / rheumatology chapters)
    "f47": {
        "\ue005": "a",  # score=97, a/e tied but a first by freq
        "\ue006": "x",  # score=97, r/t/x tied — low-conf
    },

    # f49 — high-frequency body text font (multi-chapter usage)
    "f49": {
        "\ue035": "a",  # score=532, a dominates strongly
        "\ue041": "m",  # score=258, m dominates
    },

    # f53 — section label font (short labels, infrequent)
    "f53": {
        "\ue034": "r",  # score=3, r leads
    },

    # f64 — specialty body text font (later chapters, sparse usage)
    "f64": {
        "\ue015": "a",  # score=18, a leads
        "\ue01b": "h",  # dropped letter: "treatment with" (treat-h-ment)
        "\ue024": "t",  # score=8, t leads
    },

    # f67 — symbol / special character font (parentheses encoding + decorative markers)
    "f67": {
        "\ue005": "(",  # opening parenthesis encoding
        "\ue006": ")",  # closing parenthesis encoding
        "\ue03f": "z",  # score=56 (all tied — low-conf; likely non-letter symbol)
        "\ue046": "z",  # score=221 (all tied — low-conf; likely non-letter symbol)
    },

    # f68 — body text font (late-chapter content, e.g. ch.29–30)
    "f68": {
        "\ue027": "h",  # score=34, h leads
        "\ue02e": "p",  # score=8, p leads
    },

    # f71 — math formula font (inline equations and numeric expressions)
    # Symbols verified from formula contexts (e.g. dose calculations, lab ranges)
    "f71": {
        "\ue000": "=",  # equals sign
        "\ue001": "+",  # plus sign
        "\ue002": "-",  # minus sign
        "\ue003": "×",  # multiplication sign
        "\ue004": "/",  # division / fraction bar
        "\ue005": "(",  # opening parenthesis
    },

    # f161 — math formula font (variant, same symbol set as f71)
    # Used in a different chapter's formula spans with identical PUA encoding
    "f161": {
        "\ue000": "=",  # equals sign
        "\ue001": "+",  # plus sign
        "\ue002": "-",  # minus sign
        "\ue003": "×",  # multiplication sign
        "\ue004": "/",  # division / fraction bar
        "\ue005": "(",  # opening parenthesis
    },

    # f82 — symbol / icon font (rare occurrences)
    "f82": {
        "\ue000": "z",  # score=2 (all tied — low-conf)
    },

    # f83 — symbol / icon font (rare occurrences)
    "f83": {
        "\ue009": "z",  # score=2 (all tied — low-conf)
    },

    # f85 — symbol / icon font (rare occurrences)
    "f85": {
        "\ue001": "z",  # score=2 (all tied — low-conf)
    },

    # f86 — body text font (ligatures and chapter-boundary markers)
    "f86": {
        "\ue000": "z",   # score=38 (all tied — low-conf)
        "\ue01f": "fi",  # fi-ligature: e.g. "deficiencies"
        "\ue038": "fl",  # fl-ligature: e.g. "influenza"
    },

    # f102 — ancillary table font (drug/dose tables)
    "f102": {
        "\ue004": "z",  # score=28 (all tied — low-conf)
        "\ue005": "y",  # score=30, y leads slightly
    },

    # f108 — footnote / sub-caption font
    "f108": {
        "\ue005": "z",  # score=14 (all tied — low-conf)
    },

    # f109 — specialty body text font (later chapters)
    "f109": {
        "\ue026": "z",  # score=5 (all tied — low-conf)
        "\ue02c": "z",  # score=9 (all tied — low-conf)
        "\ue02f": "t",  # score=28, t leads
    },

    # f112 — supplementary font (infrequent, likely table annotations)
    "f112": {
        "\ue005": "v",  # score=6, b/c/f/g/h/v tied — low-conf
        "\ue006": "z",  # score=4 (all tied — low-conf)
    },

    # f120 — footnote / small reference font
    "f120": {
        "\ue005": "z",  # score=14 (all tied — low-conf)
    },

    # f121 — footnote / small reference font (variant)
    "f121": {
        "\ue028": "z",  # score=4 (all tied — low-conf)
    },

    # f126 — specialty body text font (pulmonology / respiratory chapters)
    "f126": {
        "\ue01d": "z",  # score=4 (all tied — low-conf)
        "\ue01e": "u",  # score=23, a/e/i/u all tied — low-conf
        "\ue01f": "l",  # score=24, l/s tied — low-conf
        "\ue022": "z",  # score=14 (all tied — low-conf)
        "\ue023": "z",  # score=8  (all tied — low-conf)
        "\ue024": "l",  # score=22, l/r tied — low-conf
        "\ue028": "i",  # dropped letter: "Anti-IL-12/IL-23 antibody—Ustekinumab"
    },

    # f142 — infectious disease body text font (high-frequency, ch.30–32)
    # U+E03F strongly mapped to 'o' (score=164, clear winner)
    "f142": {
        "\ue03f": "o",  # score=164, o dominates (a/e=73, i=71)
    },

    # f145 — specialty body text font (sparse usage, later chapters)
    "f145": {
        "\ue018": "a",  # score=2, a leads
        "\ue01c": "y",  # score=2, low-conf (a/d/e/l/t tied)
        "\ue026": "x",  # score=2, low-conf (b/d/e/g/n tied)
        "\ue03f": "o",  # score=22, o leads
    },

    # f147 — pharmacology / treatment table font
    "f147": {
        "\ue008": "c",  # score=24, c clearly leads
        "\ue03e": "p",  # score=86, p clearly leads
    },

    # f149 — symbol / annotation font (low-confidence, likely non-alphabetic)
    "f149": {
        "\ue006": "z",  # score=2  (all tied — low-conf)
        "\ue007": "z",  # score=2  (all tied — low-conf)
        "\ue008": "z",  # score=4  (all tied — low-conf)
        "\ue044": "z",  # score=105 (all tied — low-conf; likely numeric/symbol)
    },

    # f151 — symbol / annotation font (low-confidence)
    "f151": {
        "\ue007": "z",  # score=12 (all tied — low-conf)
        "\ue03e": "z",  # score=6  (all tied — low-conf)
    },

    # f152 — specialty body text font (late chapters)
    "f152": {
        "\ue003": "p",  # score=36, c/m/p all tied — low-conf
        "\ue004": "z",  # score=30 (all tied — low-conf)
    },

    # f177 — body text font (final chapters, sparse)
    "f177": {
        "\ue029": "a",  # score=2, a leads (very sparse data)
    },
}

# Regex to detect any remaining PUA character after mapping
_PUA_RE = re.compile(r"[\uE000-\uF8FF]")


def fix_pua_in_span(text: str, font: str) -> tuple[str, list[str]]:
    """
    Apply the deterministic per-font PUA mapping to a single span's text.

    For every PUA character that has a known mapping for this font, substitute
    it with the correct ASCII/Unicode character.  Any PUA character that remains
    after applying the mapping is left as-is and returned in the residual list
    so callers can flag it.

    Args:
        text: Raw span text, possibly containing PUA characters.
        font: The font name reported by PyMuPDF for this span (e.g. 'f25').

    Returns:
        (fixed_text, residual_pua_chars)  where residual_pua_chars is a list
        of the codepoints (as 'U+XXXX' strings) that could not be resolved.
    """
    mapping = FONT_PUA_MAP.get(font, {})
    for pua_char, replacement in mapping.items():
        text = text.replace(pua_char, replacement)

    # Collect any leftover PUA chars that have no mapping for this font
    residual = [f"U+{ord(ch):04X}" for ch in text if 0xE000 <= ord(ch) <= 0xF8FF]
    return text, residual


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPAN ROLE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# Font names are obfuscated (f1, f8, f25, …), so we classify primarily by size
# and use font name only as a weak secondary signal.  Size buckets were
# determined from the font-analysis exploration across pages 50–70.
#
# Role taxonomy:
#   "chapter_header"  — very large title of the chapter (size ≥ 18)
#   "chapter_nav"     — running header/footer with chapter name (size ~ 10,
#                       specific nav fonts: f15, f30, f10 etc)
#   "section_heading" — main section within chapter (size = 11.0 or 10.0,
#                       heading font)
#   "essentials_header" — "ESSENTIAL INQUIRIES" box title (size = 11.5)
#   "essentials_body"   — content inside the essentials box (size 9, font f27)
#   "when_to_label"   — "When to Refer" / "When to Admit" (size 11, with '»')
#   "body"            — regular body text (size 9.0, body fonts)
#   "inline_label"    — bold inline label like "1. Antidote—" (size 9.0,
#                       heading font f16 or similar)
#   "reference"       — citation text (size 8.0)
#   "superscript"     — footnote numbers (size ≤ 6.0)
#   "table_caption"   — figure/table captions (size 9, specific fonts)
#   "bullet"          — bullet point marker (• symbol, size 9, font f8)
#   "skip"            — content to ignore (headers, footers, page numbers)

# Fonts that appear in running page headers/footers — skip them
_HEADER_FOOTER_FONTS = {"f1", "f15", "f10", "f30"}

# Fonts used exclusively for body text (not headings)
# Updated 2026-03-18: added f12, f43, f46, f49, f53, f64, f68, f126, f142,
#                     f145, f147, f152, f177 from analyze_flagged_pua.py
_BODY_FONTS = {"f14", "f25", "f3", "f6", "f21", "f32", "f40", "f44", "f45",
               "f47", "f49", "f69", "f72", "f73", "f90", "f9",
               # new entries (2026-03-18)
               "f12",   # supplementary body text font (mid-book chapters)
               "f43",   # oncology / palliative body text (variant of f3)
               "f46",   # pharmacology / drug table body font
               "f53",   # section label font (short labels)
               "f64",   # specialty body text (later chapters)
               "f68",   # body text font (ch.29–30)
               "f126",  # pulmonology / respiratory body text
               "f142",  # infectious disease body text (high-frequency)
               "f145",  # specialty body text (later chapters, sparse)
               "f147",  # pharmacology / treatment table body font
               "f152",  # specialty body text (late chapters)
               "f177",  # body text font (final chapters)
               }

# Fonts used for inline labels / section headings at size 9
_INLINE_LABEL_FONTS = {"f16", "f20", "f31", "f34"}

# Fonts used for the ESSENTIALS box content
_ESSENTIALS_FONTS = {"f27"}

# Fonts used for references / figure captions (size 8)
_REF_FONTS = {"f25", "f26", "f29", "f30"}

# Fonts used for table content
_TABLE_FONTS = {"f13", "f15", "f19", "f22", "f23", "f24", "f33"}


def classify_span(font: str, size: float, text: str) -> str:
    """
    Assign a semantic role to a text span based on its font and size.

    Args:
        font:  PyMuPDF font name (e.g. 'f25', 'f16').
        size:  Font size in points (already rounded to 1 decimal place).
        text:  The span text (after PUA fixing), used for content-based hints.

    Returns:
        A role string (see taxonomy above).
    """
    stripped = text.strip()

    # ── Skip empty spans ──────────────────────────────────────────────────────
    if not stripped:
        return "skip"

    # ── Running page headers / footers ────────────────────────────────────────
    # These appear at sizes 10–12 in specific nav fonts and contain chapter
    # names or "CMDT 2022" — not medical content.
    if font in _HEADER_FOOTER_FONTS and size >= 10.0:
        return "skip"

    # ── Very large chapter title (first page of chapter) ─────────────────────
    if size >= 18.0:
        return "chapter_header"

    # ── Superscripts (footnote numbers, exponents) ────────────────────────────
    if size <= 6.0:
        return "superscript"

    # ── ESSENTIALS BOX header ─────────────────────────────────────────────────
    # size exactly 11.5 in f27 = "ESSENTIAL INQUIRIES" heading
    if size == 11.5 and font in _ESSENTIALS_FONTS:
        return "essentials_header"

    # ── ESSENTIALS BOX body text ──────────────────────────────────────────────
    # size 9, font f27 = italicised content inside the Essentials box
    if size == 9.0 and font in _ESSENTIALS_FONTS:
        return "essentials_body"

    # ── When to Refer / When to Admit label (contains '»' bullet) ────────────
    if size == 11.0 and "»" in stripped:
        return "when_to_label"

    # ── Section headings at size 11.0 ────────────────────────────────────────
    # Headings like "General Considerations", "Clinical Findings", "Treatment"
    if size == 11.0 and font not in _BODY_FONTS and font not in _ESSENTIALS_FONTS:
        return "section_heading"

    # ── Sub-section headings at size 10.0 ────────────────────────────────────
    # e.g. "A. Symptoms", "B. Physical Examination", "C. Treatment of Fever"
    if size == 10.0 and font not in _HEADER_FOOTER_FONTS:
        return "section_heading"

    # ── Bullet point marker ───────────────────────────────────────────────────
    if stripped in ("•", "»") and font in ("f8", "f17"):
        return "bullet"

    # ── References (size 8.0) ─────────────────────────────────────────────────
    if size == 8.0:
        return "reference"

    # ── Inline labels at size 9 in heading-like fonts ─────────────────────────
    # e.g. "1. Antipyretic drugs—", "Table 2–5.", "Figure 2–1."
    if size == 9.0 and font in _INLINE_LABEL_FONTS:
        return "inline_label"

    # ── Body text (size 9.0, body fonts) ─────────────────────────────────────
    if size == 9.0:
        return "body"

    # ── Default: treat as body ────────────────────────────────────────────────
    return "body"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TWO-COLUMN LAYOUT HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def sort_blocks_reading_order(blocks: list[dict], page_width: float) -> list[dict]:
    """
    Sort text blocks into correct two-column reading order.

    The TMT PDF uses a two-column layout on most pages.  PyMuPDF returns blocks
    in roughly top-to-bottom, left-to-right order, but not always perfectly for
    two-column content.  We re-sort by:
        1. Assign each block to a column (left or right) based on its x-midpoint
           relative to the page midpoint.
        2. Within each column sort by top y-coordinate.
        3. If a block spans the full page width (e.g. a chapter title), it floats
           to the top before column content.

    Args:
        blocks:     List of block dicts from page.get_text("dict")["blocks"].
        page_width: Width of the page in points.

    Returns:
        Reordered list of text blocks in reading order.
    """
    mid_x = page_width / 2.0

    full_width_blocks = []
    left_col_blocks   = []
    right_col_blocks  = []

    for block in blocks:
        if block.get("type") != 0:  # only text blocks
            continue
        bbox = block.get("bbox", (0, 0, 0, 0))
        x0, y0, x1, y1 = bbox
        block_mid_x = (x0 + x1) / 2.0
        block_width = x1 - x0

        # A block is "full width" if it spans > 70% of the page
        if block_width > page_width * 0.70:
            full_width_blocks.append(block)
        elif block_mid_x < mid_x:
            left_col_blocks.append(block)
        else:
            right_col_blocks.append(block)

    # Sort each group by top y-coordinate (reading order within column)
    full_width_blocks.sort(key=lambda b: b["bbox"][1])
    left_col_blocks.sort(key=lambda b: b["bbox"][1])
    right_col_blocks.sort(key=lambda b: b["bbox"][1])

    # Full-width blocks first, then interleave columns row-by-row
    # Simple approach: left column first, then right column per page
    # (more sophisticated interleaving would require y-overlap detection)
    return full_width_blocks + left_col_blocks + right_col_blocks


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEXT EXTRACTION — single page
# ═══════════════════════════════════════════════════════════════════════════════

def extract_page_elements(page: fitz.Page) -> list[dict]:
    """
    Extract all meaningful text elements from a single PDF page.

    Uses page.get_text("dict") to get per-span font information, applies
    PUA fixing, classifies each span, and returns a flat list of element
    dicts for downstream structure building.

    Each element dict has keys:
        role          : str (from classify_span)
        text          : str (PUA-fixed span text)
        font          : str
        size          : float
        bbox          : tuple (x0, y0, x1, y1)
        residual_pua  : list[str]  (unfixed PUA codepoints, if any)

    Args:
        page: A fitz.Page object.

    Returns:
        List of element dicts in reading order (two-column aware).
    """
    data       = page.get_text("dict")
    page_width = data.get("width", 612.0)
    blocks     = data.get("blocks", [])

    # Sort blocks into correct two-column reading order
    ordered_blocks = sort_blocks_reading_order(blocks, page_width)

    elements: list[dict] = []

    for block in ordered_blocks:
        for line in block.get("lines", []):
            # Accumulate spans within a line to reconstruct whole-sentence text
            line_text_parts: list[str] = []
            line_residual:   list[str] = []
            line_role  = None
            line_font  = None
            line_size  = 0.0
            line_bbox  = block.get("bbox", (0, 0, 0, 0))

            for span in line.get("spans", []):
                raw_text = span.get("text", "")
                font     = span.get("font", "unknown")
                size     = round(span.get("size", 0.0), 1)

                # Fix PUA characters using the per-font mapping
                fixed_text, residual = fix_pua_in_span(raw_text, font)

                role = classify_span(font, size, fixed_text)

                # Assign line-level role from the dominant/first non-skip span
                if line_role is None and role != "skip":
                    line_role = role
                    line_font = font
                    line_size = size

                if role != "skip":
                    line_text_parts.append(fixed_text)
                    line_residual.extend(residual)

            # Emit a single element per line (merging spans)
            merged_text = "".join(line_text_parts).strip()
            if merged_text and line_role is not None:
                elements.append({
                    "role":         line_role,
                    "text":         merged_text,
                    "font":         line_font or "unknown",
                    "size":         line_size,
                    "bbox":         line_bbox,
                    "residual_pua": list(set(line_residual)),
                })

    return elements


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SECTION / CHUNK BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_sections_from_chapter(
    chapter_title: str,
    chapter_page_start: int,
    chapter_page_end: int,
    doc: fitz.Document,
) -> list[dict]:
    """
    Extract all sections from a single chapter's page range.

    Iterates over each page in the chapter, extracts elements, and groups
    them into section objects.  A new section is started whenever a
    "section_heading" element is encountered.

    A section object follows this schema:
    {
        "source":             "TMT_2022",
        "chapter":            str,
        "chapter_page_start": int,   (1-indexed PDF page)
        "section":            str,
        "subsection":         str,
        "text":               str,
        "page_range":         [int, int],   (1-indexed)
        "word_count":         int,
        "has_essentials_box": bool,
        "essentials_text":    str,
    }

    Args:
        chapter_title:      Title string from the TOC.
        chapter_page_start: 1-indexed first page of this chapter.
        chapter_page_end:   1-indexed last page (exclusive) of this chapter.
        doc:                Open fitz.Document object.

    Returns:
        List of section dicts for this chapter.
    """
    sections: list[dict] = []
    flagged_pua_entries:  list[dict] = []  # collected within chapter

    # -- State machine tracking current section / subsection ------------------
    current_section:    str  = chapter_title  # default = chapter name
    current_subsection: str  = ""
    current_text_lines: list[str] = []
    current_page_start: int  = chapter_page_start
    current_page_end:   int  = chapter_page_start
    current_essentials: list[str] = []
    in_essentials_box:  bool = False

    def flush_section(end_page: int) -> None:
        """
        Finalize the current section buffer and append it to `sections`.
        Called whenever a new section heading is encountered or the chapter ends.
        """
        nonlocal current_text_lines, current_essentials, in_essentials_box
        body_text = " ".join(current_text_lines).strip()
        ess_text  = " ".join(current_essentials).strip()

        # Only emit sections with some meaningful text
        if body_text or ess_text:
            word_count = len(body_text.split()) if body_text else 0
            sections.append({
                "source":             "TMT_2022",
                "chapter":            chapter_title,
                "chapter_page_start": chapter_page_start,
                "section":            current_section,
                "subsection":         current_subsection,
                "text":               body_text,
                "page_range":         [current_page_start, end_page],
                "word_count":         word_count,
                "has_essentials_box": bool(ess_text),
                "essentials_text":    ess_text,
            })

        current_text_lines.clear()
        current_essentials.clear()
        in_essentials_box = False

    # Convert 1-indexed page numbers to 0-indexed for fitz
    page_start_0 = chapter_page_start - 1
    page_end_0   = min(chapter_page_end - 1, len(doc))

    for page_idx in range(page_start_0, page_end_0):
        page_num_1indexed = page_idx + 1  # for metadata
        page = doc[page_idx]
        elements = extract_page_elements(page)

        for elem in elements:
            role = elem["role"]
            text = elem["text"]
            size = elem["size"]

            # ── Flag any spans with unresolved PUA chars ───────────────────
            if elem["residual_pua"]:
                # Build a context snippet (surrounding text from current buffer)
                context_snippet = (
                    " ".join(current_text_lines[-3:]) + " [HERE] " + text
                ).strip()
                flagged_pua_entries.append({
                    "chapter":        chapter_title,
                    "section":        current_section,
                    "page":           page_num_1indexed,
                    "font":           elem["font"],
                    "word":           text,
                    "residual_chars": elem["residual_pua"],
                    "context":        context_snippet[:300],
                })

            # ── Route by role ──────────────────────────────────────────────
            if role == "essentials_header":
                in_essentials_box = True
                # Don't append the header label itself to essentials_text
                continue

            elif role == "essentials_body":
                # Content of the ESSENTIAL INQUIRIES box
                current_essentials.append(text)
                in_essentials_box = True
                continue

            elif role == "section_heading":
                # Determine if this is a top-level section or subsection
                # Heuristic: size 11.0 = top-level section; size 10.0 = subsection
                if size >= 11.0:
                    # Flush old section, start new top-level section
                    flush_section(page_num_1indexed)
                    current_section    = text
                    current_subsection = ""
                    current_page_start = page_num_1indexed
                else:
                    # size 10.0 = subsection (e.g. "A. Clinical Findings")
                    # Flush into current section and reset subsection buffer
                    # but keep the same section
                    if current_text_lines or current_essentials:
                        flush_section(page_num_1indexed)
                        current_page_start = page_num_1indexed
                    current_subsection = text

                current_page_end = page_num_1indexed
                continue

            elif role == "when_to_label":
                # "When to Refer" / "When to Admit" are subsections within
                # the current section.  Treat like a subsection heading.
                flush_section(page_num_1indexed)
                current_subsection = text
                current_page_start = page_num_1indexed
                current_page_end   = page_num_1indexed
                continue

            elif role in ("body", "inline_label"):
                current_text_lines.append(text)
                current_page_end = page_num_1indexed

            elif role == "reference":
                # Skip reference / citation text — not medical content
                continue

            elif role in ("superscript", "bullet", "skip", "chapter_header"):
                # Skip decorative / structural elements
                continue

    # Flush whatever is left after the last page
    flush_section(page_end_0)

    return sections, flagged_pua_entries


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN PARSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_tmt(
    pdf_path: str | Path,
    test_page_range: tuple[int, int] | None = None,
) -> tuple[list[dict], list[dict], list[int]]:
    """
    Parse the TMT PDF and return structured sections and flagged PUA words.

    Steps:
        1. Open the PDF and read the embedded TOC (50 level-1 entries).
        2. Skip front-matter chapters (before page 21).
        3. For each medical chapter, call build_sections_from_chapter().
        4. Collect all flagged PUA words across all chapters.

    Args:
        pdf_path:        Path to the CMDT 2022 PDF file.
        test_page_range: Optional (start, end) tuple of 1-indexed PDF page
                         numbers.  When set, only chapters whose page ranges
                         overlap this window are processed.  All chapter
                         boundaries are still clamped to this window so the
                         parser sees only the requested pages.
                         Example: (35, 61) for the --test mode (Chapter 2).

    Returns:
        (sections, flagged_pua_words, zero_text_pages)
        sections          — list of section dicts ready to save as JSON
        flagged_pua_words — list of word entries with unresolved PUA chars
        zero_text_pages   — 1-indexed page numbers that yielded no text
    """
    pdf_path = str(pdf_path)
    print(f"\n{'='*70}")
    print(f"  TMT PDF Parser — CMDT 2022")
    if test_page_range:
        print(f"  *** TEST MODE: pages {test_page_range[0]}–{test_page_range[1]} only ***")
    print(f"{'='*70}")
    print(f"\nOpening PDF: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")

    # ── 1. Read TOC ───────────────────────────────────────────────────────────
    toc = doc.get_toc()
    print(f"TOC entries: {len(toc)}")

    # Filter to only level-1 entries (chapters)
    chapters = [(title, page) for level, title, page in toc if level == 1]
    print(f"Level-1 chapters: {len(chapters)}")

    # ── 2. Compute chapter page ranges ────────────────────────────────────────
    # Each chapter ends where the next one begins; last chapter ends at EOF
    chapter_ranges: list[tuple[str, int, int]] = []
    for i, (title, start_page) in enumerate(chapters):
        end_page = chapters[i + 1][1] if i + 1 < len(chapters) else total_pages + 1
        chapter_ranges.append((title, start_page, end_page))

    # ── 3. Identify medical content start ─────────────────────────────────────
    # Medical content begins at page 21 (1-indexed).  Everything before is
    # front matter (cover, title, copyright, contents, authors, preface,
    # dedication).  The TOC itself marks this: first medical chapter is
    # "Disease Prevention & Health Promotion" at page 21.
    MEDICAL_CONTENT_START_PAGE = 21  # 1-indexed PDF page number

    medical_chapters = [
        (title, start, end)
        for title, start, end in chapter_ranges
        if start >= MEDICAL_CONTENT_START_PAGE
    ]
    print(f"Medical chapters (page >= {MEDICAL_CONTENT_START_PAGE}): "
          f"{len(medical_chapters)}")

    # ── 3b. (Test mode) Clamp to the requested page window ───────────────────
    # When --test is active we restrict processing to chapters that overlap
    # with the test window, and clamp their start/end to that window.
    if test_page_range:
        test_start, test_end = test_page_range
        clamped = []
        for title, start, end in medical_chapters:
            # Keep chapter only if its range overlaps with the test window
            if start > test_end or end <= test_start:
                continue
            clamped_start = max(start, test_start)
            clamped_end   = min(end,   test_end + 1)  # end is exclusive
            clamped.append((title, clamped_start, clamped_end))
        medical_chapters = clamped
        print(f"After clamping to pages {test_start}–{test_end}: "
              f"{len(medical_chapters)} chapter(s) in scope")

    # ── 4. Process each chapter ───────────────────────────────────────────────
    all_sections:    list[dict] = []
    all_flagged_pua: list[dict] = []
    zero_text_pages: list[int]  = []

    for chapter_idx, (chapter_title, page_start, page_end) in \
            enumerate(medical_chapters, start=1):

        print(f"\n  Processing chapter {chapter_idx}/{len(medical_chapters)}: "
              f"{chapter_title}  (pages {page_start}–{page_end - 1})")

        sections, flagged = build_sections_from_chapter(
            chapter_title=chapter_title,
            chapter_page_start=page_start,
            chapter_page_end=page_end,
            doc=doc,
        )

        # Detect pages that produced zero text (possible parsing issues)
        for page_idx in range(page_start - 1, min(page_end - 1, total_pages)):
            page = doc[page_idx]
            raw = page.get_text("text").strip()
            if not raw:
                zero_text_pages.append(page_idx + 1)  # 1-indexed

        all_sections.extend(sections)
        all_flagged_pua.extend(flagged)

        print(f"    -> {len(sections)} sections extracted, "
              f"{len(flagged)} flagged PUA words")

    doc.close()

    print(f"\n{'─'*70}")
    print(f"  Parsing complete.")
    print(f"  Total sections: {len(all_sections)}")
    print(f"  Total flagged PUA entries: {len(all_flagged_pua)}")
    if zero_text_pages:
        print(f"  Zero-text pages: {zero_text_pages[:20]}"
              f"{'...' if len(zero_text_pages) > 20 else ''}")

    return all_sections, all_flagged_pua, zero_text_pages


# ═══════════════════════════════════════════════════════════════════════════════
# 7. QUALITY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_quality_report(
    sections: list[dict],
    flagged_pua: list[dict],
    zero_text_pages: list[int],
) -> None:
    """
    Print a summary quality report after parsing.

    Covers:
    - Total sections extracted
    - Word count distribution (min, max, mean, median)
    - Number of chapters processed
    - Number of flagged PUA words
    - Pages with zero text
    - Distribution of sections with/without essentials box
    """
    print(f"\n{'='*70}")
    print(f"  QUALITY REPORT")
    print(f"{'='*70}\n")

    if not sections:
        print("  No sections extracted — check parsing logic.")
        return

    word_counts = [s["word_count"] for s in sections]

    chapters_seen = {s["chapter"] for s in sections}
    sections_with_ess = sum(1 for s in sections if s["has_essentials_box"])

    print(f"  Total sections extracted   : {len(sections):,}")
    print(f"  Chapters processed         : {len(chapters_seen)}")
    print(f"  Sections with Essentials   : {sections_with_ess}")
    print()
    print(f"  Word count distribution:")
    print(f"    Min    : {min(word_counts):,}")
    print(f"    Max    : {max(word_counts):,}")
    print(f"    Mean   : {statistics.mean(word_counts):.1f}")
    print(f"    Median : {statistics.median(word_counts):.1f}")
    print(f"    Stdev  : {statistics.stdev(word_counts):.1f}"
          if len(word_counts) > 1 else "    Stdev  : n/a")
    print()

    # Word count histogram (rough buckets)
    buckets = {
        "0":       sum(1 for w in word_counts if w == 0),
        "1-49":    sum(1 for w in word_counts if 1 <= w < 50),
        "50-199":  sum(1 for w in word_counts if 50 <= w < 200),
        "200-499": sum(1 for w in word_counts if 200 <= w < 500),
        "500+":    sum(1 for w in word_counts if w >= 500),
    }
    print(f"  Word count buckets:")
    for label, count in buckets.items():
        bar = "█" * min(count, 60)
        print(f"    {label:>8}  {count:5d}  {bar}")
    print()

    print(f"  Flagged PUA words          : {len(flagged_pua)}")
    if flagged_pua:
        # Show most common residual chars
        from collections import Counter
        char_counter: Counter = Counter()
        for entry in flagged_pua:
            for c in entry["residual_chars"]:
                char_counter[c] += 1
        print(f"  Most common residual chars :")
        for char_code, freq in char_counter.most_common(10):
            print(f"    {char_code}  x{freq}")
    print()

    if zero_text_pages:
        print(f"  Pages with zero text ({len(zero_text_pages)} total):")
        print(f"    {zero_text_pages[:30]}"
              f"{'...' if len(zero_text_pages) > 30 else ''}")
    else:
        print(f"  Pages with zero text: none")

    print(f"\n{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_outputs(
    sections: list[dict],
    flagged_pua: list[dict],
) -> None:
    """
    Persist results to disk.

    Outputs:
        CHUNKS_DIR / tmt_raw_sections.json   — main section list
        DATA_DIR   / flagged_pua_words.json  — unresolved PUA words for review

    Args:
        sections:    List of section dicts from parse_tmt().
        flagged_pua: List of flagged word entries from parse_tmt().
    """
    # Ensure output directories exist
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Save sections ─────────────────────────────────────────────────────────
    sections_path = CHUNKS_DIR / "tmt_raw_sections.json"
    with open(sections_path, "w", encoding="utf-8") as fh:
        json.dump(sections, fh, indent=2, ensure_ascii=False)
    size_mb = sections_path.stat().st_size / 1024 / 1024
    print(f"  Saved {len(sections):,} sections -> {sections_path}  ({size_mb:.2f} MB)")

    # ── Save flagged PUA words ────────────────────────────────────────────────
    flagged_path = DATA_DIR / "flagged_pua_words.json"
    with open(flagged_path, "w", encoding="utf-8") as fh:
        json.dump(flagged_pua, fh, indent=2, ensure_ascii=False)
    print(f"  Saved {len(flagged_pua):,} flagged PUA entries -> {flagged_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── CLI argument parsing ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Parse the CMDT 2022 PDF into structured JSON sections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data_processing/tmt_parser.py            # full book\n"
            "  python data_processing/tmt_parser.py --test     # pages 35-61 only\n"
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Quick-test mode: process only pages 35–61 "
            "(Chapter 2 – Common Symptoms) instead of the full book."
        ),
    )
    args = parser.parse_args()

    # ── Verify PDF exists ─────────────────────────────────────────────────────
    if not TMT_PDF.exists():
        print(f"ERROR: PDF not found at {TMT_PDF}")
        print("Update PDF_DIR in config.py to point to the correct location.")
        sys.exit(1)

    # ── Determine page range ──────────────────────────────────────────────────
    # --test restricts parsing to pages 35–61 (Chapter 2: Common Symptoms).
    # This covers roughly 27 pages and completes in seconds, making it ideal
    # for verifying the parser logic without waiting for the full 1874-page run.
    test_range = (35, 61) if args.test else None

    if args.test:
        print("\n[TEST MODE] Only processing pages 35–61 "
              "(Chapter 2: Common Symptoms).")
        print("Run without --test to parse the full book.\n")

    # ── Run the parsing pipeline ──────────────────────────────────────────────
    sections, flagged_pua, zero_text_pages = parse_tmt(TMT_PDF, test_page_range=test_range)

    # ── Save to disk ──────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Saving outputs...")
    save_outputs(sections, flagged_pua)

    # ── Print quality report ──────────────────────────────────────────────────
    print_quality_report(sections, flagged_pua, zero_text_pages)
