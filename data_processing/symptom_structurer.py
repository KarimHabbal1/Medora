"""
symptom_structurer.py
---------------------
Extracts structured clinical data from the "Common Symptoms" chapter of the
TMT textbook chunks.

For each of the 11 symptoms, a GPT-4o (and optionally Ollama) prompt is sent
that asks the model to return a JSON object matching a fixed clinical schema.
All results are saved to data/structured_symptoms/.

Usage
-----
# Full run (GPT-4o + Ollama comparison on 3 symptoms):
    python data_processing/symptom_structurer.py

# GPT-4o only:
    python data_processing/symptom_structurer.py --gpt-only

# Ollama only:
    python data_processing/symptom_structurer.py --ollama-only

# Custom Ollama model:
    python data_processing/symptom_structurer.py --ollama-model mistral

# Process specific symptoms only:
    python data_processing/symptom_structurer.py --symptoms "CHEST PAIN,COUGH"
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests  # for Ollama API calls

# ---------------------------------------------------------------------------
# Path bootstrap — make config.py importable from the project root
# ---------------------------------------------------------------------------
# This file lives at:  <project_root>/data_processing/symptom_structurer.py
# config.py lives at:  <project_root>/config.py
# So we add the project root (one level up) to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CHUNKS_DIR, SYMPTOMS_DIR  # noqa: E402  (after sys.path fix)

# Load environment variables from the shared .env used by data_processing scripts
from dotenv import load_dotenv  # noqa: E402
import os  # noqa: E402

ENV_FILE = Path(__file__).resolve().parent.parent.parent / "data_processing" / ".env"
load_dotenv(dotenv_path=ENV_FILE)

import openai  # noqa: E402  (needs key in env before client is built)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNKS_FILE = CHUNKS_DIR / "tmt_chunks_structured.json"
OUTPUT_DIR = SYMPTOMS_DIR  # data/structured_symptoms/

# Chapter name as it appears in the JSON
COMMON_SYMPTOMS_CHAPTER = "Common Symptoms"

# The 3 symptoms used for GPT-4o vs Ollama comparison
COMPARISON_SYMPTOMS = {"CHEST PAIN", "COUGH", "DYSPNEA"}

# Required top-level keys every structured symptom object must contain
REQUIRED_FIELDS = {
    "symptom",
    "body_systems",
    "essential_questions",
    "red_flags",
    "differential_diagnosis",
    "urgency_rules",
    "specialty_routing",
    "key_history_points",
    "key_exam_findings",
    "initial_workup",
    "when_to_admit",
    "when_to_refer",
    "treatment_overview",
    "etiology",
    "epidemiology",
}

# Default Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SCHEMA_TEMPLATE = """
{
  "symptom": "string — the symptom name",
  "body_systems": ["string — body systems this symptom relates to"],
  "essential_questions": [
    "string — key questions a clinician should ask about this symptom"
  ],
  "red_flags": [
    {
      "flag": "string — the warning sign",
      "implication": "string — what it suggests",
      "urgency": "emergency | urgent | routine"
    }
  ],
  "differential_diagnosis": [
    {
      "condition": "string — condition name",
      "key_features": ["string — distinguishing features"],
      "likelihood_context": "string — when to suspect this"
    }
  ],
  "urgency_rules": [
    {
      "criteria": "string — what combination of findings",
      "urgency": "emergency | urgent | routine",
      "action": "string — recommended action"
    }
  ],
  "specialty_routing": ["string — relevant specialties"],
  "key_history_points": ["string — important history to gather"],
  "key_exam_findings": ["string — important physical exam findings"],
  "initial_workup": ["string — recommended initial tests/studies"],
  "when_to_admit": [
    "string — specific criteria or clinical scenarios that warrant hospital admission"
  ],
  "when_to_refer": [
    "string — specific criteria or clinical scenarios that warrant specialist referral"
  ],
  "treatment_overview": [
    "string — key treatment approaches, drug classes, or interventions mentioned"
  ],
  "etiology": [
    "string — common causes, mechanisms, or contributing factors"
  ],
  "epidemiology": "string — brief note on prevalence, demographics, or risk populations"
}
"""


def build_prompt(symptom_name: str, concatenated_text: str) -> str:
    """
    Build the LLM structuring prompt for a single symptom.

    Parameters
    ----------
    symptom_name : str
        Human-readable name of the symptom (e.g. "CHEST PAIN").
    concatenated_text : str
        All chunk texts for this symptom joined into a single string.

    Returns
    -------
    str
        The full prompt to send to the model.
    """
    return (
        "You are a medical knowledge extraction system. Given the following textbook "
        "section about a symptom, extract structured clinical data according to the "
        "schema below.\n\n"
        "IMPORTANT: Only extract information that is explicitly stated or directly "
        "implied in the text. Do not add information from outside knowledge.\n\n"
        f"## Textbook Section: {symptom_name}\n\n"
        f"{concatenated_text}\n\n"
        "## Output Schema (respond with valid JSON only, no other text):\n\n"
        f"{SCHEMA_TEMPLATE}"
    )


# ---------------------------------------------------------------------------
# Chunk loading and grouping
# ---------------------------------------------------------------------------

def load_symptom_chunks(chunks_file: Path) -> dict[str, list[dict]]:
    """
    Load tmt_chunks_structured.json, filter to the "Common Symptoms" chapter,
    and group chunks by their section name (each section = one symptom).

    Parameters
    ----------
    chunks_file : Path
        Absolute path to tmt_chunks_structured.json.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from symptom (section) name → list of chunk dicts, in order.
    """
    print(f"[load] Reading chunks from {chunks_file} ...")
    with open(chunks_file, encoding="utf-8") as f:
        all_chunks = json.load(f)

    print(f"[load] Total chunks in file: {len(all_chunks)}")

    # Filter to the Common Symptoms chapter only
    symptom_chunks = [
        c for c in all_chunks if c.get("chapter") == COMMON_SYMPTOMS_CHAPTER
    ]
    print(f"[load] Chunks in '{COMMON_SYMPTOMS_CHAPTER}': {len(symptom_chunks)}")

    # Group by section — each section is one symptom
    grouped: dict[str, list[dict]] = {}
    for chunk in symptom_chunks:
        section = chunk.get("section", "").strip()
        if not section:
            # Fall back to chapter name if section is missing
            section = chunk.get("chapter", "UNKNOWN")
        grouped.setdefault(section, []).append(chunk)

    # Sort chunks within each section by their chunk_id to preserve reading order.
    # chunk_id format: "tmt::<chapter>::<section>::<subsection>::<index>"
    # The trailing integer is the within-section sequence number.
    for section in grouped:
        grouped[section].sort(key=lambda c: _chunk_sort_key(c["chunk_id"]))

    print(f"[load] Unique symptoms (sections) found: {list(grouped.keys())}")
    return grouped


def _chunk_sort_key(chunk_id: str) -> tuple:
    """
    Extract a sortable key from a chunk_id string.

    The chunk_id ends with ::<integer>, which is the within-section index.
    We return a tuple (prefix, index) so chunks sort in reading order.

    Parameters
    ----------
    chunk_id : str
        e.g. "tmt::common_symptoms::chest_pain::main::3"

    Returns
    -------
    tuple
        (prefix_string, integer_index)
    """
    parts = chunk_id.rsplit("::", 1)
    prefix = parts[0]
    try:
        index = int(parts[1])
    except (IndexError, ValueError):
        index = 0
    return (prefix, index)


def concatenate_chunks(chunks: list[dict]) -> str:
    """
    Join all chunk texts for a symptom into a single string, separated by
    newlines. Subsection headers (when present) are inserted as Markdown
    headings to help the LLM understand structure.

    Parameters
    ----------
    chunks : list[dict]
        List of chunk dicts for one symptom, already sorted in reading order.

    Returns
    -------
    str
        The full text for the symptom, ready to include in the LLM prompt.
    """
    parts = []
    current_subsection = None

    for chunk in chunks:
        sub = chunk.get("subsection", "").strip()
        # Insert a heading when the subsection changes
        if sub and sub != current_subsection:
            parts.append(f"\n### {sub}\n")
            current_subsection = sub
        parts.append(chunk["text"])

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# JSON parsing (robust)
# ---------------------------------------------------------------------------

def parse_json_response(raw: str, symptom_name: str) -> dict | None:
    """
    Robustly parse a JSON object from an LLM response string.

    Tries, in order:
      1. Direct json.loads on the whole string.
      2. Extract content between ```json ... ``` or ``` ... ``` markers.
      3. Extract content between <json> ... </json> tags.
      4. Find the first {...} block in the string.

    Parameters
    ----------
    raw : str
        The raw text returned by the LLM.
    symptom_name : str
        Used only for error messages.

    Returns
    -------
    dict or None
        Parsed JSON object, or None if all strategies fail.
    """
    # Strategy 1: direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: markdown code block (```json ... ``` or ``` ... ```)
    md_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: <json> ... </json> tags
    xml_match = re.search(r"<json>\s*([\s\S]+?)\s*</json>", raw, re.IGNORECASE)
    if xml_match:
        try:
            return json.loads(xml_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 4: find the outermost {...} block
    brace_match = re.search(r"\{[\s\S]+\}", raw)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    print(f"  [parse] ERROR: Could not extract valid JSON for '{symptom_name}'.")
    print(f"  [parse] Raw response (first 300 chars): {raw[:300]}")
    return None


def validate_structured_symptom(data: dict, symptom_name: str) -> bool:
    """
    Check that a parsed JSON object contains all required fields.

    Parameters
    ----------
    data : dict
        The parsed structured symptom object.
    symptom_name : str
        Used only for error messages.

    Returns
    -------
    bool
        True if all required fields are present (even if empty).
    """
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        print(f"  [validate] WARNING: '{symptom_name}' is missing fields: {missing}")
        return False
    return True


# ---------------------------------------------------------------------------
# GPT-4o call
# ---------------------------------------------------------------------------

def call_gpt4o(prompt: str, symptom_name: str) -> dict | None:
    """
    Send the structuring prompt to GPT-4o and return the parsed JSON object.

    Uses the OPENAI_API_KEY environment variable (loaded from .env).
    Temperature is set to 0 for deterministic, reproducible output.

    Parameters
    ----------
    prompt : str
        The full structuring prompt.
    symptom_name : str
        Used for progress/error messages.

    Returns
    -------
    dict or None
        Parsed structured symptom object, or None on failure.
    """
    # Build the OpenAI client using the key loaded from .env
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise medical knowledge extraction assistant. "
                        "Always respond with valid JSON only, no prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,  # deterministic output
        )
        raw = response.choices[0].message.content
    except openai.OpenAIError as exc:
        print(f"  [gpt4o] API error for '{symptom_name}': {exc}")
        return None

    # Parse and validate
    parsed = parse_json_response(raw, symptom_name)
    if parsed is not None:
        validate_structured_symptom(parsed, symptom_name)
    return parsed


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def call_ollama(
    prompt: str, symptom_name: str, model: str = DEFAULT_OLLAMA_MODEL
) -> dict | None:
    """
    Send the structuring prompt to a locally running Ollama instance and return
    the parsed JSON object.

    The Ollama API endpoint is: POST http://localhost:11434/api/generate
    with {"model": ..., "prompt": ..., "stream": false}.

    If Ollama is not running (connection error), a warning is printed and None
    is returned — the script does NOT crash.

    Parameters
    ----------
    prompt : str
        The full structuring prompt.
    symptom_name : str
        Used for progress/error messages.
    model : str
        Ollama model tag (e.g. "llama3.1:8b").

    Returns
    -------
    dict or None
        Parsed structured symptom object, or None on failure / Ollama unavailable.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # receive the full response in one JSON body
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(
            f"  [ollama] WARNING: Cannot connect to Ollama at {OLLAMA_URL}. "
            "Is it running? Skipping Ollama for this run."
        )
        return None
    except requests.exceptions.RequestException as exc:
        print(f"  [ollama] Request error for '{symptom_name}': {exc}")
        return None

    # Ollama returns {"response": "<generated text>", ...}
    raw = resp.json().get("response", "")
    parsed = parse_json_response(raw, symptom_name)
    if parsed is not None:
        validate_structured_symptom(parsed, symptom_name)
    return parsed


# ---------------------------------------------------------------------------
# GPT-4o — process all symptoms
# ---------------------------------------------------------------------------

def run_gpt4o(
    grouped_chunks: dict[str, list[dict]],
    selected_symptoms: list[str] | None = None,
) -> dict[str, dict]:
    """
    Call GPT-4o for all (or selected) symptoms and collect results.

    Parameters
    ----------
    grouped_chunks : dict[str, list[dict]]
        Output of load_symptom_chunks — symptom name → sorted chunk list.
    selected_symptoms : list[str] or None
        If provided, only process these symptom names (case-insensitive match).
        If None, all symptoms in grouped_chunks are processed.

    Returns
    -------
    dict[str, dict]
        Mapping from symptom name → structured symptom object (may include
        None values where parsing failed).
    """
    # Determine which symptoms to process
    all_names = list(grouped_chunks.keys())
    if selected_symptoms:
        # Normalise to upper-case for matching
        sel_upper = {s.strip().upper() for s in selected_symptoms}
        target_names = [n for n in all_names if n.upper() in sel_upper]
        if not target_names:
            print(
                f"[gpt4o] WARNING: No symptoms matched the filter {selected_symptoms}. "
                "Available: " + ", ".join(all_names)
            )
            return {}
    else:
        target_names = all_names

    total = len(target_names)
    results: dict[str, dict] = {}

    for i, symptom_name in enumerate(target_names, start=1):
        print(f"[gpt4o] Structuring symptom {i}/{total}: {symptom_name} ...")

        # Concatenate all chunk texts for this symptom
        text = concatenate_chunks(grouped_chunks[symptom_name])
        prompt = build_prompt(symptom_name, text)

        # Call the API
        structured = call_gpt4o(prompt, symptom_name)

        if structured is not None:
            results[symptom_name] = structured
            print(f"  [gpt4o] OK — extracted {len(structured.get('red_flags', []))} red_flags, "
                  f"{len(structured.get('differential_diagnosis', []))} differentials.")
        else:
            # Store None so the caller knows this symptom failed
            results[symptom_name] = None
            print(f"  [gpt4o] FAILED for '{symptom_name}'.")

    return results


# ---------------------------------------------------------------------------
# Ollama — process comparison symptoms
# ---------------------------------------------------------------------------

def run_ollama(
    grouped_chunks: dict[str, list[dict]],
    model: str = DEFAULT_OLLAMA_MODEL,
    selected_symptoms: list[str] | None = None,
) -> dict[str, dict]:
    """
    Call Ollama for the 3 comparison symptoms (CHEST PAIN, COUGH, DYSPNEA),
    or a custom selection, and collect results.

    Parameters
    ----------
    grouped_chunks : dict[str, list[dict]]
        Output of load_symptom_chunks — symptom name → sorted chunk list.
    model : str
        Ollama model tag.
    selected_symptoms : list[str] or None
        If provided, restrict to the intersection with COMPARISON_SYMPTOMS.

    Returns
    -------
    dict[str, dict]
        Mapping from symptom name → structured symptom object.
    """
    # Determine which symptoms to run (always intersect with COMPARISON_SYMPTOMS)
    all_names = list(grouped_chunks.keys())

    # Build the candidate set: comparison symptoms ∩ what's in the chunks
    available_comparison = {
        n for n in all_names if n.upper() in COMPARISON_SYMPTOMS
    }

    if selected_symptoms:
        sel_upper = {s.strip().upper() for s in selected_symptoms}
        target_names = [
            n for n in all_names
            if n.upper() in available_comparison and n.upper() in sel_upper
        ]
    else:
        target_names = [n for n in all_names if n.upper() in available_comparison]

    if not target_names:
        print(
            "[ollama] No matching comparison symptoms found in chunks. "
            f"Looking for: {COMPARISON_SYMPTOMS}. Available: {all_names}"
        )
        return {}

    total = len(target_names)
    results: dict[str, dict] = {}
    ollama_unavailable = False  # track if the first call already showed it's down

    for i, symptom_name in enumerate(target_names, start=1):
        if ollama_unavailable:
            # Don't retry if we already know Ollama is down
            break

        print(f"[ollama] Structuring symptom {i}/{total}: {symptom_name} "
              f"(model={model}) ...")

        text = concatenate_chunks(grouped_chunks[symptom_name])
        prompt = build_prompt(symptom_name, text)

        structured = call_ollama(prompt, symptom_name, model=model)

        if structured is None:
            # call_ollama already printed a warning
            ollama_unavailable = True
        else:
            results[symptom_name] = structured
            print(f"  [ollama] OK — extracted {len(structured.get('red_flags', []))} red_flags, "
                  f"{len(structured.get('differential_diagnosis', []))} differentials.")

    return results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(data: dict[str, dict], output_path: Path) -> None:
    """
    Serialise the structured symptom results to JSON and write to disk.

    The output is a list of objects (one per symptom), preserving the order
    in which symptoms were processed. None entries (failed parses) are stored
    as {"symptom": "<name>", "error": "parsing failed"} so the file is always
    valid JSON.

    Parameters
    ----------
    data : dict[str, dict]
        Mapping from symptom name → structured object (or None on failure).
    output_path : Path
        Destination .json file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dict → list, replacing None with an error sentinel
    output_list = []
    for symptom_name, structured in data.items():
        if structured is None:
            output_list.append({"symptom": symptom_name, "error": "parsing_failed"})
        else:
            output_list.append(structured)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)

    print(f"[save] Written {len(output_list)} entries → {output_path}")


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

def print_comparison_summary(
    gpt4o_results: dict[str, dict],
    ollama_results: dict[str, dict],
) -> None:
    """
    Print a side-by-side summary comparing GPT-4o vs Ollama for each of the
    3 comparison symptoms.

    Metrics reported per symptom:
      - Number of essential_questions
      - Number of red_flags
      - Number of differential_diagnosis entries
      - A qualitative note when counts differ noticeably (>= 3 difference)

    Parameters
    ----------
    gpt4o_results : dict[str, dict]
        Output of run_gpt4o.
    ollama_results : dict[str, dict]
        Output of run_ollama.
    """
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY: GPT-4o vs Ollama")
    print("=" * 70)

    # We compare only the symptoms that Ollama actually ran on
    symptoms_to_compare = list(ollama_results.keys())

    if not symptoms_to_compare:
        print("(No Ollama results available — skipping comparison.)")
        return

    for symptom_name in symptoms_to_compare:
        gpt = gpt4o_results.get(symptom_name) or {}
        oll = ollama_results.get(symptom_name) or {}

        gpt_eq = len(gpt.get("essential_questions", []))
        oll_eq = len(oll.get("essential_questions", []))

        gpt_rf = len(gpt.get("red_flags", []))
        oll_rf = len(oll.get("red_flags", []))

        gpt_dd = len(gpt.get("differential_diagnosis", []))
        oll_dd = len(oll.get("differential_diagnosis", []))

        print(f"\n--- {symptom_name} ---")
        print(f"  essential_questions : GPT-4o={gpt_eq}  |  Ollama={oll_eq}")
        print(f"  red_flags           : GPT-4o={gpt_rf}  |  Ollama={oll_rf}")
        print(f"  differential_diag.  : GPT-4o={gpt_dd}  |  Ollama={oll_dd}")

        # Qualitative note if differences are large
        notes = []
        if abs(gpt_eq - oll_eq) >= 3:
            winner = "GPT-4o" if gpt_eq > oll_eq else "Ollama"
            notes.append(
                f"{winner} extracted significantly more essential_questions "
                f"({max(gpt_eq, oll_eq)} vs {min(gpt_eq, oll_eq)})."
            )
        if abs(gpt_rf - oll_rf) >= 3:
            winner = "GPT-4o" if gpt_rf > oll_rf else "Ollama"
            notes.append(
                f"{winner} extracted significantly more red_flags "
                f"({max(gpt_rf, oll_rf)} vs {min(gpt_rf, oll_rf)})."
            )
        if abs(gpt_dd - oll_dd) >= 3:
            winner = "GPT-4o" if gpt_dd > oll_dd else "Ollama"
            notes.append(
                f"{winner} extracted significantly more differential diagnoses "
                f"({max(gpt_dd, oll_dd)} vs {min(gpt_dd, oll_dd)})."
            )

        if notes:
            print("  NOTE:", " ".join(notes))
        else:
            print("  NOTE: Counts are broadly similar between models.")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: gpt_only, ollama_only,
        ollama_model, symptoms.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract structured clinical data from the 'Common Symptoms' chapter "
            "using GPT-4o and/or Ollama."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--gpt-only",
        action="store_true",
        help="Run GPT-4o only; skip Ollama.",
    )
    mode.add_argument(
        "--ollama-only",
        action="store_true",
        help="Run Ollama only; skip GPT-4o.",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        metavar="MODEL",
        help=f"Ollama model tag to use (default: {DEFAULT_OLLAMA_MODEL}).",
    )
    parser.add_argument(
        "--symptoms",
        default=None,
        metavar="SYMPTOM1,SYMPTOM2,...",
        help=(
            "Comma-separated list of symptom names to process. "
            "Names are matched case-insensitively against section names in the chunks. "
            "If omitted, all symptoms are processed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point: orchestrates loading, structuring, saving, and reporting.

    Flow
    ----
    1. Parse CLI flags.
    2. Load and group chunks from tmt_chunks_structured.json.
    3. (Unless --ollama-only) call GPT-4o for all/selected symptoms.
    4. (Unless --gpt-only) call Ollama for the 3 comparison symptoms.
    5. Save results to data/structured_symptoms/.
    6. Print comparison summary if both models ran.
    """
    args = parse_args()

    # Convert comma-separated symptom filter to a list (or None for all)
    selected_symptoms: list[str] | None = None
    if args.symptoms:
        selected_symptoms = [s.strip() for s in args.symptoms.split(",") if s.strip()]
        print(f"[main] Symptom filter active: {selected_symptoms}")

    # ------------------------------------------------------------------
    # Step 1: Load and group chunks
    # ------------------------------------------------------------------
    grouped_chunks = load_symptom_chunks(CHUNKS_FILE)

    gpt4o_results: dict[str, dict] = {}
    ollama_results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Step 2: GPT-4o — all symptoms
    # ------------------------------------------------------------------
    if not args.ollama_only:
        print("\n[main] === Running GPT-4o structuring ===")
        gpt4o_results = run_gpt4o(grouped_chunks, selected_symptoms=selected_symptoms)

        # Save GPT-4o output
        gpt4o_output_path = OUTPUT_DIR / "tmt_symptoms_gpt4o.json"
        save_results(gpt4o_results, gpt4o_output_path)
    else:
        print("[main] --ollama-only flag set: skipping GPT-4o.")

    # ------------------------------------------------------------------
    # Step 3: Ollama — 3 comparison symptoms
    # ------------------------------------------------------------------
    if not args.gpt_only:
        print(f"\n[main] === Running Ollama structuring (model={args.ollama_model}) ===")
        print(f"[main] Comparison symptoms: {sorted(COMPARISON_SYMPTOMS)}")
        ollama_results = run_ollama(
            grouped_chunks,
            model=args.ollama_model,
            selected_symptoms=selected_symptoms,
        )

        if ollama_results:
            ollama_output_path = OUTPUT_DIR / "tmt_symptoms_ollama.json"
            save_results(ollama_results, ollama_output_path)
        else:
            print("[main] No Ollama results to save (Ollama may be unavailable).")
    else:
        print("[main] --gpt-only flag set: skipping Ollama.")

    # ------------------------------------------------------------------
    # Step 4: Comparison summary
    # ------------------------------------------------------------------
    if gpt4o_results and ollama_results:
        print_comparison_summary(gpt4o_results, ollama_results)
    elif args.gpt_only:
        print("\n[main] GPT-4o-only run complete. No comparison summary generated.")
    elif args.ollama_only:
        print("\n[main] Ollama-only run complete. No comparison summary generated.")
    else:
        print(
            "\n[main] Comparison summary skipped "
            "(Ollama results unavailable or no overlap)."
        )

    print("\n[main] Done.")


if __name__ == "__main__":
    main()
