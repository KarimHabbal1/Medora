"""
Phase 2.1 — Embedding Pipeline
Embeds all TMT chunks using a SentenceTransformer model and saves the results
as a compressed numpy archive alongside a JSON metadata file.

Usage:
    python embeddings/embed_chunks.py
    python embeddings/embed_chunks.py --force
    python embeddings/embed_chunks.py --batch-size 32
    python embeddings/embed_chunks.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Project root on sys.path so config.py is importable ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHUNKS_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDINGS_DIR,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNKS_FILE = CHUNKS_DIR / "tmt_chunks_structured.json"
OUTPUT_NPZ  = EMBEDDINGS_DIR / "tmt_chunk_embeddings.npz"
OUTPUT_META = EMBEDDINGS_DIR / "embedding_metadata.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_prefix(chunk: dict) -> str:
    """Return the context-prefixed text for a single chunk."""
    chapter    = chunk.get("chapter", "")
    section    = chunk.get("section", "")
    subsection = chunk.get("subsection", "")
    text       = chunk.get("text", "")

    if subsection:
        return f"Chapter: {chapter} | Section: {section} | Subsection: {subsection} | {text}"
    return f"Chapter: {chapter} | Section: {section} | {text}"


def detect_device() -> str:
    """Return 'mps' if Apple Silicon GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_chunks(path: Path) -> list[dict]:
    """Load and return the list of chunk dicts from a JSON file."""
    print(f"Loading chunks from {path} …")
    with open(path, "r", encoding="utf-8") as fh:
        chunks = json.load(fh)
    print(f"  Loaded {len(chunks):,} chunks.")
    return chunks


def load_model(model_name: str, device: str):
    """Load and return a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading model '{model_name}' on device '{device}' …")
    model = SentenceTransformer(model_name, device=device)
    print("  Model loaded.")
    return model


def embed_chunks(model, chunks: list[dict], batch_size: int) -> tuple[np.ndarray, list[str], int]:
    """
    Embed all chunks in batches.

    Returns:
        embeddings  : np.ndarray of shape (N_embedded, EMBEDDING_DIM)
        chunk_ids   : list of chunk_id strings for successfully embedded chunks
        failed_count: number of batches that raised an exception
    """
    texts     = [build_prefix(c) for c in chunks]
    ids       = [c["chunk_id"] for c in chunks]
    total     = len(texts)
    n_batches = (total + batch_size - 1) // batch_size

    all_embeddings: list[np.ndarray] = []
    all_ids:        list[str]        = []
    failed_batches = 0

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, total)

        batch_texts = texts[start:end]
        batch_ids   = ids[start:end]

        print(
            f"  Batch {batch_idx + 1}/{n_batches} "
            f"(chunks {start + 1}–{end} of {total}) …",
            end=" ",
            flush=True,
        )

        try:
            vecs = model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.append(vecs)
            all_ids.extend(batch_ids)
            print(f"OK  shape={vecs.shape}")
        except Exception as exc:  # noqa: BLE001
            failed_batches += 1
            print(f"FAILED — {exc}")
            print(f"    Skipping {len(batch_texts)} chunks in this batch.")

    if all_embeddings:
        embeddings = np.vstack(all_embeddings).astype(np.float32)
    else:
        embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    return embeddings, all_ids, failed_batches


def save_outputs(
    embeddings:  np.ndarray,
    chunk_ids:   list[str],
    model_name:  str,
    device_used: str,
) -> None:
    """Save the .npz archive and JSON metadata file."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Compressed numpy archive
    print(f"\nSaving embeddings to {OUTPUT_NPZ} …")
    np.savez_compressed(
        OUTPUT_NPZ,
        embeddings=embeddings,
        chunk_ids=np.array(chunk_ids, dtype=object),
    )
    print(f"  Saved .npz  shape={embeddings.shape}")

    # Metadata
    meta = {
        "model_name":               model_name,
        "embedding_dim":            EMBEDDING_DIM,
        "num_chunks":               len(chunk_ids),
        "context_prefix_strategy":  "Chapter | Section | Subsection | text",
        "timestamp":                datetime.now(timezone.utc).isoformat(),
        "device_used":              device_used,
    }
    print(f"Saving metadata to {OUTPUT_META} …")
    with open(OUTPUT_META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("  Saved metadata.")


def check_existing_outputs(force: bool) -> bool:
    """
    Return True if the pipeline should proceed, False if it should abort.
    Prompts the user interactively when --force is not set and files exist.
    """
    existing = [p for p in (OUTPUT_NPZ, OUTPUT_META) if p.exists()]
    if not existing:
        return True  # nothing to overwrite

    print("\nExisting output files detected:")
    for p in existing:
        print(f"  {p}")

    if force:
        print("--force flag set — overwriting.")
        return True

    answer = input("\nOverwrite? [y/N] ").strip().lower()
    if answer in {"y", "yes"}:
        return True

    print("Aborted. Use --force to skip this prompt.")
    return False


def print_dry_run_config(batch_size: int, device: str) -> None:
    """Print the configuration that would be used and exit."""
    print("\n=== DRY RUN — configuration only, no embedding performed ===")
    print(f"  Chunks file     : {CHUNKS_FILE}")
    print(f"  Output .npz     : {OUTPUT_NPZ}")
    print(f"  Output metadata : {OUTPUT_META}")
    print(f"  Model           : {EMBEDDING_MODEL}")
    print(f"  Embedding dim   : {EMBEDDING_DIM}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Device          : {device}")
    print("=============================================================\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2.1 — Embed TMT chunks with a SentenceTransformer model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embedding files without prompting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        metavar="N",
        help=f"Number of chunks per embedding batch (default: {EMBEDDING_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the model and print config, but do not embed or save anything.",
    )
    args = parser.parse_args()

    device = detect_device()

    if args.dry_run:
        print_dry_run_config(args.batch_size, device)
        # Still load the model so the user can confirm it resolves correctly
        load_model(EMBEDDING_MODEL, device)
        print("Dry run complete — model loaded successfully.")
        return

    if not check_existing_outputs(args.force):
        sys.exit(0)

    # Ensure output directory exists early
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(CHUNKS_FILE)
    total  = len(chunks)

    model = load_model(EMBEDDING_MODEL, device)

    print(f"\nEmbedding {total:,} chunks in batches of {args.batch_size} …\n")
    embeddings, embedded_ids, failed_batches = embed_chunks(model, chunks, args.batch_size)

    n_embedded = len(embedded_ids)
    n_failed   = total - n_embedded

    print(f"\n--- Embedding summary ---")
    print(f"  Total chunks   : {total:,}")
    print(f"  Embedded       : {n_embedded:,}")
    print(f"  Failed (skipped): {n_failed:,}  (across {failed_batches} failed batch(es))")

    if n_embedded == 0:
        print("No chunks were successfully embedded. Nothing saved.")
        sys.exit(1)

    save_outputs(embeddings, embedded_ids, EMBEDDING_MODEL, device)

    print("\nPhase 2.1 complete.")


if __name__ == "__main__":
    main()
