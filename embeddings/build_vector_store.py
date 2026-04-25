"""
Phase 2.2 — Vector Store Builder
Loads Phase 2.1 embeddings and chunk metadata, then upserts everything into a
ChromaDB persistent collection called `tmt_chunks`.

Usage:
    python embeddings/build_vector_store.py
    python embeddings/build_vector_store.py --force
    python embeddings/build_vector_store.py --skip-verify
    python embeddings/build_vector_store.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Project root on sys.path so config.py is importable ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHROMA_DIR,
    CHUNKS_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDINGS_DIR,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNKS_FILE      = CHUNKS_DIR / "tmt_chunks_structured.json"
NPZ_FILE         = EMBEDDINGS_DIR / "tmt_chunk_embeddings.npz"
META_FILE        = EMBEDDINGS_DIR / "embedding_metadata.json"
COLLECTION_NAME  = "tmt_chunks"
CHROMA_BATCH_SIZE = 500

SAMPLE_QUERIES = [
    "chest pain with shortness of breath",
    "diabetes management insulin",
    "fever in immunocompromised patient",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    """Load chunks JSON and return the full list (preserving duplicates)."""
    print(f"Loading chunks from {path} …")
    with open(path, "r", encoding="utf-8") as fh:
        chunks_list = json.load(fh)
    print(f"  Loaded {len(chunks_list):,} chunks.")
    return chunks_list


def load_embeddings(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load the .npz archive and return (embeddings, chunk_ids)."""
    print(f"Loading embeddings from {path} …")
    archive    = np.load(path, allow_pickle=True)
    embeddings = archive["embeddings"].astype(np.float32)
    chunk_ids  = list(archive["chunk_ids"])
    print(f"  Loaded embeddings shape={embeddings.shape}, {len(chunk_ids):,} chunk IDs.")
    return embeddings, chunk_ids


def deduplicate_ids(chunk_ids: list[str]) -> list[str]:
    """
    Ensure all IDs are unique by appending _dup2, _dup3, etc. to duplicates.
    ChromaDB requires unique IDs per document.
    """
    from collections import Counter
    counts = Counter(chunk_ids)
    dupes = {cid for cid, n in counts.items() if n > 1}

    if dupes:
        print(f"  Found {len(dupes)} duplicate chunk_id(s) — deduplicating:")
        for d in sorted(dupes):
            print(f"    {d}  (appears {counts[d]}x)")

    seen: dict[str, int] = {}
    unique_ids = []
    for cid in chunk_ids:
        if cid in seen:
            seen[cid] += 1
            unique_ids.append(f"{cid}_dup{seen[cid]}")
        else:
            seen[cid] = 1
            unique_ids.append(cid)
    return unique_ids


def verify_alignment(chunk_ids_npz: list[str], chunks: list[dict]) -> None:
    """
    Confirm that the .npz and chunk JSON have the same count.
    Aborts on mismatch.
    """
    if len(chunk_ids_npz) != len(chunks):
        print(
            f"\nERROR: count mismatch — .npz has {len(chunk_ids_npz):,} IDs "
            f"but chunk JSON has {len(chunks):,} entries."
        )
        sys.exit(1)

    print(f"  Alignment verified — {len(chunk_ids_npz):,} entries in both sources.")


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def get_or_create_collection(chroma_dir: Path, force: bool):
    """
    Open a PersistentClient and return the `tmt_chunks` collection.
    If --force is set, delete the collection first so it is rebuilt from scratch.
    """
    import chromadb
    from chromadb.config import Settings

    chroma_dir.mkdir(parents=True, exist_ok=True)
    print(f"Opening ChromaDB client at {chroma_dir} …")
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    if force:
        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing:
            print(f"  --force: deleting existing collection '{COLLECTION_NAME}' …")
            client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Collection '{COLLECTION_NAME}' ready (current count: {collection.count():,}).")
    return collection


def build_metadata_record(chunk: dict) -> dict:
    """Return a ChromaDB-compatible metadata dict for a single chunk."""
    page_range = chunk.get("page_range")
    if not isinstance(page_range, str):
        # Convert list/tuple/int to string so ChromaDB accepts it
        page_range = str(page_range) if page_range is not None else ""

    return {
        "chapter":    str(chunk.get("chapter",    "")),
        "section":    str(chunk.get("section",    "")),
        "subsection": str(chunk.get("subsection", "")),
        "page_range": page_range,
        "word_count": int(chunk.get("word_count", 0)),
        "chunk_type": str(chunk.get("chunk_type", "")),
        "source":     str(chunk.get("source",     "")),
    }


def upsert_batches(
    collection,
    unique_ids: list[str],
    embeddings: np.ndarray,
    chunks: list[dict],
    batch_size: int = CHROMA_BATCH_SIZE,
) -> None:
    """Upsert all chunks into the collection in batches of `batch_size`."""
    total     = len(unique_ids)
    n_batches = (total + batch_size - 1) // batch_size

    print(f"\nUpserting {total:,} chunks in {n_batches} batch(es) of up to {batch_size} …\n")

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, total)

        batch_ids        = unique_ids[start:end]
        batch_embeddings = embeddings[start:end].tolist()
        batch_documents  = [chunks[i].get("text", "") for i in range(start, end)]
        batch_metadatas  = [build_metadata_record(chunks[i]) for i in range(start, end)]

        print(
            f"  Batch {batch_idx + 1}/{n_batches} "
            f"(chunks {start + 1}–{end} of {total}) …",
            end=" ",
            flush=True,
        )

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )
        print("OK")

    print()


def verify_count(collection, expected: int) -> None:
    """Check that the collection count matches the expected total."""
    actual = collection.count()
    status = "PASS" if actual == expected else "FAIL"
    print(f"Count verification [{status}]: expected {expected:,}, got {actual:,}.")
    if actual != expected:
        print(
            f"  WARNING: {abs(actual - expected):,} chunk(s) "
            f"{'missing' if actual < expected else 'extra'}."
        )


# ── Device detection ──────────────────────────────────────────────────────────

def detect_device() -> str:
    """Return 'mps' if Apple Silicon GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ── Sample queries ─────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str):
    """Load and return a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model '{model_name}' on device '{device}' …")
    model = SentenceTransformer(model_name, device=device)
    print("  Model loaded.")
    return model


def run_sample_queries(collection, model) -> None:
    """Embed each sample query and print the top-3 results."""
    print("\n=== Sample queries ===\n")
    for query in SAMPLE_QUERIES:
        print(f"Query: \"{query}\"")
        query_vec = model.encode(query, convert_to_numpy=True).tolist()
        results   = collection.query(
            query_embeddings=[query_vec],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        ids        = results["ids"][0]
        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]

        for rank, (cid, meta, dist) in enumerate(zip(ids, metadatas, distances), start=1):
            print(
                f"  [{rank}] chunk_id={cid}"
                f"  chapter={meta.get('chapter', '')!r}"
                f"  section={meta.get('section', '')!r}"
                f"  distance={dist:.4f}"
            )
        print()


# ── Dry-run summary ───────────────────────────────────────────────────────────

def print_dry_run_config() -> None:
    print("\n=== DRY RUN — configuration only, no store built ===")
    print(f"  Chunks file      : {CHUNKS_FILE}")
    print(f"  Embeddings .npz  : {NPZ_FILE}")
    print(f"  Embedding metadata: {META_FILE}")
    print(f"  ChromaDB dir     : {CHROMA_DIR}")
    print(f"  Collection name  : {COLLECTION_NAME}")
    print(f"  Embedding model  : {EMBEDDING_MODEL}")
    print(f"  Embedding dim    : {EMBEDDING_DIM}")
    print(f"  Upsert batch size: {CHROMA_BATCH_SIZE}")
    print("=====================================================\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2.2 — Build ChromaDB vector store from Phase 2.1 embeddings.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing ChromaDB collection and rebuild from scratch.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the sample query sanity checks after building.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data, print config, and exit without building the store.",
    )
    args = parser.parse_args()

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print_dry_run_config()
        chunks = load_chunks(CHUNKS_FILE)
        embeddings, chunk_ids_npz = load_embeddings(NPZ_FILE)
        verify_alignment(chunk_ids_npz, chunks)
        print("Dry run complete — data loads and alignment checks passed.")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    chunks                    = load_chunks(CHUNKS_FILE)
    embeddings, chunk_ids_npz = load_embeddings(NPZ_FILE)

    print("\nVerifying chunk_id alignment …")
    verify_alignment(chunk_ids_npz, chunks)

    # Deduplicate IDs for ChromaDB (which requires unique IDs)
    unique_ids = deduplicate_ids(chunk_ids_npz)
    expected_total = len(unique_ids)

    # ── Build collection ──────────────────────────────────────────────────────
    collection = get_or_create_collection(CHROMA_DIR, force=args.force)

    upsert_batches(collection, unique_ids, embeddings, chunks)

    # ── Post-insert count check ───────────────────────────────────────────────
    print("--- Post-insert verification ---")
    verify_count(collection, expected_total)

    # ── Sample queries ─────────────────────────────────────────────────────────
    if not args.skip_verify:
        device = detect_device()
        model  = load_model(EMBEDDING_MODEL, device)
        run_sample_queries(collection, model)
    else:
        print("\n--skip-verify set — skipping sample queries.")

    print("\nPhase 2.2 complete.")


if __name__ == "__main__":
    main()
