"""Retrieve the full content of a specific knowledge chunk."""

import time
from typing import Optional

from ..storage.knowledge_store import KnowledgeStore


def get_chunk(
    collection: str,
    chunk_id: str,
    storage_path: Optional[str] = None,
) -> dict:
    """Return the full text content of a knowledge chunk by its ID.

    Use ``search_knowledge`` first to discover relevant chunk IDs, then
    call this tool to retrieve the full text for the chunks you need.

    Args:
        collection: Knowledge collection name.
        chunk_id: Chunk ID from ``search_knowledge`` results.
        storage_path: Override the default ``~/.code-index/`` storage root.

    Returns:
        Dict with ``chunk_id``, ``title``, ``source``, ``source_type``,
        ``content``, ``summary``, ``keywords``, and ``metadata``.
    """
    start = time.perf_counter()

    store = KnowledgeStore(base_path=storage_path)
    try:
        store._safe_collection_name(collection)
    except ValueError as exc:
        return {"error": str(exc)}

    index = store.load_index(collection)
    if not index:
        return {"error": f"Knowledge collection not found: {collection!r}"}

    chunk = index.get_chunk(chunk_id)
    if not chunk:
        return {"error": f"Chunk not found: {chunk_id!r}"}

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "chunk_id": chunk["id"],
        "collection": collection,
        "source": chunk.get("source"),
        "source_type": chunk.get("source_type"),
        "title": chunk.get("title"),
        "content": chunk.get("content"),
        "summary": chunk.get("summary"),
        "keywords": chunk.get("keywords", []),
        "level": chunk.get("level"),
        "position": chunk.get("position"),
        "metadata": chunk.get("metadata", {}),
        "_meta": {"timing_ms": round(elapsed, 1)},
    }
