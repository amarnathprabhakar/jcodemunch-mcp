"""Search across a knowledge collection for relevant chunks."""

import time
from typing import Optional

from ..storage.knowledge_store import KnowledgeStore


def search_knowledge(
    collection: str,
    query: str,
    source_type: Optional[str] = None,
    max_results: int = 10,
    storage_path: Optional[str] = None,
) -> dict:
    """Search a knowledge collection for chunks relevant to a query.

    Uses weighted keyword scoring across chunk titles, summaries, keywords,
    and content bodies.  Results are sorted by relevance score (descending).

    Args:
        collection: Knowledge collection name to search.
        query: Free-text search query.
        source_type: Optional filter — ``"markdown"``, ``"url"``,
            ``"pdf"``, or ``"youtube"``.
        max_results: Maximum number of chunks to return (1–100).
        storage_path: Override the default ``~/.code-index/`` storage root.

    Returns:
        Dict with ``collection``, ``query``, ``result_count``, ``results``
        (list of chunk summaries with IDs), and ``_meta``.
    """
    start = time.perf_counter()
    max_results = max(1, min(max_results, 100))

    store = KnowledgeStore(base_path=storage_path)
    try:
        store._safe_collection_name(collection)
    except ValueError as exc:
        return {"error": str(exc)}

    index = store.load_index(collection)
    if not index:
        return {"error": f"Knowledge collection not found: {collection!r}"}

    results = index.search(query, source_type=source_type, max_results=max_results)

    # Build lightweight result summaries (no full content — agents call
    # get_chunk to retrieve the full text)
    output = []
    for chunk in results:
        output.append({
            "id": chunk["id"],
            "source_type": chunk.get("source_type"),
            "source": chunk.get("source"),
            "title": chunk.get("title"),
            "summary": chunk.get("summary"),
            "keywords": chunk.get("keywords", []),
            "level": chunk.get("level"),
            "position": chunk.get("position"),
        })

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "collection": collection,
        "query": query,
        "result_count": len(output),
        "results": output,
        "_meta": {
            "timing_ms": round(elapsed, 1),
            "total_chunks": len(index.chunks),
            "source_types": index.source_types,
        },
    }
