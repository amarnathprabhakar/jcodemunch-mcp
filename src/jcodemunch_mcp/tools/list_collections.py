"""List all indexed knowledge collections."""

from typing import Optional

from ..storage.knowledge_store import KnowledgeStore


def list_collections(storage_path: Optional[str] = None) -> dict:
    """Return a summary of every indexed knowledge collection.

    Args:
        storage_path: Override the default ``~/.code-index/`` storage root.

    Returns:
        Dict with a ``collections`` list.  Each entry contains
        ``collection``, ``indexed_at``, ``chunk_count``, ``source_count``,
        and ``source_types``.
    """
    store = KnowledgeStore(base_path=storage_path)
    collections = store.list_collections()
    return {
        "collection_count": len(collections),
        "collections": collections,
    }
