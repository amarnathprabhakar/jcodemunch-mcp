"""Index a web URL (article, blog post, documentation page) into a knowledge collection."""

import hashlib
import logging
import time
from typing import Optional

from ..ingestors.url_ingestor import ingest_url
from ..storage.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


def index_url(
    url: str,
    collection: str,
    storage_path: Optional[str] = None,
) -> dict:
    """Fetch a web page and add it to a named knowledge collection.

    The page is fetched with ``httpx``, its HTML is converted to clean text,
    and the text is split into heading-based chunks.  Calling this tool again
    with the same URL replaces the previous version of that page in the
    collection, so re-indexing is safe.

    Args:
        url: HTTP or HTTPS URL to fetch and index.
        collection: Knowledge collection name to add this page to.
        storage_path: Override the default ``~/.code-index/`` storage root.

    Returns:
        Dict with ``success``, ``collection``, ``url``, ``new_chunks``,
        ``total_chunks``, and ``_meta``.
    """
    start = time.perf_counter()

    if not url.startswith(("http://", "https://")):
        return {"success": False, "error": "URL must start with http:// or https://"}

    store = KnowledgeStore(base_path=storage_path)
    try:
        store._safe_collection_name(collection)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    try:
        new_chunks = ingest_url(url, collection)
    except Exception as exc:
        return {"success": False, "error": f"Failed to ingest URL: {exc}"}

    if not new_chunks:
        return {"success": False, "error": "No content could be extracted from the URL."}

    source_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    updated = store.add_source(
        collection=collection,
        new_chunks=new_chunks,
        source=url,
        source_type="url",
        source_hash=source_hash,
    )

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "success": True,
        "collection": collection,
        "url": url,
        "new_chunks": len(new_chunks),
        "total_chunks": len(updated.chunks),
        "_meta": {"timing_ms": round(elapsed, 1)},
    }
