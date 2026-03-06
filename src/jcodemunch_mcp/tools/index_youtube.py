"""Index a YouTube video transcript into a knowledge collection."""

import logging
import time
from typing import Optional

from ..ingestors.youtube_ingestor import ingest_youtube, extract_video_id
from ..storage.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


def index_youtube(
    url: str,
    collection: str,
    chunk_seconds: int = 120,
    languages: Optional[list[str]] = None,
    storage_path: Optional[str] = None,
) -> dict:
    """Fetch a YouTube video's transcript and add it to a knowledge collection.

    Requires the optional ``youtube-transcript-api`` package::

        pip install "jcodemunch-mcp[youtube]"

    The transcript is grouped into time-based chunks (default: 2 minutes
    each).  Re-indexing the same video replaces the existing transcript so
    the collection stays up-to-date.

    Args:
        url: YouTube video URL or bare 11-character video ID.
        collection: Knowledge collection name.
        chunk_seconds: Seconds of transcript per chunk (default: 120).
        languages: Language preference list (e.g. ``["en", "en-US"]``).
            Falls back to any available transcript.
        storage_path: Override the default ``~/.code-index/`` storage root.

    Returns:
        Dict with ``success``, ``collection``, ``video_id``, ``chunk_count``,
        ``total_chunks``, and ``_meta``.
    """
    start = time.perf_counter()

    store = KnowledgeStore(base_path=storage_path)
    try:
        store._safe_collection_name(collection)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    video_id = extract_video_id(url)
    if not video_id:
        return {"success": False, "error": f"Could not extract a video ID from: {url!r}"}

    try:
        new_chunks = ingest_youtube(
            url=url,
            collection=collection,
            chunk_seconds=chunk_seconds,
            languages=languages,
        )
    except Exception as exc:
        return {"success": False, "error": f"Failed to ingest YouTube transcript: {exc}"}

    if not new_chunks:
        return {
            "success": False,
            "error": (
                "No transcript chunks produced. The video may have no transcript, "
                "or youtube-transcript-api may not be installed."
            ),
        }

    updated = store.add_source(
        collection=collection,
        new_chunks=new_chunks,
        source=video_id,
        source_type="youtube",
    )

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "success": True,
        "collection": collection,
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "chunk_count": len(new_chunks),
        "total_chunks": len(updated.chunks),
        "_meta": {"timing_ms": round(elapsed, 1)},
    }
