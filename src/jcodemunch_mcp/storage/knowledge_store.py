"""Knowledge index storage — save/load/search knowledge collections."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..ingestors.chunks import KnowledgeChunk

KNOWLEDGE_INDEX_VERSION = 1


@dataclass
class KnowledgeIndex:
    """Index for a named knowledge collection."""

    collection: str                  # User-defined collection name
    indexed_at: str                  # ISO 8601 timestamp of last update
    sources: list[str]               # Unique source identifiers (paths / URLs / video IDs)
    source_types: dict[str, int]     # source_type → number of sources of that type
    chunks: list[dict]               # Serialised KnowledgeChunk dicts
    index_version: int = KNOWLEDGE_INDEX_VERSION
    source_hashes: dict[str, str] = field(default_factory=dict)  # source → hash

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Return a chunk dict by its ID, or ``None``."""
        for chunk in self.chunks:
            if chunk.get("id") == chunk_id:
                return chunk
        return None

    def search(
        self,
        query: str,
        source_type: Optional[str] = None,
        max_results: int = 10,
    ) -> list[dict]:
        """Return chunks ranked by relevance to *query*.

        Args:
            query: Free-text search query.
            source_type: Optional filter — only return chunks whose
                ``source_type`` matches (e.g. ``"markdown"``, ``"url"``).
            max_results: Maximum number of chunks to return.

        Returns:
            Ranked list of chunk dicts, highest score first.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[int, dict]] = []
        for chunk in self.chunks:
            if source_type and chunk.get("source_type") != source_type:
                continue
            score = self._score_chunk(chunk, query_lower, query_words)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:max_results]]

    def _score_chunk(self, chunk: dict, query_lower: str, query_words: set) -> int:
        """Weighted relevance score for a single chunk."""
        score = 0

        # 1. Title match (highest weight)
        title_lower = chunk.get("title", "").lower()
        if query_lower == title_lower:
            score += 20
        elif query_lower in title_lower:
            score += 12
        for word in query_words:
            if word in title_lower:
                score += 5

        # 2. Summary match
        summary_lower = chunk.get("summary", "").lower()
        if query_lower in summary_lower:
            score += 8
        for word in query_words:
            if word in summary_lower:
                score += 2

        # 3. Keyword match
        keywords = set(chunk.get("keywords", []))
        score += len(query_words & keywords) * 3

        # 4. Content match (lower weight — content can be long)
        content_lower = chunk.get("content", "").lower()
        if query_lower in content_lower:
            score += 5
        for word in query_words:
            if word in content_lower:
                score += 1

        return score


class KnowledgeStore:
    """Storage for knowledge indexes, kept under ``{base_path}/knowledge/``.

    Knowledge indexes are stored separately from code indexes to avoid
    polluting the ``list_repos`` output and to allow an independent
    schema version.
    """

    def __init__(self, base_path: Optional[str] = None) -> None:
        """Initialise the store.

        Args:
            base_path: Root directory shared with ``IndexStore``
                (defaults to ``~/.code-index/``).  Knowledge indexes are
                written to the ``knowledge/`` sub-directory.
        """
        root = Path(base_path) if base_path else Path.home() / ".code-index"
        self.base_path = root / "knowledge"
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _safe_collection_name(self, name: str) -> str:
        """Validate a collection name used in on-disk paths.

        Only letters, digits, ``-``, ``_``, and ``.`` are allowed.

        Raises:
            ValueError: When *name* is unsafe.
        """
        if not name or name in {".", ".."}:
            raise ValueError(f"Invalid collection name: {name!r}")
        if "/" in name or "\\" in name:
            raise ValueError(f"Invalid collection name: {name!r}")
        if not re.fullmatch(r"[A-Za-z0-9._\-]+", name):
            raise ValueError(
                f"Invalid collection name: {name!r} "
                "(use only letters, digits, '.', '_', '-')"
            )
        return name

    def _index_path(self, collection: str) -> Path:
        return self.base_path / f"{self._safe_collection_name(collection)}.json"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_index(
        self,
        collection: str,
        chunks: list[KnowledgeChunk],
        sources: list[str],
        source_types: dict[str, int],
        source_hashes: Optional[dict[str, str]] = None,
    ) -> KnowledgeIndex:
        """Persist a complete knowledge index, replacing any existing one.

        Args:
            collection: Collection name.
            chunks: All KnowledgeChunk objects to store.
            sources: Unique source identifiers.
            source_types: Count of sources per type.
            source_hashes: Optional mapping of source → content hash.

        Returns:
            The saved KnowledgeIndex.
        """
        index = KnowledgeIndex(
            collection=collection,
            indexed_at=datetime.now().isoformat(),
            sources=sorted(sources),
            source_types=source_types,
            chunks=[self._chunk_to_dict(c) for c in chunks],
            source_hashes=source_hashes or {},
        )
        self._atomic_write(collection, index)
        return index

    def add_source(
        self,
        collection: str,
        new_chunks: list[KnowledgeChunk],
        source: str,
        source_type: str,
        source_hash: str = "",
    ) -> KnowledgeIndex:
        """Add or replace chunks for a single source within a collection.

        If *collection* does not yet exist it is created.  Existing chunks
        whose ``source`` matches *source* are removed before the new chunks
        are appended, so calling this method again with the same URL will
        update (not duplicate) the content.

        Args:
            collection: Collection name.
            new_chunks: Chunks produced for *source*.
            source: Source identifier (path, URL, or video ID).
            source_type: ``"markdown"``, ``"url"``, ``"pdf"``, or ``"youtube"``.
            source_hash: Optional content hash for change-detection.

        Returns:
            Updated KnowledgeIndex.
        """
        existing = self.load_index(collection)

        if existing:
            kept_chunks = [c for c in existing.chunks if c.get("source") != source]
            kept_sources = [s for s in existing.sources if s != source]
            source_hashes = dict(existing.source_hashes)
        else:
            kept_chunks = []
            kept_sources = []
            source_hashes = {}

        all_chunk_dicts = kept_chunks + [self._chunk_to_dict(c) for c in new_chunks]
        all_sources = kept_sources + [source]
        source_types = self._source_types_from_chunks(all_chunk_dicts)
        if source_hash:
            source_hashes[source] = source_hash

        index = KnowledgeIndex(
            collection=collection,
            indexed_at=datetime.now().isoformat(),
            sources=all_sources,
            source_types=source_types,
            chunks=all_chunk_dicts,
            source_hashes=source_hashes,
        )
        self._atomic_write(collection, index)
        return index

    def load_index(self, collection: str) -> Optional[KnowledgeIndex]:
        """Load a knowledge index from disk.

        Returns ``None`` when the collection does not exist or the stored
        schema version is newer than ``KNOWLEDGE_INDEX_VERSION``.
        """
        index_path = self._index_path(collection)
        if not index_path.exists():
            return None

        with open(index_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if data.get("index_version", 1) > KNOWLEDGE_INDEX_VERSION:
            return None  # Written by a newer version of the tool

        return KnowledgeIndex(
            collection=data["collection"],
            indexed_at=data["indexed_at"],
            sources=data["sources"],
            source_types=data["source_types"],
            chunks=data["chunks"],
            index_version=data.get("index_version", 1),
            source_hashes=data.get("source_hashes", {}),
        )

    def list_collections(self) -> list[dict]:
        """Return summary metadata for every stored knowledge collection."""
        results = []
        for p in sorted(self.base_path.glob("*.json")):
            if p.name.endswith(".json.tmp"):
                continue
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if "collection" not in data:
                    continue  # Not a knowledge index
                results.append({
                    "collection": data["collection"],
                    "indexed_at": data["indexed_at"],
                    "chunk_count": len(data["chunks"]),
                    "source_count": len(data["sources"]),
                    "source_types": data["source_types"],
                })
            except Exception:
                continue
        return results

    def delete_collection(self, collection: str) -> bool:
        """Delete a knowledge collection index.

        Returns:
            ``True`` when deleted, ``False`` when not found.
        """
        index_path = self._index_path(collection)
        if index_path.exists():
            index_path.unlink()
            return True
        return False

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def _chunk_to_dict(self, chunk: KnowledgeChunk) -> dict:
        return {
            "id": chunk.id,
            "collection": chunk.collection,
            "source": chunk.source,
            "source_type": chunk.source_type,
            "title": chunk.title,
            "content": chunk.content,
            "summary": chunk.summary,
            "level": chunk.level,
            "position": chunk.position,
            "content_hash": chunk.content_hash,
            "keywords": chunk.keywords,
            "metadata": chunk.metadata,
        }

    def _index_to_dict(self, index: KnowledgeIndex) -> dict:
        return {
            "collection": index.collection,
            "indexed_at": index.indexed_at,
            "sources": index.sources,
            "source_types": index.source_types,
            "chunks": index.chunks,
            "index_version": index.index_version,
            "source_hashes": index.source_hashes,
        }

    def _atomic_write(self, collection: str, index: KnowledgeIndex) -> None:
        """Write *index* atomically (write-then-rename)."""
        index_path = self._index_path(collection)
        tmp_path = index_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._index_to_dict(index), fh, indent=2)
        tmp_path.replace(index_path)

    @staticmethod
    def _source_types_from_chunks(chunks: list[dict]) -> dict[str, int]:
        """Rebuild source_type → unique-source-count from a list of chunk dicts."""
        seen: dict[str, set] = {}
        for chunk in chunks:
            st = chunk.get("source_type", "unknown")
            src = chunk.get("source", "")
            seen.setdefault(st, set()).add(src)
        return {st: len(srcs) for st, srcs in seen.items()}
