"""Knowledge chunk data model for second-brain indexing."""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge from a document, web page, PDF, or transcript."""

    id: str               # Stable chunk ID: {collection}::{source}::{heading_slug}#{chunk_type}
    collection: str       # User-defined collection name
    source: str           # File path, URL, or YouTube video ID
    source_type: str      # "markdown", "pdf", "url", "youtube"
    title: str            # Section/chapter/heading title
    content: str          # Full text content of this chunk
    summary: str          # One-line summary (~120 chars)
    level: int            # Hierarchy level (1 = top, 2 = sub-section, etc.)
    position: int         # Ordinal position in the source (0-indexed)
    content_hash: str     # SHA-256 of content for drift detection
    keywords: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # Source-specific extras


def make_chunk_id(collection: str, source: str, heading_slug: str, chunk_type: str) -> str:
    """Create a stable, filesystem-safe chunk ID.

    Format: ``{collection}::{source}::{heading_slug}#{chunk_type}``

    Args:
        collection: User-defined collection name.
        source: Source path, URL, or video ID.
        heading_slug: Slugified heading / section identifier.
        chunk_type: One of "section", "page", "transcript".

    Returns:
        Stable string ID.
    """
    safe_source = re.sub(r"[^\w/.:\-]", "_", source)
    safe_slug = re.sub(r"[^\w/\- ]", "_", heading_slug).strip()[:60]
    return f"{collection}::{safe_source}::{safe_slug}#{chunk_type}"


def content_hash(text: str) -> str:
    """Compute SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Extract the most frequent meaningful words as keyword candidates.

    Args:
        text: Source text to analyse.
        max_keywords: Maximum keywords to return.

    Returns:
        List of keyword strings, most frequent first.
    """
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b", text.lower())

    stop_words = {
        "the", "and", "for", "that", "this", "with", "are", "from", "was",
        "have", "has", "will", "can", "you", "your", "they", "their", "its",
        "not", "but", "all", "been", "use", "how", "what", "when", "which",
        "one", "two", "any", "each", "more", "also", "into", "than", "then",
        "here", "there", "these", "those", "used", "using", "should", "would",
        "could", "may", "might", "must", "much", "many", "some", "such",
        "about", "after", "before", "above", "below", "between", "where",
        "just", "very", "most", "other", "only", "both", "during", "while",
    }

    freq: dict[str, int] = {}
    for word in words:
        if word not in stop_words:
            freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in sorted_words[:max_keywords]]


def make_summary(title: str, content: str, max_len: int = 120) -> str:
    """Generate a brief one-line summary from title and content.

    Tries the first sentence of content; falls back to the first line
    truncated to ``max_len`` characters; falls back to ``title``.

    Args:
        title: Section title (used as final fallback).
        content: Section body text.
        max_len: Maximum summary length in characters.

    Returns:
        Summary string.
    """
    text = content.strip()
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    if parts and 0 < len(parts[0]) <= max_len:
        return parts[0].strip()
    first_line = text.split("\n")[0].strip()
    if first_line:
        return first_line[:max_len].rstrip()
    return title[:max_len]
