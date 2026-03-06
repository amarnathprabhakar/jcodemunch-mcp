"""Markdown / RST / TXT document ingestor — splits files into heading-based chunks."""

import logging
import re
from pathlib import Path

from .chunks import KnowledgeChunk, make_chunk_id, content_hash, extract_keywords, make_summary

logger = logging.getLogger(__name__)

# Supported document extensions
DOC_EXTENSIONS = {
    ".md", ".markdown", ".mdx",
    ".rst", ".txt", ".adoc", ".asciidoc",
}

# Heading pattern for Markdown (and Markdown-style RST headings)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Plain-text paragraph target size (characters)
_PLAIN_CHUNK_SIZE = 2000


def is_doc_file(path: str) -> bool:
    """Return True if the file extension is a supported documentation type."""
    return Path(path).suffix.lower() in DOC_EXTENSIONS


def _parse_markdown_chunks(
    content: str,
    source: str,
    collection: str,
) -> list[KnowledgeChunk]:
    """Split a Markdown document into heading-based chunks.

    Each ATX heading (# … ######) begins a new chunk whose content is all
    text up to the next heading at any level.

    Args:
        content: Raw Markdown text.
        source: Relative file path used as source identifier.
        collection: Knowledge collection name.

    Returns:
        List of KnowledgeChunk objects.
    """
    # Locate all headings with their byte positions
    positions: list[tuple[int, int, int, str]] = []  # (start, end, level, title)
    for match in _HEADING_RE.finditer(content):
        level = len(match.group(1))
        title = match.group(2).strip()
        positions.append((match.start(), match.end(), level, title))

    # Capture intro text that appears before the first heading
    if positions and positions[0][0] > 0:
        intro = content[: positions[0][0]].strip()
        if intro:
            positions.insert(0, (0, 0, 0, Path(source).stem))

    if not positions:
        # No headings — wrap the whole file as one chunk
        chunk_id = make_chunk_id(collection, source, "content", "section")
        source_name = Path(source).name
        return [
            KnowledgeChunk(
                id=chunk_id,
                collection=collection,
                source=source,
                source_type="markdown",
                title=source_name,
                content=content.strip(),
                summary=make_summary(source_name, content),
                level=1,
                position=0,
                content_hash=content_hash(content),
                keywords=extract_keywords(content),
            )
        ]

    chunks: list[KnowledgeChunk] = []
    for i, (start, end, level, title) in enumerate(positions):
        # Determine where this section's body ends
        next_start = positions[i + 1][0] if i + 1 < len(positions) else len(content)
        if start == 0 and end == 0:
            # Intro pseudo-section
            chunk_body = content[: next_start].strip()
        else:
            chunk_body = content[end:next_start].strip()

        if not chunk_body:
            continue

        heading_slug = re.sub(r"\s+", "-", title.lower())[:60]
        chunk_id = make_chunk_id(collection, source, heading_slug, "section")

        # Disambiguate duplicate IDs
        existing_ids = {c.id for c in chunks}
        base_id = chunk_id
        suffix = 1
        while chunk_id in existing_ids:
            chunk_id = f"{base_id}~{suffix}"
            suffix += 1

        chunks.append(
            KnowledgeChunk(
                id=chunk_id,
                collection=collection,
                source=source,
                source_type="markdown",
                title=title,
                content=chunk_body,
                summary=make_summary(title, chunk_body),
                level=max(level, 1),
                position=i,
                content_hash=content_hash(chunk_body),
                keywords=extract_keywords(chunk_body),
                metadata={"heading_level": level},
            )
        )

    return chunks


def _parse_plain_text_chunks(
    content: str,
    source: str,
    collection: str,
    chunk_size: int = _PLAIN_CHUNK_SIZE,
) -> list[KnowledgeChunk]:
    """Split plain text into paragraph-sized chunks.

    Paragraphs (blank-line separated) are accumulated until the target
    ``chunk_size`` is reached, then flushed as a chunk.

    Args:
        content: Raw plain text.
        source: Relative file path used as source identifier.
        collection: Knowledge collection name.
        chunk_size: Target character count per chunk.

    Returns:
        List of KnowledgeChunk objects.
    """
    paragraphs = re.split(r"\n\s*\n", content.strip())

    chunks: list[KnowledgeChunk] = []
    current_paras: list[str] = []
    current_size = 0
    chunk_num = 0

    def _flush() -> None:
        nonlocal chunk_num, current_paras, current_size
        text = "\n\n".join(current_paras)
        if not text:
            return
        title = f"Section {chunk_num + 1}"
        cid = make_chunk_id(collection, source, f"section-{chunk_num}", "section")
        chunks.append(
            KnowledgeChunk(
                id=cid,
                collection=collection,
                source=source,
                source_type="markdown",
                title=title,
                content=text,
                summary=make_summary(title, text),
                level=1,
                position=chunk_num,
                content_hash=content_hash(text),
                keywords=extract_keywords(text),
            )
        )
        chunk_num += 1
        current_paras = []
        current_size = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_size + len(para) > chunk_size and current_paras:
            _flush()
        current_paras.append(para)
        current_size += len(para)

    _flush()  # Last chunk
    return chunks


def ingest_doc_file(
    file_path: Path,
    relative_path: str,
    collection: str,
) -> list[KnowledgeChunk]:
    """Ingest a documentation file into knowledge chunks.

    Markdown files (and files with Markdown-style headings) are split by
    heading hierarchy.  Plain text / RST / AsciiDoc files fall back to
    paragraph-based splitting.

    Args:
        file_path: Absolute path to the file.
        relative_path: Relative path used as the stable source identifier.
        collection: Knowledge collection name.

    Returns:
        List of KnowledgeChunk objects (empty on read failure).
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Failed to read %s: %s", file_path, exc)
        return []

    if not content.strip():
        return []

    ext = file_path.suffix.lower()

    if ext in {".md", ".markdown", ".mdx"}:
        return _parse_markdown_chunks(content, relative_path, collection)

    # For RST / AsciiDoc / plain text: use Markdown chunker when ATX-style
    # headings are present, otherwise fall back to paragraph splitting.
    if re.search(r"^#{1,6}\s", content, re.MULTILINE):
        return _parse_markdown_chunks(content, relative_path, collection)

    return _parse_plain_text_chunks(content, relative_path, collection)
