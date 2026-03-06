"""PDF ingestor — extracts text page-by-page using the optional ``pypdf`` library."""

import logging
import re
from pathlib import Path

from .chunks import KnowledgeChunk, make_chunk_id, content_hash, extract_keywords, make_summary

logger = logging.getLogger(__name__)

try:
    import pypdf as _pypdf
    _PYPDF_AVAILABLE = True
except ImportError:
    _pypdf = None  # type: ignore[assignment]
    _PYPDF_AVAILABLE = False


def ingest_pdf(
    file_path: Path,
    relative_path: str,
    collection: str,
) -> list[KnowledgeChunk]:
    """Extract text from a PDF file and produce one chunk per page.

    Requires the optional ``pypdf`` package::

        pip install "jcodemunch-mcp[pdf]"

    The first short line on each page is used as the chunk title when it
    looks like a heading; otherwise the title is ``"{stem} — Page N"``.

    Args:
        file_path: Absolute path to the PDF file.
        relative_path: Relative path used as the stable source identifier.
        collection: Knowledge collection name.

    Returns:
        List of KnowledgeChunk objects (empty when pypdf is not installed or
        the file cannot be read).
    """
    if not _PYPDF_AVAILABLE:
        logger.warning(
            "pypdf is not installed; PDF indexing is unavailable. "
            "Install it with: pip install \"jcodemunch-mcp[pdf]\""
        )
        return []

    try:
        reader = _pypdf.PdfReader(str(file_path))
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", file_path, exc)
        return []

    chunks: list[KnowledgeChunk] = []
    pdf_stem = Path(relative_path).stem
    total_pages = len(reader.pages)

    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "Could not extract text from page %d of %s: %s",
                page_num + 1, file_path, exc,
            )
            continue

        text = text.strip()
        if not text:
            continue

        # Detect a heading-like first line
        lines = text.splitlines()
        first_line = lines[0].strip() if lines else ""
        if first_line and len(first_line) < 100 and not first_line.endswith("."):
            title = first_line
            body = "\n".join(lines[1:]).strip() or text
        else:
            title = f"{pdf_stem} — Page {page_num + 1}"
            body = text

        slug = f"page-{page_num + 1}"
        chunk_id = make_chunk_id(collection, relative_path, slug, "page")

        chunks.append(
            KnowledgeChunk(
                id=chunk_id,
                collection=collection,
                source=relative_path,
                source_type="pdf",
                title=title,
                content=body,
                summary=make_summary(title, body),
                level=1,
                position=page_num,
                content_hash=content_hash(body),
                keywords=extract_keywords(body),
                metadata={"page": page_num + 1, "total_pages": total_pages},
            )
        )

    return chunks
