"""URL (web page) ingestor — fetches HTML and chunks it into knowledge sections."""

import logging
import re
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urlparse

import httpx

from .chunks import KnowledgeChunk, make_chunk_id, content_hash, extract_keywords, make_summary

logger = logging.getLogger(__name__)

# HTML tags whose subtree we discard entirely (nav, ads, scripts, etc.)
_SKIP_TAGS = frozenset({
    "script", "style", "nav", "footer", "header", "aside",
    "form", "button", "noscript", "svg", "canvas",
})

# Tags that mark the start of a new section
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

# Tags that imply a paragraph break (append space before their text)
_BLOCK_TAGS = frozenset({
    "p", "li", "td", "th", "dt", "dd", "blockquote",
    "article", "section", "div", "main",
})


class _HTMLTextExtractor(HTMLParser):
    """Extract clean text + heading-based sections from an HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # Finished sections: list of (heading_level, heading_title, body_text)
        self.sections: list[tuple[int, str, str]] = []
        self._page_title: str = ""
        self._in_title: bool = False
        self._skip_depth: int = 0        # depth inside a SKIP_TAGS element
        self._heading_level: int = 0
        self._in_heading: bool = False
        self._heading_buf: list[str] = []
        self._current_level: int = 0
        self._current_title: str = ""
        self._body_buf: list[str] = []

    # ------------------------------------------------------------------
    # HTMLParser callbacks
    # ------------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _HEADING_TAGS:
            # Flush current body into a section
            body = self._flush_body()
            if body or self._current_title:
                self.sections.append((self._current_level, self._current_title, body))
            self._in_heading = True
            self._heading_buf = []
            self._current_level = int(tag[1])
            return
        if tag in _BLOCK_TAGS:
            self._body_buf.append(" ")
        elif tag == "br":
            self._body_buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _HEADING_TAGS:
            self._current_title = " ".join(self._heading_buf).strip()
            self._in_heading = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._page_title += data
            return
        if self._in_heading:
            stripped = data.strip()
            if stripped:
                self._heading_buf.append(stripped)
            return
        stripped = data.strip()
        if stripped:
            self._body_buf.append(stripped)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _flush_body(self) -> str:
        text = re.sub(r"\s+", " ", " ".join(self._body_buf)).strip()
        self._body_buf = []
        return text

    def finish(self) -> list[tuple[int, str, str]]:
        """Finalise the last open section and return all sections."""
        body = self._flush_body()
        if body or self._current_title:
            self.sections.append((self._current_level, self._current_title, body))
        return self.sections


def ingest_url(
    url: str,
    collection: str,
    html_content: Optional[str] = None,
) -> list[KnowledgeChunk]:
    """Fetch a web URL and split it into heading-based knowledge chunks.

    Uses ``httpx`` (already a project dependency) to fetch the page.
    Falls back to treating the full page body as a single chunk when no
    headings are detected.

    Args:
        url: HTTP/HTTPS URL of the page to ingest.
        collection: Knowledge collection name.
        html_content: Pre-fetched HTML string (skips the HTTP request when
            provided — useful for testing and offline indexing).

    Returns:
        List of KnowledgeChunk objects (empty on fetch / parse failure).
    """
    if html_content is None:
        try:
            resp = httpx.get(
                url,
                follow_redirects=True,
                timeout=30.0,
                headers={"User-Agent": "jcodemunch-mcp knowledge-indexer"},
            )
            resp.raise_for_status()
            html_content = resp.text
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", url, exc)
            return []

    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html_content)
    except Exception as exc:
        logger.warning("HTML parse error for %s: %s", url, exc)

    sections = extractor.finish()
    page_title = re.sub(r"\s+", " ", extractor._page_title).strip() or urlparse(url).netloc

    chunks: list[KnowledgeChunk] = []

    if not sections:
        return chunks

    for i, (level, title, text) in enumerate(sections):
        text = re.sub(r"\s+", " ", text).strip()
        if not text or len(text) < 20:
            continue

        effective_title = title or page_title
        slug = re.sub(r"\s+", "-", effective_title.lower())[:60]
        chunk_id = make_chunk_id(collection, url, slug, "section")

        # Disambiguate duplicate IDs within this page
        existing_ids = {c.id for c in chunks}
        base_id = chunk_id
        n = 1
        while chunk_id in existing_ids:
            chunk_id = f"{base_id}~{n}"
            n += 1

        chunks.append(
            KnowledgeChunk(
                id=chunk_id,
                collection=collection,
                source=url,
                source_type="url",
                title=effective_title,
                content=text[:10_000],
                summary=make_summary(effective_title, text),
                level=max(level, 1),
                position=i,
                content_hash=content_hash(text),
                keywords=extract_keywords(text),
                metadata={"url": url, "page_title": page_title, "heading_level": level},
            )
        )

    return chunks
