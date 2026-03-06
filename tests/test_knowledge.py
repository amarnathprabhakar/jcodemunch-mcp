"""Tests for knowledge indexing (second-brain) functionality."""

import hashlib
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from jcodemunch_mcp.ingestors.chunks import (
    KnowledgeChunk,
    make_chunk_id,
    content_hash,
    extract_keywords,
    make_summary,
)
from jcodemunch_mcp.ingestors.doc_ingestor import (
    ingest_doc_file,
    is_doc_file,
    DOC_EXTENSIONS,
    _parse_markdown_chunks,
    _parse_plain_text_chunks,
)
from jcodemunch_mcp.ingestors.url_ingestor import ingest_url
from jcodemunch_mcp.ingestors.youtube_ingestor import extract_video_id, ingest_youtube
from jcodemunch_mcp.ingestors.pdf_ingestor import ingest_pdf
from jcodemunch_mcp.storage.knowledge_store import (
    KnowledgeStore,
    KnowledgeIndex,
    KNOWLEDGE_INDEX_VERSION,
)
from jcodemunch_mcp.tools.index_docs import index_docs
from jcodemunch_mcp.tools.index_url import index_url
from jcodemunch_mcp.tools.index_youtube import index_youtube
from jcodemunch_mcp.tools.search_knowledge import search_knowledge
from jcodemunch_mcp.tools.get_chunk import get_chunk
from jcodemunch_mcp.tools.list_collections import list_collections


# ---------------------------------------------------------------------------
# chunks.py
# ---------------------------------------------------------------------------

class TestMakeChunkId:
    def test_basic_id_format(self):
        cid = make_chunk_id("docs", "README.md", "introduction", "section")
        assert cid == "docs::README.md::introduction#section"

    def test_special_chars_in_source_are_sanitised(self):
        cid = make_chunk_id("c1", "https://example.com/page?q=1", "title", "section")
        # Should not raise; colons, slashes, dashes, dots all allowed
        assert "c1::" in cid
        assert "#section" in cid

    def test_heading_slug_truncated_to_60(self):
        long_slug = "a" * 100
        cid = make_chunk_id("c", "src.md", long_slug, "section")
        # The slug portion should be at most 60 chars (before the #)
        slug_part = cid.split("::")[-1].split("#")[0]
        assert len(slug_part) <= 60


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_inputs(self):
        assert content_hash("hello") != content_hash("world")

    def test_sha256_format(self):
        h = content_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestExtractKeywords:
    def test_returns_list(self):
        kws = extract_keywords("Python is a programming language used for data science")
        assert isinstance(kws, list)

    def test_stops_words_excluded(self):
        kws = extract_keywords("the quick brown fox")
        assert "the" not in kws

    def test_max_keywords_respected(self):
        text = " ".join([f"word{i}" for i in range(50)])
        kws = extract_keywords(text, max_keywords=5)
        assert len(kws) <= 5

    def test_non_empty_result_for_real_text(self):
        kws = extract_keywords("authentication token session login user password")
        assert len(kws) > 0


class TestMakeSummary:
    def test_first_sentence_used_when_short(self):
        result = make_summary("Title", "This is the first sentence. More text follows.")
        assert result == "This is the first sentence."

    def test_falls_back_to_first_line(self):
        result = make_summary("Title", "A very long sentence that goes on and on" * 5)
        assert len(result) <= 120

    def test_falls_back_to_title(self):
        result = make_summary("My Title", "")
        assert result == "My Title"

    def test_max_len_respected(self):
        long_text = "x" * 200
        result = make_summary("t", long_text)
        assert len(result) <= 120


# ---------------------------------------------------------------------------
# doc_ingestor.py
# ---------------------------------------------------------------------------

class TestIsDocFile:
    def test_markdown_is_doc(self):
        assert is_doc_file("README.md") is True
        assert is_doc_file("guide.markdown") is True
        assert is_doc_file("notes.mdx") is True

    def test_rst_is_doc(self):
        assert is_doc_file("index.rst") is True

    def test_txt_is_doc(self):
        assert is_doc_file("notes.txt") is True

    def test_code_is_not_doc(self):
        assert is_doc_file("main.py") is False
        assert is_doc_file("app.js") is False


class TestParseMarkdownChunks:
    def test_splits_on_headings(self):
        md = "# Title\n\nIntro text.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B."
        chunks = _parse_markdown_chunks(md, "test.md", "col1")
        titles = [c.title for c in chunks]
        assert "Title" in titles
        assert "Section A" in titles
        assert "Section B" in titles

    def test_no_headings_returns_single_chunk(self):
        md = "Just some plain markdown without any headings at all."
        chunks = _parse_markdown_chunks(md, "test.md", "col1")
        assert len(chunks) == 1
        assert chunks[0].content == md

    def test_chunk_ids_are_unique(self):
        md = "## Duplicate\n\nFirst.\n\n## Duplicate\n\nSecond."
        chunks = _parse_markdown_chunks(md, "test.md", "col1")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_source_type_is_markdown(self):
        chunks = _parse_markdown_chunks("# H1\n\nContent.", "f.md", "c")
        assert all(c.source_type == "markdown" for c in chunks)

    def test_content_hash_populated(self):
        chunks = _parse_markdown_chunks("# H1\n\nContent.", "f.md", "c")
        for c in chunks:
            assert len(c.content_hash) == 64

    def test_intro_before_first_heading_is_captured(self):
        md = "Preamble text.\n\n# Heading\n\nSection body."
        chunks = _parse_markdown_chunks(md, "test.md", "col1")
        assert any("Preamble text." in c.content for c in chunks)


class TestParsePlainTextChunks:
    def test_produces_at_least_one_chunk(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = _parse_plain_text_chunks(text, "notes.txt", "col1")
        assert len(chunks) >= 1

    def test_chunk_size_respected(self):
        big_para = "word " * 500
        text = "\n\n".join([big_para] * 5)
        chunks = _parse_plain_text_chunks(text, "big.txt", "col1", chunk_size=200)
        assert len(chunks) > 1


class TestIngestDocFile:
    def test_markdown_file(self, tmp_path):
        f = tmp_path / "guide.md"
        f.write_text("# Guide\n\nWelcome to the guide.\n\n## Setup\n\nRun `pip install`.")
        chunks = ingest_doc_file(f, "guide.md", "test-col")
        assert len(chunks) >= 2
        assert any(c.title == "Guide" or c.title == "Setup" for c in chunks)

    def test_txt_file(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("First paragraph.\n\nSecond paragraph.")
        chunks = ingest_doc_file(f, "notes.txt", "test-col")
        assert len(chunks) >= 1

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("   ")
        chunks = ingest_doc_file(f, "empty.md", "col")
        assert chunks == []


# ---------------------------------------------------------------------------
# url_ingestor.py
# ---------------------------------------------------------------------------

class TestIngestUrl:
    SAMPLE_HTML = """
    <html>
    <head><title>Test Article</title></head>
    <body>
      <h1>Introduction</h1>
      <p>Welcome to the article. This is the intro paragraph with enough text to pass the length check.</p>
      <h2>Details</h2>
      <p>Here are the details of the topic with sufficient content for a chunk.</p>
      <script>var x = 1;</script>
      <nav>Skip me</nav>
    </body>
    </html>
    """

    def test_parses_html_into_chunks(self):
        chunks = ingest_url("https://example.com/article", "test-col", html_content=self.SAMPLE_HTML)
        assert len(chunks) >= 1

    def test_headings_become_titles(self):
        chunks = ingest_url("https://example.com/article", "test-col", html_content=self.SAMPLE_HTML)
        titles = [c.title for c in chunks]
        assert any("Introduction" in t or "Details" in t for t in titles)

    def test_script_content_excluded(self):
        chunks = ingest_url("https://example.com/article", "test-col", html_content=self.SAMPLE_HTML)
        for c in chunks:
            assert "var x = 1" not in c.content

    def test_source_type_is_url(self):
        chunks = ingest_url("https://example.com/article", "test-col", html_content=self.SAMPLE_HTML)
        for c in chunks:
            assert c.source_type == "url"

    def test_source_is_the_url(self):
        url = "https://example.com/article"
        chunks = ingest_url(url, "test-col", html_content=self.SAMPLE_HTML)
        for c in chunks:
            assert c.source == url

    def test_chunk_ids_are_unique(self):
        chunks = ingest_url("https://example.com", "col", html_content=self.SAMPLE_HTML)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_html_returns_empty(self):
        chunks = ingest_url("https://example.com", "col", html_content="<html></html>")
        assert chunks == []

    def test_fetch_failure_returns_empty(self):
        """ingest_url should return [] when httpx raises."""
        import httpx as real_httpx
        with patch("jcodemunch_mcp.ingestors.url_ingestor.httpx") as mock_httpx:
            mock_httpx.get.side_effect = real_httpx.ConnectError("no route to host")
            result = ingest_url("https://unreachable.example.com", "col")
        assert result == []


# ---------------------------------------------------------------------------
# youtube_ingestor.py — extract_video_id
# ---------------------------------------------------------------------------

class TestExtractVideoId:
    def test_standard_watch_url(self):
        vid = extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

    def test_short_url(self):
        vid = extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

    def test_embed_url(self):
        vid = extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

    def test_bare_video_id(self):
        vid = extract_video_id("dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

    def test_invalid_url_returns_none(self):
        assert extract_video_id("https://notyoutube.com/video") is None

    def test_url_with_playlist(self):
        vid = extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123")
        assert vid == "dQw4w9WgXcQ"


class TestIngestYoutube:
    def test_returns_empty_when_library_missing(self):
        with patch("jcodemunch_mcp.ingestors.youtube_ingestor._YOUTUBE_AVAILABLE", False):
            chunks = ingest_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "col")
        assert chunks == []

    def test_returns_empty_for_invalid_url(self):
        chunks = ingest_youtube("not-a-url", "col")
        assert chunks == []

    def test_groups_segments_into_chunks(self):
        """Mock the API to return test segments and verify chunking."""
        fake_segments = [
            {"text": f"word{i}", "start": float(i * 10), "duration": 10.0}
            for i in range(20)  # 200 seconds total
        ]

        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = fake_segments

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_transcript.return_value = mock_transcript

        mock_yta = MagicMock()
        mock_yta.list_transcripts.return_value = mock_transcript_list

        with patch("jcodemunch_mcp.ingestors.youtube_ingestor._YOUTUBE_AVAILABLE", True):
            with patch("jcodemunch_mcp.ingestors.youtube_ingestor.YouTubeTranscriptApi", mock_yta):
                chunks = ingest_youtube(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "col",
                    chunk_seconds=120,
                )

        # 200s / 120s per chunk ≈ 2 chunks
        assert len(chunks) >= 1
        for c in chunks:
            assert c.source_type == "youtube"
            assert "Transcript" in c.title
            assert c.content

    def test_chunk_metadata_contains_video_id(self):
        fake_segments = [
            {"text": "hello world", "start": 0.0, "duration": 5.0},
        ]

        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = fake_segments

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_transcript.return_value = mock_transcript

        mock_yta = MagicMock()
        mock_yta.list_transcripts.return_value = mock_transcript_list

        with patch("jcodemunch_mcp.ingestors.youtube_ingestor._YOUTUBE_AVAILABLE", True):
            with patch("jcodemunch_mcp.ingestors.youtube_ingestor.YouTubeTranscriptApi", mock_yta):
                chunks = ingest_youtube("dQw4w9WgXcQ", "col")

        assert len(chunks) == 1
        assert chunks[0].metadata["video_id"] == "dQw4w9WgXcQ"


# ---------------------------------------------------------------------------
# pdf_ingestor.py
# ---------------------------------------------------------------------------

class TestIngestPdf:
    def test_returns_empty_when_pypdf_missing(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake")
        with patch("jcodemunch_mcp.ingestors.pdf_ingestor._PYPDF_AVAILABLE", False):
            chunks = ingest_pdf(f, "doc.pdf", "col")
        assert chunks == []

    def test_pdf_produces_page_chunks(self, tmp_path):
        """Mock _pypdf (the module-level alias) to test chunking logic."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample page content for testing purposes."

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page, mock_page]

        mock_pypdf_module = MagicMock()
        mock_pypdf_module.PdfReader.return_value = mock_reader

        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 stub")

        with patch("jcodemunch_mcp.ingestors.pdf_ingestor._PYPDF_AVAILABLE", True):
            with patch("jcodemunch_mcp.ingestors.pdf_ingestor._pypdf", mock_pypdf_module):
                chunks = ingest_pdf(f, "test.pdf", "col")

        assert len(chunks) == 2
        for i, c in enumerate(chunks):
            assert c.source_type == "pdf"
            assert c.metadata["page"] == i + 1
            assert c.metadata["total_pages"] == 2


# ---------------------------------------------------------------------------
# knowledge_store.py
# ---------------------------------------------------------------------------

def _make_chunk(collection: str, title: str, content: str, source: str = "test.md", n: int = 0) -> KnowledgeChunk:
    cid = make_chunk_id(collection, source, title.lower().replace(" ", "-"), "section")
    return KnowledgeChunk(
        id=cid,
        collection=collection,
        source=source,
        source_type="markdown",
        title=title,
        content=content,
        summary=make_summary(title, content),
        level=1,
        position=n,
        content_hash=content_hash(content),
        keywords=extract_keywords(content),
    )


class TestKnowledgeStore:
    def test_save_and_load(self, tmp_path):
        store = KnowledgeStore(base_path=str(tmp_path))
        chunk = _make_chunk("c1", "Intro", "Hello world content.")
        index = store.save_index(
            collection="c1",
            chunks=[chunk],
            sources=["test.md"],
            source_types={"markdown": 1},
        )
        assert index.collection == "c1"
        assert len(index.chunks) == 1

        loaded = store.load_index("c1")
        assert loaded is not None
        assert loaded.collection == "c1"
        assert len(loaded.chunks) == 1

    def test_load_nonexistent_returns_none(self, tmp_path):
        store = KnowledgeStore(base_path=str(tmp_path))
        assert store.load_index("does-not-exist") is None

    def test_invalid_collection_name_raises(self, tmp_path):
        store = KnowledgeStore(base_path=str(tmp_path))
        with pytest.raises(ValueError):
            store._safe_collection_name("bad/name")
        with pytest.raises(ValueError):
            store._safe_collection_name("")
        with pytest.raises(ValueError):
            store._safe_collection_name("..")

    def test_valid_collection_names(self, tmp_path):
        store = KnowledgeStore(base_path=str(tmp_path))
        for name in ["my-docs", "project_v2", "notes.2024", "Research123"]:
            assert store._safe_collection_name(name) == name

    def test_list_collections(self, tmp_path):
        store = KnowledgeStore(base_path=str(tmp_path))
        store.save_index("col-a", [_make_chunk("col-a", "T", "C")], ["f.md"], {"markdown": 1})
        store.save_index("col-b", [_make_chunk("col-b", "T2", "C2")], ["g.md"], {"markdown": 1})
        cols = store.list_collections()
        names = [c["collection"] for c in cols]
        assert "col-a" in names
        assert "col-b" in names

    def test_delete_collection(self, tmp_path):
        store = KnowledgeStore(base_path=str(tmp_path))
        store.save_index("to-del", [_make_chunk("to-del", "T", "C")], ["f.md"], {"markdown": 1})
        assert store.load_index("to-del") is not None
        deleted = store.delete_collection("to-del")
        assert deleted is True
        assert store.load_index("to-del") is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        store = KnowledgeStore(base_path=str(tmp_path))
        assert store.delete_collection("ghost") is False

    def test_atomic_write(self, tmp_path):
        """Index file should not have .tmp suffix after save."""
        store = KnowledgeStore(base_path=str(tmp_path))
        store.save_index("atomic-test", [_make_chunk("atomic-test", "T", "C")], ["f.md"], {"markdown": 1})
        index_file = store._index_path("atomic-test")
        assert index_file.exists()
        tmp_file = index_file.with_suffix(".json.tmp")
        assert not tmp_file.exists()

    def test_add_source_creates_collection(self, tmp_path):
        store = KnowledgeStore(base_path=str(tmp_path))
        chunk = _make_chunk("new-col", "Page", "Content here.", source="http://x.com")
        chunk.source_type = "url"
        index = store.add_source("new-col", [chunk], "http://x.com", "url")
        assert len(index.chunks) == 1
        assert "http://x.com" in index.sources

    def test_add_source_replaces_existing_source(self, tmp_path):
        store = KnowledgeStore(base_path=str(tmp_path))
        chunk_v1 = _make_chunk("col", "Old", "Old content.", source="http://a.com")
        store.add_source("col", [chunk_v1], "http://a.com", "url")

        chunk_v2 = _make_chunk("col", "New", "New content.", source="http://a.com")
        index = store.add_source("col", [chunk_v2], "http://a.com", "url")

        # Should only have the new chunk, not both
        assert len(index.chunks) == 1
        assert index.chunks[0]["title"] == "New"

    def test_add_source_merges_with_existing(self, tmp_path):
        store = KnowledgeStore(base_path=str(tmp_path))
        c1 = _make_chunk("col", "Doc", "Doc content.", source="a.md")
        store.save_index("col", [c1], ["a.md"], {"markdown": 1})

        c2 = _make_chunk("col", "Web", "Web content.", source="http://b.com")
        c2.source_type = "url"
        index = store.add_source("col", [c2], "http://b.com", "url")

        assert len(index.chunks) == 2
        assert len(index.sources) == 2


class TestKnowledgeIndexSearch:
    def _index(self) -> KnowledgeIndex:
        chunks = [
            {
                "id": "col::a.md::intro#section",
                "collection": "col",
                "source": "a.md",
                "source_type": "markdown",
                "title": "Introduction to Python",
                "content": "Python is a high-level programming language.",
                "summary": "Intro to Python programming.",
                "level": 1,
                "position": 0,
                "content_hash": "abc",
                "keywords": ["python", "programming", "language"],
                "metadata": {},
            },
            {
                "id": "col::a.md::setup#section",
                "collection": "col",
                "source": "a.md",
                "source_type": "markdown",
                "title": "Setup and Installation",
                "content": "Install Python using pip or conda.",
                "summary": "How to install Python.",
                "level": 2,
                "position": 1,
                "content_hash": "def",
                "keywords": ["install", "pip", "conda"],
                "metadata": {},
            },
        ]
        return KnowledgeIndex(
            collection="col",
            indexed_at="2024-01-01",
            sources=["a.md"],
            source_types={"markdown": 1},
            chunks=chunks,
        )

    def test_exact_title_match_ranks_first(self):
        idx = self._index()
        results = idx.search("Introduction to Python")
        assert results[0]["title"] == "Introduction to Python"

    def test_source_type_filter(self):
        idx = self._index()
        results = idx.search("python", source_type="url")
        assert results == []

    def test_no_match_returns_empty(self):
        idx = self._index()
        results = idx.search("javascript")
        assert results == []

    def test_keyword_match_scores(self):
        idx = self._index()
        results = idx.search("pip install")
        assert any("Setup" in r["title"] for r in results)

    def test_max_results_respected(self):
        idx = self._index()
        results = idx.search("python", max_results=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class TestIndexDocsTool:
    def test_indexes_markdown_folder(self, tmp_path):
        (tmp_path / "guide.md").write_text("# Guide\n\nWelcome.\n\n## Setup\n\nRun commands.")
        (tmp_path / "api.md").write_text("# API\n\nEndpoints listed here.")
        result = index_docs(str(tmp_path), "test-docs", storage_path=str(tmp_path / "store"))
        assert result["success"] is True
        assert result["chunk_count"] >= 2
        assert result["file_count"] == 2

    def test_missing_folder_returns_error(self, tmp_path):
        result = index_docs(str(tmp_path / "nonexistent"), "col", storage_path=str(tmp_path))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_invalid_collection_name_returns_error(self, tmp_path):
        (tmp_path / "f.md").write_text("# T\n\nC")
        result = index_docs(str(tmp_path), "bad/name", storage_path=str(tmp_path))
        assert result["success"] is False

    def test_empty_folder_returns_error(self, tmp_path):
        (tmp_path / "empty.txt").write_text("   ")
        result = index_docs(str(tmp_path), "col", storage_path=str(tmp_path))
        assert result["success"] is False

    def test_skips_non_doc_files(self, tmp_path):
        (tmp_path / "main.py").write_text("def foo(): pass")
        (tmp_path / "notes.md").write_text("# Notes\n\nSome content here to index.")
        result = index_docs(str(tmp_path), "col", storage_path=str(tmp_path / "store"))
        assert result["success"] is True
        # Only the markdown file is indexed
        assert result["file_count"] == 1


class TestIndexUrlTool:
    def test_invalid_scheme_returns_error(self, tmp_path):
        result = index_url("ftp://example.com", "col", storage_path=str(tmp_path))
        assert result["success"] is False
        assert "http" in result["error"].lower()

    def test_successful_indexing(self, tmp_path):
        html = (
            "<html><head><title>Blog</title></head><body>"
            "<h1>Article Title</h1>"
            "<p>This is a detailed article with plenty of content to be indexed properly.</p>"
            "<h2>Section One</h2>"
            "<p>Section content with enough words to pass the minimum length check.</p>"
            "</body></html>"
        )
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        with patch("jcodemunch_mcp.ingestors.url_ingestor.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = index_url(
                "https://example.com/article",
                "web-col",
                storage_path=str(tmp_path),
            )
        assert result["success"] is True
        assert result["new_chunks"] >= 1

    def test_no_content_returns_error(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.raise_for_status = MagicMock()

        with patch("jcodemunch_mcp.ingestors.url_ingestor.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = index_url("https://example.com", "col", storage_path=str(tmp_path))
        assert result["success"] is False

    def test_invalid_collection_name(self, tmp_path):
        result = index_url("https://example.com", "bad name", storage_path=str(tmp_path))
        assert result["success"] is False


class TestIndexYoutubeTool:
    def test_invalid_url_returns_error(self, tmp_path):
        result = index_youtube("not-a-video-url-at-all", "col", storage_path=str(tmp_path))
        assert result["success"] is False

    def test_library_missing_returns_error(self, tmp_path):
        with patch("jcodemunch_mcp.ingestors.youtube_ingestor._YOUTUBE_AVAILABLE", False):
            result = index_youtube("dQw4w9WgXcQ", "col", storage_path=str(tmp_path))
        assert result["success"] is False

    def test_successful_indexing(self, tmp_path):
        fake_segments = [
            {"text": f"segment {i}", "start": float(i * 30), "duration": 30.0}
            for i in range(10)  # 300 seconds
        ]

        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = fake_segments

        mock_list = MagicMock()
        mock_list.find_transcript.return_value = mock_transcript

        mock_yta = MagicMock()
        mock_yta.list_transcripts.return_value = mock_list

        with patch("jcodemunch_mcp.ingestors.youtube_ingestor._YOUTUBE_AVAILABLE", True):
            with patch("jcodemunch_mcp.ingestors.youtube_ingestor.YouTubeTranscriptApi", mock_yta):
                result = index_youtube(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "yt-col",
                    storage_path=str(tmp_path),
                )

        assert result["success"] is True
        assert result["chunk_count"] >= 1
        assert result["video_id"] == "dQw4w9WgXcQ"


class TestSearchKnowledgeTool:
    def _seed(self, tmp_path: Path, collection: str) -> None:
        store = KnowledgeStore(base_path=str(tmp_path))
        chunks = [
            _make_chunk(collection, "Authentication Guide", "Learn how to authenticate users with JWT tokens."),
            _make_chunk(collection, "Database Setup", "Set up PostgreSQL with Docker."),
        ]
        store.save_index(collection, chunks, ["docs.md"], {"markdown": 1})

    def test_returns_results(self, tmp_path):
        self._seed(tmp_path, "kbase")
        result = search_knowledge("kbase", "authentication", storage_path=str(tmp_path))
        assert "results" in result
        assert result["result_count"] >= 1

    def test_missing_collection_returns_error(self, tmp_path):
        result = search_knowledge("nope", "query", storage_path=str(tmp_path))
        assert "error" in result

    def test_source_type_filter(self, tmp_path):
        self._seed(tmp_path, "kbase")
        result = search_knowledge("authentication", "kbase", source_type="url", storage_path=str(tmp_path))
        # The query/collection args are positional in search_knowledge(collection, query, ...)
        # Call with correct order:
        result = search_knowledge(
            collection="kbase",
            query="authentication",
            source_type="url",
            storage_path=str(tmp_path),
        )
        assert result["result_count"] == 0  # All chunks are markdown, not url

    def test_max_results(self, tmp_path):
        self._seed(tmp_path, "kbase")
        result = search_knowledge(
            collection="kbase",
            query="setup authentication",
            max_results=1,
            storage_path=str(tmp_path),
        )
        assert len(result["results"]) <= 1


class TestGetChunkTool:
    def _seed_and_get_id(self, tmp_path: Path, collection: str) -> str:
        store = KnowledgeStore(base_path=str(tmp_path))
        chunk = _make_chunk(collection, "My Section", "Full text content of this section.")
        store.save_index(collection, [chunk], ["doc.md"], {"markdown": 1})
        return chunk.id

    def test_returns_chunk_content(self, tmp_path):
        chunk_id = self._seed_and_get_id(tmp_path, "col")
        result = get_chunk("col", chunk_id, storage_path=str(tmp_path))
        assert result["chunk_id"] == chunk_id
        assert "Full text content" in result["content"]
        assert result["title"] == "My Section"

    def test_missing_chunk_returns_error(self, tmp_path):
        self._seed_and_get_id(tmp_path, "col")
        result = get_chunk("col", "nonexistent::id#type", storage_path=str(tmp_path))
        assert "error" in result

    def test_missing_collection_returns_error(self, tmp_path):
        result = get_chunk("missing-col", "any-id", storage_path=str(tmp_path))
        assert "error" in result


class TestListCollectionsTool:
    def test_empty_store_returns_empty_list(self, tmp_path):
        result = list_collections(storage_path=str(tmp_path))
        assert result["collection_count"] == 0
        assert result["collections"] == []

    def test_lists_collections(self, tmp_path):
        store = KnowledgeStore(base_path=str(tmp_path))
        store.save_index("alpha", [_make_chunk("alpha", "T", "C")], ["f.md"], {"markdown": 1})
        store.save_index("beta", [_make_chunk("beta", "T2", "C2")], ["g.md"], {"markdown": 1})
        result = list_collections(storage_path=str(tmp_path))
        assert result["collection_count"] == 2
        names = [c["collection"] for c in result["collections"]]
        assert "alpha" in names
        assert "beta" in names


# ---------------------------------------------------------------------------
# Server integration — new tools present in list_tools()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_tools_registered():
    """Verify all 6 knowledge tools are registered in the MCP server."""
    from jcodemunch_mcp.server import list_tools
    tools = await list_tools()
    names = {t.name for t in tools}
    for expected in ("index_docs", "index_url", "index_youtube",
                     "search_knowledge", "get_chunk", "list_collections"):
        assert expected in names, f"Tool {expected!r} not registered"


@pytest.mark.asyncio
async def test_index_docs_tool_schema():
    from jcodemunch_mcp.server import list_tools
    tools = await list_tools()
    tool = next(t for t in tools if t.name == "index_docs")
    props = tool.inputSchema["properties"]
    assert "path" in props
    assert "collection" in props
    assert "include_pdfs" in props
    assert tool.inputSchema["required"] == ["path", "collection"]


@pytest.mark.asyncio
async def test_search_knowledge_tool_schema():
    from jcodemunch_mcp.server import list_tools
    tools = await list_tools()
    tool = next(t for t in tools if t.name == "search_knowledge")
    props = tool.inputSchema["properties"]
    assert "collection" in props
    assert "query" in props
    assert "source_type" in props
    assert "enum" in props["source_type"]
    assert set(props["source_type"]["enum"]) == {"markdown", "url", "pdf", "youtube"}


@pytest.mark.asyncio
async def test_call_tool_list_collections(tmp_path):
    """Verify list_collections is reachable via call_tool."""
    from jcodemunch_mcp.server import call_tool
    import os
    with patch.dict(os.environ, {"CODE_INDEX_PATH": str(tmp_path)}):
        results = await call_tool("list_collections", {})
    assert len(results) == 1
    data = json.loads(results[0].text)
    assert "collections" in data
