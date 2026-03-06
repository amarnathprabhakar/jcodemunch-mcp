"""Ingestors package — parse documents, URLs, PDFs, and transcripts into knowledge chunks."""

from .chunks import KnowledgeChunk, make_chunk_id, content_hash, extract_keywords, make_summary
from .doc_ingestor import ingest_doc_file, is_doc_file, DOC_EXTENSIONS
from .url_ingestor import ingest_url
from .pdf_ingestor import ingest_pdf
from .youtube_ingestor import ingest_youtube

__all__ = [
    "KnowledgeChunk",
    "make_chunk_id",
    "content_hash",
    "extract_keywords",
    "make_summary",
    "ingest_doc_file",
    "is_doc_file",
    "DOC_EXTENSIONS",
    "ingest_url",
    "ingest_pdf",
    "ingest_youtube",
]
