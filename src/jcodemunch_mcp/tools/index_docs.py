"""Index a local documentation folder into a knowledge collection."""

import hashlib
import logging
from pathlib import Path
from typing import Optional

from ..ingestors.doc_ingestor import ingest_doc_file, DOC_EXTENSIONS
from ..ingestors.pdf_ingestor import ingest_pdf
from ..security import validate_path, is_secret_file
from ..storage.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

# Hard limits for doc indexing
_MAX_DOC_FILES = 200
_MAX_DOC_FILE_SIZE = 2 * 1024 * 1024  # 2 MB

# Directory / file patterns to skip when walking a docs folder
_SKIP_PATTERNS = [
    "node_modules/", ".git/", "venv/", ".venv/", "__pycache__/",
    "dist/", "build/", ".tox/",
]


def _should_skip(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/")
    return any(pat in norm for pat in _SKIP_PATTERNS)


def index_docs(
    path: str,
    collection: str,
    include_pdfs: bool = False,
    storage_path: Optional[str] = None,
) -> dict:
    """Index a local folder of documentation files into a knowledge collection.

    Supported document types: Markdown (``.md``, ``.mdx``, ``.markdown``),
    reStructuredText (``.rst``), plain text (``.txt``), AsciiDoc
    (``.adoc``, ``.asciidoc``), and optionally PDF (``.pdf`` — requires
    the ``pypdf`` package).

    Each document is split into heading-based chunks; plain-text files fall
    back to paragraph-based splitting.  The resulting chunks are stored in a
    named *collection* under ``~/.code-index/knowledge/``.

    Args:
        path: Absolute or ``~``-prefixed path to the documentation folder.
        collection: Name for the knowledge collection (letters, digits,
            ``-``, ``_``, ``.`` only).
        include_pdfs: When ``True``, also index ``.pdf`` files.
            Requires ``pypdf``; install with
            ``pip install "jcodemunch-mcp[pdf]"``.
        storage_path: Override the default ``~/.code-index/`` storage root.

    Returns:
        Dict with ``success``, ``collection``, ``chunk_count``,
        ``source_types``, and optional ``warnings``.
    """
    folder_path = Path(path).expanduser().resolve()

    if not folder_path.exists():
        return {"success": False, "error": f"Folder not found: {path}"}
    if not folder_path.is_dir():
        return {"success": False, "error": f"Not a directory: {path}"}

    store = KnowledgeStore(base_path=storage_path)
    try:
        store._safe_collection_name(collection)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    allowed_exts = set(DOC_EXTENSIONS)
    if include_pdfs:
        allowed_exts.add(".pdf")

    all_chunks = []
    sources: list[str] = []
    source_types: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    warnings: list[str] = []
    file_count = 0

    for file_path in sorted(folder_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.is_symlink():
            continue

        if not validate_path(folder_path, file_path):
            continue

        try:
            rel_path = file_path.relative_to(folder_path).as_posix()
        except ValueError:
            continue

        if _should_skip(rel_path):
            continue

        if is_secret_file(rel_path):
            continue

        ext = file_path.suffix.lower()
        if ext not in allowed_exts:
            continue

        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        if size > _MAX_DOC_FILE_SIZE:
            warnings.append(f"Skipped large file (>{_MAX_DOC_FILE_SIZE // 1024}KB): {rel_path}")
            continue

        if file_count >= _MAX_DOC_FILES:
            warnings.append(f"Reached file limit ({_MAX_DOC_FILES}); some files were skipped.")
            break

        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
            source_hashes[rel_path] = hashlib.sha256(raw.encode()).hexdigest()
        except Exception as exc:
            warnings.append(f"Failed to read {rel_path}: {exc}")
            continue

        if ext == ".pdf":
            chunks = ingest_pdf(file_path, rel_path, collection)
            stype = "pdf"
        else:
            chunks = ingest_doc_file(file_path, rel_path, collection)
            stype = "markdown"

        if chunks:
            all_chunks.extend(chunks)
            sources.append(rel_path)
            source_types[stype] = source_types.get(stype, 0) + 1
            file_count += 1
        else:
            logger.debug("No chunks produced for: %s", rel_path)

    if not all_chunks:
        return {
            "success": False,
            "error": "No content could be extracted from the documentation files.",
            "hint": f"Supported extensions: {sorted(allowed_exts)}",
        }

    store.save_index(
        collection=collection,
        chunks=all_chunks,
        sources=sources,
        source_types=source_types,
        source_hashes=source_hashes,
    )

    result: dict = {
        "success": True,
        "collection": collection,
        "folder_path": str(folder_path),
        "file_count": len(sources),
        "chunk_count": len(all_chunks),
        "source_types": source_types,
        "sources": sources[:20],
    }
    if warnings:
        result["warnings"] = warnings
    return result
