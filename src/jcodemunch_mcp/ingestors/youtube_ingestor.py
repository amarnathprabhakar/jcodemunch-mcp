"""YouTube transcript ingestor — fetches transcripts via the optional ``youtube-transcript-api``."""

import logging
import re
from typing import Optional

from .chunks import KnowledgeChunk, make_chunk_id, content_hash, extract_keywords, make_summary

logger = logging.getLogger(__name__)

# Group transcript segments into ~2-minute chunks by default
DEFAULT_CHUNK_SECONDS = 120

try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
    _YOUTUBE_AVAILABLE = True
except ImportError:
    YouTubeTranscriptApi = None  # type: ignore[assignment,misc]
    NoTranscriptFound = None  # type: ignore[assignment,misc]
    _YOUTUBE_AVAILABLE = False


def extract_video_id(url: str) -> Optional[str]:
    """Extract the 11-character YouTube video ID from a URL or bare ID string.

    Supported formats:

    * ``https://www.youtube.com/watch?v=VIDEO_ID``
    * ``https://youtu.be/VIDEO_ID``
    * ``https://www.youtube.com/embed/VIDEO_ID``
    * Bare 11-character video ID

    Args:
        url: YouTube URL or video ID.

    Returns:
        11-character video ID string, or ``None`` if not found.
    """
    patterns = [
        r"(?:v=|v/|youtu\.be/|embed/)([A-Za-z0-9_\-]{11})",
        r"^([A-Za-z0-9_\-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def ingest_youtube(
    url: str,
    collection: str,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
    languages: Optional[list[str]] = None,
) -> list[KnowledgeChunk]:
    """Fetch a YouTube video transcript and split it into time-based chunks.

    Requires the optional ``youtube-transcript-api`` package::

        pip install "jcodemunch-mcp[youtube]"

    Each chunk covers approximately ``chunk_seconds`` of video time.  The
    chunk title contains the time range (``MM:SS-MM:SS``) so agents can
    cite the exact moment in the video.

    Args:
        url: YouTube video URL or bare 11-character video ID.
        collection: Knowledge collection name.
        chunk_seconds: Duration (seconds) covered by each chunk.  Default 120.
        languages: Language preference list, e.g. ``['en', 'en-US']``.
            Falls back to any available transcript when not found.

    Returns:
        List of KnowledgeChunk objects (empty when the library is not
        installed or no transcript is available for the video).
    """
    if not _YOUTUBE_AVAILABLE:
        logger.warning(
            "youtube-transcript-api is not installed; YouTube indexing is unavailable. "
            "Install it with: pip install \"jcodemunch-mcp[youtube]\""
        )
        return []

    video_id = extract_video_id(url)
    if not video_id:
        logger.error("Could not extract a video ID from: %s", url)
        return []

    try:
        preferred = languages or ["en", "en-US", "en-GB"]

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        transcript = None
        try:
            transcript = transcript_list.find_transcript(preferred)
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_generated_transcript(preferred)
            except NoTranscriptFound:
                for t in transcript_list:
                    transcript = t
                    break

        if transcript is None:
            logger.error("No transcript available for video: %s", video_id)
            return []

        segments = transcript.fetch()

    except Exception as exc:
        logger.error("Failed to fetch transcript for %s: %s", video_id, exc)
        return []

    # ------------------------------------------------------------------
    # Group segments into time-based chunks
    # ------------------------------------------------------------------
    chunks: list[KnowledgeChunk] = []
    current_texts: list[str] = []
    current_start = 0.0
    current_end = 0.0
    chunk_num = 0

    def _seg_attr(seg: object, name: str, default: object) -> object:
        """Access segment attribute whether it is a dict or object."""
        if isinstance(seg, dict):
            return seg.get(name, default)
        return getattr(seg, name, default)

    def _flush() -> None:
        nonlocal chunk_num, current_texts, current_start, current_end
        text = " ".join(current_texts).strip()
        if not text:
            return

        s_min, s_sec = int(current_start // 60), int(current_start % 60)
        e_min, e_sec = int(current_end // 60), int(current_end % 60)
        time_range = f"{s_min:02d}:{s_sec:02d}-{e_min:02d}:{e_sec:02d}"
        title = f"Transcript {time_range}"

        chunk_id = make_chunk_id(collection, video_id, f"segment-{chunk_num}", "transcript")

        chunks.append(
            KnowledgeChunk(
                id=chunk_id,
                collection=collection,
                source=video_id,
                source_type="youtube",
                title=title,
                content=text,
                summary=make_summary(title, text),
                level=1,
                position=chunk_num,
                content_hash=content_hash(text),
                keywords=extract_keywords(text),
                metadata={
                    "video_id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "start_seconds": current_start,
                    "end_seconds": current_end,
                    "time_range": time_range,
                },
            )
        )

        chunk_num += 1
        current_texts = []
        current_start = current_end

    for seg in segments:
        start = float(_seg_attr(seg, "start", 0))
        duration = float(_seg_attr(seg, "duration", 0))
        text = str(_seg_attr(seg, "text", "")).strip()

        if not current_texts:
            current_start = start

        if text:
            current_texts.append(text)
        current_end = start + duration

        if current_end - current_start >= chunk_seconds:
            _flush()

    _flush()  # Flush any remaining segments

    return chunks
