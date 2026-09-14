"""Persistent, crash-safe record of a generation run.

The manifest is the tool's memory.  It records the script it was built from,
the chunking parameters, the voice settings and the state of every segment, so
a re-run can resume from the first incomplete segment instead of paying for
(and re-recording) work that already succeeded.

It is rewritten atomically after every segment: a temporary file in the same
directory followed by :func:`os.replace`, which is atomic on Windows and POSIX.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterable, List, Optional

from . import ToolError

__all__ = [
    "MANIFEST_VERSION",
    "Manifest",
    "STATUS_FAILED",
    "STATUS_GENERATED",
    "STATUS_PENDING",
    "STATUS_PLACED",
    "SegmentRecord",
    "sha256_text",
]

MANIFEST_VERSION = 1

STATUS_PENDING = "pending"
STATUS_GENERATED = "generated"
STATUS_PLACED = "placed"
STATUS_FAILED = "failed"

_ALL_STATUSES = (STATUS_PENDING, STATUS_GENERATED, STATUS_PLACED, STATUS_FAILED)


def sha256_text(text: str) -> str:
    """Return the SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class SegmentRecord:
    """State of one speech segment."""

    index: int
    text: str
    char_count: int
    paragraph_index: int = 0
    status: str = STATUS_PENDING
    attempts: int = 0
    error: Optional[str] = None

    # Where Resolve put the generated audio, and our own cached copy of it.
    resolve_file_path: Optional[str] = None
    cached_file: Optional[str] = None

    # Identity of the generated MediaPoolItem, so a later run can find it again.
    media_pool_item_unique_id: Optional[str] = None
    media_id: Optional[str] = None
    clip_name: Optional[str] = None

    # Measured audio properties.  Generated clips are 48 kHz mono 32-bit float
    # (verified on Resolve Studio 21.0.1), so ``sample_format`` is "float".
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    sample_width: Optional[int] = None
    sample_format: Optional[str] = None
    source_frames: Optional[int] = None
    """Whole timeline frames the media occupies, ``floor(duration x fps)``.

    Only used for placement when ``duration_seconds`` is unknown, i.e. when the
    WAV could not be parsed and the count came from Resolve's clip properties.
    """
    leading_silence_seconds: Optional[float] = None
    trailing_silence_seconds: Optional[float] = None
    generation_seconds: Optional[float] = None

    # Timeline placement, filled in once the clip is on the timeline.
    record_frame: Optional[int] = None
    track_index: Optional[int] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    """Exclusive: the placed duration is ``end_frame - start_frame``, matching
    ``AppendToTimeline``'s ``clipInfo`` (verified on Resolve Studio 21.0.1)."""
    timeline_timecode: Optional[str] = None

    @property
    def number(self) -> int:
        """One-based segment number, as shown to the user."""
        return self.index + 1

    @property
    def is_complete(self) -> bool:
        """True when the audio exists and does not need regenerating."""
        return self.status in (STATUS_GENERATED, STATUS_PLACED)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentRecord":
        known = {f.name for f in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in known}
        record = cls(**filtered)
        if record.status not in _ALL_STATUSES:
            record.status = STATUS_PENDING
        return record


@dataclass
class Manifest:
    """A manifest file plus the segment records it holds."""

    path: str
    script_path: str
    script_sha256: str
    max_chars: int
    merge_paragraphs: bool
    voice_settings: Dict[str, Any] = field(default_factory=dict)
    segments: List[SegmentRecord] = field(default_factory=list)
    version: int = MANIFEST_VERSION
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    tool_version: str = ""

    # ---------------------------------------------------------------- I/O

    @classmethod
    def load(cls, path: str) -> "Manifest":
        """Read a manifest from disk.

        Raises:
            ToolError: The file is not readable or not valid JSON.
        """
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ToolError(
                "Could not read the manifest at %s" % path,
                "  %s: %s\n"
                "  Delete the file (or pass --force-rechunk) to start a fresh run."
                % (type(exc).__name__, exc),
            )
        if not isinstance(data, dict):
            raise ToolError("Manifest %s is not a JSON object." % path)

        manifest = cls(
            path=path,
            script_path=str(data.get("script_path", "")),
            script_sha256=str(data.get("script_sha256", "")),
            max_chars=int(data.get("max_chars", 0)),
            merge_paragraphs=bool(data.get("merge_paragraphs", False)),
            voice_settings=dict(data.get("voice_settings") or {}),
            version=int(data.get("version", 0)),
            created=str(data.get("created", _now())),
            updated=str(data.get("updated", _now())),
            tool_version=str(data.get("tool_version", "")),
        )
        manifest.segments = [
            SegmentRecord.from_dict(item)
            for item in data.get("segments", [])
            if isinstance(item, dict)
        ]
        manifest.segments.sort(key=lambda record: record.index)
        return manifest

    def save(self) -> None:
        """Write the manifest atomically (temp file then :func:`os.replace`)."""
        self.updated = _now()
        payload = {
            "version": self.version,
            "tool_version": self.tool_version,
            "script_path": self.script_path,
            "script_sha256": self.script_sha256,
            "max_chars": self.max_chars,
            "merge_paragraphs": self.merge_paragraphs,
            "voice_settings": self.voice_settings,
            "created": self.created,
            "updated": self.updated,
            "segments": [segment.to_dict() for segment in self.segments],
        }
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        temporary = self.path + ".tmp"
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def archive(self) -> str:
        """Rename the existing manifest out of the way and return the new path."""
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        root, extension = os.path.splitext(self.path)
        archived = "%s.%s%s" % (root, stamp, extension or ".json")
        os.replace(self.path, archived)
        return archived

    # ----------------------------------------------------------- queries

    def matches(
        self, script_sha256: str, max_chars: int, merge_paragraphs: bool
    ) -> bool:
        """True when this manifest was built from the same text and settings."""
        return (
            self.version == MANIFEST_VERSION
            and self.script_sha256 == script_sha256
            and self.max_chars == max_chars
            and self.merge_paragraphs == bool(merge_paragraphs)
        )

    def mismatch_reason(
        self, script_sha256: str, max_chars: int, merge_paragraphs: bool
    ) -> str:
        """Explain, in one line, why :meth:`matches` returned false."""
        if self.version != MANIFEST_VERSION:
            return "it was written by manifest version %s (this tool writes %d)" % (
                self.version,
                MANIFEST_VERSION,
            )
        if self.script_sha256 != script_sha256:
            return "the script text has changed since it was written"
        if self.max_chars != max_chars:
            return "it was chunked at --max-chars %d, not %d" % (
                self.max_chars,
                max_chars,
            )
        if self.merge_paragraphs != bool(merge_paragraphs):
            return "it was chunked with --merge-paragraphs %s" % (
                "on" if self.merge_paragraphs else "off",
            )
        return "the chunking parameters differ"

    def get(self, index: int) -> Optional[SegmentRecord]:
        """Return the segment with the given zero-based index, if present."""
        for segment in self.segments:
            if segment.index == index:
                return segment
        return None

    def count(self, *statuses: str) -> int:
        """Count segments whose status is one of ``statuses``."""
        return sum(1 for segment in self.segments if segment.status in statuses)

    def with_status(self, *statuses: str) -> List[SegmentRecord]:
        """Return every segment whose status is one of ``statuses``, in order."""
        return [segment for segment in self.segments if segment.status in statuses]

    def failed_numbers(self) -> List[int]:
        """One-based numbers of the segments currently marked failed."""
        return [s.number for s in self.segments if s.status == STATUS_FAILED]


def build_manifest(
    path: str,
    script_path: str,
    script_text: str,
    chunks: Iterable[Any],
    max_chars: int,
    merge_paragraphs: bool,
    voice_settings: Dict[str, Any],
    tool_version: str = "",
) -> Manifest:
    """Create a fresh manifest from freshly computed chunks."""
    manifest = Manifest(
        path=path,
        script_path=os.path.abspath(script_path),
        script_sha256=sha256_text(script_text),
        max_chars=max_chars,
        merge_paragraphs=bool(merge_paragraphs),
        voice_settings=dict(voice_settings),
        tool_version=tool_version,
    )
    manifest.segments = [
        SegmentRecord(
            index=chunk.index,
            text=chunk.text,
            char_count=chunk.char_count,
            paragraph_index=chunk.paragraph_index,
        )
        for chunk in chunks
    ]
    return manifest
