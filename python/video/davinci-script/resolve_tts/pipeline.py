"""Orchestration: load -> chunk -> generate (resumable) -> place on the timeline.

Every interaction with DaVinci Resolve funnels through this module and is
logged.  The behaviour encoded here was verified live against DaVinci Resolve
Studio 21.0.1.11 on Windows 11; the notable facts are:

* ``Project.GenerateSpeech`` is synchronous and returns a ``MediaPoolItem``.
  The first call of a session takes ~12 s (model warm-up), later ones 1-4 s.
* ``"Filename"`` in the settings dict is a *prefix*: Resolve writes
  ``<Filename>-<VoiceModel>-G<GenerationID>-<NNN>.wav`` into the project's
  ``Audio Files/SpeechGenerator`` folder, and the real location is read back
  from the clip's ``"File Path"`` property.
* ``MediaPool.AppendToTimeline``'s ``endFrame`` is **exclusive**: the placed
  duration is ``endFrame - startFrame``.
* The clip's ``"Frames"`` property is an empty string for generated audio, so
  the frame count is computed from the WAV instead.

When something behaves differently, ``generate_voice.log`` is where the
evidence will be.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from . import (
    PartialFailure,
    PlacementAborted,
    ResolveError,
    ToolError,
    UsageError,
    __version__,
)
from . import audio_utils, chunker, timecode as tc
from .audio_utils import AudioInfo, SilenceInfo
from .chunker import Chunk
from .manifest import (
    STATUS_FAILED,
    STATUS_GENERATED,
    STATUS_PENDING,
    STATUS_PLACED,
    Manifest,
    SegmentRecord,
    build_manifest,
    sha256_text,
)
from .resolve_connect import ResolveContext, connect

__all__ = [
    "Options",
    "Pipeline",
    "RunResult",
    "build_clip_info",
    "configure_logging",
    "frames_for_duration",
    "frames_from_clip_properties",
]

LOGGER = logging.getLogger("resolve_tts.pipeline")

#: Absolute cap Resolve places on ``TextInput``; asserted before every call.
RESOLVE_TEXT_LIMIT = chunker.RESOLVE_TEXT_LIMIT

#: Clip property keys we try, in order, when hunting for the generated WAV.
#: Verified against Resolve Studio 21.0.1: a MediaPoolItem exposes 263 property
#: keys, of which "File Path" holds the absolute path and "File Name" holds only
#: the bare filename -- hence the ``os.path.isabs`` guard on every candidate.
_FILE_PATH_KEYS = ("File Path",)

#: A ``None`` returned this quickly means Resolve rejected the settings without
#: ever starting the engine (verified: a bad voice name comes back in 0.02 s and
#: a 359-character TextInput in 0.01 s).
_FAST_FAILURE_SECONDS = 0.5

#: Console handler level of the most recent :func:`configure_logging` call.
#: Progress lines are written inline (``... done (2.4s)``) unless the console is
#: at DEBUG, where interleaved detail would break a partial line.
_CONSOLE_LEVEL = logging.INFO

_SETTINGS_FAILURE_HINT = (
    "  GenerateSpeech returned nothing almost instantly, which means the settings\n"
    "  were rejected before the engine ran. Verified causes, most likely first:\n"
    "    1. The voice name does not exist. Open the Speech Generator in Resolve and\n"
    "       copy the name exactly, or drop --voice to use the current voice.\n"
    "    2. The text was rejected (empty, or longer than 350 characters).\n"
    "    3. A custom voice file path that Resolve cannot read."
)

_ENGINE_FAILURE_HINT = (
    "  GenerateSpeech ran for a while and still returned nothing, which points at\n"
    "  the engine rather than the settings. The likely causes, most common first:\n"
    "    1. The 'AI Speech Generator' package is not installed. Install it from\n"
    "       DaVinci Resolve Studio -> Extras Download Manager, then restart Resolve.\n"
    "    2. This is the free edition of Resolve, not Studio.\n"
    "    3. The machine does not meet the feature's system requirements -- run the\n"
    "       Speech Generator once from the Resolve UI and read any error dialog."
)

_RESOLVE_PLACEMENT_FAILED = (
    "Resolve created the clip but did not place it (observed on 21.0.1 from the "
    "Edit page). Use the default `--placement append`, or `--place-only` to place "
    "the already generated segments."
)


def frames_for_duration(duration_seconds: float, frame_rate: tc.FrameRate) -> int:
    """Whole frames a clip of ``duration_seconds`` occupies on the timeline.

    Resolve **floors** partial frames: a 4.40 s clip on a 24 fps timeline is 105
    frames (4.40 x 24 = 105.6), and its ``"Duration"`` property reads
    ``00:00:04:09``.  Asking ``AppendToTimeline`` for 106 frames was accepted
    live and produced a clip reaching past the end of the media, so flooring is
    not cosmetic.
    """
    if duration_seconds is None or duration_seconds <= 0:
        return 0
    return int(math.floor(duration_seconds * frame_rate.fps))


def frames_from_clip_properties(
    properties: Any, frame_rate: tc.FrameRate
) -> Optional[int]:
    """Frame count from a MediaPoolItem's properties, when the WAV is unreadable.

    Verified on Resolve Studio 21.0.1 for generated speech clips: ``"Frames"`` is
    an **empty string**, while ``"Duration"`` is a timecode that already floors
    partial frames.  So ``"Duration"`` is tried first and ``"Frames"`` only if it
    holds something; neither is ever passed to a bare ``int()``.
    """
    if not isinstance(properties, dict):
        return None

    duration = properties.get("Duration")
    if isinstance(duration, str) and tc.is_timecode(duration):
        try:
            return tc.timecode_to_frames(duration, frame_rate)
        except ValueError as exc:
            LOGGER.debug("Duration %r is not valid at %s: %s", duration, frame_rate, exc)

    frames = _as_int(properties.get("Frames"))
    if frames and frames > 0:
        return frames
    return None


def build_clip_info(
    media_pool_item: Any,
    source_in: int,
    source_out: int,
    track_index: int,
    record_frame: int,
) -> Dict[str, Any]:
    """Build one ``AppendToTimeline`` ``clipInfo`` dictionary.

    ``source_out`` is **exclusive**, matching Resolve: the placed duration is
    ``endFrame - startFrame``.  Verified live -- ``startFrame 0, endFrame 105``
    gives a 105-frame clip, and ``startFrame 3, endFrame 103`` gives 100 frames
    with ``GetLeftOffset() == 3`` and ``GetRightOffset() == 2``.
    """
    return {
        "mediaPoolItem": media_pool_item,
        "startFrame": int(source_in),
        "endFrame": int(source_out),
        "mediaType": 2,
        "trackIndex": int(track_index),
        "recordFrame": int(record_frame),
    }


# --------------------------------------------------------------------------
# Options and results
# --------------------------------------------------------------------------


@dataclass
class Options:
    """Every knob the CLI exposes, already validated."""

    script_path: str = ""
    output_dir: str = ""
    max_chars: int = chunker.DEFAULT_MAX_CHARS
    merge_paragraphs: bool = False

    voice: Optional[str] = None
    custom_voice_file: Optional[str] = None
    speed: Optional[int] = None
    variation: Optional[int] = None
    pitch: Optional[int] = None
    generation_id: Optional[int] = None

    track_name: str = "AI Voice"
    track_index: Optional[int] = None
    bin_name: str = "AI Voice"
    start: str = "playhead"
    gap_frames: int = 0

    #: Off by default.  Measured on real generated clips at -50 dBFS: 24-32 ms
    #: of leading and 48-76 ms of trailing padding, which is less than two
    #: frames at 24 fps and shorter than a natural sentence pause.
    trim_silence: bool = False
    silence_threshold_db: float = -50.0
    keep_pad_ms: int = 40

    placement: str = "append"
    dry_run: bool = False
    skip_placement: bool = False
    place_only: bool = False
    retry_failed: bool = False
    only: Optional[Set[int]] = None
    max_attempts: int = 2
    force_rechunk: bool = False
    stop_on_error: bool = False

    @property
    def script_stem(self) -> str:
        """The script filename without directory or extension."""
        return os.path.splitext(os.path.basename(self.script_path))[0] or "script"


@dataclass
class RunResult:
    """Outcome of a whole run."""

    exit_code: int = 0
    total: int = 0
    generated: int = 0
    cached: int = 0
    failed: int = 0
    placed: int = 0
    failed_numbers: List[int] = field(default_factory=list)


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


def configure_logging(log_path: Optional[str], console_level: int) -> None:
    """Send clean messages to the console and a timestamped DEBUG log to disk.

    Args:
        log_path: File to append the DEBUG log to, or ``None`` for console only.
        console_level: Logging level for the console handler.
    """
    global _CONSOLE_LEVEL
    _CONSOLE_LEVEL = console_level
    root = logging.getLogger("resolve_tts")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _call(obj: Any, name: str, *args: Any) -> Any:
    """Call ``obj.name(*args)``, returning ``None`` instead of raising."""
    if obj is None:
        return None
    method = getattr(obj, name, None)
    if method is None:
        LOGGER.debug("object %r has no attribute %s", obj, name)
        return None
    try:
        return method(*args)
    except Exception as exc:  # noqa: BLE001 - the native bridge raises anything
        LOGGER.debug("%s%r raised %s: %s", name, args, type(exc).__name__, exc)
        return None


def _as_int(value: Any) -> Optional[int]:
    """Best-effort conversion of a Resolve return value to ``int``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(round(float(text)))
        except ValueError:
            return None
    return None


def _is_usable_path(value: Any) -> bool:
    """True when a clip property really holds an absolute path to a file."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(text) and os.path.isabs(text) and os.path.isfile(text)


def _truncate(text: str, limit: int = 60) -> str:
    """Shorten ``text`` for a single-line INFO message."""
    single_line = " ".join(text.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


class Pipeline:
    """Runs one invocation of the tool end to end."""

    def __init__(self, options: Options) -> None:
        self.options = options
        self.result = RunResult()
        self.script_text = ""
        self.chunks: List[Chunk] = []
        self.manifest: Optional[Manifest] = None
        self.context: Optional[ResolveContext] = None
        self._frame_rate: Optional[tc.FrameRate] = None
        self._target_track: Optional[int] = None
        self._next_record_frame: int = 0
        self._items: Dict[int, Any] = {}
        self._previous_folder: Optional[Any] = None
        self._open_line: Optional[str] = None

    # -------------------------------------------------------- progress lines

    @property
    def _inline_progress(self) -> bool:
        """True when a progress line may be completed in place on the console."""
        return _CONSOLE_LEVEL > logging.DEBUG

    def _begin_line(self, text: str) -> None:
        """Start a progress line such as ``Generating segment 3/29...``."""
        self._open_line = text
        if self._inline_progress:
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            LOGGER.info("%s", text)

    def _end_line(self, suffix: str) -> None:
        """Complete the open progress line with ``suffix`` (e.g. ``done (2.4s)``)."""
        full = "%s %s" % (self._open_line or "", suffix.strip())
        if self._inline_progress:
            sys.stdout.write(" %s\n" % suffix.strip())
            sys.stdout.flush()
            LOGGER.debug("%s", full.strip())
        else:
            LOGGER.info("%s", full.strip())
        self._open_line = None

    # ------------------------------------------------------------- paths

    @property
    def segments_dir(self) -> str:
        """Directory holding our cached copies of the generated WAVs."""
        return os.path.join(self.options.output_dir, "segments")

    @property
    def manifest_path(self) -> str:
        """Path to ``manifest.json`` inside the output directory."""
        return os.path.join(self.options.output_dir, "manifest.json")

    # --------------------------------------------------------------- run

    def run(self) -> RunResult:
        """Execute the pipeline and return its :class:`RunResult`.

        Raises:
            ToolError: Any expected failure; the caller renders it and uses
                :attr:`ToolError.exit_code`.
        """
        self._load_script()
        self._chunk()

        if self.options.dry_run:
            self._print_dry_run()
            return self.result

        self._prepare_manifest()
        self.context = connect(
            require_studio=True, require_project=True, require_timeline=True
        )
        self._log_environment()

        try:
            self._enter_bin()
            if not self.options.place_only:
                self._generate_all()
            else:
                LOGGER.info("Skipping generation (--place-only).")
            self._resolve_media_items()
            self._place_all()
        finally:
            self._leave_bin()
            self._save_manifest()

        self._save_project()
        self._finalise()
        return self.result

    # ------------------------------------------------------------ script

    def _load_script(self) -> None:
        path = self.options.script_path
        LOGGER.info("Loading script...")
        if not path:
            raise UsageError("No script file was given.")
        if not os.path.isfile(path):
            raise UsageError(
                "Script file not found: %s" % path,
                "  Pass the path to a UTF-8 .txt narration script.",
            )
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read()
        except UnicodeDecodeError as exc:
            raise UsageError(
                "Script file %s is not valid UTF-8." % path,
                "  %s\n  Re-save the file as UTF-8 and try again." % exc,
            )
        except OSError as exc:
            raise UsageError("Could not read %s: %s" % (path, exc))

        self.script_text = chunker.normalize_text(raw)
        if not self.script_text.strip():
            raise UsageError(
                "Script file %s is empty." % path,
                "  Add some narration text and try again.",
            )
        LOGGER.info("%s characters found.", format(len(self.script_text), ","))
        LOGGER.debug("Script sha256: %s", sha256_text(self.script_text))

    def _chunk(self) -> None:
        self.chunks = chunker.split_text(
            self.script_text,
            max_chars=self.options.max_chars,
            merge_paragraphs=self.options.merge_paragraphs,
            normalize=False,
        )
        if not self.chunks:
            raise UsageError(
                "The script produced no speech segments.",
                "  It appears to contain no readable text.",
            )
        oversized = [c for c in self.chunks if c.char_count > self.options.max_chars]
        if oversized:  # pragma: no cover - guarded structurally by the chunker
            raise ToolError(
                "Internal error: %d chunk(s) exceeded --max-chars %d."
                % (len(oversized), self.options.max_chars)
            )
        self.result.total = len(self.chunks)
        LOGGER.info("Split into %d speech segments.", len(self.chunks))
        LOGGER.debug(
            "Chunk sizes: min=%d max=%d mean=%.1f",
            min(c.char_count for c in self.chunks),
            max(c.char_count for c in self.chunks),
            sum(c.char_count for c in self.chunks) / float(len(self.chunks)),
        )

    def _print_dry_run(self) -> None:
        print("")
        for chunk in self.chunks:
            print(
                "[%3d/%3d] p%-3d %3d chars | %s"
                % (
                    chunk.index + 1,
                    len(self.chunks),
                    chunk.paragraph_index + 1,
                    chunk.char_count,
                    chunk.text,
                )
            )
        counts = [c.char_count for c in self.chunks]
        print("")
        print(
            "%d segments, %d characters, shortest %d, longest %d (limit %d)."
            % (
                len(self.chunks),
                sum(counts),
                min(counts),
                max(counts),
                self.options.max_chars,
            )
        )
        print("Dry run: DaVinci Resolve was not contacted.")

    # ---------------------------------------------------------- manifest

    def _prepare_manifest(self) -> None:
        os.makedirs(self.segments_dir, exist_ok=True)
        digest = sha256_text(self.script_text)
        voice_settings = self._voice_settings_record()

        if os.path.isfile(self.manifest_path):
            existing = Manifest.load(self.manifest_path)
            if existing.matches(digest, self.options.max_chars, self.options.merge_paragraphs):
                existing.voice_settings = voice_settings
                existing.tool_version = __version__
                self._adopt_manifest(existing)
                LOGGER.info(
                    "Resuming from %s (%d of %d segment(s) already done).",
                    os.path.basename(self.manifest_path),
                    existing.count(STATUS_GENERATED, STATUS_PLACED),
                    len(existing.segments),
                )
                return

            reason = existing.mismatch_reason(
                digest, self.options.max_chars, self.options.merge_paragraphs
            )
            if not self.options.force_rechunk:
                raise UsageError(
                    "The existing manifest cannot be reused because %s." % reason,
                    "  Manifest: %s\n"
                    "  Re-run with --force-rechunk to archive it and start over,\n"
                    "  or point --output-dir at a different directory to keep both."
                    % self.manifest_path,
                )
            archived = existing.archive()
            LOGGER.warning("Archived the previous manifest to %s.", os.path.basename(archived))

        self.manifest = build_manifest(
            path=self.manifest_path,
            script_path=self.options.script_path,
            script_text=self.script_text,
            chunks=self.chunks,
            max_chars=self.options.max_chars,
            merge_paragraphs=self.options.merge_paragraphs,
            voice_settings=voice_settings,
            tool_version=__version__,
        )
        self._save_manifest()

    def _adopt_manifest(self, existing: Manifest) -> None:
        """Reuse ``existing`` but re-sync segment text with the fresh chunks."""
        by_index = {segment.index: segment for segment in existing.segments}
        merged: List[SegmentRecord] = []
        for chunk in self.chunks:
            segment = by_index.get(chunk.index)
            if segment is None:
                segment = SegmentRecord(
                    index=chunk.index,
                    text=chunk.text,
                    char_count=chunk.char_count,
                    paragraph_index=chunk.paragraph_index,
                )
            else:
                segment.text = chunk.text
                segment.char_count = chunk.char_count
                segment.paragraph_index = chunk.paragraph_index
            merged.append(segment)
        existing.segments = merged
        self.manifest = existing

        if self.options.only:
            for segment in merged:
                if segment.number in self.options.only:
                    LOGGER.info("Resetting segment %d for regeneration.", segment.number)
                    segment.status = STATUS_PENDING
                    segment.attempts = 0
                    segment.error = None
        self._save_manifest()

    def _save_manifest(self) -> None:
        if self.manifest is not None:
            self.manifest.save()

    # -------------------------------------------------------- environment

    def _log_environment(self) -> None:
        context = self._require_context()
        timeline = context.require_timeline()
        LOGGER.debug("Project: %s", _call(context.project, "GetName"))
        LOGGER.debug("Timeline: %s", _call(timeline, "GetName"))
        LOGGER.debug(
            "Timeline frame rate setting: %r",
            _call(timeline, "GetSetting", "timelineFrameRate"),
        )
        LOGGER.debug("Timeline start frame: %r", _call(timeline, "GetStartFrame"))
        LOGGER.debug("Timeline start timecode: %r", _call(timeline, "GetStartTimecode"))
        LOGGER.debug("Playhead: %r", _call(timeline, "GetCurrentTimecode"))

    def _require_context(self) -> ResolveContext:
        if self.context is None:  # pragma: no cover - programming error
            raise ToolError("Internal error: Resolve context is not available.")
        return self.context

    def _require_manifest(self) -> Manifest:
        if self.manifest is None:  # pragma: no cover - programming error
            raise ToolError("Internal error: manifest is not available.")
        return self.manifest

    # ---------------------------------------------------------- media bin

    def _enter_bin(self) -> None:
        """Point the media pool at a dedicated bin so generated clips are tidy."""
        context = self._require_context()
        media_pool = context.require_media_pool()
        self._previous_folder = _call(media_pool, "GetCurrentFolder")

        root = _call(media_pool, "GetRootFolder")
        if root is None:
            LOGGER.warning(
                "Could not read the Media Pool root folder; generated clips will "
                "land in the current bin."
            )
            return

        target = None
        sub_folders = _call(root, "GetSubFolderList") or []
        if isinstance(sub_folders, list):
            for folder in sub_folders:
                if (_call(folder, "GetName") or "") == self.options.bin_name:
                    target = folder
                    break
        if target is None:
            target = _call(media_pool, "AddSubFolder", root, self.options.bin_name)
            if target:
                LOGGER.info("Created Media Pool bin '%s'.", self.options.bin_name)
        if target is None:
            LOGGER.warning(
                "Could not create the '%s' bin; using the current bin instead.",
                self.options.bin_name,
            )
            return
        if _call(media_pool, "SetCurrentFolder", target):
            LOGGER.debug("Current Media Pool folder set to '%s'.", self.options.bin_name)
        else:
            LOGGER.warning("SetCurrentFolder('%s') failed.", self.options.bin_name)

    def _leave_bin(self) -> None:
        if self._previous_folder is None or self.context is None:
            return
        media_pool = self.context.media_pool
        if media_pool is None:
            return
        if _call(media_pool, "SetCurrentFolder", self._previous_folder):
            LOGGER.debug("Restored the previous Media Pool folder.")
        else:
            LOGGER.debug("Could not restore the previous Media Pool folder.")

    def _voice_bin(self) -> Optional[Any]:
        context = self._require_context()
        media_pool = context.media_pool
        root = _call(media_pool, "GetRootFolder")
        for folder in _call(root, "GetSubFolderList") or []:
            if (_call(folder, "GetName") or "") == self.options.bin_name:
                return folder
        return None

    # -------------------------------------------------------- generation

    def _voice_settings_record(self) -> Dict[str, Any]:
        """The voice-related settings, for the manifest and the log."""
        options = self.options
        record: Dict[str, Any] = {}
        if options.custom_voice_file:
            record["VoiceModel"] = "Custom Voice"
            record["CustomVoiceFile"] = options.custom_voice_file
        elif options.voice:
            record["VoiceModel"] = options.voice
        for key, value in (
            ("Speed", options.speed),
            ("Variation", options.variation),
            ("Pitch", options.pitch),
            ("GenerationID", options.generation_id),
        ):
            if value is not None:
                record[key] = value
        return record

    def _speech_settings(
        self, segment: SegmentRecord, add_to_timeline: bool
    ) -> Dict[str, Any]:
        """Build one ``speechGenerationSettings`` dictionary.

        ``Filename`` is a *prefix*, not a path (verified on 21.0.1): Resolve
        writes ``<Filename>-<VoiceModel>-G<GenerationID>-<NNN>.wav`` into the
        project's ``Audio Files/SpeechGenerator`` folder, and auto-increments
        ``<NNN>`` to avoid collisions.  Omitting it gives every file the prefix
        ``SpeechGen``, which is why one is always supplied.
        """
        settings: Dict[str, Any] = {"TextInput": segment.text}
        # GenerationID is a seed: the same id and the same text give identical
        # audio.  Leaving it out lets Resolve use 1 for every segment, which
        # keeps the voice character consistent across the whole narration.
        settings.update(self._voice_settings_record())
        settings["Filename"] = "AIVoice_%s_%03d" % (
            self.options.script_stem,
            segment.number,
        )
        settings["AddToTimeline"] = bool(add_to_timeline)
        if add_to_timeline and self._target_track:
            settings["AudioTrack"] = int(self._target_track)
        return settings

    def _segments_to_generate(self) -> List[SegmentRecord]:
        manifest = self._require_manifest()
        options = self.options
        if options.only:
            return [s for s in manifest.segments if s.number in options.only]
        if options.retry_failed:
            return manifest.with_status(STATUS_FAILED)
        return manifest.with_status(STATUS_PENDING, STATUS_FAILED)

    def _generate_all(self) -> None:
        manifest = self._require_manifest()
        pending = {segment.index for segment in self._segments_to_generate()}
        total = len(manifest.segments)

        if self.options.placement == "resolve" and not self.options.skip_placement:
            self._prepare_resolve_placement()

        print("")
        first = True
        for segment in manifest.segments:
            label = "Generating segment %d/%d..." % (segment.number, total)
            if segment.index not in pending:
                self._begin_line(label)
                if segment.is_complete:
                    self.result.cached += 1
                    self._end_line("(cached)")
                else:
                    self._end_line("(skipped)")
                continue
            if first:
                first = False
                LOGGER.debug(
                    "The first GenerateSpeech call of a Resolve session warms the "
                    "model up and takes appreciably longer (~12 s was measured on "
                    "21.0.1); later calls take 1-4 s for a 300-character segment."
                )
            self._generate_segment(segment, total)
            self._save_manifest()
            if segment.status == STATUS_FAILED and self.options.stop_on_error:
                raise PartialFailure(
                    "Stopping after segment %d failed (--stop-on-error)."
                    % segment.number,
                    "  %s\n"
                    "  %d of %d segment(s) have been generated successfully so far.\n"
                    "  Progress is saved; re-run to resume, or use --retry-failed."
                    % (
                        segment.error or "no further detail",
                        manifest.count(STATUS_GENERATED, STATUS_PLACED),
                        total,
                    ),
                )

    def _generate_segment(self, segment: SegmentRecord, total: int) -> None:
        options = self.options
        attempts_allowed = max(1, options.max_attempts)
        label = "Generating segment %d/%d..." % (segment.number, total)
        for attempt in range(1, attempts_allowed + 1):
            segment.attempts += 1
            self._begin_line(label)
            try:
                self._generate_once(segment)
            except PlacementAborted:
                # The audio was generated; only the placement failed, and that
                # is fatal for the whole run.
                self.result.generated += 1
                self._end_line("generated, but Resolve placed nothing")
                raise
            except ToolError as exc:
                segment.status = STATUS_FAILED
                segment.error = exc.message
                if attempt < attempts_allowed:
                    self._end_line("failed, retrying (%s)" % exc.message)
                    continue
                self._end_line("FAILED")
                LOGGER.error("  segment %d/%d failed: %s", segment.number, total, exc.message)
                if exc.hint:
                    LOGGER.error("%s", exc.hint)
                self.result.failed += 1
                return
            segment.error = None
            self.result.generated += 1
            self._end_line("done (%.1fs)" % (segment.generation_seconds or 0.0))
            return

    def _generate_once(self, segment: SegmentRecord) -> None:
        """Run one ``GenerateSpeech`` call and record everything it produced.

        Raises:
            ToolError: The call failed, or its output could not be located.
        """
        context = self._require_context()
        project = context.require_project()
        timeline = context.require_timeline()

        text = segment.text
        if len(text) > RESOLVE_TEXT_LIMIT:  # belt and braces
            raise ToolError(
                "Segment %d is %d characters; Resolve accepts at most %d."
                % (segment.number, len(text), RESOLVE_TEXT_LIMIT)
            )
        if not text.strip():
            raise ToolError("Segment %d is empty." % segment.number)

        place_with_resolve = (
            self.options.placement == "resolve" and not self.options.skip_placement
        )
        settings = self._speech_settings(segment, add_to_timeline=place_with_resolve)
        position = self._resolve_mode_timecode() if place_with_resolve else self._playhead_timecode()

        loggable = dict(settings)
        loggable["TextInput"] = _truncate(text)
        LOGGER.debug("  settings %s at %s", loggable, position)
        LOGGER.debug("  full TextInput (%d chars): %r", len(text), text)

        before: Optional[List[Any]] = None
        if place_with_resolve:
            before = self._track_item_ids(self._target_track or 0)

        started = time.monotonic()
        try:
            item = project.GenerateSpeech(settings, position)
        except Exception as exc:  # noqa: BLE001 - the native bridge raises anything
            raise ToolError(
                "GenerateSpeech raised %s: %s" % (type(exc).__name__, exc),
                _ENGINE_FAILURE_HINT,
            )
        elapsed = time.monotonic() - started
        segment.generation_seconds = round(elapsed, 3)
        LOGGER.debug("  GenerateSpeech returned %r after %.2fs", item, elapsed)

        if not item:
            # Verified on 21.0.1: a rejected setting (bad voice name, or text over
            # 350 characters) comes back as None in hundredths of a second, while
            # a real engine problem takes seconds first.
            fast = elapsed < _FAST_FAILURE_SECONDS
            raise ToolError(
                "GenerateSpeech returned %r for segment %d after %.2fs."
                % (item, segment.number, elapsed),
                _SETTINGS_FAILURE_HINT if fast else _ENGINE_FAILURE_HINT,
            )
        if not hasattr(item, "GetClipProperty"):
            raise ToolError(
                "GenerateSpeech returned a %s, not a MediaPoolItem."
                % type(item).__name__,
                "  Enable --log-level DEBUG and inspect generate_voice.log; the API\n"
                "  contract may have changed in this Resolve build.",
            )

        segment.clip_name = _call(item, "GetName") or None
        segment.media_pool_item_unique_id = _call(item, "GetUniqueId") or None
        segment.media_id = _call(item, "GetMediaId") or None
        LOGGER.debug(
            "  generated '%s' in %.2fs", segment.clip_name or "(unnamed)", elapsed
        )

        file_path = self._discover_file_path(item, segment)
        segment.resolve_file_path = file_path
        LOGGER.debug("  file: %s", file_path)

        self._record_audio_properties(segment, file_path, item)
        self._cache_file(segment, file_path)
        self._items[segment.index] = item

        # The audio exists and is cached from here on, whatever placement does,
        # so record that before the placement check can abort the run.
        segment.status = STATUS_GENERATED
        if place_with_resolve:
            self._verify_resolve_placement(segment, before)
            self._record_resolve_placement(segment)
            segment.status = STATUS_PLACED
            self.result.placed += 1

    def _discover_file_path(self, item: Any, segment: SegmentRecord) -> str:
        """Find the WAV that Resolve just wrote.

        Raises:
            ToolError: No usable path could be read from the clip properties.
        """
        for key in _FILE_PATH_KEYS:
            value = _call(item, "GetClipProperty", key)
            if _is_usable_path(value):
                LOGGER.debug("Clip property %r gave %r", key, value)
                return value.strip()

        properties = _call(item, "GetClipProperty")
        LOGGER.debug("Full clip property dict for segment %d: %r", segment.number, properties)
        if isinstance(properties, dict):
            for key, value in properties.items():
                if not isinstance(key, str) or "path" not in key.lower():
                    continue
                if key in _FILE_PATH_KEYS or not _is_usable_path(value):
                    continue
                LOGGER.warning(
                    "Used clip property %r instead of 'File Path' for segment %d.",
                    key,
                    segment.number,
                )
                return value.strip()

        raise ToolError(
            "Could not determine where Resolve wrote the audio for segment %d."
            % segment.number,
            "  The returned MediaPoolItem has no readable 'File Path' property.\n"
            "  The full property dictionary was written to generate_voice.log at\n"
            "  DEBUG level; check it for the key that holds the file location.",
        )

    def _record_audio_properties(
        self, segment: SegmentRecord, file_path: str, item: Any
    ) -> None:
        info: AudioInfo = audio_utils.inspect_wav(file_path)
        LOGGER.debug("  audio: %s", info.describe())
        if info.ok:
            segment.duration_seconds = info.duration_seconds
            segment.sample_rate = info.sample_rate
            segment.channels = info.channels
            segment.sample_width = info.sample_width
            segment.sample_format = info.sample_format
            segment.source_frames = frames_for_duration(
                info.duration_seconds or 0.0, self._frame_rate_for_timeline()
            )
        else:
            LOGGER.warning(
                "  could not parse the WAV (%s); falling back to clip properties.",
                info.error,
            )
            segment.duration_seconds = None
            segment.source_frames = self._frames_from_clip_properties(item)
            if segment.source_frames is None:
                LOGGER.warning(
                    "  Resolve's clip properties gave no usable frame count either; "
                    "segment %d will be placed as a single frame.",
                    segment.number,
                )

        silence: SilenceInfo = audio_utils.detect_silence(
            file_path, threshold_db=self.options.silence_threshold_db, info=info
        )
        if silence.analyzed and not silence.all_silent:
            segment.leading_silence_seconds = round(silence.leading_seconds, 4)
            segment.trailing_silence_seconds = round(silence.trailing_seconds, 4)
            LOGGER.debug(
                "  silence: %.3fs leading, %.3fs trailing (peak %.1f dBFS)",
                silence.leading_seconds,
                silence.trailing_seconds,
                silence.peak_dbfs if silence.peak_dbfs is not None else 0.0,
            )
        elif silence.all_silent:
            LOGGER.warning(
                "  the whole of segment %d is below %.1f dBFS.",
                segment.number,
                self.options.silence_threshold_db,
            )
        else:
            LOGGER.debug("  silence: not analysed (%s)", silence.reason)

    def _frames_from_clip_properties(self, item: Any) -> Optional[int]:
        """Fall back to Resolve's own frame count when the WAV is unparseable."""
        properties = _call(item, "GetClipProperty")
        LOGGER.debug("Clip properties used for the frame-count fallback: %r", properties)
        return frames_from_clip_properties(properties, self._frame_rate_for_timeline())

    def _cache_file(self, segment: SegmentRecord, file_path: str) -> None:
        """Copy Resolve's WAV into ``segments/`` as a safety net.

        Resolve writes its own audio into the project's working folder
        (``<media location>/Audio Files/SpeechGenerator/``, verified on 21.0.1)
        and keeps referencing it there.  This copy is never given to Resolve
        unless the original disappears, in which case it is re-imported.
        """
        destination = os.path.join(self.segments_dir, "%03d.wav" % segment.number)
        try:
            os.makedirs(self.segments_dir, exist_ok=True)
            shutil.copy2(file_path, destination)
        except OSError as exc:
            LOGGER.warning("  could not cache %s: %s", file_path, exc)
            segment.cached_file = None
            return
        segment.cached_file = os.path.relpath(destination, self.options.output_dir)
        LOGGER.debug("Cached segment %d to %s", segment.number, destination)

    # ---------------------------------------------------------- timecode

    def _frame_rate_for_timeline(self) -> tc.FrameRate:
        if self._frame_rate is not None:
            return self._frame_rate
        context = self._require_context()
        timeline = context.require_timeline()
        raw = _call(timeline, "GetSetting", "timelineFrameRate")
        source = "timeline"
        if not raw:
            raw = _call(context.project, "GetSetting", "timelineFrameRate")
            source = "project"
        try:
            frame_rate = tc.parse_frame_rate(raw)
        except ValueError as exc:
            raise ResolveError(
                "Could not understand the timeline frame rate %r." % (raw,),
                "  %s\n"
                "  Check Project Settings -> Master Settings -> Timeline frame rate."
                % exc,
            )
        LOGGER.debug(
            "Timeline frame rate: %s (%s setting %r)", frame_rate, source, raw
        )
        self._frame_rate = frame_rate
        return frame_rate

    def _playhead_timecode(self) -> str:
        """The timecode string passed as ``GenerateSpeech``'s second argument."""
        timeline = self._require_context().require_timeline()
        current = _call(timeline, "GetCurrentTimecode")
        if isinstance(current, str) and tc.is_timecode(current):
            return current
        start = _call(timeline, "GetStartTimecode")
        if isinstance(start, str) and tc.is_timecode(start):
            LOGGER.debug("Playhead unavailable; using start timecode %s", start)
            return start
        LOGGER.debug("No timecode available from the timeline; using 00:00:00:00")
        return "00:00:00:00"

    def _start_frame(self) -> int:
        """Absolute timeline frame where the first clip should land.

        Raises:
            ResolveError: The requested start position cannot be resolved.
        """
        timeline = self._require_context().require_timeline()
        frame_rate = self._frame_rate_for_timeline()
        setting = self.options.start

        if setting == "timeline-start":
            start = _as_int(_call(timeline, "GetStartFrame"))
            if start is None:
                raise ResolveError("Timeline.GetStartFrame() returned no usable value.")
            return start

        if setting == "playhead":
            current = _call(timeline, "GetCurrentTimecode")
            if isinstance(current, str) and tc.is_timecode(current):
                return tc.timecode_to_frames(current, frame_rate)
            start = _as_int(_call(timeline, "GetStartFrame"))
            if start is None:
                raise ResolveError(
                    "Could not read the playhead position from the timeline.",
                    "  Open the Edit, Cut or Fairlight page so Resolve reports a\n"
                    "  playhead timecode, or use --start timeline-start.",
                )
            LOGGER.warning(
                "Playhead timecode unavailable; starting at the timeline start instead."
            )
            return start

        try:
            return tc.timecode_to_frames(setting, frame_rate)
        except ValueError as exc:
            raise ResolveError("Invalid --start timecode %r: %s" % (setting, exc))

    def _resolve_mode_timecode(self) -> str:
        """Next insert timecode for ``--placement resolve``."""
        frame_rate = self._frame_rate_for_timeline()
        return tc.frames_to_timecode(self._next_record_frame, frame_rate)

    # --------------------------------------------------------- placement

    def _resolve_media_items(self) -> None:
        """Make sure every completed segment has a usable MediaPoolItem."""
        manifest = self._require_manifest()
        candidates = [s for s in manifest.segments if s.is_complete]
        if not candidates or self.options.skip_placement:
            return
        if self.options.placement == "resolve":
            return

        LOGGER.info("")
        LOGGER.info("Importing audio...")
        bin_folder = self._voice_bin()
        clips = _call(bin_folder, "GetClipList") or [] if bin_folder else []
        by_unique: Dict[str, Any] = {}
        by_media_id: Dict[str, Any] = {}
        by_name: Dict[str, Any] = {}
        by_path: Dict[str, Any] = {}
        if isinstance(clips, list):
            for clip in clips:
                unique = _call(clip, "GetUniqueId")
                if unique:
                    by_unique[str(unique)] = clip
                media_id = _call(clip, "GetMediaId")
                if media_id:
                    by_media_id[str(media_id)] = clip
                name = _call(clip, "GetName")
                if name:
                    by_name[str(name)] = clip
                path = _call(clip, "GetClipProperty", "File Path")
                if isinstance(path, str) and path:
                    by_path[os.path.normcase(os.path.abspath(path))] = clip

        for segment in candidates:
            if segment.index in self._items:
                continue
            item = None
            if segment.media_pool_item_unique_id:
                item = by_unique.get(segment.media_pool_item_unique_id)
            if item is None and segment.media_id:
                item = by_media_id.get(segment.media_id)
            if item is None and segment.resolve_file_path:
                item = by_path.get(
                    os.path.normcase(os.path.abspath(segment.resolve_file_path))
                )
            if item is None and segment.clip_name:
                item = by_name.get(segment.clip_name)
            if item is not None:
                LOGGER.debug("Reused bin clip for segment %d.", segment.number)
                self._items[segment.index] = item
                continue

            path = self._existing_audio_path(segment)
            if path is None:
                LOGGER.error(
                    "Segment %d: neither Resolve's file nor the cached copy exists; "
                    "it must be regenerated.",
                    segment.number,
                )
                segment.status = STATUS_FAILED
                segment.error = "generated audio file is missing on disk"
                self.result.failed += 1
                continue
            LOGGER.info("  re-importing segment %d from %s", segment.number, path)
            imported = _call(
                self._require_context().require_media_pool(), "ImportMedia", [path]
            )
            if isinstance(imported, list) and imported:
                self._items[segment.index] = imported[0]
                segment.media_pool_item_unique_id = (
                    _call(imported[0], "GetUniqueId") or segment.media_pool_item_unique_id
                )
                segment.clip_name = _call(imported[0], "GetName") or segment.clip_name
            else:
                LOGGER.error(
                    "Segment %d: MediaPool.ImportMedia(%r) returned %r.",
                    segment.number,
                    path,
                    imported,
                )
                segment.status = STATUS_FAILED
                segment.error = "ImportMedia failed for %s" % path
                self.result.failed += 1
        self._save_manifest()

    def _existing_audio_path(self, segment: SegmentRecord) -> Optional[str]:
        if segment.resolve_file_path and os.path.isfile(segment.resolve_file_path):
            return segment.resolve_file_path
        if segment.cached_file:
            cached = os.path.join(self.options.output_dir, segment.cached_file)
            if os.path.isfile(cached):
                return cached
        return None

    def _place_all(self) -> None:
        options = self.options
        if options.skip_placement:
            LOGGER.info("")
            LOGGER.info("Skipping timeline placement (--skip-placement).")
            return
        if options.only:
            LOGGER.info("")
            LOGGER.warning(
                "Skipping timeline placement because --only was used; existing clips "
                "are never moved or replaced."
            )
            LOGGER.warning(
                "Re-run with --place-only (and a clear target track) to lay the "
                "segments down again."
            )
            return
        if options.placement == "resolve":
            LOGGER.info("")
            LOGGER.info("Building timeline...")
            LOGGER.info(
                "Clips were placed by Resolve itself during generation "
                "(--placement resolve)."
            )
            return

        manifest = self._require_manifest()
        failed = manifest.failed_numbers()
        if failed:
            LOGGER.info("")
            LOGGER.error(
                "Not placing anything: %d segment(s) failed to generate (%s).",
                len(failed),
                ", ".join(str(number) for number in failed),
            )
            LOGGER.error(
                "  %d of %d segment(s) have been generated successfully so far.",
                manifest.count(STATUS_GENERATED, STATUS_PLACED),
                len(manifest.segments),
            )
            LOGGER.error(
                "  Placing the rest would put an incomplete narration on your "
                "timeline. Fix the failures first:"
            )
            LOGGER.error(
                '    python generate_voice.py "%s" --retry-failed',
                self.options.script_path,
            )
            LOGGER.error(
                '    python generate_voice.py "%s" --place-only',
                self.options.script_path,
            )
            return

        ready = [
            s
            for s in manifest.segments
            if s.status == STATUS_GENERATED and s.index in self._items
        ]
        already = manifest.count(STATUS_PLACED)
        if ready and already:
            LOGGER.warning(
                "%d segment(s) are already on the timeline from an earlier run; "
                "only the remaining %d will be placed.",
                already,
                len(ready),
            )
            resume = self._resume_frame()
            if resume is not None and self.options.start == "playhead":
                LOGGER.warning(
                    "  If the placement below is refused, re-run with "
                    "--start %s to continue after the last placed clip.",
                    tc.frames_to_timecode(resume, self._frame_rate_for_timeline()),
                )
        if not ready:
            LOGGER.info("")
            if already:
                LOGGER.info(
                    "Building timeline... nothing to do: all %d placed segment(s) "
                    "are already on the timeline.",
                    already,
                )
            else:
                LOGGER.warning("Building timeline... nothing to place.")
            return

        LOGGER.info("")
        LOGGER.info("Building timeline...")
        frame_rate = self._frame_rate_for_timeline()
        track_index = self._ensure_track(ready)
        start_frame = self._start_frame()
        self._guard_track_range(track_index, start_frame, ready, frame_rate)

        media_pool = self._require_context().require_media_pool()
        record_frame = start_frame
        for segment in ready:
            source_in, source_out = self._source_range(segment, frame_rate)
            length = source_out - source_in
            clip_info = build_clip_info(
                self._items[segment.index],
                source_in,
                source_out,
                track_index,
                record_frame,
            )
            LOGGER.info(
                "  segment %d -> A%d at %s (frame %d), source %d..%d exclusive "
                "(%d frames)",
                segment.number,
                track_index,
                tc.frames_to_timecode(record_frame, frame_rate),
                record_frame,
                source_in,
                source_out,
                length,
            )
            LOGGER.debug("  AppendToTimeline clipInfo: %r", clip_info)
            placed = _call(media_pool, "AppendToTimeline", [clip_info])
            if not placed or not isinstance(placed, list):
                raise PlacementAborted(
                    "MediaPool.AppendToTimeline returned %r for segment %d."
                    % (placed, segment.number),
                    "  Nothing was placed for this segment. Things to check:\n"
                    "    - the target track (A%d) is an audio track and is unlocked;\n"
                    "    - the track layout matches the clip (mono clip on a mono track);\n"
                    "    - the segment's audio file still exists on disk.\n"
                    "  %d of %d segment(s) were placed before this one."
                    % (track_index, self.result.placed, len(ready)),
                )

            item = placed[0]
            placed_start = _as_int(_call(item, "GetStart"))
            placed_end = _as_int(_call(item, "GetEnd"))
            LOGGER.debug(
                "  TimelineItem '%s' start=%r end=%r duration=%r",
                _call(item, "GetName"),
                placed_start,
                placed_end,
                _call(item, "GetDuration"),
            )
            segment.status = STATUS_PLACED
            segment.track_index = track_index
            segment.record_frame = record_frame
            segment.start_frame = source_in
            segment.end_frame = source_out
            segment.timeline_timecode = tc.frames_to_timecode(record_frame, frame_rate)
            self.result.placed += 1
            self._save_manifest()

            # GetEnd() is exclusive too (verified: start 515, end 621 is a
            # 106-frame clip), so the next recordFrame is exactly GetEnd().
            if placed_start is not None and placed_end is not None and placed_end > placed_start:
                actual = placed_end - placed_start
                if actual != length:
                    LOGGER.warning(
                        "  segment %d was placed as %d frames, not the %d requested; "
                        "following clips follow the actual length.",
                        segment.number,
                        actual,
                        length,
                    )
                record_frame = placed_end + self.options.gap_frames
            else:
                record_frame = record_frame + length + self.options.gap_frames

    def _resume_frame(self) -> Optional[int]:
        """Frame just past the last clip an earlier run placed, if any."""
        manifest = self._require_manifest()
        end = None
        for segment in manifest.segments:
            if segment.status != STATUS_PLACED or segment.record_frame is None:
                continue
            length = 1
            if segment.start_frame is not None and segment.end_frame is not None:
                # end_frame is exclusive.
                length = max(1, segment.end_frame - segment.start_frame)
            candidate = segment.record_frame + length
            if end is None or candidate > end:
                end = candidate
        return end

    def _prepare_resolve_placement(self) -> None:
        """Set up the track and the running timecode for ``--placement resolve``."""
        frame_rate = self._frame_rate_for_timeline()
        manifest = self._require_manifest()
        LOGGER.warning(
            "--placement resolve is EXPERIMENTAL: on Resolve Studio 21.0.1 the "
            "AddToTimeline setting created the clip but placed nothing on any "
            "audio track. Each call is checked, and the run stops if the clip "
            "does not appear."
        )
        self._target_track = self._ensure_track(manifest.segments)
        self._next_record_frame = self._start_frame()
        self._guard_track_open_end(self._target_track, self._next_record_frame, frame_rate)
        if self.options.trim_silence:
            LOGGER.warning(
                "--trim-silence has no effect with --placement resolve: Resolve places "
                "each whole clip itself."
            )

    def _track_item_ids(self, track_index: int) -> List[Any]:
        """Identity of every clip currently on an audio track, for a before/after."""
        if not track_index:
            return []
        timeline = self._require_context().require_timeline()
        items = _call(timeline, "GetItemListInTrack", "audio", track_index)
        if not isinstance(items, list):
            return []
        return [
            (
                _call(item, "GetUniqueId") or _call(item, "GetName"),
                _as_int(_call(item, "GetStart")),
            )
            for item in items
        ]

    def _verify_resolve_placement(
        self, segment: SegmentRecord, before: Optional[List[Any]]
    ) -> None:
        """Confirm ``AddToTimeline`` really put a clip on the target track.

        Raises:
            PlacementAborted: The track has no new clip, which is what Resolve
                Studio 21.0.1 was observed to do.
        """
        if before is None:
            return
        after = self._track_item_ids(self._target_track or 0)
        if len(after) > len(before):
            LOGGER.debug(
                "  A%s went from %d to %d clip(s).",
                self._target_track,
                len(before),
                len(after),
            )
            return
        raise PlacementAborted(
            "%s (segment %d)" % (_RESOLVE_PLACEMENT_FAILED, segment.number),
            "  Audio track A%s still holds %d clip(s) after the call, and the\n"
            "  generated audio is safely cached, so nothing was lost:\n"
            '    python generate_voice.py "%s" --place-only'
            % (self._target_track, len(after), self.options.script_path),
        )

    def _record_resolve_placement(self, segment: SegmentRecord) -> None:
        """Advance the running timecode after Resolve placed a clip itself."""
        frame_rate = self._frame_rate_for_timeline()
        segment.track_index = self._target_track
        segment.record_frame = self._next_record_frame
        segment.timeline_timecode = tc.frames_to_timecode(
            self._next_record_frame, frame_rate
        )
        length = self._total_source_frames(segment, frame_rate)
        if length <= 0:
            LOGGER.warning(
                "  unknown duration for segment %d; assuming 1 frame so the next "
                "clip does not overlap it.",
                segment.number,
            )
            length = 1
        segment.start_frame = 0
        segment.end_frame = length
        self._next_record_frame += length + self.options.gap_frames
        LOGGER.debug(
            "  placed by Resolve on A%s at %s (%d frames)",
            self._target_track,
            segment.timeline_timecode,
            length,
        )

    # --------------------------------------------------------- track work

    def _ensure_track(self, segments: Sequence[SegmentRecord]) -> int:
        """Find, or create, the audio track to place clips on.

        Raises:
            PlacementAborted: The requested track does not exist or is locked.
        """
        options = self.options
        timeline = self._require_context().require_timeline()
        count = _as_int(_call(timeline, "GetTrackCount", "audio")) or 0

        if options.track_index is not None:
            index = options.track_index
            if not 1 <= index <= count:
                raise PlacementAborted(
                    "--track-index %d is out of range; the timeline has %d audio track(s)."
                    % (index, count),
                    "  Use --track-name to have the track created for you, or pick an\n"
                    "  index between 1 and %d." % max(count, 1),
                )
            self._check_track_unlocked(timeline, index)
            LOGGER.info(
                "Using audio track A%d ('%s').",
                index,
                _call(timeline, "GetTrackName", "audio", index) or "unnamed",
            )
            return index

        for index in range(1, count + 1):
            if (_call(timeline, "GetTrackName", "audio", index) or "") == options.track_name:
                self._check_track_unlocked(timeline, index)
                LOGGER.info("Reusing existing audio track A%d ('%s').", index, options.track_name)
                return index

        audio_type = self._track_sub_type(segments)
        LOGGER.info(
            "Adding a %s audio track named '%s'.", audio_type, options.track_name
        )
        # Verified on 21.0.1: AddTrack returns True, GetTrackCount immediately
        # reflects the new track, SetTrackName works, and GetTrackSubType then
        # reports "mono".
        if not _call(timeline, "AddTrack", "audio", audio_type):
            raise PlacementAborted(
                "Timeline.AddTrack('audio', %r) failed." % audio_type,
                "  Add an audio track manually in Resolve, name it '%s', then re-run\n"
                "  with --place-only." % options.track_name,
            )
        new_index = _as_int(_call(timeline, "GetTrackCount", "audio")) or (count + 1)
        if not _call(timeline, "SetTrackName", "audio", new_index, options.track_name):
            LOGGER.warning(
                "Created track A%d but could not rename it to '%s'.",
                new_index,
                options.track_name,
            )
        self._check_track_unlocked(timeline, new_index)
        return new_index

    def _check_track_unlocked(self, timeline: Any, index: int) -> None:
        if _call(timeline, "GetIsTrackLocked", "audio", index):
            raise PlacementAborted(
                "Audio track A%d is locked." % index,
                "  Unlock it in Resolve (click the padlock on the track header) or\n"
                "  choose another track with --track-index / --track-name.",
            )

    def _track_sub_type(self, segments: Sequence[SegmentRecord]) -> str:
        """Match the new track layout to the generated audio's channel count."""
        channels = {s.channels for s in segments if s.channels}
        if channels == {2}:
            return "stereo"
        if channels and channels != {1}:
            LOGGER.warning(
                "Generated clips have mixed channel counts %s; creating a mono track.",
                sorted(c for c in channels if c),
            )
        return "mono"

    def _total_source_frames(
        self, segment: SegmentRecord, frame_rate: tc.FrameRate
    ) -> int:
        """Whole frames of media available for ``segment``.

        The WAV duration is authoritative and is **floored**: Resolve itself
        floors partial frames, and asking ``AppendToTimeline`` for the ceiling
        was verified to produce a clip that reaches past the end of the media.
        """
        duration = segment.duration_seconds
        if duration and duration > 0:
            frames = frames_for_duration(duration, frame_rate)
            if frames > 0:
                return frames
        if segment.source_frames and segment.source_frames > 0:
            return segment.source_frames
        return 0

    def _source_range(
        self, segment: SegmentRecord, frame_rate: tc.FrameRate
    ) -> Tuple[int, int]:
        """Return ``(startFrame, endFrame)`` for the clip, ``endFrame`` exclusive."""
        total = self._total_source_frames(segment, frame_rate)
        if total <= 0:
            LOGGER.warning(
                "  unknown duration for segment %d; placing a single frame.",
                segment.number,
            )
            return 0, 1

        if not self.options.trim_silence:
            return 0, total

        duration = segment.duration_seconds or (total / frame_rate.fps)
        pad = max(0, self.options.keep_pad_ms) / 1000.0
        leading = max(0.0, (segment.leading_silence_seconds or 0.0) - pad)
        trailing = max(0.0, (segment.trailing_silence_seconds or 0.0) - pad)
        # Floor the leading trim and ceil the trailing keep so a rounding error
        # can only ever leave silence in, never clip speech out.
        first = min(total - 1, tc.seconds_to_frames(leading, frame_rate, "floor"))
        last = min(total, tc.seconds_to_frames(duration - trailing, frame_rate, "ceil"))
        if last <= first:
            LOGGER.warning(
                "  silence trimming would empty segment %d; using the whole clip.",
                segment.number,
            )
            return 0, total
        if first or last != total:
            LOGGER.debug(
                "  segment %d trimmed to frames %d-%d of %d",
                segment.number,
                first,
                last - 1,
                total,
            )
        return first, last

    def _guard_track_range(
        self,
        track_index: int,
        start_frame: int,
        segments: Sequence[SegmentRecord],
        frame_rate: tc.FrameRate,
    ) -> None:
        """Refuse to place clips where the user already has audio.

        Raises:
            PlacementAborted: An existing clip overlaps the intended range.
        """
        total = 0
        for segment in segments:
            source_in, source_out = self._source_range(segment, frame_rate)
            total += (source_out - source_in) + self.options.gap_frames
        end_frame = start_frame + max(total, 1)
        LOGGER.debug(
            "Intended range on A%d: frames %d-%d", track_index, start_frame, end_frame - 1
        )

        conflicts = self._overlapping_items(track_index, start_frame, end_frame)
        if not conflicts:
            return
        raise PlacementAborted(
            "Audio track A%d already has %d clip(s) between %s and %s."
            % (
                track_index,
                len(conflicts),
                tc.frames_to_timecode(start_frame, frame_rate),
                tc.frames_to_timecode(end_frame, frame_rate),
            ),
            "  Nothing was changed -- this tool never deletes or overwrites your clips.\n"
            "  First conflicting clip: %s\n"
            "  Move the playhead somewhere clear (--start <timecode>), pick a different\n"
            "  track (--track-name / --track-index), or place the voice on a new track."
            % conflicts[0],
        )

    def _guard_track_open_end(
        self, track_index: int, start_frame: int, frame_rate: tc.FrameRate
    ) -> None:
        """Same guard, for when the total length is not yet known.

        Raises:
            PlacementAborted: Any existing clip ends after ``start_frame``.
        """
        conflicts = self._overlapping_items(track_index, start_frame, None)
        if not conflicts:
            return
        raise PlacementAborted(
            "Audio track A%d already has %d clip(s) at or after %s."
            % (track_index, len(conflicts), tc.frames_to_timecode(start_frame, frame_rate)),
            "  Nothing was changed. With --placement resolve the total length is not\n"
            "  known in advance, so the whole tail of the track must be clear.\n"
            "  First conflicting clip: %s" % conflicts[0],
        )

    def _overlapping_items(
        self, track_index: int, start_frame: int, end_frame: Optional[int]
    ) -> List[str]:
        timeline = self._require_context().require_timeline()
        items = _call(timeline, "GetItemListInTrack", "audio", track_index)
        if not isinstance(items, list):
            LOGGER.warning(
                "GetItemListInTrack('audio', %d) returned %r; cannot verify the track "
                "is clear.",
                track_index,
                items,
            )
            return []
        conflicts: List[str] = []
        for item in items:
            item_start = _as_int(_call(item, "GetStart"))
            item_end = _as_int(_call(item, "GetEnd"))
            if item_start is None or item_end is None:
                continue
            if end_frame is None:
                overlaps = item_end > start_frame
            else:
                overlaps = item_start < end_frame and item_end > start_frame
            if overlaps:
                conflicts.append(
                    "'%s' frames %d-%d"
                    % (_call(item, "GetName") or "unnamed", item_start, item_end)
                )
        return conflicts

    # ------------------------------------------------------------- finish

    def _save_project(self) -> None:
        context = self._require_context()
        saved = _call(context.project_manager, "SaveProject")
        if saved:
            LOGGER.info("Project saved.")
        else:
            LOGGER.warning(
                "ProjectManager.SaveProject() returned %r; save the project in "
                "Resolve to keep the new clips.",
                saved,
            )

    def _finalise(self) -> None:
        manifest = self._require_manifest()
        failed = manifest.failed_numbers()
        self.result.failed = len(failed)
        self.result.failed_numbers = failed
        self.result.placed = manifest.count(STATUS_PLACED)
        LOGGER.info("Complete.")
        if failed:
            self.result.exit_code = 3

    # -------------------------------------------------------- reporting

    def summary_lines(self) -> List[str]:
        """The final summary table shown after a run."""
        result = self.result
        lines = [
            "",
            "Summary",
            "-------",
            "  segments   %d" % result.total,
            "  generated  %d" % result.generated,
            "  cached     %d" % result.cached,
            "  placed     %d" % result.placed,
            "  failed     %d" % result.failed,
        ]
        if result.failed_numbers:
            lines.append(
                "  failed segments: %s"
                % ", ".join(str(number) for number in result.failed_numbers)
            )
            lines.append("")
            lines.append("  Retry them with:")
            lines.append(
                '    python generate_voice.py "%s" --retry-failed'
                % self.options.script_path
            )
        if not self.options.dry_run:
            lines.append("")
            lines.append("  output   %s" % os.path.abspath(self.options.output_dir))
            lines.append("  log      %s" % os.path.join(
                os.path.abspath(self.options.output_dir), "generate_voice.log"
            ))
        return lines
