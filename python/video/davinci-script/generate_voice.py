#!/usr/bin/env python3
"""Generate a long narration with DaVinci Resolve's AI Speech Generator.

Splits a narration script into chunks that fit Resolve's 350-character
``GenerateSpeech`` limit, generates each chunk, caches the results so re-runs
resume, and lays the clips back-to-back on a dedicated "AI Voice" audio track.

Run ``python generate_voice.py --help`` for the full option list, or
``python generate_voice.py --probe`` to check the environment.

Exit codes
    0  success
    1  usage / input error
    2  Resolve connection or validation failure
    3  some segments failed to generate
    4  timeline placement refused for safety
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from typing import Optional, Sequence, Set

from resolve_tts import ToolError, UsageError, __version__
from resolve_tts import chunker, timecode as tc
from resolve_tts.pipeline import Options, Pipeline, configure_logging
from resolve_tts.resolve_connect import connect, probe_report

LOGGER = logging.getLogger("resolve_tts.cli")

_MIN_MAX_CHARS = 50
_MAX_MAX_CHARS = chunker.RESOLVE_TEXT_LIMIT

#: Voice names documented in Resolve's scripting README.  The API does not
#: expose the full list, so anything else has to be typed exactly as it appears
#: in the Speech Generator UI.
KNOWN_VOICE_EXAMPLES = ("Female 1", "Male 1", "Custom Voice")


# --------------------------------------------------------------------------
# argparse helpers
# --------------------------------------------------------------------------


def _env(name: str, fallback: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value else fallback


def _max_chars(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number, got %r" % value)
    if not _MIN_MAX_CHARS <= number <= _MAX_MAX_CHARS:
        raise argparse.ArgumentTypeError(
            "must be between %d and %d (Resolve's hard limit), got %d"
            % (_MIN_MAX_CHARS, _MAX_MAX_CHARS, number)
        )
    return number


def _start_position(value: str) -> str:
    text = value.strip()
    if text in ("playhead", "timeline-start"):
        return text
    if tc.is_timecode(text):
        return text
    raise argparse.ArgumentTypeError(
        "must be 'playhead', 'timeline-start' or a timecode such as 01:00:05:00, "
        "got %r" % value
    )


def parse_segment_selection(value: str) -> Set[int]:
    """Parse ``"5,7-9"`` into ``{5, 7, 8, 9}`` (one-based segment numbers).

    Raises:
        argparse.ArgumentTypeError: The selection is malformed or empty.
    """
    selected: Set[int] = set()
    for part in value.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece.lstrip("-"):
            low_text, _, high_text = piece.partition("-")
            try:
                low, high = int(low_text), int(high_text)
            except ValueError:
                raise argparse.ArgumentTypeError("bad range %r in --only" % piece)
            if low < 1 or high < low:
                raise argparse.ArgumentTypeError("bad range %r in --only" % piece)
            selected.update(range(low, high + 1))
            continue
        try:
            number = int(piece)
        except ValueError:
            raise argparse.ArgumentTypeError("bad segment number %r in --only" % piece)
        if number < 1:
            raise argparse.ArgumentTypeError("segment numbers start at 1, got %d" % number)
        selected.add(number)
    if not selected:
        raise argparse.ArgumentTypeError("--only did not select any segments")
    return selected


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser."""
    parser = argparse.ArgumentParser(
        prog="generate_voice.py",
        description=(
            "Generate a long narration with DaVinci Resolve Studio's AI Speech "
            "Generator and lay it out on a dedicated audio track."
        ),
        epilog=(
            "Examples:\n"
            "  py -3.14 generate_voice.py script.txt\n"
            "  py -3.14 generate_voice.py script.txt --dry-run\n"
            "  py -3.14 generate_voice.py --probe\n"
            "  py -3.14 generate_voice.py script.txt --voice \"Male 1\" "
            "--start timeline-start\n"
            "  py -3.14 generate_voice.py script.txt --retry-failed\n"
            "  py -3.14 generate_voice.py script.txt --only 5,7-9\n"
            "  py -3.14 generate_voice.py script.txt --place-only\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="UTF-8 .txt narration script. Omit it to pick a file in a dialog.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)

    chunking = parser.add_argument_group("chunking")
    chunking.add_argument(
        "--max-chars",
        type=_max_chars,
        default=int(_env("RESOLVE_TTS_MAX_CHARS", str(chunker.DEFAULT_MAX_CHARS))),
        metavar="N",
        help="Maximum characters per segment, %d-%d (default: %%(default)s). "
        "Resolve rejects anything over %d."
        % (_MIN_MAX_CHARS, _MAX_MAX_CHARS, _MAX_MAX_CHARS),
    )
    chunking.add_argument(
        "--merge-paragraphs",
        action="store_true",
        help="Allow short consecutive paragraphs to share a segment. Off by "
        "default so paragraph breaks stay audible as pauses.",
    )

    voice = parser.add_argument_group("voice settings (passed straight to Resolve)")
    voice.add_argument(
        "--voice",
        default=_env("RESOLVE_TTS_VOICE"),
        metavar="NAME",
        help="VoiceModel name, exactly as shown in the Speech Generator UI "
        "(documented examples: %s). Omitted by default so Resolve uses its "
        "current voice." % ", ".join('"%s"' % v for v in KNOWN_VOICE_EXAMPLES),
    )
    voice.add_argument(
        "--custom-voice-file",
        metavar="PATH",
        help="Full path to a custom voice file. Implies --voice \"Custom Voice\".",
    )
    voice.add_argument(
        "--speed",
        type=int,
        metavar="N",
        help="Speed setting, passed through unvalidated. Resolve 21.0.1 accepted "
        "2, 50, 100 and 1000 without complaint and the ranges are undocumented, "
        "so try a value on one segment first (--only 1).",
    )
    voice.add_argument(
        "--variation",
        type=int,
        metavar="N",
        help="Variation setting, passed through unvalidated (see --speed).",
    )
    voice.add_argument(
        "--pitch",
        type=int,
        metavar="N",
        help="Pitch setting, passed through unvalidated (see --speed); -50 was "
        "accepted.",
    )
    voice.add_argument(
        "--generation-id",
        type=int,
        metavar="N",
        help="GenerationID: verified to act as a seed, so the same id and the "
        "same text give identical audio. Omit it and Resolve uses 1 for every "
        "segment, which keeps one voice character across the narration.",
    )

    timeline = parser.add_argument_group("timeline placement")
    timeline.add_argument(
        "--track-name",
        default=_env("RESOLVE_TTS_TRACK_NAME", "AI Voice"),
        metavar="NAME",
        help="Audio track to use; created if it does not exist (default: %(default)s).",
    )
    timeline.add_argument(
        "--track-index",
        type=int,
        metavar="N",
        help="Place on this existing 1-based audio track instead of matching by "
        "name. No track is created.",
    )
    timeline.add_argument(
        "--start",
        type=_start_position,
        default="playhead",
        metavar="WHERE",
        help="playhead (default), timeline-start, or an explicit HH:MM:SS:FF timecode.",
    )
    timeline.add_argument(
        "--gap-frames",
        type=int,
        default=0,
        metavar="N",
        help="Frames of silence between segments (default: %(default)s, i.e. "
        "back-to-back). The generated clips already carry ~25 ms of leading and "
        "~60 ms of trailing padding, so back-to-back sounds continuous; use "
        "6-12 frames at 24 fps (250-500 ms) if you want a fuller pause.",
    )
    timeline.add_argument(
        "--placement",
        choices=("append", "resolve"),
        default="append",
        help="append: place clips ourselves with MediaPool.AppendToTimeline "
        "(default, exact ordering). resolve: EXPERIMENTAL -- let GenerateSpeech "
        "place each clip itself via AddToTimeline. On Resolve Studio 21.0.1 this "
        "created the clip but placed nothing on any audio track, so the run stops "
        "if the clip does not appear.",
    )
    timeline.add_argument(
        "--bin-name",
        default="AI Voice",
        metavar="NAME",
        help="Media Pool bin for the generated clips (default: %(default)s).",
    )

    silence = parser.add_argument_group("silence trimming (in/out points only)")
    silence.add_argument(
        "--trim-silence",
        action="store_true",
        help="Trim leading/trailing silence using clip in/out points. OFF by "
        "default: measured on real generated clips, the padding is only 24-32 ms "
        "leading and 48-76 ms trailing, which is under two frames at 24 fps and "
        "shorter than a natural sentence pause. Silence is measured and recorded "
        "in the manifest either way.",
    )
    silence.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-50.0,
        metavar="DB",
        help="Peak level at or below which audio counts as silence "
        "(default: %(default)s dBFS).",
    )
    silence.add_argument(
        "--keep-pad-ms",
        type=int,
        default=40,
        metavar="MS",
        help="Silence kept either side of the speech when trimming "
        "(default: %(default)s ms).",
    )

    workflow = parser.add_argument_group("workflow")
    workflow.add_argument(
        "--output-dir",
        default=_env("RESOLVE_TTS_OUTPUT_DIR"),
        metavar="DIR",
        help="Where the manifest, cached WAVs and log go "
        "(default: ./output/<script-name>/).",
    )
    workflow.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the segments and exit without contacting Resolve.",
    )
    workflow.add_argument(
        "--probe",
        action="store_true",
        help="Print a Resolve environment diagnostic report and exit.",
    )
    workflow.add_argument(
        "--skip-placement", action="store_true", help="Generate and cache only."
    )
    workflow.add_argument(
        "--place-only",
        action="store_true",
        help="Place already-generated segments without generating anything.",
    )
    workflow.add_argument(
        "--retry-failed", action="store_true", help="Re-attempt only failed segments."
    )
    workflow.add_argument(
        "--only",
        type=parse_segment_selection,
        metavar="LIST",
        help="Regenerate only these 1-based segments, e.g. 5,7-9. Placement is "
        "skipped so existing clips are never disturbed.",
    )
    workflow.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        metavar="N",
        help="Attempts per segment per run (default: %(default)s).",
    )
    workflow.add_argument(
        "--force-rechunk",
        action="store_true",
        help="Archive a mismatched manifest and start over.",
    )
    workflow.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed segment instead of continuing.",
    )
    workflow.add_argument(
        "--log-level",
        default=_env("RESOLVE_TTS_LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console verbosity (default: %(default)s). The log file is always DEBUG.",
    )
    return parser


# --------------------------------------------------------------------------
# Input selection
# --------------------------------------------------------------------------


def choose_script_file() -> str:
    """Ask for a script path: a tkinter dialog, or an ``input()`` prompt.

    Raises:
        UsageError: No path was supplied.
    """
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        try:
            selected = filedialog.askopenfilename(
                title="Choose a narration script",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )
        finally:
            root.destroy()
        if selected:
            return selected
        raise UsageError("No script file was chosen.")
    except UsageError:
        raise
    except Exception as exc:  # noqa: BLE001 - tkinter is optional everywhere
        LOGGER.debug("tkinter unavailable (%s: %s); falling back to a prompt.",
                     type(exc).__name__, exc)

    try:
        typed = input("Path to the narration script (.txt): ").strip().strip('"')
    except EOFError:
        raise UsageError(
            "No script file was given.",
            "  Usage: python generate_voice.py script.txt",
        )
    if not typed:
        raise UsageError(
            "No script file was given.",
            "  Usage: python generate_voice.py script.txt",
        )
    return typed


def build_options(namespace: argparse.Namespace, script_path: str) -> Options:
    """Turn parsed arguments into a validated :class:`Options`.

    Raises:
        UsageError: Two mutually exclusive flags were combined, or a path is bad.
    """
    if namespace.place_only and namespace.skip_placement:
        raise UsageError("--place-only and --skip-placement cannot be combined.")
    if namespace.place_only and namespace.retry_failed:
        raise UsageError("--place-only and --retry-failed cannot be combined.")
    if namespace.place_only and namespace.only:
        raise UsageError("--place-only and --only cannot be combined.")
    if namespace.place_only and namespace.placement == "resolve":
        raise UsageError(
            "--place-only cannot be combined with --placement resolve.",
            "  --placement resolve asks Resolve to place each clip as it is generated,\n"
            "  so there is nothing for --place-only to do. Drop --placement resolve:\n"
            "  the default append mode places the already generated segments.",
        )
    if not os.path.isfile(script_path):
        raise UsageError(
            "Script file not found: %s" % script_path,
            "  Pass the path to a UTF-8 .txt narration script.",
        )
    if namespace.max_attempts < 1:
        raise UsageError("--max-attempts must be at least 1.")
    if namespace.gap_frames < 0:
        raise UsageError("--gap-frames must not be negative.")
    if namespace.keep_pad_ms < 0:
        raise UsageError("--keep-pad-ms must not be negative.")
    if namespace.track_index is not None and namespace.track_index < 1:
        raise UsageError("--track-index is 1-based, so it must be at least 1.")
    if namespace.custom_voice_file and not os.path.isfile(namespace.custom_voice_file):
        raise UsageError(
            "Custom voice file not found: %s" % namespace.custom_voice_file
        )

    output_dir = namespace.output_dir
    if not output_dir:
        stem = os.path.splitext(os.path.basename(script_path))[0] or "script"
        output_dir = os.path.join("output", stem)

    return Options(
        script_path=script_path,
        output_dir=output_dir,
        max_chars=namespace.max_chars,
        merge_paragraphs=namespace.merge_paragraphs,
        voice=namespace.voice,
        custom_voice_file=namespace.custom_voice_file,
        speed=namespace.speed,
        variation=namespace.variation,
        pitch=namespace.pitch,
        generation_id=namespace.generation_id,
        track_name=namespace.track_name,
        track_index=namespace.track_index,
        bin_name=namespace.bin_name,
        start=namespace.start,
        gap_frames=namespace.gap_frames,
        trim_silence=namespace.trim_silence,
        silence_threshold_db=namespace.silence_threshold_db,
        keep_pad_ms=namespace.keep_pad_ms,
        placement=namespace.placement,
        dry_run=namespace.dry_run,
        skip_placement=namespace.skip_placement,
        place_only=namespace.place_only,
        retry_failed=namespace.retry_failed,
        only=namespace.only,
        max_attempts=namespace.max_attempts,
        force_rechunk=namespace.force_rechunk,
        stop_on_error=namespace.stop_on_error,
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _run_probe(console_level: int) -> int:
    configure_logging(None, console_level)
    context = connect(
        require_studio=False, require_project=False, require_timeline=False
    )
    print(probe_report(context))
    if not context.is_studio:
        print("")
        print(
            "WARNING: this is not DaVinci Resolve Studio; GenerateSpeech is "
            "unavailable in the free edition."
        )
    if context.project is not None and not context.has_generate_speech():
        print("")
        print(
            "WARNING: Project.GenerateSpeech is missing. It requires Resolve "
            "Studio 20.x or 21.x."
        )
    return 0


def _use_utf8_output() -> None:
    """Make stdout/stderr survive non-ASCII scripts, even when redirected.

    On Windows a redirected stream defaults to the ANSI code page, which raises
    ``UnicodeEncodeError`` on the CJK, emoji or curly-quote text a narration
    script may legitimately contain.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):  # pragma: no cover - already-closed stream
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line tool and return a process exit code."""
    _use_utf8_output()
    parser = build_parser()
    namespace = parser.parse_args(argv)
    console_level = getattr(logging, namespace.log_level)
    log_path: Optional[str] = None
    pipeline: Optional[Pipeline] = None

    try:
        if namespace.probe:
            return _run_probe(console_level)

        script_path = namespace.script or choose_script_file()
        options = build_options(namespace, script_path)
        if not options.dry_run:
            os.makedirs(options.output_dir, exist_ok=True)
            log_path = os.path.join(options.output_dir, "generate_voice.log")
        configure_logging(log_path, console_level)

        pipeline = Pipeline(options)
        result = pipeline.run()
        if not options.dry_run:
            for line in pipeline.summary_lines():
                print(line)
        return result.exit_code

    except ToolError as exc:
        logging.getLogger("resolve_tts").debug(
            "Expected failure", exc_info=True
        )
        print("")
        print("ERROR: %s" % exc.render(), file=sys.stderr)
        if pipeline is not None and pipeline.result.total:
            for line in pipeline.summary_lines():
                print(line)
        return exc.exit_code

    except KeyboardInterrupt:
        print("")
        print("Interrupted. Progress is saved in the manifest; re-run to resume.")
        return 130

    except Exception:  # noqa: BLE001 - last-resort handler
        logging.getLogger("resolve_tts").debug("Unexpected failure", exc_info=True)
        detail = traceback.format_exc().strip().splitlines()[-1]
        print("")
        print("ERROR: unexpected failure: %s" % detail, file=sys.stderr)
        if log_path:
            print("Full traceback: %s" % os.path.abspath(log_path), file=sys.stderr)
        else:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
