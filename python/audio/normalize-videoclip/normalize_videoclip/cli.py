"""Command-line interface.

All user-facing output (progress lines, the summary, JSON) lives here —
the library itself never prints. Exit codes are deliberate and documented
in README.md so the tool is safe to script around:

  0  everything succeeded (or there was nothing to do / dry run)
  1  one or more files failed to encode
  2  usage / argument error (argparse default) or invalid configuration
  3  ffmpeg is not installed / not on PATH
  4  the input directory does not exist
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .config import NormalizeConfig, load_config
from .core import discover_video_files, normalize_directory
from .exceptions import ConfigError, FFmpegNotFoundError, InvalidInputDirectoryError
from .logging_utils import setup_logging
from .models import TaskResult, TaskStatus

_STATUS_LABEL = {
    TaskStatus.SUCCESS: "done",
    TaskStatus.SKIPPED: "skipped (exists)",
    TaskStatus.FAILED: "FAILED",
    TaskStatus.DRY_RUN: "would encode",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="normalize-videoclip",
        description=(
            "Batch-normalize audio loudness across every video in a folder, "
            "re-encoding legacy containers (mpg/mpeg/webm) to MP4 along the way."
        ),
    )
    parser.add_argument("input_dir", type=Path, help="folder containing source video files")
    parser.add_argument("output_dir", type=Path, help="folder to write normalized output into")

    parser.add_argument("--config", type=Path, default=None, help="optional TOML config file")

    parser.add_argument("--crf", type=int, default=None, help="x264 CRF for re-encoded files (default: 18)")
    parser.add_argument("--preset", default=None, help="x264 preset for re-encoded files (default: medium)")
    parser.add_argument("--audio-bitrate", default=None, help="AAC bitrate (default: 192k)")
    parser.add_argument("--audio-channels", type=int, default=None, help="output audio channel count (default: 2)")
    parser.add_argument("--loudnorm-i", type=float, default=None, help="loudnorm integrated loudness target (default: -16.0)")
    parser.add_argument("--loudnorm-lra", type=float, default=None, help="loudnorm loudness range target (default: 11.0)")
    parser.add_argument("--loudnorm-tp", type=float, default=None, help="loudnorm true peak target (default: -1.5)")
    parser.add_argument("--extensions", default=None, help="comma-separated list of extensions to process, e.g. .mp4,.mkv")
    parser.add_argument(
        "-j", "--workers", type=int, default=None,
        help=f"parallel ffmpeg processes (default: 4; this machine has {os.cpu_count()} logical cores)",
    )
    parser.add_argument("--overwrite", action="store_true", default=None, help="re-encode even if the destination already exists")
    parser.add_argument("--dry-run", action="store_true", default=None, help="print the plan without invoking ffmpeg or writing files")
    parser.add_argument("--ffmpeg-path", default=None, help="path to the ffmpeg binary (default: ffmpeg on PATH)")

    parser.add_argument("--json", action="store_true", help="print a machine-readable JSON summary instead of text")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase log verbosity (-v, -vv)")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress progress lines; only errors and the final summary")
    parser.add_argument("--version", action="version", version=f"normalize-videoclip {__version__}")

    return parser


def _config_from_args(args: argparse.Namespace) -> NormalizeConfig:
    return load_config(
        args.config,
        video_crf=args.crf,
        video_preset=args.preset,
        audio_bitrate=args.audio_bitrate,
        audio_channels=args.audio_channels,
        loudnorm_i=args.loudnorm_i,
        loudnorm_lra=args.loudnorm_lra,
        loudnorm_tp=args.loudnorm_tp,
        extensions=args.extensions,
        workers=args.workers,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        ffmpeg_path=args.ffmpeg_path,
    )


def _format_result_line(index: int, total: int, result: TaskResult) -> str:
    label = _STATUS_LABEL[result.status]
    line = f"[{index}/{total}] {result.task.source.name} -> {label} ({result.duration_seconds:.1f}s)"
    if result.status == TaskStatus.FAILED and result.message:
        line += f"\n    {result.message.splitlines()[-1]}"
    return line


def _format_summary(summary) -> str:
    parts = [
        f"succeeded={summary.succeeded}",
        f"skipped={summary.skipped}",
        f"failed={summary.failed}",
    ]
    if summary.dry_run_count:
        parts.append(f"dry_run={summary.dry_run_count}")
    return "Done: " + ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose, args.quiet)

    try:
        config = _config_from_args(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        total = len(discover_video_files(args.input_dir, config))
    except InvalidInputDirectoryError as exc:
        print(str(exc), file=sys.stderr)
        return 4

    seen = 0

    def on_result(result: TaskResult) -> None:
        nonlocal seen
        seen += 1
        if not args.json and not args.quiet:
            print(_format_result_line(seen, total, result))

    try:
        summary = normalize_directory(args.input_dir, args.output_dir, config, on_result=on_result)
    except FFmpegNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except InvalidInputDirectoryError as exc:
        print(str(exc), file=sys.stderr)
        return 4

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
    elif not summary.results:
        print(f"No matching video files found in {args.input_dir}")
    else:
        print(_format_summary(summary))

    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
