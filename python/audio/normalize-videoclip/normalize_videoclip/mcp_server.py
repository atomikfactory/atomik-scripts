"""MCP adapter: exposes the core library to MCP-compatible agents (Claude Code, etc).

This module is a thin translation layer only — it imports the same
core.py the CLI uses and adds no normalization logic of its own. That's
deliberate: the MCP server should never be able to do anything the CLI
can't, and behavior can't drift between the two surfaces.

Three tools, split by blast radius:

  check_environment   read-only, no filesystem access beyond running `ffmpeg -version`
  plan_normalization   read-only, lists what would happen, writes nothing
  normalize_videos      mutating, writes files to output_dir

An agent can always call plan_normalization first to preview a batch before
committing to normalize_videos — the same directory, config, and matching
logic both tools share guarantees the preview matches what actually runs.

Runs over stdio (local only, no network exposure). The server has exactly
the filesystem access of the user account running it: any directory you can
already read/write from a terminal, an agent can ask this server to
read/write. There is no path sandboxing — that's a deliberate scope
decision for a single-user local tool, documented in docs/DESIGN.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .config import NormalizeConfig
from .core import discover_video_files, normalize_directory, plan_tasks
from .exceptions import FFmpegNotFoundError, InvalidInputDirectoryError, NormalizeVideoclipError
from .ffmpeg_utils import check_ffmpeg_available

logging.basicConfig(level=logging.WARNING)

mcp = FastMCP("normalize-videoclip")


@mcp.tool()
def check_environment() -> dict[str, Any]:
    """Check whether ffmpeg is installed and reachable, before planning a batch."""
    try:
        check_ffmpeg_available(NormalizeConfig())
        return {"ffmpeg_available": True}
    except FFmpegNotFoundError as exc:
        return {"ffmpeg_available": False, "error": str(exc)}


@mcp.tool()
def plan_normalization(input_dir: str, output_dir: str, overwrite: bool = False) -> dict[str, Any]:
    """Preview what normalize_videos would do for a directory. Read-only; writes nothing.

    Returns the list of discovered source files, their planned destination
    paths, whether each requires a full re-encode (legacy container) or a
    fast stream-copy, and whether the destination already exists.
    """
    config = NormalizeConfig(overwrite=overwrite)
    try:
        files = discover_video_files(Path(input_dir), config)
    except InvalidInputDirectoryError as exc:
        return {"error": str(exc)}

    tasks = plan_tasks(files, Path(output_dir), config)
    return {
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "file_count": len(tasks),
        "tasks": [
            {**t.to_dict(), "already_exists": t.destination.exists()}
            for t in tasks
        ],
    }


@mcp.tool()
def normalize_videos(
    input_dir: str,
    output_dir: str,
    crf: int = 18,
    preset: str = "medium",
    audio_bitrate: str = "192k",
    workers: int = 4,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Normalize audio loudness across every video in input_dir, writing results to output_dir.

    Legacy containers (.mpg, .mpeg, .webm) are fully re-encoded to MP4
    (H.264 + AAC) since they can't hold AAC audio; everything else is
    stream-copied and only re-encoded on the audio track. Files whose
    destination already exists are skipped unless overwrite=True. Set
    dry_run=True to get the same summary shape without touching ffmpeg or
    the filesystem — useful to confirm a plan before committing to it.
    """
    config = NormalizeConfig(
        video_crf=crf,
        video_preset=preset,
        audio_bitrate=audio_bitrate,
        workers=workers,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    try:
        summary = normalize_directory(input_dir, output_dir, config)
    except (FFmpegNotFoundError, InvalidInputDirectoryError, NormalizeVideoclipError) as exc:
        return {"error": str(exc)}
    return summary.to_dict()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
