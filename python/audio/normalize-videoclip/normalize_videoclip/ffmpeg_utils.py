"""ffmpeg command construction and environment checks.

Isolated from core.py so the exact command line — the part most likely to
need tweaking (codecs, flags, filter graph) — lives in one obvious place.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import NormalizeConfig
from .exceptions import FFmpegNotFoundError
from .models import VideoTask


def check_ffmpeg_available(config: NormalizeConfig) -> None:
    """Raise FFmpegNotFoundError with an actionable message if ffmpeg is missing.

    Run once per batch (not per file) — cheap enough not to matter, and it
    turns a confusing FileNotFoundError deep inside a worker thread into a
    single clear failure before any work starts.
    """
    try:
        subprocess.run(
            [config.ffmpeg_path, "-version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(
            f"'{config.ffmpeg_path}' was not found on PATH. "
            "Install ffmpeg (https://ffmpeg.org/download.html) and make sure "
            "it's on your PATH, or pass --ffmpeg-path / ffmpeg_path=... "
            "pointing at the binary."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise FFmpegNotFoundError(
            f"'{config.ffmpeg_path} -version' exited with an error; "
            "the binary at that path may be broken."
        ) from exc


def _loudnorm_filter(config: NormalizeConfig) -> str:
    return f"loudnorm=I={config.loudnorm_i}:LRA={config.loudnorm_lra}:TP={config.loudnorm_tp}"


def build_ffmpeg_command(
    task: VideoTask,
    config: NormalizeConfig,
    output_path: Path | None = None,
) -> list[str]:
    """Build the ffmpeg argv for one task.

    ``output_path`` lets the caller redirect output to a temp file (for
    atomic writes) while keeping the codec/format decisions tied to
    ``task.requires_reencode``. ``-y`` is required because atomic writes
    pre-create the temp file via ``tempfile.mkstemp``.
    """
    out = output_path if output_path is not None else task.destination
    loudnorm = _loudnorm_filter(config)

    if task.requires_reencode:
        # Legacy container (mpg/mpeg/webm): can't hold AAC, so re-encode
        # video + all audio tracks to a Plex/browser-friendly MP4.
        return [
            config.ffmpeg_path,
            "-y",
            "-i", str(task.source),
            "-map", "0:v:0",
            "-map", "0:a?",
            "-map_metadata", "0",
            "-c:v", "libx264",
            "-preset", config.video_preset,
            "-crf", str(config.video_crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", config.audio_bitrate,
            "-ac", str(config.audio_channels),
            "-af", loudnorm,
            "-movflags", "+faststart",
            "-hide_banner",
            "-loglevel", "error",
            str(out),
        ]

    # Everything else: keep video (and subtitles) bit-identical, only touch
    # audio. -map 0 (instead of relying on ffmpeg's default stream picker)
    # preserves every audio/subtitle track instead of silently dropping all
    # but the first of each. -dn drops data streams (e.g. GoPro metadata)
    # that can't be stream-copied and would otherwise abort the whole job.
    return [
        config.ffmpeg_path,
        "-y",
        "-i", str(task.source),
        "-map", "0",
        "-dn",
        "-map_metadata", "0",
        "-c:v", "copy",
        "-c:s", "copy",
        "-c:a", "aac",
        "-b:a", config.audio_bitrate,
        "-ac", str(config.audio_channels),
        "-af", loudnorm,
        "-hide_banner",
        "-loglevel", "error",
        str(out),
    ]
