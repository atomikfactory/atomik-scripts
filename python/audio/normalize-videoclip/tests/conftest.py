from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

requires_ffmpeg = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not found on PATH")


@pytest.fixture
def tiny_video_factory(tmp_path):
    """Return a function that generates a short synthetic video file via ffmpeg.

    Real ffmpeg-backed tests need real (if trivial) video content rather than
    an empty file with the right extension, so encode paths are exercised
    for real. Callers that don't need ffmpeg (discovery/planning tests) skip
    this fixture entirely and just touch() empty files instead.
    """

    def _make(name: str, video_codec: str = "libx264", audio_codec: str = "aac") -> Path:
        path = tmp_path / name
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", video_codec,
            "-c:a", audio_codec,
            "-hide_banner",
            "-loglevel", "error",
            str(path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return path

    return _make
