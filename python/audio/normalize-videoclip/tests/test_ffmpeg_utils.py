from __future__ import annotations

from pathlib import Path

import pytest

from normalize_videoclip.config import NormalizeConfig
from normalize_videoclip.exceptions import FFmpegNotFoundError
from normalize_videoclip.ffmpeg_utils import build_ffmpeg_command, check_ffmpeg_available
from normalize_videoclip.models import VideoTask


def test_check_ffmpeg_available_raises_for_missing_binary():
    config = NormalizeConfig(ffmpeg_path="definitely-not-a-real-binary-xyz")
    with pytest.raises(FFmpegNotFoundError):
        check_ffmpeg_available(config)


def test_build_command_legacy_reencodes_video_and_audio():
    task = VideoTask(Path("in/old.webm"), Path("out/old.mp4"), requires_reencode=True)
    cmd = build_ffmpeg_command(task, NormalizeConfig())

    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "libx264"
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "aac"
    assert any("loudnorm=" in arg for arg in cmd)
    assert cmd[-1] == str(Path("out/old.mp4"))


def test_build_command_standard_stream_copies_video():
    task = VideoTask(Path("in/clip.mkv"), Path("out/clip.mkv"), requires_reencode=False)
    cmd = build_ffmpeg_command(task, NormalizeConfig())

    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    assert "-c:s" in cmd and cmd[cmd.index("-c:s") + 1] == "copy"
    assert "-map" in cmd and cmd[cmd.index("-map") + 1] == "0"


def test_build_command_respects_output_path_override():
    task = VideoTask(Path("in/clip.mkv"), Path("out/clip.mkv"), requires_reencode=False)
    cmd = build_ffmpeg_command(task, NormalizeConfig(), output_path=Path("out/.tmp123.mkv"))

    assert cmd[-1] == str(Path("out/.tmp123.mkv"))


def test_build_command_uses_config_quality_settings():
    task = VideoTask(Path("in/old.mpg"), Path("out/old.mp4"), requires_reencode=True)
    config = NormalizeConfig(video_crf=23, video_preset="fast", audio_bitrate="128k")
    cmd = build_ffmpeg_command(task, config)

    assert cmd[cmd.index("-crf") + 1] == "23"
    assert cmd[cmd.index("-preset") + 1] == "fast"
    assert cmd[cmd.index("-b:a") + 1] == "128k"
