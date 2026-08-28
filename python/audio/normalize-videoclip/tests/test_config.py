from __future__ import annotations

import pytest

from normalize_videoclip.config import NormalizeConfig, load_config
from normalize_videoclip.exceptions import ConfigError


def test_defaults_match_original_script_behavior():
    config = NormalizeConfig()
    assert config.video_crf == 18
    assert config.video_preset == "medium"
    assert config.audio_bitrate == "192k"
    assert config.audio_channels == 2
    assert config.loudnorm_i == -16.0
    assert ".webm" in config.legacy_extensions
    assert ".mp4" in config.extensions


def test_negative_workers_rejected():
    with pytest.raises(ConfigError):
        NormalizeConfig(workers=0)


def test_out_of_range_crf_rejected():
    with pytest.raises(ConfigError):
        NormalizeConfig(video_crf=99)


def test_load_config_applies_overrides_and_ignores_none():
    config = load_config(None, video_crf=20, video_preset=None)
    assert config.video_crf == 20
    assert config.video_preset == "medium"  # untouched, since override was None


def test_load_config_rejects_unknown_key():
    with pytest.raises(ConfigError):
        load_config(None, not_a_real_field=123)


def test_load_config_from_toml_file(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'video_crf = 22\nworkers = 8\nextensions = ".mp4,.mkv"\n',
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.video_crf == 22
    assert config.workers == 8
    assert config.extensions == (".mp4", ".mkv")


def test_load_config_cli_overrides_beat_toml_file(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("video_crf = 22\n", encoding="utf-8")
    config = load_config(config_path, video_crf=30)
    assert config.video_crf == 30


def test_load_config_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.toml")
