"""Configuration for a normalization run.

A single frozen dataclass holds every tunable. Defaults reproduce the
original script's behavior exactly (CRF 18, medium preset, -16 LUFS
loudnorm, stereo 192k AAC) so upgrading is a no-op unless you opt into
something new.
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
from typing import Any

from .exceptions import ConfigError

# Containers ffmpeg can stream-copy video into unchanged; only audio is touched.
DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".mkv",
    ".mp4",
    ".avi",
    ".mov",
    ".mpg",
    ".mpeg",
    ".webm",
)

# Containers that cannot legally hold AAC audio (MPEG-PS: mp2/ac3/dts only;
# WebM: Vorbis/Opus only) and therefore must be fully re-encoded to MP4
# (H.264 + AAC) rather than stream-copied.
DEFAULT_LEGACY_EXTENSIONS: tuple[str, ...] = (".mpg", ".mpeg", ".webm")


@dataclasses.dataclass(frozen=True, slots=True)
class NormalizeConfig:
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    legacy_extensions: tuple[str, ...] = DEFAULT_LEGACY_EXTENSIONS

    video_crf: int = 18
    video_preset: str = "medium"

    audio_bitrate: str = "192k"
    audio_channels: int = 2

    loudnorm_i: float = -16.0
    loudnorm_lra: float = 11.0
    loudnorm_tp: float = -1.5

    workers: int = 4
    overwrite: bool = False
    dry_run: bool = False

    ffmpeg_path: str = "ffmpeg"

    def __post_init__(self) -> None:
        if isinstance(self.extensions, list):
            object.__setattr__(self, "extensions", tuple(self.extensions))
        if isinstance(self.legacy_extensions, list):
            object.__setattr__(self, "legacy_extensions", tuple(self.legacy_extensions))
        if self.workers < 1:
            raise ConfigError("workers must be >= 1")
        if not (0 <= self.video_crf <= 51):
            raise ConfigError("video_crf must be between 0 and 51")


def load_config(config_path: Path | str | None = None, **overrides: Any) -> NormalizeConfig:
    """Build a NormalizeConfig from an optional TOML file plus keyword overrides.

    Overrides take precedence over the file, and ``None``-valued overrides are
    ignored (so CLI flags that default to ``None`` never clobber a config file
    value just because the user didn't pass them).
    """
    data: dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except FileNotFoundError as exc:
            raise ConfigError(f"config file not found: {path}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    merged = {**data, **{k: v for k, v in overrides.items() if v is not None}}

    valid_fields = {f.name for f in dataclasses.fields(NormalizeConfig)}
    unknown = set(merged) - valid_fields
    if unknown:
        raise ConfigError(f"unknown config key(s): {', '.join(sorted(unknown))}")

    for key in ("extensions", "legacy_extensions"):
        if isinstance(merged.get(key), str):
            merged[key] = tuple(part.strip() for part in merged[key].split(",") if part.strip())

    try:
        return NormalizeConfig(**merged)
    except TypeError as exc:
        raise ConfigError(str(exc)) from exc
