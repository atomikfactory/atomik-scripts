"""normalize_videoclip: batch-normalize audio loudness across a folder of video files.

Public API surface. The library is intentionally print-free and side-effect-free
except for filesystem reads/writes explicitly requested by the caller — this is
what lets the same core power a CLI, a library import, and an MCP server without
duplicating logic.
"""

from .config import NormalizeConfig, load_config
from .core import discover_video_files, normalize_directory, plan_tasks
from .exceptions import (
    ConfigError,
    FFmpegNotFoundError,
    InvalidInputDirectoryError,
    NormalizeVideoclipError,
)
from .models import NormalizeSummary, TaskResult, TaskStatus, VideoTask

__version__ = "1.0.0"

__all__ = [
    "NormalizeConfig",
    "load_config",
    "discover_video_files",
    "plan_tasks",
    "normalize_directory",
    "NormalizeSummary",
    "TaskResult",
    "TaskStatus",
    "VideoTask",
    "NormalizeVideoclipError",
    "FFmpegNotFoundError",
    "InvalidInputDirectoryError",
    "ConfigError",
    "__version__",
]
