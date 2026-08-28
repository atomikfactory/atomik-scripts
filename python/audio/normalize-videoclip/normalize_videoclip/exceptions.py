"""Exception hierarchy for normalize_videoclip.

Kept flat and small on purpose: every caller (CLI, MCP server, or a script
importing the library directly) needs to catch a short, predictable list of
error types rather than guessing which stdlib exception might leak out.
"""


class NormalizeVideoclipError(Exception):
    """Base class for all errors raised deliberately by this package."""


class FFmpegNotFoundError(NormalizeVideoclipError):
    """ffmpeg (or ffprobe) is not installed or not reachable on PATH."""


class InvalidInputDirectoryError(NormalizeVideoclipError):
    """The requested input directory does not exist or is not a directory."""


class ConfigError(NormalizeVideoclipError):
    """A configuration file or set of overrides could not be resolved."""
