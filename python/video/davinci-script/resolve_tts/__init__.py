"""Long-form narration to DaVinci Resolve AI Speech Generator.

This package splits a long narration script into chunks that fit Resolve's
350-character ``GenerateSpeech`` limit, generates each chunk with Resolve's
native AI Speech Generator, caches the results so re-runs resume, and places
the clips back-to-back on a dedicated audio track.

Only the Python standard library is used.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "ToolError",
    "UsageError",
    "ResolveError",
    "PartialFailure",
    "PlacementAborted",
    "MANIFEST_LIMIT",
]

__version__ = "1.0.0"

#: Hard limit imposed by Resolve on ``speechGenerationSettings["TextInput"]``.
MANIFEST_LIMIT = 350


class ToolError(Exception):
    """Base class for expected, user-facing failures.

    Expected failures are reported as a clean one-line message plus guidance;
    a traceback is only ever written to the log file.
    """

    #: Process exit code associated with this class of failure.
    exit_code: int = 1

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def render(self) -> str:
        """Return the multi-line message to show the user."""
        if self.hint:
            return "{0}\n{1}".format(self.message, self.hint)
        return self.message


class UsageError(ToolError):
    """Bad or missing input supplied by the user (exit code 1)."""

    exit_code = 1


class ResolveError(ToolError):
    """Resolve could not be reached or failed validation (exit code 2)."""

    exit_code = 2


class PartialFailure(ToolError):
    """Some segments could not be generated (exit code 3)."""

    exit_code = 3


class PlacementAborted(ToolError):
    """Timeline placement was refused to protect existing clips (exit code 4)."""

    exit_code = 4
