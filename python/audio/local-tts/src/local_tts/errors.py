"""Exception types that carry user-facing messages.

Anything derived from :class:`UserError` is printed as a short, readable
message (no traceback) unless ``--debug`` is enabled.  Everything else is a
genuine bug and gets the full traceback.
"""

from __future__ import annotations


class UserError(Exception):
    """A problem the user can fix (bad input, missing file, wrong option...)."""

    exit_code = 1

    def __init__(self, message: str, hints: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hints = hints or []

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message

    def format(self) -> str:
        lines = [f"Error: {self.message}"]
        for hint in self.hints:
            lines.append(f"  - {hint}")
        return "\n".join(lines)


class InputError(UserError):
    """Invalid or missing input file / text."""


class VoiceError(UserError):
    """Unknown voice, invalid reference audio, bad voice profile."""


class ModelError(UserError):
    """Model not found, not installed, failed to load."""

    exit_code = 2


class DeviceError(UserError):
    """CUDA requested but unavailable, unsupported device string..."""

    exit_code = 2


class DependencyError(UserError):
    """A required Python package is missing for the selected engine."""

    exit_code = 2


class OutOfMemoryError(UserError):
    """GPU ran out of memory."""

    exit_code = 3


class GenerationError(UserError):
    """The model failed to produce usable audio for a chunk."""

    exit_code = 4


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True if *exc* looks like a CUDA out-of-memory error."""
    name = type(exc).__name__
    if name == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def oom_hints(model_id: str | None = None) -> list[str]:
    hints = [
        "Close other applications that use the GPU (browsers, games, other AI tools).",
        "Try a smaller model, e.g. --model chatterbox-nano or --model kokoro (see --list-models).",
        "Use shorter chunks: --max-chunk-chars 150",
        "Fall back to the CPU (slow but works): --device cpu",
    ]
    if model_id and model_id.startswith("qwen3"):
        hints.insert(1, "Use the 0.6B Qwen3-TTS model instead of the 1.7B one: --model qwen3-tts")
    return hints
