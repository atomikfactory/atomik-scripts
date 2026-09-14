"""Common interface every TTS engine implements.

An *engine* wraps one model family.  The rest of the application only talks
to :class:`TTSEngine`, so adding a model means adding one module under
``local_tts/engines`` and one line in the registry (``engines/__init__.py``).

Engine modules must **not** import torch or model packages at import time:
``--list-models`` and the unit tests have to work without them.  Import heavy
dependencies inside ``load()``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from ..audio import AudioClip
from ..errors import DependencyError, VoiceError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """Static facts about a model, shown by ``--list-models`` and used for defaults."""

    id: str
    name: str
    family: str
    description: str
    license: str
    license_url: str
    homepage: str
    languages: tuple[str, ...]  # ISO-639-1 codes
    supports_cloning: bool
    has_preset_voices: bool
    has_default_voice: bool  # works with no --voice/--voice-sample at all
    native_speed: bool  # engine can change speaking rate itself
    cpu_supported: bool
    cpu_note: str
    approx_vram_gb: float
    download_gb: float
    sample_rate: int
    max_chunk_chars: int  # recommended text length per generate() call
    min_reference_seconds: float
    requirements_file: str
    notes: str = ""


@dataclass(frozen=True)
class PresetVoice:
    id: str
    name: str
    language: str
    description: str = ""


@dataclass
class VoiceRequest:
    """What the user asked for.  Exactly one of reference_audio / preset, or neither (default voice)."""

    reference_audio: Path | None = None  # pre-processed reference WAV
    reference_text: str | None = None  # transcript of the reference (optional, helps some models)
    preset: str | None = None
    language: str = "en"
    cache_dir: Path | None = None  # where the engine may store speaker embeddings
    options: dict = field(default_factory=dict)  # exaggeration, cfg_weight, temperature, speed, instruct...
    label: str = ""  # human readable, for logging

    def fingerprint(self) -> str:
        """Stable id of the voice configuration, used for the chunk cache."""
        from ..audio import file_fingerprint

        parts = {
            "ref": file_fingerprint(self.reference_audio) if self.reference_audio else None,
            "ref_text": self.reference_text,
            "preset": self.preset,
            "language": self.language,
            "options": {k: v for k, v in sorted(self.options.items())},
        }
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


class TTSEngine(ABC):
    """Base class for all engines."""

    info: ClassVar[ModelInfo]

    def __init__(self, device: str, models_dir: Path | None = None) -> None:
        self.device = device
        self.models_dir = models_dir
        self.voice: VoiceRequest | None = None
        self._loaded = False

    # -- lifecycle ----------------------------------------------------------- #
    @classmethod
    def missing_dependency(cls) -> str | None:
        """Return an explanation if the engine's Python package is not installed."""
        return None

    @classmethod
    def require_dependencies(cls) -> None:
        problem = cls.missing_dependency()
        if problem:
            raise DependencyError(
                f"{cls.info.name} is not installed: {problem}",
                hints=[
                    f"pip install -r {cls.info.requirements_file}",
                    "See README.md -> Installation for the full steps (PyTorch with CUDA first).",
                ],
            )

    @abstractmethod
    def load(self) -> None:
        """Download (first time) and load the model onto ``self.device``."""

    def unload(self) -> None:
        self._loaded = False

    # -- voices -------------------------------------------------------------- #
    def list_voices(self) -> list[PresetVoice]:
        return []

    def resolve_preset(self, name: str) -> PresetVoice:
        for v in self.list_voices():
            if v.id.lower() == name.lower():
                return v
        available = ", ".join(v.id for v in self.list_voices()) or "(none)"
        raise VoiceError(
            f"Voice '{name}' is not available for model '{self.info.id}'.",
            hints=[f"Available voices: {available}", "Run: python tts.py --list-voices --model " + self.info.id],
        )

    @abstractmethod
    def set_voice(self, voice: VoiceRequest) -> str:
        """Configure the voice for subsequent synthesize() calls. Returns a description."""

    # -- synthesis ----------------------------------------------------------- #
    @abstractmethod
    def synthesize(self, text: str, *, seed: int | None = None) -> AudioClip:
        """Generate speech for one chunk of text."""

    # -- helpers ------------------------------------------------------------- #
    @staticmethod
    def seed_everything(seed: int | None) -> None:
        if seed is None:
            return
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed % (2**32 - 1))
        except ImportError:  # pragma: no cover
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:  # pragma: no cover
            pass

    @staticmethod
    @contextmanager
    def quiet_progress_bars() -> Iterator[None]:
        """Hide the per-token tqdm bars and stray prints some models emit while generating.

        Our own progress lines are unaffected: the Reporter and the logging
        handler hold references to the real stdout/stderr.  ``--debug`` shows everything.
        """
        if log.isEnabledFor(logging.DEBUG):
            yield
            return
        previous = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        try:
            with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            if previous is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = previous

    def embedding_cache_path(self, voice: VoiceRequest, suffix: str = ".pt") -> Path | None:
        """Where a speaker embedding for *voice* may be cached (None if not cacheable)."""
        if voice.cache_dir is None or voice.reference_audio is None:
            return None
        from ..audio import file_fingerprint

        key = file_fingerprint(voice.reference_audio, voice.reference_text or "")
        folder = voice.cache_dir / "embeddings"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{self.info.id}__{key}{suffix}"


def check_import(*module_names: str) -> str | None:
    """Return None if all modules import, else a short message naming the first missing one."""
    import importlib.util

    for name in module_names:
        if importlib.util.find_spec(name) is None:
            return f"Python package '{name}' not found"
    return None
