"""Project paths, defaults and (optional) ``.env`` loading.

Nothing here is machine specific.  All paths are relative to the project
root (the folder that contains ``tts.py``) unless overridden through
environment variables or CLI options.

Supported environment variables (see ``.env.example``):

``LOCAL_TTS_MODEL``        default model id
``LOCAL_TTS_DEVICE``       auto | cuda | cpu
``LOCAL_TTS_LANGUAGE``     default language code (e.g. en)
``LOCAL_TTS_MODELS_DIR``   where downloaded model weights are cached
``LOCAL_TTS_VOICES_DIR``   where voice profiles live
``LOCAL_TTS_OUTPUT_DIR``   default output folder
``LOCAL_TTS_CACHE_DIR``    chunk cache (lets an interrupted run resume)
``HF_HUB_OFFLINE``         set to 1 to forbid any network access by huggingface_hub
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = "chatterbox-turbo"
DEFAULT_DEVICE = "auto"
DEFAULT_LANGUAGE = "en"
DEFAULT_OUTPUT_NAME = "voiceover.wav"

ENV_PREFIX = "LOCAL_TTS_"


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ``.

    Tiny, dependency-free replacement for python-dotenv.  Lines starting
    with ``#`` and blank lines are ignored; surrounding quotes are stripped.
    Existing environment variables win unless *override* is True.
    Returns the values that were read (useful for tests).
    """
    path = path or (PROJECT_ROOT / ".env")
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def _env(name: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(ENV_PREFIX + name)
    if not raw:
        return default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass
class Settings:
    """Resolved project settings (defaults <- .env <- environment)."""

    project_root: Path = PROJECT_ROOT
    model: str = DEFAULT_MODEL
    device: str = DEFAULT_DEVICE
    language: str = DEFAULT_LANGUAGE
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    voices_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "voices")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / ".cache")

    @classmethod
    def from_env(cls, dotenv: bool = True) -> Settings:
        if dotenv:
            load_dotenv()
        return cls(
            model=_env("MODEL", DEFAULT_MODEL),
            device=_env("DEVICE", DEFAULT_DEVICE),
            language=_env("LANGUAGE", DEFAULT_LANGUAGE),
            models_dir=_env_path("MODELS_DIR", PROJECT_ROOT / "models"),
            voices_dir=_env_path("VOICES_DIR", PROJECT_ROOT / "voices"),
            output_dir=_env_path("OUTPUT_DIR", PROJECT_ROOT / "output"),
            cache_dir=_env_path("CACHE_DIR", PROJECT_ROOT / ".cache"),
        )


def configure_model_cache(models_dir: Path) -> None:
    """Point huggingface_hub at *models_dir* so weights are stored in the project.

    Must run *before* ``huggingface_hub`` / any engine module is imported.
    If the user already set ``HF_HOME`` or ``HF_HUB_CACHE`` we respect that.
    """
    if "HF_HOME" in os.environ or "HF_HUB_CACHE" in os.environ:
        return
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(models_dir)
    # Silence the symlink warning on Windows; files are simply copied.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
