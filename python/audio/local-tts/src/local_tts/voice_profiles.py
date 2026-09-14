"""Reusable local voice profiles.

A profile is just a folder inside ``voices/``::

    voices/
    └── my_voice/
        ├── reference.wav      # required: the reference recording (any common format)
        ├── reference.txt      # optional: exact transcript of the recording
        ├── profile.json       # optional: {"language": "en", "description": "...", "options": {...}}
        └── .cache/            # generated: pre-processed audio + per-model embeddings

Nothing in ``voices/`` is committed to Git (see ``.gitignore``) because
reference recordings are personal data.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import audio
from .errors import VoiceError

log = logging.getLogger(__name__)

PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
REFERENCE_STEMS = ("reference", "sample", "voice")
TRANSCRIPT_NAMES = ("reference.txt", "transcript.txt", "sample.txt")
PROFILE_FILE = "profile.json"
CACHE_DIRNAME = ".cache"

# Practical guidance (documented in README -> Voice Cloning).
MIN_REFERENCE_SECONDS = 3.0
RECOMMENDED_MIN_SECONDS = 8.0
RECOMMENDED_MAX_SECONDS = 20.0
DEFAULT_MAX_REFERENCE_SECONDS = 30.0


@dataclass
class VoiceProfile:
    name: str
    path: Path
    reference_audio: Path | None
    transcript: str | None = None
    language: str | None = None
    description: str | None = None
    options: dict = field(default_factory=dict)

    @property
    def cache_dir(self) -> Path:
        return self.path / CACHE_DIRNAME

    @property
    def is_valid(self) -> bool:
        return self.reference_audio is not None


def _find_reference(folder: Path) -> Path | None:
    candidates = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in audio.AUDIO_EXTS
    ]
    if not candidates:
        return None
    for stem in REFERENCE_STEMS:
        for p in candidates:
            if p.stem.lower() == stem:
                return p
    return candidates[0]


def _read_profile(folder: Path) -> VoiceProfile:
    reference = _find_reference(folder)
    transcript = None
    for name in TRANSCRIPT_NAMES:
        t = folder / name
        if t.is_file():
            transcript = t.read_text(encoding="utf-8-sig").strip() or None
            break
    meta: dict = {}
    pj = folder / PROFILE_FILE
    if pj.is_file():
        try:
            meta = json.loads(pj.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log.warning("Ignoring malformed %s in profile '%s': %s", PROFILE_FILE, folder.name, exc)
    return VoiceProfile(
        name=folder.name,
        path=folder,
        reference_audio=reference,
        transcript=transcript or meta.get("transcript"),
        language=meta.get("language"),
        description=meta.get("description"),
        options=dict(meta.get("options") or {}),
    )


def list_profiles(voices_dir: Path) -> list[VoiceProfile]:
    """All profile folders inside *voices_dir* (including ones missing audio)."""
    if not voices_dir.is_dir():
        return []
    profiles = []
    for folder in sorted(voices_dir.iterdir()):
        if folder.is_dir() and not folder.name.startswith(".") and not folder.name.startswith("_"):
            profiles.append(_read_profile(folder))
    return profiles


def load_profile(voices_dir: Path, name: str) -> VoiceProfile:
    folder = voices_dir / name
    if not folder.is_dir():
        available = [p.name for p in list_profiles(voices_dir) if p.is_valid]
        hints = [f"Available profiles: {', '.join(available)}"] if available else [
            f"No profiles found in {voices_dir}. Create one with: "
            "python tts.py --save-voice-profile NAME --voice-sample path/to/recording.wav"
        ]
        raise VoiceError(f"Voice profile '{name}' not found.", hints=hints)
    profile = _read_profile(folder)
    if not profile.is_valid:
        raise VoiceError(
            f"Voice profile '{name}' has no reference recording.",
            hints=[f"Put a file named reference.wav (or .mp3/.flac/...) into {folder}"],
        )
    return profile


def validate_reference_audio(path: Path, *, min_seconds: float = MIN_REFERENCE_SECONDS) -> audio.AudioClip:
    """Load and sanity-check a reference recording.  Returns the clip."""
    try:
        clip = audio.load_audio(path)
    except VoiceError:
        raise
    except Exception as exc:
        raise VoiceError(f"Invalid reference audio '{path}': {exc}") from exc
    if clip.duration < min_seconds:
        raise VoiceError(
            f"Reference audio is too short ({clip.duration:.1f}s). At least {min_seconds:g}s are required; "
            f"{RECOMMENDED_MIN_SECONDS:.0f}-{RECOMMENDED_MAX_SECONDS:.0f}s of clean speech is recommended."
        )
    if clip.peak < 0.01:
        raise VoiceError(f"Reference audio '{path.name}' is silent or extremely quiet.")
    if clip.duration > 120:
        log.warning("Reference audio is %.0fs long; only the beginning is used by the models.", clip.duration)
    return clip


def create_profile(
    voices_dir: Path,
    name: str,
    sample_path: Path,
    *,
    transcript: str | None = None,
    language: str | None = None,
    description: str | None = None,
    overwrite: bool = False,
) -> VoiceProfile:
    """Create ``voices/<name>/`` from a recording.  The original file is copied untouched."""
    if not PROFILE_NAME_RE.match(name):
        raise VoiceError(
            f"Invalid profile name '{name}'. Use letters, digits, '-' and '_' only (max 64 chars)."
        )
    sample_path = Path(sample_path)
    if not sample_path.is_file():
        raise VoiceError(f"Voice sample not found: {sample_path}")
    clip = validate_reference_audio(sample_path)
    folder = voices_dir / name
    if folder.exists() and any(folder.iterdir()) and not overwrite:
        raise VoiceError(f"Profile '{name}' already exists in {folder}.", hints=["Use --overwrite to replace it."])
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():
        if old.is_file() and old.stem.lower() in REFERENCE_STEMS:
            old.unlink()
    shutil.rmtree(folder / CACHE_DIRNAME, ignore_errors=True)

    ext = sample_path.suffix.lower() or ".wav"
    target = folder / f"reference{ext}"
    shutil.copy2(sample_path, target)
    if transcript:
        (folder / "reference.txt").write_text(transcript.strip() + "\n", encoding="utf-8")
    meta = {k: v for k, v in {"language": language, "description": description}.items() if v}
    meta["source_duration_seconds"] = round(clip.duration, 2)
    meta["source_sample_rate"] = clip.sample_rate
    (folder / PROFILE_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log.info("Saved voice profile '%s' (%.1fs @ %d Hz) -> %s", name, clip.duration, clip.sample_rate, folder)
    return _read_profile(folder)


def prepare_reference(
    source: Path,
    cache_dir: Path,
    *,
    preprocess: bool = True,
    max_seconds: float = DEFAULT_MAX_REFERENCE_SECONDS,
    trim: bool = True,
    min_seconds: float = MIN_REFERENCE_SECONDS,
) -> Path:
    """Return a clean mono WAV version of *source* suitable for the models.

    Processing is deliberately conservative: convert to mono, trim leading /
    trailing silence, cap the length.  Sample rate and loudness are left alone
    (each model resamples/normalises itself).  The original file is never
    modified; results are cached in *cache_dir* keyed by content hash.
    """
    source = Path(source)
    clip = validate_reference_audio(source, min_seconds=max(MIN_REFERENCE_SECONDS, min_seconds))
    if not preprocess:
        if source.suffix.lower() == ".wav":
            return source
        # Models expect a readable file; formats like m4a are re-encoded losslessly.
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"ref_{audio.file_fingerprint(source, 'raw')}.wav"
        if not target.exists():
            audio.save_audio(clip, target, bit_depth=32)
        return target

    key = audio.file_fingerprint(source, f"prep|trim={trim}|max={max_seconds}|min={min_seconds}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"ref_{key}.wav"
    if target.exists():
        return target

    processed = clip
    if trim:
        processed = audio.trim_silence(processed, threshold_db=-45.0, pad_ms=150.0)
        if processed.duration < max(MIN_REFERENCE_SECONDS, min_seconds):
            processed = clip  # trimming was too aggressive (noisy file) -> keep original
    if processed.duration > max_seconds:
        log.info("Reference is %.1fs; using the first %.0fs (models only use the beginning).", processed.duration, max_seconds)
        processed = audio.truncate(processed, max_seconds)
    audio.save_audio(processed, target, bit_depth=32)
    return target
