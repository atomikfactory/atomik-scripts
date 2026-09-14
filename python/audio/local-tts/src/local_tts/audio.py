"""Audio helpers: load/save, mono, resample, trim, normalise, concatenate.

Everything works on :class:`AudioClip` objects holding mono ``float32``
samples in the range [-1, 1].  ``soundfile`` (libsndfile) handles WAV, FLAC,
OGG and - with recent libsndfile builds - MP3.  ``ffmpeg`` is used as an
optional fallback for anything else (M4A/AAC input, MP3 export, time-stretch).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import DependencyError, InputError, UserError

log = logging.getLogger(__name__)

SUPPORTED_OUTPUT_FORMATS = ("wav", "flac", "ogg", "mp3")
SOUNDFILE_INPUT_EXTS = {".wav", ".flac", ".ogg", ".oga", ".opus", ".aiff", ".aif", ".mp3", ".w64", ".caf"}
FFMPEG_ONLY_EXTS = {".m4a", ".aac", ".mp4", ".wma", ".webm", ".mkv", ".mov"}
AUDIO_EXTS = SOUNDFILE_INPUT_EXTS | FFMPEG_ONLY_EXTS

_SUBTYPES = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}


@dataclass
class AudioClip:
    samples: np.ndarray  # mono float32
    sample_rate: int

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples)
        if arr.ndim > 1:
            arr = to_mono(arr)
        self.samples = np.ascontiguousarray(arr, dtype=np.float32)
        self.sample_rate = int(self.sample_rate)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if len(self.samples) else 0.0

    def rms_dbfs(self) -> float:
        if not len(self.samples):
            return -np.inf
        rms = float(np.sqrt(np.mean(np.square(self.samples, dtype=np.float64))))
        return 20 * np.log10(rms) if rms > 0 else -np.inf


# --------------------------------------------------------------------------- #
# Basics
# --------------------------------------------------------------------------- #
def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average channels.  Accepts (n,), (n, ch) or (ch, n) arrays."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"Unsupported audio array shape {arr.shape}")
    # Channel axis is the small one.
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return arr.mean(axis=1, dtype=np.float32)


def silence(seconds: float, sample_rate: int) -> AudioClip:
    n = max(0, int(round(seconds * sample_rate)))
    return AudioClip(np.zeros(n, dtype=np.float32), sample_rate)


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def file_fingerprint(path: str | Path, extra: str = "") -> str:
    """Stable hash of a file's bytes (+ optional extra string)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    h.update(extra.encode("utf-8"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #
def _import_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise DependencyError("The 'soundfile' package is missing.", hints=["pip install -r requirements.txt"]) from exc
    return sf


def load_audio(path: str | Path) -> AudioClip:
    """Load any common audio file as mono float32 at its native sample rate."""
    p = Path(path)
    if not p.is_file():
        raise InputError(f"Audio file not found: {p}")
    sf = _import_soundfile()
    try:
        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception as exc:
        if p.suffix.lower() in FFMPEG_ONLY_EXTS or ffmpeg_path():
            return _load_via_ffmpeg(p)
        raise InputError(
            f"Could not read audio file '{p.name}': {exc}",
            hints=[
                "Supported without ffmpeg: WAV, FLAC, OGG/Opus, AIFF, MP3.",
                "Install ffmpeg (https://ffmpeg.org) to read M4A/AAC and other formats, "
                "or convert the file to WAV first.",
            ],
        ) from exc
    if data.size == 0:
        raise InputError(f"Audio file '{p.name}' contains no samples.")
    return AudioClip(to_mono(data), sr)


def _load_via_ffmpeg(p: Path) -> AudioClip:
    exe = ffmpeg_path()
    if not exe:
        raise InputError(
            f"'{p.suffix}' files need ffmpeg to be decoded and ffmpeg was not found on PATH.",
            hints=["Install ffmpeg (https://ffmpeg.org) or convert the file to WAV/FLAC."],
        )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "decoded.wav"
        cmd = [exe, "-v", "error", "-y", "-i", str(p), "-ac", "1", "-f", "wav", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise InputError(f"ffmpeg could not decode '{p.name}': {proc.stderr.strip()[:400]}")
        sf = _import_soundfile()
        data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    return AudioClip(to_mono(data), sr)


def save_audio(clip: AudioClip, path: str | Path, *, fmt: str | None = None, bit_depth: int = 16) -> Path:
    """Write *clip* to *path*.  Format is inferred from the extension unless given."""
    p = Path(path)
    fmt = (fmt or p.suffix.lstrip(".") or "wav").lower()
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise UserError(f"Unsupported output format '{fmt}'. Choose one of: {', '.join(SUPPORTED_OUTPUT_FORMATS)}")
    if p.suffix.lower() != f".{fmt}":
        p = p.with_suffix(f".{fmt}")
    p.parent.mkdir(parents=True, exist_ok=True)
    if bit_depth not in _SUBTYPES:
        raise UserError("--bit-depth must be 16, 24 or 32.")

    samples = np.clip(clip.samples, -1.0, 1.0)
    sf = _import_soundfile()
    if fmt == "wav":
        sf.write(str(p), samples, clip.sample_rate, subtype=_SUBTYPES[bit_depth])
    elif fmt == "flac":
        sf.write(str(p), samples, clip.sample_rate, subtype="PCM_24" if bit_depth >= 24 else "PCM_16")
    elif fmt == "ogg":
        sf.write(str(p), samples, clip.sample_rate, format="OGG", subtype="VORBIS")
    elif fmt == "mp3":
        _save_mp3(samples, clip.sample_rate, p)
    return p


def _save_mp3(samples: np.ndarray, sr: int, p: Path) -> None:
    sf = _import_soundfile()
    try:
        if "MP3" in sf.available_formats():
            sf.write(str(p), samples, sr, format="MP3")
            return
    except Exception as exc:  # pragma: no cover - depends on libsndfile build
        log.debug("soundfile MP3 export failed (%s); trying ffmpeg", exc)
    exe = ffmpeg_path()
    if not exe:
        raise DependencyError(
            "MP3 export needs either a libsndfile build with MP3 support or ffmpeg on PATH.",
            hints=["Install ffmpeg (https://ffmpeg.org) or use --format wav / flac / ogg."],
        )
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "tmp.wav"
        sf.write(str(wav), samples, sr, subtype="PCM_16")
        cmd = [exe, "-v", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-q:a", "2", str(p)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise UserError(f"ffmpeg failed to encode MP3: {proc.stderr.strip()[:400]}")


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #
def resample(clip: AudioClip, target_sr: int) -> AudioClip:
    """High-quality resampling via soxr, falling back to torchaudio."""
    if clip.sample_rate == target_sr:
        return clip
    try:
        import soxr

        out = soxr.resample(clip.samples, clip.sample_rate, target_sr, quality="VHQ")
        return AudioClip(np.asarray(out, dtype=np.float32), target_sr)
    except ImportError:
        pass
    try:
        import torch
        import torchaudio.functional as F

        tensor = torch.from_numpy(clip.samples).unsqueeze(0)
        out = F.resample(tensor, clip.sample_rate, target_sr)
        return AudioClip(out.squeeze(0).numpy(), target_sr)
    except ImportError as exc:
        raise DependencyError(
            "Resampling needs the 'soxr' package (or torchaudio).", hints=["pip install soxr"]
        ) from exc


def trim_silence(clip: AudioClip, *, threshold_db: float = -45.0, pad_ms: float = 60.0) -> AudioClip:
    """Remove leading/trailing silence below *threshold_db*, keeping *pad_ms* padding."""
    if not len(clip.samples):
        return clip
    thresh = 10 ** (threshold_db / 20)
    loud = np.flatnonzero(np.abs(clip.samples) > thresh)
    if loud.size == 0:
        return AudioClip(clip.samples[:0], clip.sample_rate)
    pad = int(pad_ms / 1000 * clip.sample_rate)
    start = max(0, int(loud[0]) - pad)
    end = min(len(clip.samples), int(loud[-1]) + pad + 1)
    return AudioClip(clip.samples[start:end], clip.sample_rate)


def truncate(clip: AudioClip, max_seconds: float) -> AudioClip:
    n = int(max_seconds * clip.sample_rate)
    if len(clip.samples) <= n:
        return clip
    return AudioClip(clip.samples[:n], clip.sample_rate)


def apply_fade(clip: AudioClip, ms: float = 5.0) -> AudioClip:
    """Short linear fade-in/out to avoid clicks at chunk joins."""
    n = int(ms / 1000 * clip.sample_rate)
    if n <= 0 or len(clip.samples) < 2 * n:
        return clip
    out = clip.samples.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return AudioClip(out, clip.sample_rate)


def peak_normalize(clip: AudioClip, peak_dbfs: float = -1.0) -> AudioClip:
    """Scale so the loudest sample sits at *peak_dbfs* (never boosts pure silence)."""
    peak = clip.peak
    if peak <= 0:
        return clip
    target = 10 ** (peak_dbfs / 20)
    return AudioClip(clip.samples * np.float32(target / peak), clip.sample_rate)


def concatenate(clips: list[AudioClip], gaps: list[float] | None = None) -> AudioClip:
    """Join clips in order, inserting ``gaps[i]`` seconds of silence after clip *i*."""
    if not clips:
        raise ValueError("No clips to concatenate")
    sr = clips[0].sample_rate
    gaps = gaps or [0.0] * len(clips)
    if len(gaps) != len(clips):
        raise ValueError("gaps must have one entry per clip")
    parts: list[np.ndarray] = []
    for clip, gap in zip(clips, gaps, strict=True):
        if clip.sample_rate != sr:
            clip = resample(clip, sr)
        parts.append(clip.samples)
        if gap > 0:
            parts.append(silence(gap, sr).samples)
    return AudioClip(np.concatenate(parts), sr)


def time_stretch(clip: AudioClip, speed: float) -> AudioClip:
    """Change tempo without changing pitch using ffmpeg's ``atempo`` filter."""
    if abs(speed - 1.0) < 1e-3:
        return clip
    if not 0.5 <= speed <= 2.0:
        raise UserError("--speed must be between 0.5 and 2.0.")
    exe = ffmpeg_path()
    if not exe:
        raise DependencyError(
            "Changing speed for this model is done as a post-process and needs ffmpeg on PATH.",
            hints=["Install ffmpeg (https://ffmpeg.org) or use a model with native speed control (kokoro)."],
        )
    sf = _import_soundfile()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / "out.wav"
        sf.write(str(src), clip.samples, clip.sample_rate, subtype="FLOAT")
        cmd = [exe, "-v", "error", "-y", "-i", str(src), "-filter:a", f"atempo={speed:.4f}", "-c:a", "pcm_f32le", str(dst)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise UserError(f"ffmpeg atempo failed: {proc.stderr.strip()[:400]}")
        data, sr = sf.read(str(dst), dtype="float32", always_2d=True)
    return AudioClip(to_mono(data), sr)
