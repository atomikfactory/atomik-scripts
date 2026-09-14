"""End-to-end generation: text -> chunks -> model -> stitched audio file.

::

    script text
      -> chunk_text()                       (text_processing)
      -> engine.load() / engine.set_voice() (engines)
      -> per chunk: cache lookup | synthesize (+ retries & sanity checks)
      -> trim + fade + gaps -> concatenate  (audio)
      -> optional speed / resample / normalise
      -> save_audio() + optional manifest
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from . import audio
from .audio import AudioClip
from .devices import DeviceInfo, free_cuda_memory, resolve_device
from .engines import TTSEngine, VoiceRequest, get_engine_class
from .errors import GenerationError, OutOfMemoryError, UserError, is_cuda_oom, oom_hints
from .text_processing import TextChunk, chunk_text, estimate_seconds

log = logging.getLogger(__name__)

CACHE_VERSION = "v1"


@dataclass
class GenerationOptions:
    model_id: str
    output_path: Path
    voice: VoiceRequest = field(default_factory=VoiceRequest)
    device: str = "auto"
    models_dir: Path | None = None
    cache_dir: Path | None = None  # None disables the resumable chunk cache
    max_chunk_chars: int | None = None  # None -> model default
    newline_is_break: bool = False
    sentence_gap: float = 0.25  # seconds of silence between chunks inside a paragraph
    paragraph_gap: float = 0.6  # seconds of silence between paragraphs
    trim_chunks: bool = True
    normalize: bool = True
    peak_dbfs: float = -1.0
    speed: float = 1.0
    seed: int | None = None
    retries: int = 2
    output_format: str = "wav"
    bit_depth: int = 16
    output_sample_rate: int | None = None
    write_manifest: bool = False


@dataclass
class ChunkResult:
    chunk: TextChunk
    clip: AudioClip
    cached: bool
    seconds: float  # wall-clock generation time
    attempts: int
    warning: str | None = None
    start: float = 0.0  # position in the final file
    end: float = 0.0


@dataclass
class GenerationResult:
    output_path: Path
    duration: float
    sample_rate: int
    chunks: list[ChunkResult]
    elapsed: float
    model_id: str
    device: DeviceInfo
    voice_description: str
    manifest_path: Path | None = None

    @property
    def cached_chunks(self) -> int:
        return sum(1 for c in self.chunks if c.cached)


class Reporter:
    """Plain-text progress output.  Replace with a silent instance in tests."""

    def __init__(self, stream: TextIO | None = None, quiet: bool = False, verbose: bool = False) -> None:
        self.stream = stream or sys.stdout
        self.quiet = quiet
        self.verbose = verbose

    def _emit(self, message: str, end: str = "\n") -> None:
        if not self.quiet:
            print(message, file=self.stream, end=end, flush=True)

    def stage(self, message: str) -> None:
        self._emit(message)

    def info(self, message: str) -> None:
        self._emit(message)

    def chunk_start(self, index: int, total: int, chunk: TextChunk) -> None:
        width = len(str(total))
        if self.verbose:
            self._emit(f"      | {chunk.text}")
        self._emit(f"[{index + 1:>{width}}/{total}] Generating ({len(chunk.text)} chars)... ", end="")

    def chunk_done(self, index: int, total: int, result: ChunkResult) -> None:
        status = "cached" if result.cached else f"{result.seconds:.1f}s"
        extra = f"  WARNING: {result.warning}" if result.warning else ""
        self._emit(f"done ({status} -> {result.clip.duration:.1f}s audio){extra}")


# --------------------------------------------------------------------------- #
# Engine setup
# --------------------------------------------------------------------------- #
def build_engine(options: GenerationOptions) -> tuple[TTSEngine, DeviceInfo]:
    engine_cls = get_engine_class(options.model_id)
    engine_cls.require_dependencies()
    device = resolve_device(options.device, cpu_supported=engine_cls.info.cpu_supported, model_label=engine_cls.info.name)
    return engine_cls(device.device, options.models_dir), device


# --------------------------------------------------------------------------- #
# Chunk cache
# --------------------------------------------------------------------------- #
def chunk_cache_key(model_id: str, voice_fingerprint: str, text: str, seed: int | None) -> str:
    payload = json.dumps([CACHE_VERSION, model_id, voice_fingerprint, text, seed], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _cache_path(cache_dir: Path | None, key: str) -> Path | None:
    return (cache_dir / "chunks" / f"{key}.wav") if cache_dir else None


def clear_chunk_cache(cache_dir: Path) -> int:
    folder = cache_dir / "chunks"
    if not folder.is_dir():
        return 0
    n = 0
    for f in folder.glob("*.wav"):
        f.unlink()
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Synthesis with retries
# --------------------------------------------------------------------------- #
def looks_suspicious(clip: AudioClip, text: str) -> str | None:
    """Heuristic detection of runaway / truncated generations."""
    expected = estimate_seconds(text)
    if clip.duration < 0.15 or clip.peak < 1e-4:
        return "model returned (almost) no audio"
    if clip.duration > expected * 2.5 + 3.0:
        return f"audio is {clip.duration:.1f}s but ~{expected:.0f}s were expected (possible hallucination)"
    if len(text) > 40 and clip.duration < expected * 0.3:
        return f"audio is only {clip.duration:.1f}s for ~{expected:.0f}s of text (possible truncation)"
    return None


def _chunk_seed(base_seed: int | None, chunk_index: int, attempt: int) -> int | None:
    if base_seed is None:
        return None
    return (base_seed + chunk_index + attempt * 7919) % (2**31 - 1)


def synthesize_chunk(engine: TTSEngine, chunk: TextChunk, *, retries: int, base_seed: int | None) -> ChunkResult:
    t0 = time.perf_counter()
    best: tuple[float, AudioClip, str | None] | None = None
    expected = estimate_seconds(chunk.text)
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        seed = _chunk_seed(base_seed, chunk.index, attempt)
        try:
            clip = engine.synthesize(chunk.text, seed=seed)
        except UserError:
            raise
        except Exception as exc:  # model-level failure
            if is_cuda_oom(exc):
                free_cuda_memory()
                raise OutOfMemoryError(
                    f"The GPU ran out of memory while generating chunk {chunk.index + 1}.",
                    hints=oom_hints(engine.info.id),
                ) from exc
            last_error = exc
            log.warning("Chunk %d attempt %d failed: %s", chunk.index + 1, attempt + 1, exc)
            continue

        reason = looks_suspicious(clip, chunk.text)
        if reason is None:
            return ChunkResult(chunk, clip, cached=False, seconds=time.perf_counter() - t0, attempts=attempt + 1)
        score = abs(math.log(max(clip.duration, 1e-3) / expected))
        if best is None or score < best[0]:
            best = (score, clip, reason)
        log.warning("Chunk %d attempt %d: %s - retrying with a different seed", chunk.index + 1, attempt + 1, reason)

    if best is not None:
        _, clip, reason = best
        return ChunkResult(chunk, clip, cached=False, seconds=time.perf_counter() - t0,
                           attempts=retries + 1, warning=f"kept best of {retries + 1} attempts: {reason}")
    raise GenerationError(
        f"Chunk {chunk.index + 1} failed after {retries + 1} attempt(s): {last_error}",
        hints=[
            "Run again - finished chunks are cached and will be reused.",
            "Try --max-chunk-chars 150 or a different --model.",
            "Run with --debug for the full traceback.",
        ],
    )


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def generate_voiceover(text: str, options: GenerationOptions, reporter: Reporter | None = None) -> GenerationResult:
    reporter = reporter or Reporter()
    t_start = time.perf_counter()

    engine_cls = get_engine_class(options.model_id)
    info = engine_cls.info
    max_chars = options.max_chunk_chars or info.max_chunk_chars
    if max_chars > info.max_chunk_chars:
        log.warning("--max-chunk-chars %d is above the recommended %d for %s; expect more artifacts.",
                    max_chars, info.max_chunk_chars, info.name)

    reporter.stage("Processing script...")
    chunks = chunk_text(text, max_chars, newline_is_break=options.newline_is_break)
    est_minutes = sum(c.estimated_seconds for c in chunks) / 60
    reporter.info(f"Chunks: {len(chunks)} (~{est_minutes:.1f} min of speech, rough estimate)")

    engine, device = build_engine(options)
    reporter.stage(f"Loading model {info.name} on {device.name}...")
    t0 = time.perf_counter()
    try:
        engine.load()
    except Exception as exc:
        if is_cuda_oom(exc):
            raise OutOfMemoryError("The GPU ran out of memory while loading the model.", hints=oom_hints(info.id)) from exc
        raise
    reporter.info(f"Model loaded ({time.perf_counter() - t0:.1f}s).")

    try:
        voice_description = engine.set_voice(options.voice)
    except Exception as exc:
        if is_cuda_oom(exc):
            raise OutOfMemoryError("The GPU ran out of memory while preparing the voice.", hints=oom_hints(info.id)) from exc
        raise
    reporter.info(f"Voice: {voice_description}")
    voice_fp = options.voice.fingerprint()

    results: list[ChunkResult] = []
    total = len(chunks)
    try:
        for chunk in chunks:
            reporter.chunk_start(chunk.index, total, chunk)
            seed = _chunk_seed(options.seed, chunk.index, 0)
            cache_file = _cache_path(options.cache_dir, chunk_cache_key(info.id, voice_fp, chunk.text, seed))
            result: ChunkResult | None = None
            if cache_file is not None and cache_file.exists():
                try:
                    clip = audio.load_audio(cache_file)
                    result = ChunkResult(chunk, clip, cached=True, seconds=0.0, attempts=0)
                except Exception as exc:  # corrupt cache entry -> regenerate
                    log.debug("Discarding unreadable cache file %s: %s", cache_file, exc)
                    cache_file.unlink(missing_ok=True)
            if result is None:
                result = synthesize_chunk(engine, chunk, retries=options.retries, base_seed=options.seed)
                if cache_file is not None:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    audio.save_audio(result.clip, cache_file, bit_depth=32)
            results.append(result)
            reporter.chunk_done(chunk.index, total, result)
        if reporter.verbose and device.is_cuda:
            peak = _peak_vram_gb()
            if peak is not None:
                reporter.info(f"Peak GPU memory used: {peak:.1f} GB")
    finally:
        engine.unload()

    reporter.stage("Combining audio...")
    combined = stitch(results, options)

    if abs(options.speed - 1.0) > 1e-3 and not info.native_speed:
        reporter.info(f"Applying speed x{options.speed:.2f} (post-process)...")
        combined = audio.time_stretch(combined, options.speed)
    if options.output_sample_rate and options.output_sample_rate != combined.sample_rate:
        combined = audio.resample(combined, options.output_sample_rate)
    if options.normalize:
        combined = audio.peak_normalize(combined, options.peak_dbfs)

    out_path = audio.save_audio(combined, options.output_path, fmt=options.output_format, bit_depth=options.bit_depth)
    manifest_path = write_manifest(out_path, results, info.id, voice_description, combined) if options.write_manifest else None

    return GenerationResult(
        output_path=out_path,
        duration=combined.duration,
        sample_rate=combined.sample_rate,
        chunks=results,
        elapsed=time.perf_counter() - t_start,
        model_id=info.id,
        device=device,
        voice_description=voice_description,
        manifest_path=manifest_path,
    )


def _peak_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_reserved() / 1024**3
    except Exception:  # pragma: no cover
        pass
    return None


def stitch(results: list[ChunkResult], options: GenerationOptions) -> AudioClip:
    """Trim, fade and join chunk audio with sentence/paragraph gaps; fills in start/end times."""
    clips: list[AudioClip] = []
    gaps: list[float] = []
    for i, r in enumerate(results):
        clip = r.clip
        if options.trim_chunks:
            trimmed = audio.trim_silence(clip, threshold_db=-50.0, pad_ms=40.0)
            if trimmed.duration > 0.1:
                clip = trimmed
        clip = audio.apply_fade(clip, ms=5.0)
        clips.append(clip)
        is_last = i == len(results) - 1
        gaps.append(0.0 if is_last else (options.paragraph_gap if r.chunk.ends_paragraph else options.sentence_gap))

    combined = audio.concatenate(clips, gaps)
    cursor = 0.0
    for r, clip, gap in zip(results, clips, gaps, strict=True):
        r.start = cursor
        r.end = cursor + clip.duration
        cursor = r.end + gap
    return combined


def write_manifest(out_path: Path, results: list[ChunkResult], model_id: str, voice: str, combined: AudioClip) -> Path:
    manifest = {
        "output": out_path.name,
        "model": model_id,
        "voice": voice,
        "sample_rate": combined.sample_rate,
        "duration_seconds": round(combined.duration, 3),
        "chunks": [
            {
                "index": r.chunk.index + 1,
                "paragraph": r.chunk.paragraph + 1,
                "start": round(r.start, 3),
                "end": round(r.end, 3),
                "text": r.chunk.text,
                "cached": r.cached,
                "warning": r.warning,
            }
            for r in results
        ],
    }
    path = out_path.with_suffix(out_path.suffix + ".json")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
