"""End-to-end orchestration: pick a model, plan the chain, run it, write the file.

Both the CLI and the MCP server drive this module; neither contains any
image logic of its own.
"""
from __future__ import annotations

import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from . import engine, imaging, models, planner
from .hardware import Hardware
from .paths import ARCHIVE_DIR, IMAGE_SUFFIXES, INPUT_DIR, OUTPUT_DIR

ProgressFn = Callable[[str, float], None]


@dataclass
class Options:
    target: str = "4k"
    model: str | None = None          # None -> automatic
    tier: str = "balanced"            # fast | balanced | quality
    fmt: str = "png"
    quality: int = 95
    bits: int = 8
    tile: int | None = None
    overlap: int = engine.DEFAULT_OVERLAP
    force_fp32: bool = False
    channels_last: bool = False
    keep_alpha: bool = True
    archive: bool = False
    overwrite: bool = False
    strip_metadata: bool = False
    suffix: str = "{stem}__{label}"
    max_passes: int = 3


@dataclass
class Result:
    src: Path
    dst: Path | None
    ok: bool
    src_size: tuple[int, int] = (0, 0)
    dst_size: tuple[int, int] = (0, 0)
    model: str = ""
    content: str = ""
    plan: str = ""
    seconds: float = 0.0
    tiles: int = 0
    dtype: str = ""
    tile_size: int = 0
    peak_vram_mb: int = 0
    skipped: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)
    pending_write: Future | None = None

    def finalize(self) -> "Result":
        """Block until the deferred encode lands, folding failures into the result."""
        fut, self.pending_write = self.pending_write, None
        if fut is not None:
            try:
                fut.result()
            except Exception as exc:
                self.ok = False
                self.error = f"write failed: {type(exc).__name__}: {exc}"
        return self

    def summary(self) -> str:
        if not self.ok:
            return f"{self.src.name}: {self.skipped or self.error}"
        sw, sh = self.src_size
        dw, dh = self.dst_size
        return (f"{self.src.name}  {sw}x{sh} -> {dw}x{dh}  "
                f"[{self.model}, {self.dtype}, {self.tiles} tiles, {self.seconds:.1f}s]")


# --- model selection ---------------------------------------------------------

def choose_model(img: imaging.LoadedImage, src: Path, tier: str,
                 override: str | None) -> tuple[str, str, list[str]]:
    """Return (model_id, detected_content, notes)."""
    notes: list[str] = []
    content = imaging.classify(img.rgb)

    if override:
        return override, content, notes

    if content == "anime":
        notes.append("detected illustration/anime -> line-art model")
        return "anime", content, notes

    compressed = imaging.looks_compressed(src, img.rgb)
    if compressed:
        notes.append("source looks lossy-compressed -> degradation-aware model")

    if tier == "fast":
        return ("fast2x-avc" if compressed else "fast2x"), content, notes
    if tier == "quality":
        return ("nomos8k-srformer" if compressed else "nomos8k-dat"), content, notes
    return ("nomos-webphoto" if compressed else "nomos2-realplksr"), content, notes


def output_path(src: Path, target_label: str, opts: Options,
                out_dir: Path) -> Path:
    stem = opts.suffix.format(stem=src.stem, label=target_label.replace(" ", ""),
                              model=opts.model or "auto")
    ext = "jpg" if opts.fmt in ("jpg", "jpeg") else opts.fmt
    dst = out_dir / f"{stem}.{ext}"
    if dst.exists() and not opts.overwrite:
        n = 2
        while (cand := out_dir / f"{stem}_{n}.{ext}").exists():
            n += 1
        dst = cand
    return dst


# --- single image ------------------------------------------------------------

def process(src: Path, hw: Hardware, opts: Options, *,
            out_dir: Path | None = None,
            dst: Path | None = None,
            save_pool: ThreadPoolExecutor | None = None,
            progress: ProgressFn | None = None) -> Result:
    out_dir = out_dir or OUTPUT_DIR
    res = Result(src=src, dst=None, ok=False)
    t0 = time.perf_counter()

    def note(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    try:
        note("loading", 0.0)
        img = imaging.load(src)
        res.src_size = img.size

        model_id, content, notes = choose_model(img, src, opts.tier, opts.model)
        res.model, res.content, res.notes = model_id, content, list(notes)
        spec = models.get(model_id)

        target = planner.parse_target(opts.target, img.width, img.height)
        plan = planner.plan(target, model_id, spec.scale,
                            models.CHAIN_2X, models.get(models.CHAIN_2X).scale,
                            max_passes=opts.max_passes)
        res.plan = plan.describe()
        res.dst_size = (target.width, target.height)

        if not plan.steps:
            res.skipped = (f"already {img.width}x{img.height}, at or above "
                           f"{target.label} — nothing to upscale")
            return res

        peak = planner.peak_megapixels(img.width, img.height, plan)
        if peak > 400:
            res.notes.append(f"large job: peaks at {peak:.0f} MPix in RAM")

        rgb = img.rgb
        alpha = img.alpha if (opts.keep_alpha and img.alpha is not None) else None
        total_steps = len(plan.steps) * (2 if alpha is not None else 1)
        done_steps = 0
        precisions: list[str] = []

        for step in plan.steps:
            step_spec = models.get(step.model_id)
            up = engine.get_upscaler(
                step_spec, hw,
                tile=opts.tile, overlap=opts.overlap,
                force_fp32=opts.force_fp32, channels_last=opts.channels_last,
            )

            def tile_progress(done: int, total: int, _s=done_steps, _t=total_steps,
                              _n=step.model_id) -> None:
                frac = (_s + done / max(total, 1)) / max(_t, 1)
                note(f"{_n} x{step.scale}  tile {done}/{total}", 0.05 + 0.9 * frac)

            rgb, stats = up.run(rgb, tile_progress)
            done_steps += 1
            res.tiles += stats.tiles
            # A chain can mix precisions (bf16 main model, fp16 2x step), so
            # record each distinct one rather than letting the last pass
            # overwrite what the model that mattered actually used.
            if stats.dtype not in precisions:
                precisions.append(stats.dtype)
            res.dtype = "+".join(precisions)
            res.tile_size = stats.tile_size
            res.peak_vram_mb = max(res.peak_vram_mb, stats.peak_vram_mb)
            if stats.fell_back_to_fp32:
                res.notes.append(f"{step.model_id}: fp16 overflowed, redone in fp32")

            if alpha is not None:
                note(f"{step.model_id} x{step.scale}  alpha", 0.05 + 0.9 * done_steps / total_steps)
                alpha = up.run_alpha(alpha)
                done_steps += 1

        if plan.final_resize:
            note(f"resampling to {target.width}x{target.height}", 0.95)
            rgb = imaging.resize(rgb, (target.width, target.height))
            if alpha is not None:
                alpha = imaging.resize(alpha, (target.width, target.height))

        res.dst_size = (rgb.shape[1], rgb.shape[0])

        note("encoding", 0.97)
        dst = dst or output_path(src, target.label, opts, out_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        res.dst = dst

        def write() -> None:
            written = imaging.save(
                dst, rgb, alpha,
                fmt=opts.fmt, quality=opts.quality, bits=opts.bits,
                exif=None if opts.strip_metadata else img.exif,
                icc=None if opts.strip_metadata else img.icc,
            )
            if written != dst:
                # 16-bit colour is redirected to TIFF; report where it landed.
                res.dst = written
                res.notes.append(f"16-bit colour written as TIFF: {written.name}")
            if opts.archive:
                ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
                moved = ARCHIVE_DIR / src.name
                n = 2
                while moved.exists():
                    moved = ARCHIVE_DIR / f"{src.stem}_{n}{src.suffix}"
                    n += 1
                shutil.move(str(src), str(moved))
                res.notes.append(f"source archived to {moved.name}")

        if save_pool is not None:
            # Hand the encode to a worker thread; Pillow drops the GIL inside
            # its codecs, so the GPU can start the next image immediately.
            res.pending_write = save_pool.submit(write)
            res.ok = True
        else:
            write()
            res.ok = True

        res.seconds = time.perf_counter() - t0
        note("done", 1.0)
        return res

    except (engine.OutOfVramError, planner.PlanError) as exc:
        res.seconds = time.perf_counter() - t0
        res.error = str(exc)
        return res
    except Exception as exc:  # keep a batch alive when one file is broken
        res.seconds = time.perf_counter() - t0
        res.error = f"{type(exc).__name__}: {exc}"
        return res


# --- batch -------------------------------------------------------------------

def discover(paths: Sequence[Path] | None = None, *,
             recursive: bool = False) -> list[Path]:
    """Collect image files from explicit paths, or from the inbox."""
    roots = list(paths) if paths else [INPUT_DIR]
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if root.is_dir():
            it = root.rglob("*") if recursive else root.glob("*")
            found += [p for p in it
                      if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        elif root.is_file():
            found.append(root)
    # Stable, and smallest first so a batch shows progress early.
    return sorted(set(found), key=lambda p: (p.stat().st_size, str(p)))


def process_many(files: Iterable[Path], hw: Hardware, opts: Options, *,
                 out_dir: Path | None = None,
                 on_start: Callable[[Path, int, int], None] | None = None,
                 on_result: Callable[[Result], None] | None = None,
                 progress: ProgressFn | None = None) -> list[Result]:
    """Run a batch. The GPU stays serialised; encoding overlaps the next image.

    With 16 cores available, PNG encoding of an 8K frame is worth pushing off
    the critical path -- it costs seconds the GPU could otherwise be using.
    """
    files = list(files)
    results: list[Result] = []
    started = time.perf_counter()

    workers = min(4, max(1, (hw.cpu_threads // 4) or 1))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="atomik-io") as pool:
        inflight: list[Result] = []
        for i, src in enumerate(files, 1):
            if on_start:
                on_start(src, i, len(files))
            res = process(src, hw, opts, out_dir=out_dir,
                          save_pool=pool, progress=progress)
            results.append(res)
            inflight.append(res)
            # Keep at most `workers` encodes queued so a long batch cannot pile
            # up finished 8K frames in RAM faster than they are written.
            while len(inflight) > workers:
                done = inflight.pop(0).finalize()
                if on_result:
                    on_result(done)
        for res in inflight:
            done = res.finalize()
            if on_result:
                on_result(done)

    if results:
        results[-1].notes.append(f"batch took {time.perf_counter() - started:.1f}s")
    return results


def warm_up(hw: Hardware, opts: Options) -> None:
    """Load and JIT the likely model before a watch session starts."""
    model_id = opts.model or models.DEFAULT_BY_TIER.get(opts.tier, "nomos-webphoto")
    spec = models.get(model_id)
    up = engine.get_upscaler(spec, hw, tile=opts.tile, overlap=opts.overlap,
                             force_fp32=opts.force_fp32,
                             channels_last=opts.channels_last)
    probe = np.zeros((64, 64, 3), dtype=np.float32)
    up.run(probe)
