"""Tiled super-resolution inference.

One :class:`Upscaler` owns one loaded network. It knows how to split an image
into overlapping tiles that fit in 8 GB, run them in fp16, and stitch the
results back without seams.
"""
from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import hardware, models
from .hardware import Hardware
from .models import ModelSpec

# Overlap between neighbouring tiles, in input pixels. Enough to cover the
# receptive field of every arch in the registry; the feathered blend hides
# whatever context error is left.
DEFAULT_OVERLAP = 32
MIN_TILE = 64
# Ceiling on the autotuned tile. Past this the per-tile launch overhead is
# already amortised, and huge tiles only make an OOM retry more expensive.
MAX_TILE = 1024
# Side length used for the one-off activation-cost measurement.
PROBE_TILE = 128
# Wall-clock ceiling for the one-off tile timing sweep, per model.
TUNE_TIME_BUDGET = 20.0


class OutOfVramError(RuntimeError):
    pass


_DTYPE_ERROR_MARKERS = (
    "expected scalar type",
    "same dtype",
    "input type",
    "expected m1 and m2",
    "but found",
)


def _is_dtype_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _DTYPE_ERROR_MARKERS)


@dataclass
class RunStats:
    tiles: int
    seconds: float
    tile_size: int
    dtype: str
    fell_back_to_fp32: bool = False
    peak_vram_mb: int = 0


def _load_arches() -> None:
    """Register the non-commercial architectures (DAT, SRFormer, ...)."""
    try:
        import spandrel_extra_arches
        spandrel_extra_arches.install()
    except Exception:
        pass


_ARCHES_READY = False


class Upscaler:
    def __init__(self, spec: ModelSpec, hw: Hardware, *,
                 tile: int | None = None,
                 overlap: int = DEFAULT_OVERLAP,
                 force_fp32: bool = False,
                 channels_last: bool = False):
        global _ARCHES_READY
        if not _ARCHES_READY:
            _load_arches()
            _ARCHES_READY = True

        from spandrel import ImageModelDescriptor, ModelLoader

        self.spec = spec
        self.hw = hw
        self.overlap = overlap
        self.channels_last = channels_last

        models.download(spec)
        before_mb = torch.cuda.memory_allocated() / 1024**2 if hw.is_cuda else 0.0
        descriptor = ModelLoader(device=torch.device(hw.device)).load_from_file(str(spec.path))
        if not isinstance(descriptor, ImageModelDescriptor):
            raise RuntimeError(f"{spec.id}: not a single-image super-resolution model")

        self.model = descriptor
        self.scale: int = int(descriptor.scale)
        if self.scale != spec.scale:  # registry metadata must match the weights
            self.scale = int(descriptor.scale)

        # Several of the strongest architectures (DAT, SRFormer, PLKSR) are
        # flagged fp16-unsafe -- their attention/large-kernel blocks overflow
        # the fp16 range. They are fine in bf16, which has the same exponent
        # range as fp32 and still runs on Ampere's tensor cores, so the quality
        # tier keeps tensor-core throughput instead of dropping to fp32.
        self.precision = "fp32"
        if not force_fp32:
            if hw.supports_fp16 and bool(getattr(descriptor, "supports_half", False)):
                self.precision = "fp16"
            elif hw.supports_bf16 and bool(getattr(descriptor, "supports_bfloat16", False)):
                self.precision = "bf16"

        descriptor.eval()
        if self.precision == "fp16":
            descriptor.half()
        elif self.precision == "bf16":
            descriptor.bfloat16()
        if channels_last and hw.is_cuda:
            try:
                descriptor.model.to(memory_format=torch.channels_last)
            except Exception:
                self.channels_last = False

        req = getattr(descriptor, "size_requirements", None)
        self.multiple_of = int(getattr(req, "multiple_of", 1) or 1)
        self.min_size = int(getattr(req, "minimum", 0) or 0)
        self.needs_square = bool(getattr(req, "square", False))

        tiling = str(getattr(descriptor, "tiling", ""))
        self.tiling_internal = "INTERNAL" in tiling.upper()

        # VRAM held by *these* weights, so the autotuner can price activations
        # separately. Measured as a delta: another model may already be
        # resident, and charging its memory to this one shrinks the tile for
        # no reason.
        self.weights_mb = (
            int(torch.cuda.memory_allocated() / 1024**2 - before_mb)
            if hw.is_cuda else 0
        )
        # An explicit tile short-circuits tuning; otherwise the cost curve is
        # measured (or loaded) on the first real run.
        self.tile: int | None = tile
        self._profile: dict[int, float] | None = None

        self.precision_downgraded_from: str | None = None
        self._validate_precision()

    # -- precision --------------------------------------------------------
    def _set_precision(self, precision: str) -> None:
        self.precision = precision
        if precision == "fp16":
            self.model.half()
        elif precision == "bf16":
            self.model.bfloat16()
        else:
            self.model.float()
        self.free()

    def _downgrade_precision(self) -> bool:
        """Step down one rung: fp16 -> bf16 -> fp32. False when already fp32."""
        if self.precision == "fp32":
            return False
        nxt = "fp32"
        if self.precision == "fp16" and self.hw.supports_bf16 and bool(
                getattr(self.model, "supports_bfloat16", False)):
            nxt = "bf16"
        if self.precision_downgraded_from is None:
            self.precision_downgraded_from = self.precision
        self._set_precision(nxt)
        return True

    def _validate_precision(self) -> None:
        """Prove the chosen precision actually runs before committing to it.

        spandrel's ``supports_bfloat16`` flag is per-architecture and not
        always right -- SRFormer, for one, leaves an internal tensor in fp32
        and dies on the first attention matmul. A 64px probe surfaces that
        here, cheaply, instead of part-way through someone's batch.
        """
        if self.precision == "fp32":
            return
        probe = np.zeros((64, 64, 3), dtype=np.float32)
        while True:
            try:
                self._forward(probe)
                return
            except torch.cuda.OutOfMemoryError:
                self.free()
                return  # a memory problem, not a dtype one; tiling handles it
            except RuntimeError as exc:
                if not _is_dtype_error(exc) or not self._downgrade_precision():
                    raise

    # -- tile sizing ------------------------------------------------------
    def _memory_ceiling(self) -> int:
        """Largest tile whose activations fit the free-VRAM budget."""
        if not self.hw.is_cuda:
            return min(self.spec.default_tile, 256)

        probe = PROBE_TILE
        try:
            # Warm-up pass, discarded. With cudnn.benchmark on, the first
            # convolution at any new shape trials several algorithms and can
            # transiently allocate an order of magnitude more than the winner
            # ever will (2.4 GB vs 128 MB for ESRGAN here). Measuring that
            # would size the tiles against the autotuner instead of the model.
            self._forward(np.zeros((probe, probe, 3), dtype=np.float32))
            self.free()
            torch.cuda.reset_peak_memory_stats()
            # Baseline is whatever is already resident (these weights, plus any
            # other model still loaded). The difference to the peak is this
            # forward pass's transient cost, which is what scales with tile area.
            base_mb = torch.cuda.memory_allocated() / 1024**2
            self._forward(np.zeros((probe, probe, 3), dtype=np.float32))
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        except torch.cuda.OutOfMemoryError:
            self.free()
            return MIN_TILE

        activation_mb = max(peak_mb - base_mb, 1.0)
        # Release the probe's blocks before asking how much VRAM is free: the
        # caching allocator still holds them, and they would otherwise be
        # counted as unavailable and shrink the budget for no reason.
        self.free()
        budget = max(self.hw.vram_budget_mb(), 128)

        # Activation memory grows with tile area, so the side length scales
        # with the square root of the memory ratio.
        tile = int(probe * math.sqrt(budget / activation_mb))
        tile = max(MIN_TILE, min(tile, MAX_TILE))
        tile = (tile // 32) * 32 or MIN_TILE
        self.free()
        return tile

    def _starts(self, extent: int, tile: int) -> list[int]:
        """Tile origins along one axis. Shared with the cost predictor so the
        two can never disagree about how many tiles a grid really has."""
        stride = max(MIN_TILE // 2, tile - self.overlap)
        raw = range(0, max(1, extent - self.overlap), stride)
        return sorted({min(s, max(0, extent - tile)) for s in raw})

    def _measure_profile(self) -> dict[int, float]:
        """Time one tile at each candidate size: {tile -> seconds}.

        Costs a few seconds per model, once, then lives in .cache/ keyed by
        model and GPU.
        """
        ceiling = self._memory_ceiling()
        if not self.hw.is_cuda:
            return {ceiling: 1.0}

        candidates = sorted({t for t in (128, 192, 256, 384, 512, 768, ceiling)
                             if MIN_TILE <= t <= ceiling})
        profile: dict[int, float] = {}
        spent = 0.0

        for tile in candidates:
            try:
                chunk = np.zeros((tile, tile, 3), dtype=np.float32)
                self._forward(chunk)          # warm cuDNN for this shape
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                self._forward(chunk)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
            except torch.cuda.OutOfMemoryError:
                self.free()
                break                          # nothing larger will fit either
            finally:
                self.free()

            profile[tile] = elapsed
            spent += elapsed * 2
            if spent > TUNE_TIME_BUDGET:
                break

        return profile or {MIN_TILE: 1.0}

    def _resolve_tile(self, h: int, w: int) -> int:
        """Choose the tile that minimises predicted time for *this* image.

        Throughput per tile is only half the story. A 768px tile on a 1024px
        image lands two rows and two columns, so it recomputes 2.25x the
        pixels -- nothing like the 1.09x an infinitely large image would
        suggest. Scoring against the real grid is what makes the choice
        correct at every image size instead of only for huge ones.
        """
        if self.tile is not None:
            return max(MIN_TILE, min(self.tile, max(h, w)))

        if self._profile is None:
            self._profile = (hardware.remembered_profile(self.spec.id, self.hw)
                             or self._measure_profile())
            hardware.remember_profile(self.spec.id, self.hw, self._profile)

        best, best_cost = None, float("inf")
        for tile, per_tile in sorted(self._profile.items()):
            capped = max(MIN_TILE, min(tile, max(h, w)))
            n = len(self._starts(h, capped)) * len(self._starts(w, capped))
            cost = n * per_tile
            if cost < best_cost:
                best, best_cost = capped, cost
        return best or MIN_TILE

    @property
    def dtype(self) -> torch.dtype:
        return {"fp16": torch.float16,
                "bf16": torch.bfloat16}.get(self.precision, torch.float32)

    # -- core -------------------------------------------------------------
    def _to_tensor(self, chunk: np.ndarray) -> torch.Tensor:
        h, w = chunk.shape[:2]
        pad_h = self._pad_to(h)
        pad_w = self._pad_to(w)
        if self.needs_square:
            pad_h = pad_w = max(pad_h, pad_w)

        t = torch.from_numpy(chunk).permute(2, 0, 1).unsqueeze(0)
        t = t.to(self.hw.device, dtype=self.dtype, non_blocking=True)
        if pad_h != h or pad_w != w:
            # Reflect needs the source to be larger than the padding.
            mode = "reflect" if (h > pad_h - h and w > pad_w - w) else "replicate"
            t = torch.nn.functional.pad(t, (0, pad_w - w, 0, pad_h - h), mode=mode)
        if self.channels_last:
            t = t.contiguous(memory_format=torch.channels_last)
        return t

    def _forward(self, chunk: np.ndarray) -> np.ndarray:
        """Run one padded tile. ``chunk`` is float32 HWC in [0, 1].

        Retries at a lower precision on a dtype mismatch. Some architectures
        only take the offending code path at certain input sizes -- SRFormer
        runs fine in bf16 at 64px and dies at 128px -- so the guard has to sit
        on every forward, not only on a start-up probe.
        """
        h, w = chunk.shape[:2]
        while True:
            t = self._to_tensor(chunk)
            try:
                with torch.inference_mode():
                    out = self.model(t)
                    # Stay inside inference mode for the crop/cast: the result
                    # is an inference tensor and rejects in-place ops outside.
                    out = out[..., : h * self.scale, : w * self.scale]
                    out = out.squeeze(0).permute(1, 2, 0).float().clamp(0.0, 1.0)
                    return out.cpu().numpy().copy()
            except torch.cuda.OutOfMemoryError:
                # A subclass of RuntimeError, but it is the tiling loop's
                # problem, not the precision ladder's.
                del t
                raise
            except RuntimeError as exc:
                del t
                if not _is_dtype_error(exc) or not self._downgrade_precision():
                    raise

    def _pad_to(self, n: int) -> int:
        n = max(n, self.min_size)
        if self.multiple_of > 1:
            n = int(math.ceil(n / self.multiple_of) * self.multiple_of)
        return n

    @staticmethod
    def _window(length: int, ramp: int) -> np.ndarray:
        """Raised-cosine ramp used to feather tile edges."""
        w = np.ones(length, dtype=np.float32)
        r = min(ramp, length // 2)
        if r > 0:
            t = (np.arange(r, dtype=np.float32) + 0.5) / r
            edge = (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)
            w[:r] = edge
            w[length - r:] = edge[::-1]
        return np.maximum(w, 1e-3)

    def _tiled(self, rgb: np.ndarray, tile: int,
               progress=None) -> tuple[np.ndarray, int]:
        h, w = rgb.shape[:2]
        s = self.scale
        # Shared with the cost predictor; the last tile sits flush with the
        # edge rather than being padded.
        ys = self._starts(h, tile)
        xs = self._starts(w, tile)

        acc = np.zeros((h * s, w * s, 3), dtype=np.float32)
        wacc = np.zeros((h * s, w * s, 1), dtype=np.float32)
        ramp = self.overlap * s // 2
        count = 0

        for y in ys:
            y1 = min(y + tile, h)
            for x in xs:
                x1 = min(x + tile, w)
                out = self._forward(rgb[y:y1, x:x1])
                th, tw = out.shape[:2]
                win = (self._window(th, ramp)[:, None]
                       * self._window(tw, ramp)[None, :])[..., None]
                acc[y * s:y * s + th, x * s:x * s + tw] += out * win
                wacc[y * s:y * s + th, x * s:x * s + tw] += win
                count += 1
                if progress:
                    progress(count, len(ys) * len(xs))

        np.maximum(wacc, 1e-6, out=wacc)
        return acc / wacc, count

    def run(self, rgb: np.ndarray, progress=None) -> tuple[np.ndarray, RunStats]:
        """Upscale one float32 HWC image by this model's native scale."""
        h, w = rgb.shape[:2]
        started = time.perf_counter()
        if self.hw.is_cuda:
            torch.cuda.reset_peak_memory_stats()

        tile = self._resolve_tile(h, w)
        fell_back = False

        while True:
            try:
                if self.tiling_internal or (h <= tile and w <= tile):
                    out = self._forward(rgb)
                    tiles = 1
                    if progress:
                        progress(1, 1)
                else:
                    out, tiles = self._tiled(rgb, tile, progress)
                break
            except torch.cuda.OutOfMemoryError:
                self.free()
                if tile <= MIN_TILE:
                    raise OutOfVramError(
                        f"{self.spec.id} cannot run even at {MIN_TILE}px tiles on "
                        f"{self.hw.gpu_name}. Try a lighter model (--model nomos-webphoto) "
                        f"or close other GPU applications."
                    ) from None
                tile = max(MIN_TILE, (tile // 2 // 32) * 32 or MIN_TILE)
                self.tile = tile   # pin it; the measured curve was optimistic

        # Reduced-precision overflow shows up as non-finite output. Rather than
        # ship a corrupt image, redo this one in fp32 and stay there.
        if self.precision != "fp32" and not np.isfinite(out).all():
            self.precision = "fp32"
            fell_back = True
            self.model.float()
            self.free()
            out, tiles = (self._forward(rgb), 1) if (h <= tile and w <= tile) \
                else self._tiled(rgb, tile, progress)

        peak = int(torch.cuda.max_memory_allocated() / 1024**2) if self.hw.is_cuda else 0
        return out, RunStats(
            tiles=tiles,
            seconds=time.perf_counter() - started,
            tile_size=tile,
            dtype=self.precision,
            fell_back_to_fp32=fell_back,
            peak_vram_mb=peak,
        )

    def run_alpha(self, alpha: np.ndarray, progress=None) -> np.ndarray:
        """Upscale an alpha plane through the same network.

        Running the model on a replicated alpha keeps cut-out edges hard, which
        a plain Lanczos resize would soften.
        """
        rgb = np.repeat(alpha[..., None], 3, axis=2)
        out, _ = self.run(rgb, progress)
        return out.mean(axis=2)

    def free(self) -> None:
        gc.collect()
        if self.hw.is_cuda:
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self.model = None  # type: ignore[assignment]
        self.free()


# --- a tiny cache so batch runs load each network once -----------------------
_CACHE: dict[tuple, Upscaler] = {}


def get_upscaler(spec: ModelSpec, hw: Hardware, **kw) -> Upscaler:
    key = (spec.id, hw.device, kw.get("force_fp32", False),
           kw.get("channels_last", False), kw.get("tile"))
    up = _CACHE.get(key)
    if up is None:
        up = Upscaler(spec, hw, **kw)
        _CACHE[key] = up
    return up


def clear_cache() -> None:
    for up in _CACHE.values():
        up.unload()
    _CACHE.clear()
