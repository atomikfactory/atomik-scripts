"""Image load/save, colour handling, and content classification.

Pillow owns decode/encode; numpy owns everything in between. Images move
through the pipeline as float32 HWC arrays in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Big source files are legitimate here; the guard exists for untrusted input.
Image.MAX_IMAGE_PIXELS = 1_000_000_000

RESAMPLE = Image.Resampling.LANCZOS


@dataclass
class LoadedImage:
    rgb: np.ndarray               # float32 HWC, 3 channels, [0, 1]
    alpha: np.ndarray | None      # float32 HW, [0, 1]
    exif: bytes | None
    icc: bytes | None
    mode: str                     # original PIL mode
    source_bits: int              # 8 or 16

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def load(path: Path) -> LoadedImage:
    with Image.open(path) as im:
        im.load()
        exif = im.info.get("exif")
        icc = im.info.get("icc_profile")
        mode = im.mode
        bits = 16 if mode in ("I;16", "I;16B", "I", "I;16L") or (
            mode == "RGB" and im.info.get("bits") == 16
        ) else 8

        # Honour EXIF orientation before anything else touches the pixels.
        try:
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im) or im
        except Exception:
            pass

        alpha = None
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            rgba = im.convert("RGBA")
            arr = np.asarray(rgba, dtype=np.uint8)
            rgb_u8, a_u8 = arr[..., :3], arr[..., 3]
            alpha = a_u8.astype(np.float32) / 255.0
            rgb = rgb_u8.astype(np.float32) / 255.0
        elif im.mode in ("I;16", "I;16B", "I;16L", "I"):
            gray = np.asarray(im.convert("I;16"), dtype=np.uint16).astype(np.float32) / 65535.0
            rgb = np.repeat(gray[..., None], 3, axis=2)
            bits = 16
        else:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8).astype(np.float32) / 255.0

    return LoadedImage(rgb=np.ascontiguousarray(rgb), alpha=alpha,
                       exif=exif, icc=icc, mode=mode, source_bits=bits)


def resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resample a float32 array to (width, height) with a properly filtered Lanczos.

    Pillow scales the filter support on reduction, so this stays correct for
    both the small trims and the large downscales the chain planner produces.
    """
    w, h = size
    if arr.shape[1] == w and arr.shape[0] == h:
        return arr
    is_2d = arr.ndim == 2
    src = arr[..., None] if is_2d else arr
    out = np.empty((h, w, src.shape[2]), dtype=np.float32)
    for c in range(src.shape[2]):
        plane = Image.fromarray(src[..., c], mode="F")
        out[..., c] = np.asarray(plane.resize((w, h), RESAMPLE), dtype=np.float32)
    return out[..., 0] if is_2d else out


def save(path: Path, rgb: np.ndarray, alpha: np.ndarray | None,
         *, fmt: str = "png", quality: int = 95, bits: int = 8,
         exif: bytes | None = None, icc: bytes | None = None,
         compress_level: int = 6) -> Path:
    fmt = fmt.lower()
    rgb = np.clip(rgb, 0.0, 1.0)

    if bits == 16:
        data = (rgb * 65535.0 + 0.5).astype(np.uint16)
        is_gray = (np.array_equal(data[..., 0], data[..., 1])
                   and np.array_equal(data[..., 1], data[..., 2]))

        if fmt == "png" and is_gray:
            # Pillow writes 16-bit PNG only from single-channel I;16.
            Image.fromarray(data[..., 0], mode="I;16").save(
                path, format="PNG", compress_level=compress_level)
            return path

        # Pillow cannot construct a 16-bit *colour* image at all, so 16-bit
        # colour goes out as TIFF via tifffile. Rewriting the extension is
        # better than the alternative -- silently dropping to 8 bits under a
        # name that claims 16.
        import tifffile

        if fmt != "tiff":
            path = path.with_suffix(".tiff")
        kw: dict = {"photometric": "rgb", "compression": "adobe_deflate"}
        if icc:
            kw["iccprofile"] = icc
        if alpha is not None:
            a16 = (np.clip(alpha, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
            data = np.dstack([data, a16])
            kw["extrasamples"] = ["unassalpha"]
        tifffile.imwrite(str(path), data, **kw)
        return path

    u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    # JPEG is the only format here that cannot carry alpha.
    if alpha is not None and fmt in ("png", "webp", "tif", "tiff"):
        a8 = (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        im = Image.fromarray(np.dstack([u8, a8]), mode="RGBA")
    else:
        im = Image.fromarray(u8, mode="RGB")

    params: dict = {}
    if icc:
        params["icc_profile"] = icc
    if exif and fmt in ("jpg", "jpeg", "webp", "png"):
        params["exif"] = exif

    if fmt in ("jpg", "jpeg"):
        im.save(path, format="JPEG", quality=quality, subsampling=0,
                optimize=True, progressive=True, **params)
    elif fmt == "webp":
        im.save(path, format="WEBP", quality=quality, method=6, **params)
    elif fmt in ("tif", "tiff"):
        im.save(path, format="TIFF", compression="tiff_lzw", **params)
    else:
        im.save(path, format="PNG", compress_level=compress_level, **params)
    return path


# --- content classification --------------------------------------------------

def measure(rgb: np.ndarray) -> tuple[float, float]:
    """Return (unique-colour ratio, exact-flatness) on a strided sample.

    Striding rather than resampling matters: a filtered downscale invents
    intermediate colours and smooths away the very flatness being measured.
    """
    h, w = rgb.shape[:2]
    step = max(1, max(h, w) // 600)
    small = (np.clip(rgb[::step, ::step], 0.0, 1.0) * 255.0).astype(np.uint8)

    n = small.shape[0] * small.shape[1]
    unique_ratio = len(np.unique(small.reshape(-1, 3), axis=0)) / max(n, 1)

    lum = small.astype(np.int16).sum(axis=2)
    dx = np.diff(lum, axis=1)
    dy = np.diff(lum, axis=0)
    exact_flat = ((dx == 0).mean() + (dy == 0).mean()) / 2.0
    return float(unique_ratio), float(exact_flat)


def classify(rgb: np.ndarray) -> str:
    """Detect flat-shaded artwork, defaulting to "photo" whenever unsure.

    Deliberately hard to trigger. Measured over the Windows wallpaper set
    (photographs and abstract renders) the unique-colour ratio never fell
    below 0.11 and exact flatness never exceeded 0.55, while genuinely
    flat-shaded art sits at ~0.00 and ~0.99. The gap between those is wide,
    but real anime -- which has gradients, shading and encoder noise -- lands
    somewhere inside it, and nothing here can place it reliably.

    So the bar is set where only unambiguous vector-like art clears it. The
    asymmetry is deliberate: a line-art model smears the texture out of a
    photograph, whereas a photo model on illustration is merely unexciting.
    Pass --model anime (or tier/model in MCP) when the call is wrong.
    """
    unique_ratio, exact_flat = measure(rgb)
    return "anime" if (unique_ratio < 0.03 and exact_flat > 0.85) else "photo"


def looks_compressed(path: Path, rgb: np.ndarray) -> bool:
    """Cheap proxy for "this source has been through a lossy encoder"."""
    if path.suffix.lower() in (".jpg", ".jpeg", ".jpe", ".jfif", ".webp"):
        return True

    # Otherwise look for JPEG's 8x8 block grid, which survives a transcode to
    # PNG. Needs real texture to be meaningful: on near-flat art the ratio is
    # a division of noise by noise and reads high for no reason.
    gray = rgb.mean(axis=2)
    if gray.shape[0] < 64 or gray.shape[1] < 64:
        return False
    dx = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    if dx.size < 16:
        return False
    edges = dx[7::8]
    other = float(dx.mean())
    if not edges.size or other < 0.01:
        return False
    return bool(float(edges.mean()) / other > 1.5)
