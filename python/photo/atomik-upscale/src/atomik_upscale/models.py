"""Curated model registry.

Every entry here was picked for an 8 GB Ampere card: the architectures all fit
in VRAM at a workable tile size and run in fp16 without falling apart. Weights
are fetched lazily on first use and cached in ``models/``.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .paths import MODELS_DIR

USER_AGENT = "atomik-upscale/1.0"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    filename: str
    url: str
    scale: int
    arch: str
    content: str          # "photo" | "anime" | "any"
    tier: str             # "fast" | "balanced" | "quality"
    size_mb: int
    default_tile: int     # starting point; autotuned on first run
    cost_weight: float    # relative GPU cost per output pixel (fast2x = 1.0)
    blurb: str
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename

    @property
    def available(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 1024


REGISTRY: dict[str, ModelSpec] = {
    m.id: m
    for m in [
        # ---- quality tier: transformer archs, slow but the best detail -------
        ModelSpec(
            id="nomos8k-dat",
            filename="4xNomos8kDAT.pth",
            url="https://github.com/Phhofm/models/releases/download/4xNomos8kDAT/4xNomos8kDAT.pth",
            scale=4, arch="DAT", content="photo", tier="quality",
            size_mb=147, default_tile=192,
            cost_weight=36.0,
            blurb="Best photographic detail. Transformer arch, so it is slow; worth it for stills.",
            tags=("photo", "detail", "sharp"),
        ),
        ModelSpec(
            id="nomos8k-srformer",
            filename="4xNomos8kSCSRFormer.pth",
            url="https://github.com/Phhofm/models/releases/download/4xNomos8kSCSRFormer/4xNomos8kSCSRFormer.pth",
            scale=4, arch="SRFormer", content="photo", tier="quality",
            size_mb=92, default_tile=192,
            cost_weight=41.0,
            blurb="Alternative to DAT; handles compression and mild blur more gracefully.",
            tags=("photo", "jpeg", "blur"),
        ),
        # ---- balanced tier: the everyday workhorses --------------------------
        ModelSpec(
            id="nomos-webphoto",
            filename="4xNomosWebPhoto_RealPLKSR.pth",
            url="https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.pth",
            scale=4, arch="RealPLKSR", content="photo", tier="balanced",
            size_mb=28, default_tile=384,
            cost_weight=3.0,
            blurb="Default for real-world photos: trained on web-degraded images "
                  "(JPEG, resizing, noise). Fast and very clean.",
            tags=("photo", "web", "jpeg", "noisy"),
        ),
        ModelSpec(
            id="nomos2-realplksr",
            filename="4xNomos2_realplksr_dysample.pth",
            url="https://github.com/Phhofm/models/releases/download/4xNomos2_realplksr_dysample/4xNomos2_realplksr_dysample.pth",
            scale=4, arch="RealPLKSR", content="photo", tier="balanced",
            size_mb=28, default_tile=384,
            cost_weight=3.2,
            blurb="For already-clean sources (RAW exports, originals). Sharper than "
                  "webphoto, but it will amplify any artifacts that are already there.",
            tags=("photo", "clean", "sharp"),
        ),
        ModelSpec(
            id="ultrasharp",
            filename="4x-UltraSharp.pth",
            url="https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth",
            scale=4, arch="ESRGAN", content="any", tier="balanced",
            size_mb=64, default_tile=320,
            cost_weight=7.3,
            blurb="Aggressively crisp. Good for graphics, text and product shots.",
            tags=("graphics", "text", "crisp"),
        ),
        ModelSpec(
            id="realesrgan",
            filename="RealESRGAN_x4plus.pth",
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            scale=4, arch="ESRGAN", content="any", tier="balanced",
            size_mb=64, default_tile=320,
            cost_weight=7.3,
            blurb="The safe baseline. Never spectacular, never wrong.",
            tags=("general", "safe"),
        ),
        # ---- illustration ----------------------------------------------------
        ModelSpec(
            id="anime",
            filename="RealESRGAN_x4plus_anime_6B.pth",
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            scale=4, arch="ESRGAN", content="anime", tier="balanced",
            size_mb=17, default_tile=512,
            cost_weight=3.4,
            blurb="Anime, illustration, flat-shaded art. Keeps line weight clean.",
            tags=("anime", "illustration", "lineart"),
        ),
        # ---- fast tier / 2x chain steps -------------------------------------
        ModelSpec(
            id="fast2x",
            filename="2xHFA2kCompact.pth",
            url="https://github.com/Phhofm/models/releases/download/2xHFA2kCompact/2xHFA2kCompact.pth",
            scale=2, arch="Compact", content="any", tier="fast",
            size_mb=2, default_tile=768,
            cost_weight=1.0,
            blurb="Tiny 2x net. Second step of x6 chains, and a standalone option "
                  "when throughput beats fidelity.",
            tags=("fast", "chain", "2x"),
        ),
        ModelSpec(
            id="fast2x-avc",
            filename="2xHFA2kAVCCompact.pth",
            url="https://github.com/Phhofm/models/releases/download/2xHFA2kAVCCompact/2xHFA2kAVCCompact.pth",
            scale=2, arch="Compact", content="any", tier="fast",
            size_mb=2, default_tile=768,
            cost_weight=1.0,
            blurb="Same speed as fast2x but trained on H.264 artifacts; use it for "
                  "video stills and screen grabs.",
            tags=("fast", "video", "compressed", "2x"),
        ),
    ]
}

# The 2x model used to finish odd chains (x6) without a wasteful second 4x pass.
CHAIN_2X = "fast2x"

DEFAULT_BY_TIER = {
    "fast": "fast2x",
    "balanced": "nomos-webphoto",
    "quality": "nomos8k-dat",
}


def get(model_id: str) -> ModelSpec:
    try:
        return REGISTRY[model_id]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown model {model_id!r}. Known models: {known}") from None


def by_tier(tier: str) -> list[ModelSpec]:
    return [m for m in REGISTRY.values() if m.tier == tier]


def download(spec: ModelSpec,
             on_progress: Callable[[int, int], None] | None = None,
             force: bool = False) -> Path:
    """Fetch weights into ``models/``. Atomic: writes .part, then renames."""
    if spec.available and not force:
        return spec.path

    tmp = spec.path.with_suffix(spec.path.suffix + ".part")
    req = urllib.request.Request(spec.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)

    if total and tmp.stat().st_size != total:
        got = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{spec.id}: truncated download ({got} of {total} bytes)")
    tmp.replace(spec.path)
    return spec.path




def disk_usage_bytes() -> int:
    return sum(m.path.stat().st_size for m in REGISTRY.values() if m.available)
