"""MCP server exposing the upscaler as tools.

Written against the MCP Python SDK v2 API (``MCPServer``; v1 called this
``FastMCP``). Runs on stdio, so any MCP client can spawn it directly.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp.server.mcpserver import MCPServer

from . import engine, hardware, imaging, models, planner, pipeline
from .paths import ARCHIVE_DIR, INPUT_DIR, MODELS_DIR, OUTPUT_DIR

server = MCPServer(
    name="atomik-upscale",
    title="Atomik Upscale",
    version="1.0.0",
    instructions=(
        "GPU image upscaling on the user's local machine.\n\n"
        f"Drop images into {INPUT_DIR} and call `upscale_inbox`, or call "
        "`upscale` with explicit paths.\n\n"
        "Targets accept x2 / x4 / x6, presets (4k, 8k, 1080p, 2k, 5k, 6k), an "
        "explicit 3840x2160, or a bare pixel width. The default target is 4k, "
        "which means the long edge reaches at least 3840.\n\n"
        "Leave `model` unset to let the server pick one from image content; use "
        "`tier` (fast/balanced/quality) to bias that choice. Call `list_models` "
        "to see the registry and `hardware_info` for GPU limits."
    ),
)

_HW = None


def _hw():
    global _HW
    if _HW is None:
        _HW = hardware.detect()
        hardware.apply_global_tuning(_HW)
    return _HW


def _result_dict(r: pipeline.Result) -> dict[str, Any]:
    return {
        "source": str(r.src),
        "output": str(r.dst) if r.dst and r.ok else None,
        "ok": r.ok,
        "source_size": f"{r.src_size[0]}x{r.src_size[1]}",
        "output_size": f"{r.dst_size[0]}x{r.dst_size[1]}" if r.ok else None,
        "model": r.model,
        "detected_content": r.content,
        "plan": r.plan,
        "seconds": round(r.seconds, 2),
        "tiles": r.tiles,
        "precision": r.dtype,
        "peak_vram_mb": r.peak_vram_mb,
        "skipped": r.skipped or None,
        "error": r.error or None,
        "notes": r.notes,
    }


def _options(target: str, model: str | None, tier: str, fmt: str,
             quality: int, archive: bool, overwrite: bool) -> pipeline.Options:
    if model is not None and model not in models.REGISTRY:
        raise ValueError(f"unknown model {model!r}; "
                         f"known: {', '.join(sorted(models.REGISTRY))}")
    return pipeline.Options(
        target=target, model=model, tier=tier, fmt=fmt, quality=quality,
        archive=archive, overwrite=overwrite,
    )


# --- tools -------------------------------------------------------------------

@server.tool(
    title="Upscale images",
    description="Upscale specific image files or folders on the local GPU.",
)
async def upscale(
    paths: Annotated[list[str], Field(description="Image files or folders to upscale.")],
    target: Annotated[str, Field(
        description="x2 | x4 | x6 | 4k | 8k | 1080p | 2k | 5k | 6k | 3840x2160 | "
                    "a pixel width. Default 4k = long edge at least 3840.")] = "4k",
    model: Annotated[str | None, Field(
        description="Model id to force. Leave unset for automatic selection.")] = None,
    tier: Annotated[Literal["fast", "balanced", "quality"], Field(
        description="Bias for automatic model choice.")] = "balanced",
    output_dir: Annotated[str | None, Field(
        description=f"Where to write. Default {OUTPUT_DIR}.")] = None,
    fmt: Annotated[Literal["png", "jpg", "webp", "tiff"], Field(
        description="Output format.")] = "png",
    quality: Annotated[int, Field(ge=1, le=100,
        description="JPEG/WebP quality.")] = 95,
    archive: Annotated[bool, Field(
        description=f"Move each source into {ARCHIVE_DIR} after success.")] = False,
    overwrite: Annotated[bool, Field(
        description="Overwrite an existing output instead of numbering it.")] = False,
) -> dict[str, Any]:
    files = pipeline.discover([Path(p) for p in paths], recursive=True)
    if not files:
        return {"ok": False, "error": "no readable images at the given paths",
                "paths": paths}

    opts = _options(target, model, tier, fmt, quality, archive, overwrite)
    out = Path(output_dir) if output_dir else None

    results = await asyncio.to_thread(
        pipeline.process_many, files, _hw(), opts, out_dir=out
    )
    return {
        "ok": all(r.ok or r.skipped for r in results),
        "count": len(results),
        "succeeded": sum(1 for r in results if r.ok),
        "failed": sum(1 for r in results if not r.ok and not r.skipped),
        "output_dir": str(out or OUTPUT_DIR),
        "results": [_result_dict(r) for r in results],
    }


@server.tool(
    title="Upscale the inbox",
    description=f"Upscale every image currently in the drop folder ({INPUT_DIR}).",
)
async def upscale_inbox(
    target: Annotated[str, Field(
        description="x2 | x4 | x6 | 4k | 8k | 3840x2160 | pixel width.")] = "4k",
    model: Annotated[str | None, Field(description="Force a model id.")] = None,
    tier: Annotated[Literal["fast", "balanced", "quality"], Field(
        description="Bias for automatic model choice.")] = "balanced",
    fmt: Annotated[Literal["png", "jpg", "webp", "tiff"], Field(
        description="Output format.")] = "png",
    archive: Annotated[bool, Field(
        description="Move sources into archive/ after success.")] = True,
) -> dict[str, Any]:
    files = pipeline.discover(None)
    if not files:
        return {"ok": True, "count": 0,
                "message": f"inbox is empty: {INPUT_DIR}"}
    opts = _options(target, model, tier, fmt, 95, archive, False)
    results = await asyncio.to_thread(pipeline.process_many, files, _hw(), opts)
    return {
        "ok": all(r.ok or r.skipped for r in results),
        "count": len(results),
        "succeeded": sum(1 for r in results if r.ok),
        "output_dir": str(OUTPUT_DIR),
        "results": [_result_dict(r) for r in results],
    }


@server.tool(
    title="Preview a plan",
    description="Report which model and pass chain would be used, without running it.",
)
async def plan_upscale(
    path: Annotated[str, Field(description="Image file to inspect.")],
    target: Annotated[str, Field(description="Desired target.")] = "4k",
    model: Annotated[str | None, Field(description="Force a model id.")] = None,
    tier: Annotated[Literal["fast", "balanced", "quality"], Field(
        description="Bias for automatic model choice.")] = "balanced",
) -> dict[str, Any]:
    src = Path(path)
    if not src.is_file():
        return {"ok": False, "error": f"not a file: {path}"}

    img = await asyncio.to_thread(imaging.load, src)
    model_id, content, notes = pipeline.choose_model(img, src, tier, model)
    spec = models.get(model_id)
    tgt = planner.parse_target(target, img.width, img.height)
    p = planner.plan(tgt, model_id, spec.scale, models.CHAIN_2X,
                     models.get(models.CHAIN_2X).scale)
    return {
        "ok": True,
        "source": str(src),
        "source_size": f"{img.width}x{img.height}",
        "output_size": f"{tgt.width}x{tgt.height}",
        "multiplier": round(tgt.multiplier, 3),
        "detected_content": content,
        "chosen_model": model_id,
        "model_arch": spec.arch,
        "model_note": spec.blurb,
        "passes": p.passes,
        "plan": p.describe(),
        "peak_megapixels": round(planner.peak_megapixels(img.width, img.height, p), 1),
        "weights_cached": spec.available,
        "notes": notes,
    }


@server.tool(
    title="List models",
    description="The curated model registry, with what each one is good for.",
)
async def list_models() -> dict[str, Any]:
    return {
        "models_dir": str(MODELS_DIR),
        "chain_helper": models.CHAIN_2X,
        "defaults_by_tier": models.DEFAULT_BY_TIER,
        "models": [
            {
                "id": s.id, "scale": s.scale, "arch": s.arch, "tier": s.tier,
                "content": s.content, "size_mb": s.size_mb,
                "downloaded": s.available, "tags": list(s.tags),
                "description": s.blurb,
            }
            for s in models.REGISTRY.values()
        ],
    }


@server.tool(
    title="Hardware info",
    description="GPU, VRAM and precision limits that constrain what can be run.",
)
async def hardware_info() -> dict[str, Any]:
    hw = _hw()
    rep = hardware.hardware_report(hw)
    rep["input_dir"] = str(INPUT_DIR)
    rep["output_dir"] = str(OUTPUT_DIR)
    rep["archive_dir"] = str(ARCHIVE_DIR)
    rep["models_cached"] = [m.id for m in models.REGISTRY.values() if m.available]
    rep["compute_cap"] = ".".join(map(str, hw.compute_cap or ()))
    return rep


@server.tool(
    title="Download model weights",
    description="Fetch weights ahead of time so the first upscale does not stall.",
)
async def download_model(
    model: Annotated[str, Field(description="Model id from list_models.")],
) -> dict[str, Any]:
    if model not in models.REGISTRY:
        return {"ok": False, "error": f"unknown model {model!r}",
                "known": sorted(models.REGISTRY)}
    spec = models.get(model)
    path = await asyncio.to_thread(models.download, spec)
    return {"ok": True, "model": model, "path": str(path),
            "size_mb": round(path.stat().st_size / 1024**2, 1)}


@server.tool(
    title="List the inbox",
    description="What is currently waiting in the drop folder.",
)
async def list_inbox() -> dict[str, Any]:
    files = pipeline.discover(None)
    out = []
    for f in files:
        entry: dict[str, Any] = {"path": str(f),
                                 "size_mb": round(f.stat().st_size / 1024**2, 2)}
        try:
            from PIL import Image
            with Image.open(f) as im:
                entry["size"] = f"{im.width}x{im.height}"
                entry["scale_for_4k"] = round(
                    planner.ceil_multiplier_for_4k(im.width, im.height), 2)
        except Exception as exc:
            entry["error"] = str(exc)
        out.append(entry)
    return {"input_dir": str(INPUT_DIR), "count": len(out), "images": out}


def run_stdio() -> None:
    try:
        server.run(transport="stdio")
    finally:
        engine.clear_cache()


if __name__ == "__main__":
    run_stdio()
