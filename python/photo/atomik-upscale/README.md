# atomik-upscale

GPU image upscaling tuned for **this** machine: RTX 3060 Ti (8 GB, Ampere sm_86),
Ryzen 9 3950X, 64 GB RAM, Windows 11.

Drop images in `input/`, run one command, get at least 4K out.

**Full CLI reference: [SOP.md](SOP.md)** — every flag, target form, exit code
and standard procedure.

```
atomik run                 # inbox + interactive size picker
atomik run -t x4           # everything in input/, 4x
atomik run photo.jpg -t 8k -q quality
atomik watch --archive     # process files as they land
```

---

## Why Python

The heavy work is CUDA kernels; Python only schedules them. Rewriting the driver
in Rust or C++ would move maybe 2% of the wall clock, and would give up the
entire PyTorch/spandrel model ecosystem — the thing that actually decides output
quality. The measured cost of the Python layer here is well under a tenth of a
second per image against 1–20 s of GPU time.

Where Python *was* the bottleneck it has been dealt with: tile stitching is
vectorised numpy, and PNG encoding runs on worker threads so it overlaps the
next image's GPU work rather than blocking it.

## Install

Already done in this directory. To reproduce elsewhere:

```powershell
uv sync                    # torch 2.9.1+cu128, pinned for sm_86
```

`uv sync` installs the `atomik` command into `.venv\Scripts\`. Either use
`.\atomik-upscale.cmd` from this folder, or add `.venv\Scripts` to `PATH`.

If you **move this folder**, re-run `uv sync --reinstall-package atomik-upscale`
in the new location. The venv bakes in absolute paths at install time, so a move
otherwise breaks it with `uv trampoline failed to canonicalize script path`.

## Choosing the size

`atomik run` with no `-t` shows a picker:

```
  1  x2
  2  x4
  3  x6
  4  4K  (long edge 3840)
  5  8K  (long edge 7680)
  6  custom
```

Non-interactively, `-t` / `--to` accepts:

| form | meaning |
|---|---|
| `x2` `x4` `x6` `x3.5` | multiplier, up to x32 |
| `4k` | long edge reaches 3840 (the default) |
| `1080p` `2k` `1440p` `5k` `6k` `8k` `12k` `16k` | other long-edge presets |
| `3840x2160` | aspect preserved, scaled to **cover** the box |
| `5000` | target width in pixels |

A source already at or above the target is skipped rather than re-encoded.

### How x6 is reached

A 4x network asked for x6 could do one pass and Lanczos *up* (soft, wastes the
network) or two passes to x16 (four times the pixels anyone asked for). Neither
is good, so the planner chains a 4x pass with the tiny 2x net to land on x8 and
resamples **down** to x6. Downscaling after super-resolution is nearly free and
keeps the detail the network produced.

The chosen model always takes the first pass, on the original pixels — it is
what decides quality, and it must not inherit another network's artifacts.
Later passes go to the cheap 2x net, which is ~30x cheaper per pixel. On a
400x400 source to 4K that choice is 11.8 s instead of ~19 s, for the same size.

## Models

`atomik models` lists them; weights download on first use into `models/`.

| id | x | arch | precision | for |
|---|---|---|---|---|
| `nomos8k-dat` | 4 | DAT | bf16 | best photographic detail, slow |
| `nomos8k-srformer` | 4 | SRFormer | fp32 | photos with compression or mild blur |
| `nomos-webphoto` | 4 | RealPLKSR | bf16 | **default** — real-world/web photos |
| `nomos2-realplksr` | 4 | RealPLKSR | bf16 | already-clean sources, sharper |
| `ultrasharp` | 4 | ESRGAN | fp16 | graphics, text, product shots |
| `realesrgan` | 4 | ESRGAN | fp16 | safe general baseline |
| `anime` | 4 | ESRGAN-6B | fp16 | anime, illustration, flat art |
| `fast2x` | 2 | Compact | fp16 | speed, and the x6/x16 chain step |
| `fast2x-avc` | 2 | Compact | fp16 | video stills, screen grabs |

`--tier fast|balanced|quality` steers automatic selection; `-m <id>` forces one.

### Measured on this GPU

`atomik bench --size 1024` (1024x1024 source, output MPix/s):

| model | tile | s | MPix/s | peak VRAM |
|---|---|---|---|---|
| nomos8k-dat | 384 | 20.5 | 0.8 | 1255 MB |
| nomos8k-srformer | 384 | 24.8 | 0.7 | 1861 MB |
| nomos-webphoto | 1024 | 1.8 | 9.5 | 925 MB |
| nomos2-realplksr | 1024 | 1.9 | 9.0 | 2269 MB |
| ultrasharp | 384 | 4.3 | 3.9 | 1194 MB |
| realesrgan | 384 | 4.4 | 3.9 | 1194 MB |
| anime | 384 | 2.0 | 8.5 | 1170 MB |
| fast2x / fast2x-avc | 1024 | 0.15 | 28.5 | 528 MB |

## What "tuned for this hardware" actually means

**bf16 on the quality tier.** DAT, SRFormer and RealPLKSR are all flagged
fp16-unsafe — their attention and large-kernel blocks overflow the fp16 range.
The naive response is fp32. But this card is Ampere, which has full bf16 tensor
cores, and bf16 carries fp32's exponent range. The quality models therefore keep
tensor-core throughput instead of dropping to fp32. Precision is also verified
at load and steps down automatically (fp16 → bf16 → fp32) if a model turns out
to disagree — SRFormer does exactly this, and lands on fp32 on its own.

**Tiles chosen by measurement, not by table.** The first run of each model times
one tile at several sizes and caches the curve in `.cache/`. At run time the
tile that minimises predicted time *for that image's dimensions* is picked.

This matters more than it sounds. The largest tile that fits is not the
fastest: a 768 px tile on a 1024 px image lands two rows and two columns, so it
recomputes 2.25x the pixels — nothing like the 1.09x an infinitely large image
would suggest. Scoring against the real grid took DAT from 74 s to 20 s on the
same image, UltraSharp from 11.2 s to 4.3 s, and cut peak VRAM from 5.5 GB to
1.2 GB.

One subtlety worth recording: the memory probe runs a warm-up pass first.
With `cudnn.benchmark` on, the *first* convolution at any new shape trials
several algorithms and can transiently allocate 2.4 GB where the winning
algorithm only ever needs 128 MB. Measuring that would have sized every tile
against the autotuner instead of the model.

**Everything else.** TF32 matmuls (Ampere), `cudnn.benchmark` for the fixed tile
shapes, `inference_mode`, pinned non-blocking transfers, and PNG encoding on
worker threads so the GPU starts the next image while the last one is still
being written.

## Quality details

- **Seamless tiling.** Tiles overlap by 32 px and are blended with a
  raised-cosine window into a weight-normalised accumulator, so borders need no
  special case. Verified: versus a single-tile reference, tiled output is
  46–50 dB PSNR, and error at tile boundaries is 1.15x the error elsewhere —
  i.e. no seam structure.
- **Transparency.** Alpha is upscaled through the same network rather than
  resized, which keeps cut-out edges hard. `--no-alpha` to drop it.
- **Colour and metadata.** EXIF orientation is applied on load; EXIF and ICC
  profiles are carried to the output unless `--strip-metadata`.
- **Automatic model choice** looks at unique-colour ratio and exact flatness to
  spot flat-shaded artwork. It is deliberately conservative and defaults to the
  photo model whenever unsure, because a line-art model smears the texture out
  of a photograph while a photo model on illustration is merely unexciting.
  Validated at 18/18 on the Windows wallpaper set plus synthetic art. It has
  **not** been validated against real anime — if the call is wrong, pass
  `-m anime`.

## Commands

| | |
|---|---|
| `atomik run [paths...]` | upscale files/folders, or `input/` |
| `atomik watch [dir]` | process images as they are dropped in |
| `atomik models` | list the registry; `--download <id>` to prefetch |
| `atomik doctor` | hardware report + smoke test |
| `atomik bench` | time each model on this GPU |
| `atomik mcp` | run the MCP server on stdio |

Useful flags: `-o <dir>`, `-f png|jpg|webp|tiff`, `--quality N`, `--bits 16`,
`--archive` (move sources to `archive/` on success), `--overwrite`, `--tile N`,
`--fp32`, `--json`, `--suffix "{stem}__{label}"`.

`--bits 16` on a colour image writes a 16-bit TIFF (alpha included) regardless
of `-f`, and says so in the output — Pillow cannot encode 16-bit colour in any
format, and silently dropping to 8 bits under a filename that claims 16 would
be worse than changing the extension. 16-bit greyscale still writes as PNG.

## MCP server

Register with Claude Code:

```powershell
claude mcp add atomik -- "D:\- Dev -\atomik-factory\atomik-upscale\.venv\Scripts\python.exe" -m atomik_upscale mcp
```

Or use the included `.mcp.json`. Tools exposed:

| tool | does |
|---|---|
| `upscale` | upscale given files/folders |
| `upscale_inbox` | upscale everything in `input/` |
| `plan_upscale` | report model + pass chain without running |
| `list_models` | the registry and what each is for |
| `hardware_info` | GPU, VRAM, precision limits |
| `download_model` | prefetch weights |
| `list_inbox` | what is waiting, and each file's scale-to-4K |

## Layout

```
input/     drop images here        models/    downloaded weights
output/    results                 .cache/    measured tile curves
archive/   sources, with --archive
```

Override with `ATOMIK_INPUT`, `ATOMIK_OUTPUT`, `ATOMIK_ARCHIVE`,
`ATOMIK_MODELS`, `ATOMIK_CACHE`, or `ATOMIK_HOME` for all of them.

## Notes

- First use of a model downloads weights (2–147 MB) and spends a few seconds
  timing tiles. Both are cached; later runs start immediately.
- Out-of-memory is handled by halving the tile and retrying, so a background
  game or another CUDA process degrades speed rather than failing the run.
- Model licences vary. The `spandrel_extra_arches` architectures (DAT,
  SRFormer) are not licensed for commercial use; check before shipping output
  commercially.
