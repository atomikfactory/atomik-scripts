# atomik-upscale — CLI Standard Operating Procedure

Complete operating reference for the `atomik` command line tool.

Install location: `D:\- Dev -\atomik-factory\atomik-upscale`


---

## Contents

1. [Invoking the tool](#1-invoking-the-tool)
2. [The five-second version](#2-the-five-second-version)
3. [Folder layout](#3-folder-layout)
4. [Command: `run`](#4-command-run)
5. [Command: `watch`](#5-command-watch)
6. [Command: `models`](#6-command-models)
7. [Command: `doctor`](#7-command-doctor)
8. [Command: `bench`](#8-command-bench)
9. [Command: `mcp`](#9-command-mcp)
10. [Target syntax reference](#10-target-syntax-reference)
11. [Model reference](#11-model-reference)
12. [Output files](#12-output-files)
13. [Exit codes](#13-exit-codes)
14. [JSON output](#14-json-output)
15. [Environment variables](#15-environment-variables)
16. [Standard procedures](#16-standard-procedures)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. Invoking the tool

Three equivalent ways. Pick one and stay with it.

**A — the wrapper (simplest).** From the project folder:

```powershell
cd ".\atomik-factory\atomik-upscale"
.\atomik-upscale.cmd run
```

Works from any working directory as long as you give the full path:

```powershell
& ".\atomik-factory\atomik-upscale\atomik-upscale.cmd" run
```

**B — the executable directly.** No venv activation needed:

```powershell
& ".\atomik-factory\atomik-upscale\.venv\Scripts\atomik.exe" run
```

**C — put it on PATH once**, then just type `atomik` anywhere:

```powershell
# current session only
$env:PATH += ";C:\tmp\atomik-upscale\.venv\Scripts"

# permanent (new terminals pick it up)
[Environment]::SetEnvironmentVariable(
  "PATH",
  [Environment]::GetEnvironmentVariable("PATH","User") + ";C:\tmp\atomik-upscale\.venv\Scripts",
  "User")
```

> Throughout this document commands are written as `atomik ...`. Substitute
> `.\atomik-upscale.cmd ...` if you have not put it on PATH.

**Getting help at any time:**

```powershell
atomik --help              # list commands
atomik run --help          # every flag for one command
```

---

## 2. The five-second version

```powershell
# 1. put images in the inbox
copy C:\photos\*.jpg "D:\- Dev -\atomik-factory\atomik-upscale\input\"

# 2. run; pick a size when asked
atomik run

# 3. collect from C:\tmp\atomik-upscale\output\
```

---

## 3. Folder layout

| Folder | Purpose |
|---|---|
| `input\` | **Drop images here.** `atomik run` with no arguments processes this folder. |
| `output\` | Results land here. |
| `archive\` | Sources are moved here after success, but **only** with `--archive`. |
| `models\` | Downloaded model weights (~444 MB when all 9 are cached). |
| `.cache\` | Measured tile-cost curves. Safe to delete; it just re-measures. |

Accepted input types: `.png` `.jpg` `.jpeg` `.jpe` `.jfif` `.webp` `.bmp`
`.tif` `.tiff` `.ppm`. Anything else in `input\` is ignored, not an error.

---

## 4. Command: `run`

Upscale files. This is the command you will use 95% of the time.

```
atomik run [paths...] [options]
```

With no `paths`, it processes everything in `input\`.
With no `-t`, it asks you interactively what size you want.

### 4.1 The interactive size picker

Running bare `atomik run` shows:

```
┌───────────── Choose output size ─────────────┐
│    target                  result            │
│  1 x2                      1080x720          │
│  2 x4                      2160x1440         │
│  3 x6                      3240x2160         │
│  4 4K  (long edge 3840)    3840x2560         │
│  5 8K  (long edge 7680)    7680x5120         │
│  6 custom                  e.g. 3840x2160    │
└──── 3 image(s) · first is 540x360 ───────────┘
Selection [1/2/3/4/5/6] (4):
```

- The **result** column shows real output dimensions for the *first* image in
  the batch, so you can see what each option actually means.
- If an option says `(source already larger)`, that file will be skipped.
- Press **Enter** to take the default, **4K**.
- Choosing **6** prompts for free-form input — anything from
  [section 10](#10-target-syntax-reference).

The picker only appears when the terminal is interactive **and** `--json` is not
used. In a script, a pipe, or a scheduled task it is skipped and the target
defaults to `4k`. If stdin looks interactive but has nothing to read, it prints
`no input available — using 4K` and continues rather than failing.

### 4.2 Selecting what to process

| Form | Effect |
|---|---|
| `atomik run` | everything directly inside `input\` |
| `atomik run photo.jpg` | one file |
| `atomik run a.jpg b.png c.webp` | several files |
| `atomik run C:\photos` | every image directly inside that folder |
| `atomik run C:\photos -r` | that folder **and all subfolders** |
| `atomik run input C:\photos x.jpg` | mix freely |

`-r` / `--recursive` only affects folder arguments; it does nothing for files.

Files are processed **smallest first**, so a big batch shows progress early.

### 4.3 Size and quality options

| Flag | Default | Meaning |
|---|---|---|
| `-t`, `--to TARGET` | asks, else `4k` | output size — see [section 10](#10-target-syntax-reference) |
| `-q`, `--tier {fast,balanced,quality}` | `balanced` | steers **automatic** model choice |
| `-m`, `--model ID` | automatic | force one specific model, overrides `--tier` |

**Tiers** only matter when you have *not* passed `-m`. The tier picks a model
based on what the image looks like:

| tier | clean source | lossy/compressed source | speed on a 1024px image |
|---|---|---|---|
| `fast` | `fast2x` | `fast2x-avc` | ~0.15 s |
| `balanced` | `nomos2-realplksr` | `nomos-webphoto` | ~1.8 s |
| `quality` | `nomos8k-dat` | `nomos8k-srformer` | ~20–25 s |

Flat-shaded artwork is routed to the `anime` model in every tier. See
[section 11.2](#112-how-automatic-selection-works).

### 4.4 Output options

| Flag | Default | Meaning |
|---|---|---|
| `-o`, `--out DIR` | `output\` | where to write |
| `-f`, `--format {png,jpg,webp,tiff}` | `png` | output format |
| `--quality N` | `95` | JPEG/WebP quality, 1–100. Ignored for PNG/TIFF |
| `--bits {8,16}` | `8` | bit depth — see [12.3](#123-16-bit-output) |
| `--suffix TEMPLATE` | `{stem}__{label}` | output filename pattern |
| `--overwrite` | off | replace an existing output instead of numbering it |
| `--no-alpha` | off | discard transparency |
| `--strip-metadata` | off | drop EXIF and ICC profile |
| `--archive` | off | move each source to `archive\` after it succeeds |

> `--archive` **moves** your source files. It only fires after a successful
> write, and never on a skip or failure — but the originals do leave `input\`.
> Run once without it if you want to be sure of the results first.

### 4.5 Performance options

You should not normally need any of these. The defaults are measured on your
GPU at first use and cached.

| Flag | Default | When to touch it |
|---|---|---|
| `--tile N` | autotuned | Pin the tile size. Use a smaller value if something else is using the GPU. |
| `--overlap N` | `32` | Pixels shared between neighbouring tiles. Raising it costs time; lowering it risks seams. |
| `--fp32` | off | Forces full precision. Only for diagnosing a suspected precision artifact — it is slower and rarely changes anything. |
| `--channels-last` | off | Alternate memory layout. May help some conv models; measure with `bench` before adopting. |

### 4.6 Reading the output

```
  ok   photo_scene.jpg 540x360 -> 3840x2560 · nomos-webphoto · bf16+fp16 · 8 tiles @768px · 3.3s · 1314 MB VRAM
       · source looks lossy-compressed -> degradation-aware model
```

| Field | Meaning |
|---|---|
| `540x360 -> 3840x2560` | source and final dimensions |
| `nomos-webphoto` | model used for the **first** (quality-defining) pass |
| `bf16+fp16` | every precision used across the chain, in order |
| `8 tiles @768px` | total tiles processed and the tile size |
| `3.3s` | wall clock for this image |
| `1314 MB VRAM` | peak GPU memory |
| indented `·` lines | notes: why a model was chosen, warnings, archive confirmations |

Status words: `ok` succeeded · `skip` already at or above target · `FAIL` errored
(the batch continues).

### 4.7 Worked examples

```powershell
# everything in the inbox to at least 4K, interactive picker
atomik run

# whole inbox at 4x, no prompt
atomik run -t x4

# one file to 8K with the best model available
atomik run photo.jpg -t 8k -q quality

# a folder tree to 4K as high-quality JPEGs, into a custom location
atomik run C:\shoot -r -t 4k -f jpg --quality 95 -o D:\delivery

# force the illustration model regardless of detection
atomik run artwork.png -t x4 -m anime

# fast pass over a big batch, filing sources away as it goes
atomik run -t x2 -q fast --archive

# exactly 6x, overwriting previous results
atomik run input -t x6 --overwrite

# transparent PNG, keeping alpha (default) but stripping metadata
atomik run logo.png -t 4k --strip-metadata
```

---

## 5. Command: `watch`

Monitors a folder and upscales each image as it arrives. Runs until you press
**Ctrl-C**.

```
atomik watch [dir] [same options as run] [--settle SECONDS] [--no-warm]
```

`dir` defaults to `input\`.

```powershell
# watch the inbox, everything to 4K, file sources away afterwards
atomik watch --archive

# watch a shared drop folder, 2x, fast tier, output elsewhere
atomik watch D:\dropbox\incoming -t x2 -q fast -o D:\dropbox\done
```

Behaviour worth knowing:

- **Anything already in the folder when you start is processed immediately.**
- A file is only picked up once it has **stopped changing** for `--settle`
  seconds (default `1.5`). This prevents a half-copied file from being read.
  Raise it for slow network drives: `--settle 5`.
- Each file is processed **exactly once**, tracked by size and modification
  time. Re-saving the same filename with different content does re-process it.
- On start it preloads the model so the first image is not slow. `--no-warm`
  skips that if you want the prompt back instantly.
- `watch` accepts every `run` option except `-r`/`--recursive`; it watches one
  folder, not a tree.

Because `watch` does not prompt, `-t` defaults to `4k` unless you pass it.

---

## 6. Command: `models`

```powershell
atomik models                                    # list everything
atomik models --download nomos8k-dat anime       # prefetch specific weights
```

The listing shows id, scale, architecture, tier, intended content, size, a `✓`
if the weights are already downloaded, and a one-line description.

Weights download automatically the first time a model is used, so `--download`
is only for prefetching — useful before going offline, or to get the wait out of
the way before a large batch.

Downloads are atomic (written to `.part`, then renamed) and size-verified, so an
interrupted download cannot leave a corrupt model behind. To force a re-download,
delete the file from `models\` and run again.

---

## 7. Command: `doctor`

Health check. **Run this first if anything misbehaves.**

```powershell
atomik doctor
atomik doctor --json     # machine-readable
```

Prints GPU, VRAM total/free, compute capability, CUDA and torch versions,
precision support, CPU, RAM, and the inbox/outbox paths — then runs a real
512×512 upscale as a smoke test.

Expected healthy output on this machine:

```
│ GPU                 NVIDIA GeForce RTX 3060 Ti     │
│ VRAM                8191 MB total, 7140 MB free    │
│ compute             (8, 6)                         │
│ CUDA / torch        12.8 / 2.9.1+cu128             │
│ precision           fp16 tensor cores              │
│ TF32                on                             │
ok nomos-webphoto: 512x512 -> 2048x2048 in 0.46s (bf16, tile 512, peak 239 MB VRAM)
```

If it says **CUDA is not available**, everything will run on CPU roughly 30–60x
slower, and `doctor` exits `1`. See [troubleshooting](#17-troubleshooting).

---

## 8. Command: `bench`

Times each model on your GPU.

```powershell
atomik bench                          # all models, 512x512 source
atomik bench --size 1024              # more representative of real work
atomik bench nomos8k-dat anime        # only these two
atomik bench --size 1024 --fp32       # compare against full precision
```

Reference numbers already measured on this machine (1024×1024 source):

| model | tile | sec | MPix/s | peak VRAM |
|---|---|---|---|---|
| `nomos8k-dat` | 384 | 20.5 | 0.8 | 1255 MB |
| `nomos8k-srformer` | 384 | 24.8 | 0.7 | 1861 MB |
| `nomos-webphoto` | 1024 | 1.8 | 9.5 | 925 MB |
| `nomos2-realplksr` | 1024 | 1.9 | 9.0 | 2269 MB |
| `ultrasharp` | 384 | 4.3 | 3.9 | 1194 MB |
| `realesrgan` | 384 | 4.4 | 3.9 | 1194 MB |
| `anime` | 384 | 2.0 | 8.5 | 1170 MB |
| `fast2x` / `fast2x-avc` | 1024 | 0.15 | 28.5 | 528 MB |

`MPix/s` counts **output** pixels. Only compare within the column: the 2x models
produce a quarter as many pixels per pass as the 4x ones.

Run `bench` after a driver update or if speed regresses. It re-measures rather
than trusting the cache.

---

## 9. Command: `mcp`

Runs the MCP server on stdio, so Claude Code (or any MCP client) can drive the
upscaler as a tool.

```powershell
atomik mcp
```

You do not normally run this by hand — the client launches it. Register it once:

```powershell
claude mcp add atomik -- "D:\- Dev -\atomik-factory\atomik-upscale\.venv\Scripts\python.exe" -m atomik_upscale mcp
```

The project also ships a `.mcp.json` that clients reading project-level config
will pick up automatically.

Tools exposed: `upscale`, `upscale_inbox`, `plan_upscale`, `list_models`,
`hardware_info`, `download_model`, `list_inbox`.

`plan_upscale` is the useful one for checking intent — it reports which model and
pass chain *would* be used, without spending GPU time.

---

## 10. Target syntax reference

Everything `-t` / `--to` accepts. **Aspect ratio is always preserved; nothing is
ever cropped.**

| You type | Meaning | 540×360 source becomes |
|---|---|---|
| `x2` `x4` `x6` | multiplier | `1080x720` / `2160x1440` / `3240x2160` |
| `x3.5` | fractional multiplier, up to `x32` | `1890x1260` |
| `2` `4` `6` | bare number under 100 = multiplier | same as `x2` `x4` `x6` |
| `4k` | long edge reaches 3840 **(default)** | `3840x2560` |
| `1080p` | long edge 1920 | `1920x1280` |
| `2k` / `1440p` | long edge 2560 | `2560x1707` |
| `5k` | long edge 5120 | `5120x3413` |
| `6k` | long edge 6144 | `6144x4096` |
| `8k` | long edge 7680 | `7680x5120` |
| `12k` / `16k` | long edge 11520 / 15360 | very large |
| `3840x2160` | scaled to **cover** that box | `3840x2560` |
| `5000` | bare number ≥ 100 = target **width** | `5000x3333` |

Three points that catch people out:

1. **"4K" means the long edge, not a fixed frame.** A portrait image at `4k`
   gets a 3840px *height*. This is what you want for prints and wallpapers.
2. **`3840x2160` covers, it does not fit.** The image is scaled so both
   dimensions are *at least* the requested box, preserving aspect. A 3:2 source
   asked for `2560x1440` comes out `2560x1707` — taller than 1440, because
   cropping to match would throw pixels away. Crop afterwards if you need exact
   framing.
3. **A source already at or above the target is skipped**, not re-encoded. The
   batch reports it as `skip` and exits `0`.

### 10.1 How x6 is reached

Worth understanding because it explains the timings you will see.

The models are natively 4x or 2x. For x6 there is no single pass, so the planner
chains: **a 4x pass, then the small 2x net → x8, then resamples down to x6.**
Downscaling after super-resolution is nearly free and keeps the detail.

Two rules govern the chain:

- **Your chosen model always takes the first pass**, on the original pixels. It
  is what decides quality, and it must not inherit another network's artifacts.
  Running it while the image is still small is also the cheapest place to spend
  it.
- **Later passes go to the cheap 2x net**, which is ~30x cheaper per pixel.

So `-t x2` with a 4x model does a full 4x pass then downscales — deliberately.
That is higher quality than a weaker native-2x model, and the planner knows it.

You can see any plan without running it:

```powershell
atomik run photo.jpg -t x6 --json   # look at the "plan" field
```

Typical plans for a 540×360 source:

| target | plan |
|---|---|
| `x2` | `nomos-webphoto x4 (x4) -> Lanczos to 1080x720` |
| `x4` | `nomos-webphoto x4 (x4)` |
| `x6` | `nomos-webphoto x4 -> fast2x x2 (x8) -> Lanczos to 3240x2160` |
| `4k` | `nomos-webphoto x4 -> fast2x x2 (x8) -> Lanczos to 3840x2560` |

---

## 11. Model reference

### 11.1 The registry

| id | x | arch | precision | best for |
|---|---|---|---|---|
| `nomos8k-dat` | 4 | DAT | bf16 | maximum photographic detail; slow |
| `nomos8k-srformer` | 4 | SRFormer | fp32 | photos with compression or mild blur |
| `nomos-webphoto` | 4 | RealPLKSR | bf16 | **default** — real-world/web photos, JPEG, noise |
| `nomos2-realplksr` | 4 | RealPLKSR | bf16 | already-clean sources; sharper, but amplifies existing artifacts |
| `ultrasharp` | 4 | ESRGAN | fp16 | graphics, text, product shots; aggressively crisp |
| `realesrgan` | 4 | ESRGAN | fp16 | safe general baseline |
| `anime` | 4 | ESRGAN-6B | fp16 | anime, illustration, flat-shaded art |
| `fast2x` | 2 | Compact | fp16 | speed; also the chain step for x6/x16 |
| `fast2x-avc` | 2 | Compact | fp16 | video stills and screen grabs (H.264 artifacts) |

Precision is chosen per model and verified at load. If a model turns out to
disagree with its own declared support it steps down automatically
(fp16 → bf16 → fp32); `nomos8k-srformer` does exactly this and settles on fp32.

### 11.2 How automatic selection works

With no `-m`, each image is measured and routed:

1. **Flat-shaded artwork → `anime`.** Detected from unique-colour ratio and
   exact pixel flatness.
2. **Otherwise → the tier's photo model**, choosing the compression-aware
   variant if the source is a JPEG/WebP or shows JPEG's 8×8 block grid.

The artwork detector is **deliberately conservative and defaults to "photo"
whenever unsure.** A line-art model smears the texture out of a photograph,
whereas a photo model on illustration is merely unexciting — so the bar is set
where only unambiguous vector-like art clears it. It was validated 18/18 on the
Windows wallpaper set plus synthetic artwork, but **has not been validated
against real anime**.

**If it calls one wrong, override it:** `-m anime` or `-m nomos-webphoto`. The
chosen model and the reason are always printed, so you can see the decision.

### 11.3 Choosing a model by hand

| Situation | Use |
|---|---|
| Photos off a phone or the web | `-q balanced` (default) |
| A hero image where time does not matter | `-q quality` |
| Hundreds of files, speed matters | `-q fast` |
| Anime, illustration, logos, flat art | `-m anime` |
| Screenshots, UI, text, product cutouts | `-m ultrasharp` |
| Frames grabbed from video | `-m fast2x-avc` |
| Unsure / want a safe result | `-m realesrgan` |

---

## 12. Output files

### 12.1 Naming

Default pattern `{stem}__{label}`:

| Source | Target | Output |
|---|---|---|
| `photo_scene.jpg` | `x2` | `photo_scene__x2.png` |
| `photo_scene.jpg` | `x6` | `photo_scene__x6.png` |
| `photo_scene.jpg` | `4k` | `photo_scene__4K.png` |
| `photo_scene.jpg` | `3840x2160` | `photo_scene__3840x2160.png` |
| `photo_scene.jpg` | `5000` | `photo_scene__5000pxwide.png` |

Customise with `--suffix`, which accepts `{stem}`, `{label}`, and `{model}`:

```powershell
atomik run -t 4k --suffix "{stem}_upscaled"          # photo_scene_upscaled.png
atomik run -t 4k --suffix "{stem}__{label}__{model}" # photo_scene__4K__auto.png
atomik run -t 4k --suffix "{stem}"                   # photo_scene.png
```

> `{model}` renders as `auto` unless you passed `-m` explicitly — it reflects
> what you asked for, not what was chosen.

**Collisions are never overwritten by default.** A second run appends `_2`,
`_3`, and so on. Pass `--overwrite` to replace instead.

### 12.2 Formats

| `-f` | Notes |
|---|---|
| `png` | Default. Lossless, keeps alpha. Large files (an 8K PNG can exceed 15 MB). |
| `jpg` | Use `--quality` (default 95). No alpha — transparency is dropped. 4:4:4 chroma, progressive. |
| `webp` | Keeps alpha, much smaller than PNG at `--quality 95`. |
| `tiff` | LZW-compressed. Keeps alpha. Best for handing to an editor. |

### 12.3 16-bit output

`--bits 16` on a **colour** image writes a **16-bit TIFF regardless of `-f`**,
renames the extension to `.tiff`, and says so in the notes. This is deliberate:
Pillow cannot encode 16-bit colour in any format, and silently dropping to 8 bits
under a filename claiming 16 would be worse than changing the extension. Alpha is
preserved as a fourth 16-bit channel.

16-bit **greyscale** still writes as a normal 16-bit PNG.

There is little to gain from 16-bit if the source is 8-bit; the main benefit is
avoiding banding in the final downscale step of chained plans.

### 12.4 Metadata

EXIF orientation is applied on load, so rotated phone photos come out upright.
EXIF and ICC colour profiles are carried into the output unless you pass
`--strip-metadata`.

---

## 13. Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. **Also returned when every file was skipped** for already being large enough. |
| `1` | No images found at the given paths, **or** at least one file failed. |
| `2` | Usage error — unknown model id, bad argument value. Nothing was processed. |
| `130` | Interrupted with Ctrl-C. |

Scripting note: `0` does not prove work was done, only that nothing failed. To
confirm files were actually produced, use `--json` and check `succeeded`.

```powershell
atomik run -t 4k
if ($LASTEXITCODE -ne 0) { Write-Error "upscale failed"; exit 1 }
```

---

## 14. JSON output

`--json` on `run` (or `watch`) suppresses the progress display and prints an
array, one object per input file. The interactive picker is also suppressed, so
always pass `-t` alongside it.

```powershell
atomik run input -t x2 --json > results.json
```

```json
[
  {
    "src": "input\\photo_scene.jpg",
    "dst": "D:\\- Dev -\\atomik-factory\\atomik-upscale\\output\\photo_scene__x2.png",
    "ok": true,
    "src_size": [540, 360],
    "dst_size": [1080, 720],
    "model": "fast2x-avc",
    "content": "photo",
    "plan": "fast2x-avc x2 (x2)",
    "seconds": 1.964,
    "tiles": 2,
    "dtype": "fp16",
    "tile_size": 384,
    "peak_vram_mb": 1291,
    "skipped": "",
    "error": "",
    "notes": ["source looks lossy-compressed -> degradation-aware model"]
  }
]
```

| Field | Meaning |
|---|---|
| `ok` | `true` only if the file was written successfully |
| `skipped` | non-empty string when the source was already large enough |
| `error` | non-empty string when the file failed; `ok` is then `false` |
| `content` | `photo` or `anime`, as detected |
| `model` | the model actually used for the first pass |
| `plan` | the full pass chain, human-readable |
| `dtype` | precisions used across the chain, e.g. `bf16+fp16` |

`atomik doctor --json` emits a flat object with `device`, `gpu_name`,
`vram_total_mb`, `vram_free_mb`, `compute_cap`, `cuda_version`, `torch`,
`supports_fp16` / `_bf16` / `_tf32`, `cpu`, `cpu_threads`, `ram_gb`, and
`models_cached`.

---

## 15. Environment variables

Override any folder without editing code:

| Variable | Overrides |
|---|---|
| `ATOMIK_HOME` | the project root — moves **all** folders at once |
| `ATOMIK_INPUT` | inbox |
| `ATOMIK_OUTPUT` | output folder |
| `ATOMIK_ARCHIVE` | archive folder |
| `ATOMIK_MODELS` | weights cache |
| `ATOMIK_CACHE` | tile-timing cache |

```powershell
# one-off run against a different working set
$env:ATOMIK_INPUT  = "D:\job42\raw"
$env:ATOMIK_OUTPUT = "D:\job42\4k"
atomik run -t 4k
```

Missing folders are created automatically.

---

## 16. Standard procedures

### SOP-1 — Routine batch to 4K

```powershell
copy D:\shoot\*.jpg "D:\- Dev -\atomik-factory\atomik-upscale\input\"
cd "D:\- Dev -\atomik-factory\atomik-upscale"
.\atomik-upscale.cmd run -t 4k
explorer output
```

### SOP-2 — Highest quality, single important image

```powershell
atomik run hero.jpg -t 8k -q quality -f png
```

Expect roughly 20–60 s. Check the note line to see which model was chosen; if
the image is a clean original you may prefer `-m nomos8k-dat` explicitly.

### SOP-3 — Large batch, throughput first

```powershell
atomik run D:\bulk -r -t x2 -q fast -f jpg --quality 92 -o D:\bulk_out
```

### SOP-4 — Continuous drop folder

Leave this running in its own terminal:

```powershell
atomik watch --archive
```

Anything dropped into `input\` is upscaled to 4K and the source is filed into
`archive\`. Stop with Ctrl-C.

### SOP-5 — Preparing for offline work

```powershell
atomik models --download nomos-webphoto nomos8k-dat anime fast2x
atomik bench --size 1024      # also warms the tile cache for each model
```

### SOP-6 — After a GPU driver update

```powershell
atomik doctor
Remove-Item "D:\- Dev -\atomik-factory\atomik-upscale\.cache\tile_sizes.json"
atomik bench --size 1024
```

Clearing the cache forces tile timings to be re-measured against the new driver.

### SOP-7 — Verifying before committing to a big job

```powershell
# see the plan and model choice for one representative file, cheaply
atomik run sample.jpg -t x6 --json
```

Read the `plan` and `model` fields, then run the full batch with the same flags.

---

## 17. Troubleshooting

**`atomik` is not recognised**
You have not put it on PATH. Use `.\atomik-upscale.cmd` from
`D:\- Dev -\atomik-factory\atomik-upscale`, or follow
[section 1C](#1-invoking-the-tool).

**`error: uv trampoline failed to canonicalize script path`**
**or `No module named atomik_upscale`**
The project folder has been moved. Both the generated `.venv\Scripts\*.exe`
launchers and the editable install record absolute paths at install time, so a
move leaves them pointing at the old location. Re-link the venv in place:

```powershell
cd "D:\- Dev -\atomik-factory\atomik-upscale"
uv sync --reinstall-package atomik-upscale
```

That rewrites `atomik.exe` and the `.pth` file. If other venv scripts
(`torchrun.exe`, `mcp.exe`, ...) are also needed, use `uv sync --reinstall`
instead, which rebuilds every launcher. `.\atomik-upscale.cmd` itself is
move-proof — it resolves the venv relative to its own location — but the
package still has to be re-linked before it can import.

**"No images found in ..."**
The inbox is empty, or the files have an unsupported extension. Check
[section 3](#3-folder-layout) for the accepted list. Remember that folder
arguments are non-recursive unless you pass `-r`.

**Everything is skipped**
Your sources are already at or above the target. Ask for more: `-t x2` scales
relative to the source and always does work, whereas `-t 4k` is a ceiling.

**"CUDA is not available" in `doctor`**
The GPU is not being seen. Check `nvidia-smi` runs, that no other process has
exhausted VRAM, and that the driver is current. Until fixed, everything runs on
CPU at roughly 30–60x slower.

**Out of memory**
Handled automatically — the tile size is halved and the run retries, so a
background game or another CUDA process degrades speed rather than failing. If
it persists, pin a smaller tile: `--tile 256`. If it still fails at the minimum,
switch to a lighter model: `-m nomos-webphoto`.

**First run of a model is slow**
Expected, once per model. It downloads weights (2–147 MB) and spends a few
seconds timing tile sizes. Both are cached in `models\` and `.cache\`. Prefetch
with SOP-5 to get it out of the way.

**A photo came out looking painted or smeared**
The artwork detector misfired. Force the photo model: `-m nomos-webphoto`, or
`-q quality` for `nomos8k-dat`.

**Illustration came out soft**
The opposite case: `-m anime`.

**Output is larger on disk than expected**
PNG is lossless; an 8K PNG can exceed 15 MB. Use `-f jpg --quality 95` or
`-f webp --quality 95` for delivery.

**Transparency was lost**
JPEG cannot store alpha. Use `-f png` or `-f webp`, and do not pass `--no-alpha`.

**Garbled characters in the terminal**
The tool forces UTF-8 output, but a very old console host may still struggle.
Use Windows Terminal, or add `--json` for plain ASCII.

**Results changed after a driver update**
Re-run SOP-6 to re-measure tile timings.

---

## Appendix — full flag matrix

Flags shared by `run` and `watch`:

```
-t, --to TARGET          x2|x4|x6|4k|8k|3840x2160|5000     (default: 4k)
-m, --model ID           force a model                     (default: automatic)
-q, --tier TIER          fast|balanced|quality             (default: balanced)
-f, --format FMT         png|jpg|webp|tiff                 (default: png)
    --quality N          JPEG/WebP quality 1-100           (default: 95)
    --bits N             8|16                              (default: 8)
-o, --out DIR            output directory                  (default: output\)
    --tile N             force tile size                   (default: autotuned)
    --overlap N          tile overlap in pixels            (default: 32)
    --fp32               disable reduced precision
    --channels-last      alternate memory layout
    --no-alpha           discard transparency
    --archive            move sources to archive\ on success
    --overwrite          replace existing outputs
    --strip-metadata     drop EXIF and ICC
    --suffix TEMPLATE    {stem} {label} {model}            (default: {stem}__{label})
    --json               machine-readable output
```

`run` only: `paths...`, `-r` / `--recursive`
`watch` only: `dir`, `--settle SECONDS` (default 1.5), `--no-warm`
`models` only: `--download ID [ID ...]`
`doctor` only: `--json`
`bench` only: `ID...`, `--size N` (default 512), `--fp32`
