"""Command line interface."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn, TaskProgressColumn,
                           TextColumn, TimeElapsedColumn)
from rich.prompt import Prompt
from rich.table import Table

from . import engine, hardware, models, planner, pipeline
from .paths import INPUT_DIR, MODELS_DIR, OUTPUT_DIR

# The legacy Windows console defaults to cp1252, which cannot encode the
# spinner glyphs or the separators used below. Ask for UTF-8 before rich
# captures the stream, and degrade gracefully if the terminal refuses.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

console = Console()

TIERS = ("fast", "balanced", "quality")
FORMATS = ("png", "jpg", "webp", "tiff")


# --- shared option wiring ----------------------------------------------------

def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-t", "--to", dest="target", default=None,
                   help="target: x2 | x4 | x6 | 4k | 8k | 3840x2160 | 5000 "
                        "(default: 4k, i.e. long edge >= 3840)")
    p.add_argument("-m", "--model", default=None,
                   help="force a model id (see `atomik models`); default is automatic")
    p.add_argument("-q", "--tier", choices=TIERS, default="balanced",
                   help="quality/speed tier used by automatic model choice")
    p.add_argument("-f", "--format", dest="fmt", choices=FORMATS, default="png")
    p.add_argument("--quality", type=int, default=95, help="JPEG/WebP quality")
    p.add_argument("--bits", type=int, choices=(8, 16), default=8)
    p.add_argument("-o", "--out", type=Path, default=None,
                   help=f"output directory (default: {OUTPUT_DIR})")
    p.add_argument("--tile", type=int, default=None,
                   help="force tile size in pixels; default autotunes to free VRAM")
    p.add_argument("--overlap", type=int, default=engine.DEFAULT_OVERLAP)
    p.add_argument("--fp32", action="store_true", help="disable fp16 (slower, rarely needed)")
    p.add_argument("--channels-last", action="store_true",
                   help="channels-last memory format; helps some conv archs")
    p.add_argument("--no-alpha", action="store_true", help="discard transparency")
    p.add_argument("--archive", action="store_true",
                   help="move each source into archive/ once it succeeds")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--strip-metadata", action="store_true")
    p.add_argument("--suffix", default="{stem}__{label}",
                   help="output name template; fields: {stem} {label} {model}")
    p.add_argument("--json", action="store_true", help="emit machine-readable results")


def options_from(args) -> pipeline.Options:
    # Catch a bad model id once, here, rather than letting every file in the
    # batch fail separately with a raw KeyError.
    if args.model and args.model not in models.REGISTRY:
        console.print(f"[red]unknown model[/] {args.model!r}")
        console.print(f"[dim]known: {', '.join(sorted(models.REGISTRY))}[/]")
        raise SystemExit(2)
    return pipeline.Options(
        target=args.target or "4k",
        model=args.model,
        tier=args.tier,
        fmt=args.fmt,
        quality=args.quality,
        bits=args.bits,
        tile=args.tile,
        overlap=args.overlap,
        force_fp32=args.fp32,
        channels_last=args.channels_last,
        keep_alpha=not args.no_alpha,
        archive=args.archive,
        overwrite=args.overwrite,
        strip_metadata=args.strip_metadata,
        suffix=args.suffix,
    )


# --- interactive picker ------------------------------------------------------

def pick_target(files: list[Path]) -> str:
    """Ask for the scale, showing what each choice means for the first image."""
    try:
        with Image.open(files[0]) as im:
            w, h = im.size
    except Exception:
        w = h = 0

    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column(" ", style="bold cyan", width=3)
    table.add_column("target")
    table.add_column("result" if w else "", style="dim")

    choices = ["1", "2", "3", "4", "5", "6"]
    rows = [
        ("1", "x2", 2.0), ("2", "x4", 4.0), ("3", "x6", 6.0),
    ]
    long_edge = max(w, h) or 1
    rows += [
        ("4", "4K  (long edge 3840)", planner.PRESETS["4k"] / long_edge),
        ("5", "8K  (long edge 7680)", planner.PRESETS["8k"] / long_edge),
    ]
    for key, label, mult in rows:
        size = f"{round(w * mult)}x{round(h * mult)}" if w else ""
        if w and mult <= 1.0:
            size += "  (source already larger)"
        table.add_row(key, label, size)
    table.add_row("6", "custom", "e.g. 3840x2160, x3, 5000")

    header = f"{len(files)} image(s)"
    if w:
        header += f" · first is {w}x{h}"
    console.print(Panel(table, title="[bold]Choose output size[/]",
                        subtitle=f"[dim]{header}[/]", border_style="cyan"))

    mapping = {"1": "x2", "2": "x4", "3": "x6", "4": "4k", "5": "8k"}
    try:
        sel = Prompt.ask("Selection", choices=choices, default="4", console=console)
        if sel != "6":
            return mapping[sel]
        return Prompt.ask("Target (x3, 3840x2160, 5000, 6k, ...)",
                          default="4k", console=console)
    except (EOFError, KeyboardInterrupt):
        # stdin can look interactive and still have nothing to read (piped
        # runs, CI, some terminal emulators). Take the default rather than
        # dying on a traceback.
        console.print("[dim]no input available — using 4K[/]")
        return "4k"


# --- progress plumbing -------------------------------------------------------

def run_batch(files: list[Path], hw, opts: pipeline.Options,
              out_dir: Path | None, as_json: bool) -> int:
    if as_json:
        results = pipeline.process_many(files, hw, opts, out_dir=out_dir)
        payload = [
            {
                "src": str(r.src), "dst": str(r.dst) if r.dst else None,
                "ok": r.ok, "src_size": list(r.src_size), "dst_size": list(r.dst_size),
                "model": r.model, "content": r.content, "plan": r.plan,
                "seconds": round(r.seconds, 3), "tiles": r.tiles, "dtype": r.dtype,
                "tile_size": r.tile_size, "peak_vram_mb": r.peak_vram_mb,
                "skipped": r.skipped, "error": r.error, "notes": r.notes,
            }
            for r in results
        ]
        print(json.dumps(payload, indent=2))
        return 0 if all(r.ok or r.skipped for r in results) else 1

    ok = failed = skipped = 0
    started = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.fields[name]}"),
        BarColumn(bar_width=28),
        TaskProgressColumn(),
        TextColumn("[dim]{task.fields[stage]}"),
        TimeElapsedColumn(),
        console=console, transient=False,
    ) as prog:
        task = prog.add_task("upscale", total=len(files) * 100,
                             name="starting", stage="")
        state = {"index": 0}

        def on_start(src: Path, i: int, n: int) -> None:
            state["index"] = i - 1
            prog.update(task, name=f"[{i}/{n}] {src.name}", stage="loading",
                        completed=(i - 1) * 100)

        def on_progress(stage: str, frac: float) -> None:
            prog.update(task, stage=stage,
                        completed=state["index"] * 100 + frac * 100)

        def on_result(r: pipeline.Result) -> None:
            nonlocal ok, failed, skipped
            if r.skipped:
                skipped += 1
                console.print(f"  [yellow]skip[/] {r.src.name} — {r.skipped}")
            elif r.ok:
                ok += 1
                sw, sh = r.src_size
                dw, dh = r.dst_size
                console.print(
                    f"  [green]ok[/]   {r.src.name} [dim]{sw}x{sh} ->[/] "
                    f"[bold]{dw}x{dh}[/] [dim]· {r.model} · {r.dtype} · "
                    f"{r.tiles} tiles @{r.tile_size}px · {r.seconds:.1f}s "
                    f"· {r.peak_vram_mb} MB VRAM[/]"
                )
                for n in r.notes:
                    console.print(f"       [dim]· {n}[/]")
            else:
                failed += 1
                console.print(f"  [red]FAIL[/] {r.src.name} — {r.error}")

        pipeline.process_many(files, hw, opts, out_dir=out_dir,
                              on_start=on_start, on_result=on_result,
                              progress=on_progress)
        prog.update(task, completed=len(files) * 100, stage="done")

    elapsed = time.perf_counter() - started
    console.print(
        f"\n[bold]{ok} upscaled[/]"
        + (f", [yellow]{skipped} skipped[/]" if skipped else "")
        + (f", [red]{failed} failed[/]" if failed else "")
        + f" in {elapsed:.1f}s → [cyan]{out_dir or OUTPUT_DIR}[/]"
    )
    return 0 if failed == 0 else 1


# --- commands ----------------------------------------------------------------

def cmd_run(args) -> int:
    hw = hardware.detect()
    hardware.apply_global_tuning(hw)

    files = pipeline.discover([Path(p) for p in args.paths] or None,
                              recursive=args.recursive)
    if not files:
        where = ", ".join(args.paths) if args.paths else str(INPUT_DIR)
        console.print(f"[yellow]No images found in[/] {where}")
        console.print(f"[dim]Drop files into {INPUT_DIR} and run `atomik run`.[/]")
        return 1

    if args.target is None and sys.stdin.isatty() and not args.json:
        args.target = pick_target(files)
    opts = options_from(args)

    if not args.json:
        console.print(f"[dim]{hw.summary()}[/]\n")
    return run_batch(files, hw, opts, args.out, args.json)


def cmd_models(args) -> int:
    if args.download:
        for mid in args.download:
            spec = models.get(mid)
            with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                          BarColumn(), TaskProgressColumn(),
                          console=console) as prog:
                task = prog.add_task(f"{spec.id}", total=spec.size_mb * 1024 * 1024)
                models.download(spec, lambda d, t: prog.update(task, completed=d,
                                                               total=t or None))
        console.print("[green]done[/]")
        return 0

    table = Table(title="Model registry", header_style="bold", box=None, pad_edge=False)
    table.add_column("id", style="cyan bold")
    table.add_column("x", justify="center")
    table.add_column("arch")
    table.add_column("tier")
    table.add_column("for")
    table.add_column("MB", justify="right")
    table.add_column("", justify="center")
    table.add_column("notes", style="dim", max_width=52)

    order = {"quality": 0, "balanced": 1, "fast": 2}
    for spec in sorted(models.REGISTRY.values(),
                       key=lambda s: (order.get(s.tier, 9), s.id)):
        table.add_row(
            spec.id, str(spec.scale), spec.arch, spec.tier, spec.content,
            str(spec.size_mb),
            "[green]✓[/]" if spec.available else "[dim]–[/]",
            spec.blurb,
        )
    console.print(table)
    have = models.disk_usage_bytes() / 1024**2
    console.print(f"\n[dim]{MODELS_DIR} · {have:.0f} MB cached. "
                  f"Missing weights download on first use, or "
                  f"`atomik models --download <id>`.[/]")
    return 0


def cmd_doctor(args) -> int:
    hw = hardware.detect()
    hardware.apply_global_tuning(hw)
    rep = hardware.hardware_report(hw)

    if args.json:
        rep["models_cached"] = [m.id for m in models.REGISTRY.values() if m.available]
        print(json.dumps(rep, indent=2, default=str))
        return 0

    t = Table(box=None, show_header=False, pad_edge=False)
    t.add_column(style="dim", width=18)
    t.add_column()
    t.add_row("GPU", f"{rep['gpu_name']}")
    t.add_row("VRAM", f"{rep['vram_total_mb']} MB total, "
                      f"{rep.get('vram_free_mb', '?')} MB free")
    t.add_row("compute", str(rep["compute_cap"]))
    t.add_row("CUDA / torch", f"{rep['cuda_version']} / {rep['torch']}")
    t.add_row("precision", "fp16 tensor cores" if hw.supports_fp16 else "fp32 only")
    t.add_row("TF32", "on" if hw.supports_tf32 else "unavailable")
    t.add_row("CPU", f"{rep['cpu']} · {rep['cpu_threads']} threads")
    t.add_row("RAM", f"{rep['ram_gb']} GB")
    t.add_row("inbox", str(INPUT_DIR))
    t.add_row("outbox", str(OUTPUT_DIR))
    console.print(Panel(t, title="[bold]atomik doctor[/]", border_style="cyan"))

    if not hw.is_cuda:
        console.print("[red]CUDA is not available[/] — everything will run on CPU, "
                      "roughly 30-60x slower.")
        return 1

    console.print("[dim]running a 512x512 smoke test through the default model…[/]")
    import numpy as np
    spec = models.get(models.DEFAULT_BY_TIER["balanced"])
    up = engine.get_upscaler(spec, hw)
    probe = np.random.rand(512, 512, 3).astype("float32")
    # First call may spend a few seconds measuring the tile-cost curve; timing
    # that would report tuning cost as though it were inference cost.
    up.run(probe[:128, :128])
    out, stats = up.run(probe)
    console.print(
        f"[green]ok[/] {spec.id}: 512x512 -> {out.shape[1]}x{out.shape[0]} "
        f"in {stats.seconds:.2f}s ({stats.dtype}, tile {stats.tile_size}, "
        f"peak {stats.peak_vram_mb} MB VRAM)"
    )
    return 0


def cmd_bench(args) -> int:
    import numpy as np
    hw = hardware.detect()
    hardware.apply_global_tuning(hw)

    ids = args.models or [m.id for m in models.REGISTRY.values()]
    w = h = args.size
    probe = np.random.rand(h, w, 3).astype("float32")

    table = Table(title=f"{w}x{h} source · {hw.gpu_name}", header_style="bold", box=None)
    table.add_column("model", style="cyan")
    table.add_column("x", justify="center")
    table.add_column("dtype")
    table.add_column("tile", justify="right")
    table.add_column("tiles", justify="right")
    table.add_column("sec", justify="right")
    table.add_column("MPix/s", justify="right")
    table.add_column("VRAM MB", justify="right")

    for mid in ids:
        spec = models.get(mid)
        try:
            up = engine.get_upscaler(spec, hw, force_fp32=args.fp32)
            up.run(probe[:64, :64])          # warm-up, excluded from timing
            out, st = up.run(probe)
            mpix = (out.shape[0] * out.shape[1] / 1e6) / max(st.seconds, 1e-6)
            table.add_row(mid, str(up.scale), st.dtype, str(st.tile_size),
                          str(st.tiles), f"{st.seconds:.2f}", f"{mpix:.1f}",
                          str(st.peak_vram_mb))
        except Exception as exc:
            table.add_row(mid, "-", "-", "-", "-", "-", "-", f"[red]{type(exc).__name__}[/]")
        engine.clear_cache()

    console.print(table)
    console.print("[dim]MPix/s counts output pixels. Compare within a column only; "
                  "the 2x models produce a quarter the pixels per pass.[/]")
    return 0


def cmd_watch(args) -> int:
    from .watch import watch_folder
    hw = hardware.detect()
    hardware.apply_global_tuning(hw)
    opts = options_from(args)
    if args.target is None:
        opts.target = "4k"
    return watch_folder(Path(args.dir or INPUT_DIR), hw, opts,
                        out_dir=args.out, console=console,
                        settle=args.settle, warm=not args.no_warm)


def cmd_mcp(args) -> int:
    from .mcp_server import run_stdio
    run_stdio()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="atomik",
        description="GPU image upscaler tuned for this machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  atomik run                     # inbox + interactive size picker\n"
            "  atomik run -t x4               # everything in input/ at 4x\n"
            "  atomik run photo.jpg -t 8k -q quality\n"
            "  atomik watch --archive         # process files as they land\n"
            "  atomik models                  # what is installed\n"
            "  atomik doctor                  # hardware + smoke test\n"
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    run = sub.add_parser("run", help="upscale files, or everything in input/")
    run.add_argument("paths", nargs="*", help="files or folders (default: input/)")
    run.add_argument("-r", "--recursive", action="store_true")
    add_common(run)
    run.set_defaults(func=cmd_run)

    w = sub.add_parser("watch", help="watch a folder and upscale on arrival")
    w.add_argument("dir", nargs="?", default=None)
    w.add_argument("--settle", type=float, default=1.5,
                   help="seconds a file must stop changing before it is picked up")
    w.add_argument("--no-warm", action="store_true", help="skip model preload")
    add_common(w)
    w.set_defaults(func=cmd_watch)

    m = sub.add_parser("models", help="list or fetch models")
    m.add_argument("--download", nargs="+", metavar="ID")
    m.set_defaults(func=cmd_models)

    d = sub.add_parser("doctor", help="hardware report and smoke test")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_doctor)

    b = sub.add_parser("bench", help="time each model on this GPU")
    b.add_argument("models", nargs="*", metavar="ID")
    b.add_argument("--size", type=int, default=512)
    b.add_argument("--fp32", action="store_true")
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("mcp", help="run the MCP server on stdio")
    s.set_defaults(func=cmd_mcp)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw)
    if not getattr(args, "cmd", None):
        # Bare `atomik`, or plain flags, behave like `atomik run`.
        args = parser.parse_args(["run", *raw])
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        return 130
    except KeyError as exc:
        console.print(f"[red]{exc.args[0] if exc.args else exc}[/]")
        return 2
    finally:
        engine.clear_cache()


if __name__ == "__main__":
    raise SystemExit(main())
