"""Drop-folder watcher.

Files copied into the inbox are picked up once they stop growing -- a large
PNG dragged in from another drive is not readable the instant Windows creates
the directory entry.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from . import pipeline
from .hardware import Hardware
from .paths import IMAGE_SUFFIXES, OUTPUT_DIR


class _Handler(FileSystemEventHandler):
    def __init__(self, sink: queue.Queue):
        self.sink = sink

    def _offer(self, path_str: str) -> None:
        p = Path(path_str)
        if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."):
            self.sink.put(p)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._offer(str(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._offer(str(event.dest_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._offer(str(event.src_path))


def _wait_until_stable(path: Path, settle: float, timeout: float = 300.0) -> bool:
    """Return True once the file has held the same size for `settle` seconds."""
    deadline = time.time() + timeout
    last = -1
    stable_since = 0.0
    while time.time() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        now = time.time()
        if size == last and size > 0:
            if now - stable_since >= settle:
                return True
        else:
            last, stable_since = size, now
        time.sleep(0.25)
    return False


def watch_folder(folder: Path, hw: Hardware, opts: pipeline.Options, *,
                 out_dir: Path | None = None, console=None,
                 settle: float = 1.5, warm: bool = True) -> int:
    from rich.console import Console
    console = console or Console()
    folder.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir or OUTPUT_DIR

    if warm:
        console.print("[dim]preloading model…[/]")
        try:
            pipeline.warm_up(hw, opts)
        except Exception as exc:
            console.print(f"[yellow]warm-up skipped:[/] {exc}")

    console.print(
        f"[bold cyan]watching[/] {folder}\n"
        f"[dim]target {opts.target} · tier {opts.tier} · "
        f"model {opts.model or 'auto'} · out {out_dir}\n"
        f"Ctrl-C to stop.[/]\n"
    )

    sink: queue.Queue[Path] = queue.Queue()
    handler = _Handler(sink)
    observer = Observer()
    observer.schedule(handler, str(folder), recursive=False)
    observer.start()

    # Anything already sitting in the folder counts as pending work.
    for existing in pipeline.discover([folder]):
        sink.put(existing)

    # Fingerprint of every file already handled, so the several events that a
    # single copy emits (created, then one or more modified) cannot produce
    # several outputs. A time-based debounce cannot do this job: waiting for
    # the file to stop growing already outlasts any sane debounce window.
    done: dict[Path, tuple[int, float]] = {}
    stop = threading.Event()
    processed = 0

    def fingerprint(p: Path) -> tuple[int, float] | None:
        try:
            st = p.stat()
            return (st.st_size, st.st_mtime)
        except OSError:
            return None

    try:
        while not stop.is_set():
            try:
                path = sink.get(timeout=0.5)
            except queue.Empty:
                continue

            if not path.exists() or fingerprint(path) == done.get(path):
                continue
            if not _wait_until_stable(path, settle):
                console.print(f"[yellow]skip[/] {path.name} — still being written")
                continue
            # Re-check after settling: the file that just stopped changing may
            # be the one already processed, arriving via a second event.
            fp = fingerprint(path)
            if fp is None or fp == done.get(path):
                continue
            done[path] = fp

            console.print(f"[bold]{path.name}[/] [dim]…[/]")
            res = pipeline.process(path, hw, opts, out_dir=out_dir).finalize()
            if res.ok:
                processed += 1
                sw, sh = res.src_size
                dw, dh = res.dst_size
                console.print(
                    f"  [green]ok[/] {sw}x{sh} -> [bold]{dw}x{dh}[/] "
                    f"[dim]{res.model} · {res.seconds:.1f}s -> {res.dst.name}[/]"
                )
            elif res.skipped:
                console.print(f"  [yellow]skip[/] {res.skipped}")
            else:
                console.print(f"  [red]FAIL[/] {res.error}")

            # Drop bookkeeping for files that are gone (e.g. moved by --archive),
            # so re-dropping the same name later is treated as new work.
            for gone in [p for p in done if not p.exists()]:
                done.pop(gone, None)

    except KeyboardInterrupt:
        console.print(f"\n[dim]stopped after {processed} image(s)[/]")
    finally:
        observer.stop()
        observer.join(timeout=5)

    return 0
