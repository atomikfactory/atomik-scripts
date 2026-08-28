"""Core library: discovery, planning, and execution.

No printing, no argparse, no MCP — this module is the single source of
truth for *what* normalization does, imported by both cli.py and
mcp_server.py. Keeping it print-free is what makes it directly unit
testable and safely embeddable in another program or agent.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from .config import NormalizeConfig
from .exceptions import InvalidInputDirectoryError
from .ffmpeg_utils import build_ffmpeg_command, check_ffmpeg_available
from .models import NormalizeSummary, TaskResult, TaskStatus, VideoTask

logger = logging.getLogger("normalize_videoclip")

OnResult = Callable[[TaskResult], None]


def discover_video_files(input_dir: Path | str, config: NormalizeConfig) -> list[Path]:
    """Find video files directly inside ``input_dir`` (non-recursive), deduplicated.

    Case-insensitive filesystems (Windows, default macOS) would otherwise
    return the same file twice for a lowercase and uppercase glob pattern;
    dedup by resolved path avoids double-counting or double-processing it.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise InvalidInputDirectoryError(f"input directory does not exist: {input_dir}")

    seen: set[Path] = set()
    files: list[Path] = []
    for ext in config.extensions:
        for pattern in (f"*{ext}", f"*{ext.upper()}"):
            for candidate in input_dir.glob(pattern):
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                files.append(candidate)

    return sorted(files, key=lambda p: p.name.lower())


def plan_tasks(files: list[Path], output_dir: Path | str, config: NormalizeConfig) -> list[VideoTask]:
    """Pair each discovered file with its destination and re-encode requirement.

    Pure function, no filesystem access beyond what ``files`` already
    represents — this is what lets an MCP client call it as a safe,
    read-only "preview" before anything is written.
    """
    output_dir = Path(output_dir)
    tasks = []
    for f in files:
        is_legacy = f.suffix.lower() in config.legacy_extensions
        dest_name = f"{f.stem}.mp4" if is_legacy else f.name
        tasks.append(VideoTask(source=f, destination=output_dir / dest_name, requires_reencode=is_legacy))
    return tasks


def _run_single(task: VideoTask, config: NormalizeConfig) -> TaskResult:
    start = time.monotonic()

    if task.destination.exists() and not config.overwrite:
        return TaskResult(task, TaskStatus.SKIPPED, "destination already exists", time.monotonic() - start)

    if config.dry_run:
        return TaskResult(task, TaskStatus.DRY_RUN, "would encode", time.monotonic() - start)

    task.destination.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory (same filesystem, so the
    # final rename is atomic) and only os.replace() it into place once
    # ffmpeg has exited 0 and produced non-empty output. This is the fix for
    # the original script's biggest correctness bug: if a run is killed
    # mid-encode, no partial file is ever left at the destination path, so a
    # re-run can't mistake a truncated file for a completed one.
    fd, tmp_name = tempfile.mkstemp(
        dir=task.destination.parent,
        prefix=f".{task.destination.stem}.",
        suffix=f".part{task.destination.suffix}",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    cmd = build_ffmpeg_command(task, config, output_path=tmp_path)
    logger.debug("running: %s", " ".join(cmd))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
        if proc.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
            message = (proc.stderr or "").strip()
            if not message:
                message = f"ffmpeg exited with code {proc.returncode}"
            return TaskResult(task, TaskStatus.FAILED, message[-2000:], time.monotonic() - start)
        os.replace(tmp_path, task.destination)
        return TaskResult(task, TaskStatus.SUCCESS, "", time.monotonic() - start)
    except OSError as exc:
        return TaskResult(task, TaskStatus.FAILED, str(exc), time.monotonic() - start)
    finally:
        tmp_path.unlink(missing_ok=True)


def normalize_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    config: NormalizeConfig,
    on_result: OnResult | None = None,
) -> NormalizeSummary:
    """Discover, plan, and execute normalization for every video in ``input_dir``.

    Files are processed concurrently (bounded by ``config.workers``) and
    ``on_result`` — if given — is invoked as each one finishes, in
    completion order, so a caller can stream progress. The returned summary
    is always sorted by source filename for deterministic output.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = discover_video_files(input_dir, config)
    if not files:
        return NormalizeSummary(results=[])

    if not config.dry_run:
        check_ffmpeg_available(config)
        output_dir.mkdir(parents=True, exist_ok=True)

    tasks = plan_tasks(files, output_dir, config)

    results: list[TaskResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
        futures = [pool.submit(_run_single, task, config) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            if on_result is not None:
                on_result(result)

    results.sort(key=lambda r: r.task.source.name.lower())
    return NormalizeSummary(results=results)
