#!/usr/bin/env python3
"""
transcribe_folder.py
=====================

Recursively transcribe every video file in a folder using OpenAI's
open-source Whisper model, running entirely locally (no API calls).

------------------------------------------------------------------
INSTALLATION
------------------------------------------------------------------

1. Install Python 3.9+.

2. Install FFmpeg (required by Whisper to decode audio/video):
     - Windows:  https://ffmpeg.org/download.html  (or `winget install ffmpeg`
                 / `choco install ffmpeg`), then ensure ffmpeg.exe is on PATH.
     - macOS:    brew install ffmpeg
     - Linux:    sudo apt install ffmpeg   (or your distro's package manager)

3. Install PyTorch appropriate for your machine (CPU-only or CUDA build).
   See https://pytorch.org/get-started/locally/ for the exact command for
   your platform/GPU. Example (CPU only):
     pip install torch --index-url https://download.pytorch.org/whl/cpu

4. Install openai-whisper:
     pip install -U openai-whisper

   (Optional, for automatic hardware-based model suggestions)
     pip install psutil

------------------------------------------------------------------
EXAMPLE USAGE
------------------------------------------------------------------

    python transcribe_folder.py "D:\\Videos"

    python transcribe_folder.py "/home/user/videos" --model large

    python transcribe_folder.py "./Videos" --language English

    python transcribe_folder.py "./Videos" --force

    python transcribe_folder.py "./Videos" --device cuda --model small

------------------------------------------------------------------
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# Video extensions we will look for (case-insensitive).
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".wmv",
    ".flv", ".webm", ".mpeg", ".mpg", ".m4v",
}

VALID_MODELS = ["tiny", "base", "small", "medium", "large"]


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Recursively transcribe every video file in a folder using "
            "local OpenAI Whisper (no API calls)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python transcribe_folder.py "D:\\Videos"\n'
            '  python transcribe_folder.py "/home/user/videos" --model large\n'
            '  python transcribe_folder.py "./Videos" --language English\n'
        ),
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the folder to scan recursively for video files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=VALID_MODELS,
        help="Whisper model size to use (default: medium).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help=(
            "Force a specific language (e.g. 'English', 'Spanish'). "
            "If omitted, Whisper auto-detects the language."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help=(
            "Device to run inference on. If omitted, CUDA is used "
            "automatically when available, otherwise CPU."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe and overwrite .txt files that already exist.",
    )
    return parser.parse_args()


def suggest_model(device: str) -> None:
    """
    Print a suggested Whisper model size based on detected hardware.
    This is advisory only -- it never overrides the user's --model choice.
    """
    try:
        import torch
    except ImportError:
        print("[Hardware check] PyTorch not importable yet -- skipping model suggestion.")
        return

    suggestion = "small"
    details = []

    if device == "cuda" and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            details.append(f"GPU: {props.name} ({vram_gb:.1f} GB VRAM)")
            if vram_gb >= 10:
                suggestion = "large"
            elif vram_gb >= 5:
                suggestion = "medium"
            elif vram_gb >= 3:
                suggestion = "small"
            else:
                suggestion = "base"
        except Exception:
            suggestion = "medium"
    else:
        # CPU-based suggestion using available system RAM, if psutil is present.
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            details.append(f"System RAM: {ram_gb:.1f} GB")
            if ram_gb >= 16:
                suggestion = "medium"
            elif ram_gb >= 8:
                suggestion = "small"
            else:
                suggestion = "base"
        except ImportError:
            details.append("psutil not installed -- defaulting suggestion to 'small' for CPU")
            suggestion = "small"

    detail_str = "; ".join(details) if details else "no details available"
    print(f"[Hardware check] {detail_str}")
    print(f"[Hardware check] Suggested model for this machine: '{suggestion}' "
          f"(you can pass --model {suggestion} to use it; your current choice is respected either way)")


def resolve_device(requested_device: str) -> str:
    """Determine which device to run on, auto-selecting CUDA if available."""
    if requested_device:
        return requested_device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def find_video_files(root: Path) -> list:
    """Recursively find all video files under root matching VIDEO_EXTENSIONS."""
    videos = []
    for path in root.rglob("*"):
        try:
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path)
        except OSError as exc:
            # Permission errors, broken symlinks, etc. -- skip but report.
            print(f"[WARN] Could not access '{path}': {exc}")
    return sorted(videos)


def transcribe_video(model, video_path: Path, language: str) -> str:
    """Run Whisper transcription on a single video file and return the text."""
    transcribe_kwargs = {"verbose": False}
    if language:
        transcribe_kwargs["language"] = language

    result = model.transcribe(str(video_path), **transcribe_kwargs)
    return result.get("text", "").strip()


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def main() -> int:
    args = parse_args()

    input_folder = Path(args.input_folder).expanduser().resolve()
    if not input_folder.exists():
        print(f"[ERROR] Input folder does not exist: {input_folder}")
        return 1
    if not input_folder.is_dir():
        print(f"[ERROR] Input path is not a directory: {input_folder}")
        return 1

    device = resolve_device(args.device)
    print(f"[Info] Using device: {device}")
    suggest_model(device)

    print(f"[Info] Scanning '{input_folder}' recursively for video files...")
    video_files = find_video_files(input_folder)
    total_found = len(video_files)

    if total_found == 0:
        print("[Info] No video files found. Nothing to do.")
        return 0

    print(f"[Info] Found {total_found} video file(s).")

    # Load the Whisper model once and reuse it for every file.
    print(f"[Info] Loading Whisper model '{args.model}' (this may take a while on first run)...")
    try:
        import whisper
    except ImportError:
        print(
            "[ERROR] The 'openai-whisper' package is not installed.\n"
            "        Install it with: pip install -U openai-whisper"
        )
        return 1

    try:
        model = whisper.load_model(args.model, device=device)
    except Exception as exc:
        print(f"[ERROR] Failed to load Whisper model '{args.model}': {exc}")
        traceback.print_exc()
        return 1

    print(f"[Info] Model loaded successfully.\n")

    succeeded = 0
    skipped = 0
    failed = 0
    failed_files = []

    start_time = time.monotonic()

    for index, video_path in enumerate(video_files, start=1):
        progress_prefix = f"[{index}/{total_found}]"
        output_path = video_path.with_suffix(".txt")

        print(f"{progress_prefix} Processing: {video_path}")

        if output_path.exists() and not args.force:
            print(f"{progress_prefix} SKIPPED (transcript already exists): {output_path}")
            skipped += 1
            continue

        file_start = time.monotonic()
        try:
            transcript_text = transcribe_video(model, video_path, args.language)

            # Write as plain UTF-8 text, no other formats.
            output_path.write_text(transcript_text, encoding="utf-8")

            elapsed = time.monotonic() - file_start
            print(
                f"{progress_prefix} SUCCESS ({format_duration(elapsed)}): "
                f"wrote '{output_path.name}'"
            )
            succeeded += 1

        except Exception as exc:
            elapsed = time.monotonic() - file_start
            print(
                f"{progress_prefix} FAILED ({format_duration(elapsed)}): "
                f"'{video_path.name}' -- {exc}"
            )
            traceback.print_exc()
            failed += 1
            failed_files.append(video_path)
            # Continue processing remaining videos even if this one failed.
            continue

    total_elapsed = time.monotonic() - start_time

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total videos found:      {total_found}")
    print(f"Successfully transcribed: {succeeded}")
    print(f"Skipped (already done):  {skipped}")
    print(f"Failed:                  {failed}")
    if failed_files:
        print("Failed files:")
        for f in failed_files:
            print(f"  - {f}")
    print(f"Total execution time:    {format_duration(total_elapsed)}")
    print("=" * 60)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
