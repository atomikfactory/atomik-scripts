"""Project directory layout. Override the root with ATOMIK_HOME."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ATOMIK_HOME") or Path(__file__).resolve().parents[2])

INPUT_DIR = Path(os.environ.get("ATOMIK_INPUT") or ROOT / "input")
OUTPUT_DIR = Path(os.environ.get("ATOMIK_OUTPUT") or ROOT / "output")
ARCHIVE_DIR = Path(os.environ.get("ATOMIK_ARCHIVE") or ROOT / "archive")
MODELS_DIR = Path(os.environ.get("ATOMIK_MODELS") or ROOT / "models")
CACHE_DIR = Path(os.environ.get("ATOMIK_CACHE") or ROOT / ".cache")

for _d in (INPUT_DIR, OUTPUT_DIR, ARCHIVE_DIR, MODELS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".jpe", ".webp", ".bmp", ".tif", ".tiff", ".jfif", ".ppm",
}
