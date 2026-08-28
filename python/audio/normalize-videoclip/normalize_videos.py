"""Legacy entry point, kept for backward compatibility.

The implementation this originally contained moved to the
`normalize_videoclip` package (see normalize_videoclip/core.py) — same
ffmpeg behavior, plus atomic writes, parallelism, a real CLI, and an
optional MCP server. See README.md and docs/DESIGN.md for the full story
and migration notes.

Install the package (`pip install -e .`) and use the `normalize-videoclip`
command directly going forward. This file just forwards to it so old
muscle memory (`python normalize_videos.py`, with no arguments, defaulting
to ./Library -> ./Library_Normalized) still works.
"""

from __future__ import annotations

import sys

from normalize_videoclip.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += ["Library", "Library_Normalized"]
    raise SystemExit(main())
