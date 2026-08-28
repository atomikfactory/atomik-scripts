# normalize-videoclip

Batch-normalize audio loudness across a folder of video files with
`ffmpeg`, re-encoding legacy containers (`.mpg`, `.mpeg`, `.webm`) to MP4
along the way since they can't legally hold AAC audio. Point it at a
folder, get back a folder of consistently-loud, broadly-compatible video
files.

Library, CLI, and an optional MCP server for agent use, all built on the
same core so behavior never drifts between them. See
[`docs/DESIGN.md`](docs/DESIGN.md) for the full engineering rationale,
architecture decisions, and what was deliberately left out.

## Why

Video from different sources (screen recordings, phone clips, downloads,
old camcorder footage) tends to have wildly inconsistent loudness and a
grab-bag of containers. This normalizes loudness to a consistent target
(`-16 LUFS` by default) across an entire library in one pass, and fixes up
containers that don't play well with modern players/AAC audio.

## Install

```bash
git clone <this-repo>
cd normalize-videoclip
pip install -e .            # CLI only, zero extra dependencies
pip install -e ".[mcp]"     # CLI + MCP server
```

Requires Python 3.11+ and `ffmpeg` on your `PATH`
(<https://ffmpeg.org/download.html>).

## CLI usage

```bash
normalize-videoclip INPUT_DIR OUTPUT_DIR [options]
```

```bash
# Preview what would happen — no files touched, ffmpeg not even required
normalize-videoclip ./videos ./videos_normalized --dry-run

# Run it
normalize-videoclip ./videos ./videos_normalized

# Re-encode with a different quality target, 8 parallel workers
normalize-videoclip ./videos ./videos_normalized --crf 20 -j 8

# Machine-readable output (for scripts / CI)
normalize-videoclip ./videos ./videos_normalized --json

# Reuse settings via a config file (CLI flags still override it)
normalize-videoclip ./videos ./videos_normalized --config my.toml
```

See `examples/config.example.toml` for every configurable key, or run
`normalize-videoclip --help` for the full flag list.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Everything succeeded (or there was nothing to do / dry run) |
| 1 | One or more files failed to encode |
| 2 | Usage error or invalid configuration |
| 3 | ffmpeg is not installed / not on `PATH` |
| 4 | The input directory does not exist |

### Safe to re-run

Re-running against the same output directory skips files that already
exist there (pass `--overwrite` to force a redo). Every encode is written
to a temp file and only moved into place atomically after it succeeds, so
an interrupted run (Ctrl+C, crash, power loss) never leaves a corrupt file
that a later run would mistake for a finished one.

## Library usage

```python
from normalize_videoclip import NormalizeConfig, normalize_directory

summary = normalize_directory(
    "videos",
    "videos_normalized",
    NormalizeConfig(video_crf=20, workers=8),
)
print(summary.succeeded, summary.failed)
for result in summary.results:
    print(result.task.source.name, result.status.value)
```

## MCP server (Claude Code / other MCP agents)

```bash
pip install -e ".[mcp]"
claude mcp add normalize-videoclip -- normalize-videoclip-mcp
```

Exposes three tools — `check_environment`, `plan_normalization`
(read-only preview) and `normalize_videos` (writes files) — so an agent can
inspect what a batch would do before committing to it. See
[`docs/DESIGN.md §4`](docs/DESIGN.md#4-mcp-evaluation) for the tool
boundaries and security notes (local-only, no path sandboxing — the server
has exactly the filesystem access of the account running it).

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests that invoke real ffmpeg encodes are skipped automatically if
`ffmpeg` isn't on `PATH`.

## Migrating from the original script

`normalize_videos.py` still exists and still runs with no arguments
(`Library -> Library_Normalized`, same as before) — it's now a thin shim
that forwards to this package. See
[`docs/DESIGN.md §7`](docs/DESIGN.md#7-migration-notes-from-the-original-script)
for the small set of behavior changes (multi-track audio/subtitles are now
preserved on the stream-copy path; exit code is now `1` on any failure
instead of always `0`).

## License

MIT — see [`LICENSE`](LICENSE). Fill in the copyright holder before
publishing.
