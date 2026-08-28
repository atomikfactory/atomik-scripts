from __future__ import annotations

import pytest

from conftest import requires_ffmpeg
from normalize_videoclip.config import NormalizeConfig
from normalize_videoclip.core import discover_video_files, normalize_directory, plan_tasks
from normalize_videoclip.exceptions import InvalidInputDirectoryError
from normalize_videoclip.models import TaskStatus


def test_discover_video_files_missing_dir(tmp_path):
    with pytest.raises(InvalidInputDirectoryError):
        discover_video_files(tmp_path / "does-not-exist", NormalizeConfig())


def test_discover_video_files_dedupes_case_variants(tmp_path):
    (tmp_path / "clip.mp4").touch()
    (tmp_path / "other.MKV").touch()
    (tmp_path / "not_a_video.txt").touch()

    files = discover_video_files(tmp_path, NormalizeConfig())

    names = sorted(f.name for f in files)
    assert names == ["clip.mp4", "other.MKV"]


def test_discover_video_files_is_not_recursive(tmp_path):
    (tmp_path / "top.mp4").touch()
    nested = tmp_path / "subdir"
    nested.mkdir()
    (nested / "nested.mp4").touch()

    files = discover_video_files(tmp_path, NormalizeConfig())

    assert [f.name for f in files] == ["top.mp4"]


def test_plan_tasks_legacy_container_maps_to_mp4(tmp_path):
    src = tmp_path / "old.webm"
    src.touch()
    tasks = plan_tasks([src], tmp_path / "out", NormalizeConfig())

    assert len(tasks) == 1
    task = tasks[0]
    assert task.requires_reencode is True
    assert task.destination.name == "old.mp4"


def test_plan_tasks_standard_container_keeps_name(tmp_path):
    src = tmp_path / "clip.mkv"
    src.touch()
    tasks = plan_tasks([src], tmp_path / "out", NormalizeConfig())

    assert tasks[0].requires_reencode is False
    assert tasks[0].destination.name == "clip.mkv"


def test_normalize_directory_no_files_returns_empty_summary(tmp_path):
    (tmp_path / "in").mkdir()
    summary = normalize_directory(tmp_path / "in", tmp_path / "out", NormalizeConfig())
    assert summary.results == []
    assert summary.exit_code == 0


def test_normalize_directory_dry_run_touches_nothing(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "clip.mp4").touch()

    summary = normalize_directory(in_dir, tmp_path / "out", NormalizeConfig(dry_run=True))

    assert len(summary.results) == 1
    assert summary.results[0].status == TaskStatus.DRY_RUN
    assert not (tmp_path / "out").exists()


@requires_ffmpeg
def test_normalize_directory_skips_existing_destination(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    (in_dir / "clip.mp4").touch()
    existing = out_dir / "clip.mp4"
    existing.write_bytes(b"already here")

    summary = normalize_directory(in_dir, out_dir, NormalizeConfig())

    assert summary.skipped == 1
    assert summary.failed == 0
    assert existing.read_bytes() == b"already here"  # untouched, not re-encoded


@requires_ffmpeg
def test_normalize_directory_real_encode_is_atomic_and_clean(tmp_path, tiny_video_factory):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    src = tiny_video_factory("clip.mp4")
    src.rename(in_dir / "clip.mp4")

    out_dir = tmp_path / "out"
    summary = normalize_directory(in_dir, out_dir, NormalizeConfig(workers=2))

    assert summary.succeeded == 1
    assert summary.failed == 0
    output_file = out_dir / "clip.mp4"
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    # No leftover temp files from the atomic-write dance.
    leftovers = [p for p in out_dir.iterdir() if p.name.startswith(".")]
    assert leftovers == []


@requires_ffmpeg
def test_normalize_directory_legacy_container_reencodes_to_mp4(tmp_path, tiny_video_factory):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    src = tiny_video_factory("clip.webm", video_codec="libvpx", audio_codec="libvorbis")
    src.rename(in_dir / "clip.webm")

    out_dir = tmp_path / "out"
    summary = normalize_directory(in_dir, out_dir, NormalizeConfig())

    assert summary.succeeded == 1
    assert (out_dir / "clip.mp4").exists()
    assert not (out_dir / "clip.webm").exists()


@requires_ffmpeg
def test_normalize_directory_failure_leaves_no_partial_output(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    # A file with a video extension but garbage content: ffmpeg will fail on it.
    (in_dir / "broken.mp4").write_bytes(b"not a real video file")

    out_dir = tmp_path / "out"
    summary = normalize_directory(in_dir, out_dir, NormalizeConfig())

    assert summary.failed == 1
    assert summary.results[0].status == TaskStatus.FAILED
    assert not (out_dir / "broken.mp4").exists()
    leftovers = list(out_dir.glob(".*"))
    assert leftovers == []
