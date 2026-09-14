"""Tests for :mod:`resolve_tts.manifest`."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resolve_tts import ToolError
from resolve_tts.chunker import split_text
from resolve_tts.manifest import (
    MANIFEST_VERSION,
    STATUS_FAILED,
    STATUS_GENERATED,
    STATUS_PENDING,
    STATUS_PLACED,
    Manifest,
    SegmentRecord,
    build_manifest,
    sha256_text,
)

SCRIPT = (
    "The first paragraph is short and simple.\n\n"
    "The second paragraph carries a little more weight, and it runs on for "
    "long enough to become two segments when the limit is small enough."
)


class ManifestTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="resolve_tts_manifest_")
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.path = os.path.join(self.directory, "manifest.json")

    def make(self, text=SCRIPT, max_chars=80, merge=False, voice=None):
        chunks = split_text(text, max_chars=max_chars, merge_paragraphs=merge)
        return build_manifest(
            path=self.path,
            script_path=os.path.join(self.directory, "script.txt"),
            script_text=text,
            chunks=chunks,
            max_chars=max_chars,
            merge_paragraphs=merge,
            voice_settings=voice or {"VoiceModel": "Female 1"},
            tool_version="test",
        )


class TestSha256(unittest.TestCase):
    def test_stable_and_unicode_safe(self):
        self.assertEqual(sha256_text("abc"), sha256_text("abc"))
        self.assertNotEqual(sha256_text("abc"), sha256_text("abd"))
        self.assertEqual(len(sha256_text("café")), 64)


class TestBuildAndSave(ManifestTestCase):
    def test_builds_one_record_per_chunk(self):
        manifest = self.make()
        self.assertGreater(len(manifest.segments), 2)
        self.assertEqual([s.index for s in manifest.segments], list(range(len(manifest.segments))))
        self.assertEqual([s.number for s in manifest.segments],
                         list(range(1, len(manifest.segments) + 1)))
        for segment in manifest.segments:
            self.assertEqual(segment.status, STATUS_PENDING)
            self.assertEqual(segment.attempts, 0)
            self.assertEqual(segment.char_count, len(segment.text))
            self.assertFalse(segment.is_complete)

    def test_save_creates_readable_json(self):
        manifest = self.make()
        manifest.save()
        self.assertTrue(os.path.isfile(self.path))
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data["version"], MANIFEST_VERSION)
        self.assertEqual(data["max_chars"], 80)
        self.assertEqual(data["voice_settings"], {"VoiceModel": "Female 1"})
        self.assertEqual(len(data["segments"]), len(manifest.segments))
        self.assertIn("script_sha256", data)

    def test_save_leaves_no_temporary_file(self):
        manifest = self.make()
        manifest.save()
        self.assertFalse(os.path.isfile(self.path + ".tmp"))
        self.assertEqual(os.listdir(self.directory), ["manifest.json"])

    def test_save_is_idempotent_and_overwrites(self):
        manifest = self.make()
        manifest.save()
        manifest.segments[0].status = STATUS_GENERATED
        manifest.save()
        reloaded = Manifest.load(self.path)
        self.assertEqual(reloaded.segments[0].status, STATUS_GENERATED)

    def test_save_creates_missing_directories(self):
        nested = os.path.join(self.directory, "a", "b", "manifest.json")
        manifest = self.make()
        manifest.path = nested
        manifest.save()
        self.assertTrue(os.path.isfile(nested))


class TestRoundTrip(ManifestTestCase):
    def test_all_fields_survive(self):
        manifest = self.make()
        segment = manifest.segments[0]
        segment.status = STATUS_PLACED
        segment.attempts = 3
        segment.error = None
        segment.resolve_file_path = r"C:\Speech\SpeechGen-Male 1-G1-001.wav"
        segment.cached_file = os.path.join("segments", "001.wav")
        segment.media_pool_item_unique_id = "unique-abc"
        segment.media_id = "media-abc"
        segment.clip_name = "SpeechGen-Male 1-G1-001.wav"
        segment.duration_seconds = 4.125
        segment.sample_rate = 48000
        segment.channels = 1
        segment.sample_width = 2
        segment.leading_silence_seconds = 0.12
        segment.trailing_silence_seconds = 0.34
        segment.generation_seconds = 2.5
        segment.record_frame = 86400
        segment.track_index = 3
        segment.start_frame = 2
        segment.end_frame = 98
        segment.timeline_timecode = "01:00:00:00"
        manifest.save()

        reloaded = Manifest.load(self.path)
        restored = reloaded.segments[0]
        for field_name in (
            "status", "attempts", "resolve_file_path", "cached_file",
            "media_pool_item_unique_id", "media_id", "clip_name",
            "duration_seconds", "sample_rate", "channels", "sample_width",
            "leading_silence_seconds", "trailing_silence_seconds",
            "generation_seconds", "record_frame", "track_index",
            "start_frame", "end_frame", "timeline_timecode",
        ):
            with self.subTest(field=field_name):
                self.assertEqual(
                    getattr(restored, field_name), getattr(segment, field_name)
                )
        self.assertEqual(reloaded.script_sha256, manifest.script_sha256)
        self.assertEqual(reloaded.voice_settings, manifest.voice_settings)

    def test_unicode_text_survives(self):
        text = "Le réalisateur a dit « c'est fini ».\n\n今日は晴れ。"
        manifest = self.make(text=text, max_chars=100)
        manifest.save()
        reloaded = Manifest.load(self.path)
        self.assertEqual(
            [s.text for s in reloaded.segments], [s.text for s in manifest.segments]
        )

    def test_unknown_fields_are_ignored(self):
        manifest = self.make()
        manifest.save()
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["segments"][0]["something_from_the_future"] = 42
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        reloaded = Manifest.load(self.path)
        self.assertEqual(reloaded.segments[0].index, 0)

    def test_bad_status_falls_back_to_pending(self):
        record = SegmentRecord.from_dict(
            {"index": 0, "text": "x", "char_count": 1, "status": "nonsense"}
        )
        self.assertEqual(record.status, STATUS_PENDING)


class TestLoadFailures(ManifestTestCase):
    def test_missing_file(self):
        with self.assertRaises(ToolError):
            Manifest.load(os.path.join(self.directory, "absent.json"))

    def test_corrupt_json(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("{not json at all")
        with self.assertRaises(ToolError) as caught:
            Manifest.load(self.path)
        self.assertIn("--force-rechunk", caught.exception.render())

    def test_json_that_is_not_an_object(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("[1, 2, 3]")
        with self.assertRaises(ToolError):
            Manifest.load(self.path)


class TestResumeLogic(ManifestTestCase):
    def test_matches_identical_settings(self):
        manifest = self.make()
        self.assertTrue(manifest.matches(sha256_text(SCRIPT), 80, False))

    def test_rejects_changed_script(self):
        manifest = self.make()
        self.assertFalse(manifest.matches(sha256_text(SCRIPT + " More."), 80, False))
        self.assertIn(
            "script text has changed",
            manifest.mismatch_reason(sha256_text(SCRIPT + " More."), 80, False),
        )

    def test_rejects_changed_max_chars(self):
        manifest = self.make()
        self.assertFalse(manifest.matches(sha256_text(SCRIPT), 300, False))
        self.assertIn("--max-chars", manifest.mismatch_reason(sha256_text(SCRIPT), 300, False))

    def test_rejects_changed_merge_setting(self):
        manifest = self.make()
        self.assertFalse(manifest.matches(sha256_text(SCRIPT), 80, True))
        self.assertIn(
            "--merge-paragraphs", manifest.mismatch_reason(sha256_text(SCRIPT), 80, True)
        )

    def test_rejects_old_manifest_version(self):
        manifest = self.make()
        manifest.version = MANIFEST_VERSION - 1
        self.assertFalse(manifest.matches(sha256_text(SCRIPT), 80, False))
        self.assertIn("manifest version", manifest.mismatch_reason(sha256_text(SCRIPT), 80, False))

    def test_archive_moves_the_file_aside(self):
        manifest = self.make()
        manifest.save()
        archived = manifest.archive()
        self.assertFalse(os.path.isfile(self.path))
        self.assertTrue(os.path.isfile(archived))
        self.assertTrue(os.path.basename(archived).startswith("manifest."))
        self.assertTrue(archived.endswith(".json"))


class TestQueries(ManifestTestCase):
    def test_counts_and_selection(self):
        manifest = self.make()
        manifest.segments[0].status = STATUS_GENERATED
        manifest.segments[1].status = STATUS_FAILED
        manifest.segments[1].error = "GenerateSpeech returned None"

        self.assertEqual(manifest.count(STATUS_GENERATED), 1)
        self.assertEqual(manifest.count(STATUS_FAILED), 1)
        self.assertEqual(
            manifest.count(STATUS_PENDING), len(manifest.segments) - 2
        )
        self.assertEqual(
            [s.index for s in manifest.with_status(STATUS_GENERATED, STATUS_FAILED)],
            [0, 1],
        )
        self.assertEqual(manifest.failed_numbers(), [2])
        self.assertTrue(manifest.segments[0].is_complete)
        self.assertFalse(manifest.segments[1].is_complete)

    def test_get_by_index(self):
        manifest = self.make()
        self.assertIs(manifest.get(1), manifest.segments[1])
        self.assertIsNone(manifest.get(9999))

    def test_placed_counts_as_complete(self):
        manifest = self.make()
        manifest.segments[0].status = STATUS_PLACED
        self.assertTrue(manifest.segments[0].is_complete)
        self.assertEqual(manifest.count(STATUS_GENERATED, STATUS_PLACED), 1)


class TestCrashSafety(ManifestTestCase):
    def test_a_stale_temp_file_does_not_corrupt_the_manifest(self):
        manifest = self.make()
        manifest.save()
        with open(self.path + ".tmp", "w", encoding="utf-8") as handle:
            handle.write("half-written garbage")
        reloaded = Manifest.load(self.path)
        self.assertEqual(len(reloaded.segments), len(manifest.segments))

    def test_progress_written_after_every_segment_survives(self):
        manifest = self.make()
        for segment in manifest.segments:
            segment.status = STATUS_GENERATED
            manifest.save()
            snapshot = Manifest.load(self.path)
            self.assertEqual(
                snapshot.count(STATUS_GENERATED), segment.index + 1
            )


if __name__ == "__main__":
    unittest.main()
