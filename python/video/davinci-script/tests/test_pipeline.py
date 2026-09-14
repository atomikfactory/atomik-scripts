"""Tests for the frame arithmetic and timeline placement in :mod:`resolve_tts.pipeline`.

Every number here comes from a live run against DaVinci Resolve Studio
21.0.1.11:

* a 4.40 s clip on a 24 fps timeline is **105** frames (4.40 x 24 = 105.6, and
  Resolve floors it -- its ``"Duration"`` property reads ``00:00:04:09``);
* ``"Frames"`` is an empty string for generated audio clips;
* ``AppendToTimeline``'s ``endFrame`` is **exclusive**: ``startFrame 0,
  endFrame 105`` places 105 frames, and ``startFrame 3, endFrame 103`` places
  100 with ``GetLeftOffset() == 3`` and ``GetRightOffset() == 2``;
* ``TimelineItem.GetEnd()`` is exclusive too, so the next clip's
  ``recordFrame`` is exactly the previous ``GetEnd()`` (515, 621, 728, 828, 935
  was the observed sequence).
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_voice
from test_audio_utils import chunk, encode, fmt_body, riff, tone_samples

from resolve_tts import PlacementAborted, ToolError
from resolve_tts import timecode as tc
from resolve_tts.manifest import (
    STATUS_GENERATED,
    STATUS_PENDING,
    STATUS_PLACED,
    Manifest,
    SegmentRecord,
)
from resolve_tts.pipeline import (
    Options,
    Pipeline,
    build_clip_info,
    frames_for_duration,
    frames_from_clip_properties,
)
from resolve_tts.resolve_connect import ResolveContext

FPS_24 = tc.parse_frame_rate("24")

# The pipeline logs through this logger; keep its warnings out of the test
# output without hiding them from the code under test.
logging.getLogger("resolve_tts").addHandler(logging.NullHandler())
logging.getLogger("resolve_tts").propagate = False

#: The property dictionary Resolve Studio 21.0.1 returns for a generated clip,
#: reduced to the keys this tool reads.  Note the empty "Frames".
GENERATED_CLIP_PROPERTIES = {
    "Frames": "",
    "Duration": "00:00:04:09",
    "FPS": 24.0,
    "Sample Rate": "48000",
    "Audio Ch": "1",
    "Audio Bit Depth": "32",
    "Format": "Wave",
    "Audio Codec": "Linear PCM",
    "Type": "Audio",
    "File Name": "AIVoiceTest_001-Male 1-G1-001.wav",
}


# --------------------------------------------------------------------------
# Pure frame arithmetic
# --------------------------------------------------------------------------


class TestFramesForDuration(unittest.TestCase):
    def test_partial_frames_are_floored(self):
        # 4.40 s x 24 = 105.6.  Resolve reports 105 and asking for 106 was
        # accepted live but reached past the end of the media.
        self.assertEqual(frames_for_duration(4.40, FPS_24), 105)

    def test_the_other_measured_clip(self):
        self.assertEqual(frames_for_duration(4.48, FPS_24), 107)

    def test_exact_multiples_are_unchanged(self):
        self.assertEqual(frames_for_duration(1.0, FPS_24), 24)
        self.assertEqual(frames_for_duration(10.0, FPS_24), 240)

    def test_zero_and_negative(self):
        self.assertEqual(frames_for_duration(0.0, FPS_24), 0)
        self.assertEqual(frames_for_duration(-1.0, FPS_24), 0)

    def test_pulldown_rate_uses_the_true_fps(self):
        rate = tc.parse_frame_rate("23.976")
        self.assertEqual(frames_for_duration(1.0, rate), 23)
        self.assertEqual(frames_for_duration(4.40, rate), 105)

    def test_never_rounds_up(self):
        for milliseconds in range(0, 2000):
            seconds = milliseconds / 1000.0
            frames = frames_for_duration(seconds, FPS_24)
            self.assertLessEqual(frames / FPS_24.fps, seconds + 1e-9)


class TestFramesFromClipProperties(unittest.TestCase):
    def test_duration_timecode_is_preferred(self):
        self.assertEqual(
            frames_from_clip_properties(GENERATED_CLIP_PROPERTIES, FPS_24), 105
        )

    def test_empty_frames_string_never_reaches_int(self):
        properties = dict(GENERATED_CLIP_PROPERTIES)
        del properties["Duration"]
        self.assertIsNone(frames_from_clip_properties(properties, FPS_24))

    def test_frames_is_used_when_it_holds_a_number(self):
        properties = {"Frames": "105"}
        self.assertEqual(frames_from_clip_properties(properties, FPS_24), 105)

    def test_duration_wins_over_frames(self):
        properties = {"Duration": "00:00:04:09", "Frames": "9999"}
        self.assertEqual(frames_from_clip_properties(properties, FPS_24), 105)

    def test_unusable_input(self):
        self.assertIsNone(frames_from_clip_properties(None, FPS_24))
        self.assertIsNone(frames_from_clip_properties({}, FPS_24))
        self.assertIsNone(frames_from_clip_properties({"Duration": "not a timecode"}, FPS_24))
        self.assertIsNone(frames_from_clip_properties({"Frames": "0"}, FPS_24))
        self.assertIsNone(frames_from_clip_properties({"Frames": None}, FPS_24))

    def test_a_frame_field_beyond_the_rate_is_rejected(self):
        # "00:00:04:30" cannot exist at 24 fps; do not silently mis-convert it.
        self.assertIsNone(frames_from_clip_properties({"Duration": "00:00:04:30"}, FPS_24))


class TestBuildClipInfo(unittest.TestCase):
    def test_end_frame_is_exclusive(self):
        info = build_clip_info("item", 0, 105, 3, 515)
        self.assertEqual(info["startFrame"], 0)
        self.assertEqual(info["endFrame"], 105)
        self.assertEqual(info["endFrame"] - info["startFrame"], 105)

    def test_trimmed_range_matches_the_observed_offsets(self):
        total = 105
        info = build_clip_info("item", 3, 103, 1, 0)
        self.assertEqual(info["endFrame"] - info["startFrame"], 100)
        self.assertEqual(info["startFrame"], 3)  # GetLeftOffset() == 3
        self.assertEqual(total - info["endFrame"], 2)  # GetRightOffset() == 2

    def test_fixed_fields(self):
        info = build_clip_info("item", 0, 10, 2, 7)
        self.assertEqual(info["mediaType"], 2)
        self.assertEqual(info["trackIndex"], 2)
        self.assertEqual(info["recordFrame"], 7)
        self.assertEqual(info["mediaPoolItem"], "item")


# --------------------------------------------------------------------------
# A minimal fake Resolve, honouring the verified semantics
# --------------------------------------------------------------------------


class FakeTimelineItem:
    """A placed clip.  ``GetEnd()`` is exclusive, as Resolve's is."""

    def __init__(self, name, start, duration, left_offset=0):
        self._name = name
        self._start = start
        self._duration = duration
        self._left = left_offset

    def GetName(self):
        return self._name

    def GetStart(self):
        return self._start

    def GetEnd(self):
        return self._start + self._duration

    def GetDuration(self):
        return self._duration

    def GetLeftOffset(self):
        return self._left

    def GetUniqueId(self):
        return "item-%s-%d" % (self._name, self._start)


class FakeTimeline:
    def __init__(self, tracks=None, timecode="00:00:21:11"):
        # Each track: {"name": str, "sub_type": str, "locked": bool, "items": []}
        self.tracks = list(tracks or [])
        self.timecode = timecode
        self.added_tracks = []

    # -- settings
    def GetSetting(self, key):
        return 24.0 if key == "timelineFrameRate" else None

    def GetName(self):
        return "Fake Timeline"

    def GetStartFrame(self):
        return 0

    def GetStartTimecode(self):
        return "00:00:00:00"

    def GetCurrentTimecode(self):
        return self.timecode

    # -- tracks
    def GetTrackCount(self, track_type):
        return len(self.tracks) if track_type == "audio" else 0

    def GetTrackName(self, track_type, index):
        return self.tracks[index - 1]["name"]

    def GetTrackSubType(self, track_type, index):
        return self.tracks[index - 1].get("sub_type", "mono")

    def GetIsTrackLocked(self, track_type, index):
        return self.tracks[index - 1].get("locked", False)

    def GetIsTrackEnabled(self, track_type, index):
        return True

    def GetItemListInTrack(self, track_type, index):
        return list(self.tracks[index - 1]["items"])

    def AddTrack(self, track_type, sub_type):
        self.added_tracks.append((track_type, sub_type))
        self.tracks.append({"name": "Audio %d" % (len(self.tracks) + 1),
                            "sub_type": sub_type, "items": []})
        return True

    def SetTrackName(self, track_type, index, name):
        self.tracks[index - 1]["name"] = name
        return True


class FakeMediaPool:
    """``AppendToTimeline`` with the verified duration rule."""

    def __init__(self, timeline, fail=False):
        self.timeline = timeline
        self.calls = []
        self.fail = fail

    def AppendToTimeline(self, clip_infos):
        placed = []
        for info in clip_infos:
            self.calls.append(dict(info))
            if self.fail:
                return None
            duration = info["endFrame"] - info["startFrame"]
            item = FakeTimelineItem(
                str(info["mediaPoolItem"]),
                info["recordFrame"],
                duration,
                info["startFrame"],
            )
            self.timeline.tracks[info["trackIndex"] - 1]["items"].append(item)
            placed.append(item)
        return placed


class FakeMediaPoolItem:
    """A generated clip, with the property set Resolve 21.0.1 really returns."""

    def __init__(self, file_path, name="AIVoice_myscript_001-Male 1-G1-001.wav"):
        self.properties = dict(GENERATED_CLIP_PROPERTIES)
        self.properties["File Path"] = file_path
        self.properties["Clip Directory"] = os.path.dirname(file_path)
        self.properties["File Name"] = os.path.basename(file_path)
        self._name = name

    def GetClipProperty(self, key=None):
        return dict(self.properties) if key is None else self.properties.get(key, "")

    def GetName(self):
        return self._name

    def GetUniqueId(self):
        return "unique-1"

    def GetMediaId(self):
        return "media-1"


class FakeProject:
    """``GenerateSpeech`` that optionally places a clip, like AddToTimeline should."""

    def __init__(self, speech_result=None, timeline=None):
        self.speech_result = speech_result
        self.speech_calls = []
        self.timeline = timeline
        self.place_on_track = None

    def GetName(self):
        return "Fake Project"

    def GenerateSpeech(self, settings, position):
        self.speech_calls.append((dict(settings), position))
        if self.place_on_track and self.timeline is not None:
            track = self.timeline.tracks[self.place_on_track - 1]
            track["items"].append(FakeTimelineItem("placed", 0, 12))
        return self.speech_result


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="resolve_tts_pipeline_")
        self.addCleanup(shutil.rmtree, self.directory, True)

    def make_pipeline(self, tracks=None, timecode="00:00:21:11", fail_append=False, **overrides):
        options = Options(
            script_path=os.path.join(self.directory, "myscript.txt"),
            output_dir=self.directory,
            **overrides
        )
        pipeline = Pipeline(options)
        timeline = FakeTimeline(tracks=tracks, timecode=timecode)
        media_pool = FakeMediaPool(timeline, fail=fail_append)
        project = FakeProject(timeline=timeline)
        pipeline.context = ResolveContext(
            resolve=None,
            project_manager=None,
            project=project,
            media_pool=media_pool,
            timeline=timeline,
            product_name="DaVinci Resolve Studio",
            version_string="21.0.1.11",
            version_fields=[21, 0, 1, 11],
        )
        pipeline.manifest = Manifest(
            path=os.path.join(self.directory, "manifest.json"),
            script_path=options.script_path,
            script_sha256="0" * 64,
            max_chars=300,
            merge_paragraphs=False,
        )
        return pipeline, timeline, media_pool, project

    def add_segments(self, pipeline, durations, status=STATUS_GENERATED):
        for index, duration in enumerate(durations):
            segment = SegmentRecord(
                index=index,
                text="segment %d" % (index + 1),
                char_count=9,
                status=status,
                duration_seconds=duration,
                sample_rate=48000,
                channels=1,
                sample_width=4,
                sample_format="float",
                source_frames=frames_for_duration(duration, FPS_24),
                resolve_file_path=os.path.join(self.directory, "%03d.wav" % (index + 1)),
            )
            pipeline.manifest.segments.append(segment)
            pipeline._items[index] = "clip%d" % (index + 1)
        return pipeline.manifest.segments

    def write_generated_wav(self, name="001.wav", lead=0.028, tone=0.412, trail=0.060):
        """A WAV in Resolve's real format: extensible float, 48 kHz, mono."""
        values = tone_samples(
            lead,
            tone,
            trail,
            sample_rate=48000,
            channels=1,
            noise_amplitude=10 ** (-70.0 / 20.0),
        )
        payload = riff(
            chunk(b"JUNK", b"\x00" * 28)
            + chunk(b"fmt ", fmt_body("float", 32, 1, 48000, extensible=True))
            + chunk(b"bext", b"\x00" * 968)
            + chunk(b"data", encode(values, "float", 32))
        )
        path = os.path.join(self.directory, name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return path


# --------------------------------------------------------------------------
# Source ranges
# --------------------------------------------------------------------------


class TestSourceRange(PipelineTestCase):
    def test_untrimmed_range_is_the_whole_floored_clip(self):
        pipeline, _, _, _ = self.make_pipeline()
        segment = self.add_segments(pipeline, [4.40])[0]
        self.assertEqual(pipeline._source_range(segment, FPS_24), (0, 105))

    def test_real_measured_padding_trims_nothing_at_24_fps(self):
        # 28 ms leading and 60 ms trailing, the middle of the measured range.
        pipeline, _, _, _ = self.make_pipeline(trim_silence=True)
        segment = self.add_segments(pipeline, [4.40])[0]
        segment.leading_silence_seconds = 0.028
        segment.trailing_silence_seconds = 0.060
        self.assertEqual(pipeline._source_range(segment, FPS_24), (0, 105))

    def test_even_with_no_keep_pad_at_most_one_frame_can_be_trimmed(self):
        # The worst case measured live: 32 ms leading, 76 ms trailing.  One
        # frame at 24 fps is 41.7 ms, so quantisation can only ever take a
        # single frame off the tail -- which is why trimming defaults to off.
        pipeline, _, _, _ = self.make_pipeline(trim_silence=True, keep_pad_ms=0)
        segment = self.add_segments(pipeline, [4.40])[0]
        segment.leading_silence_seconds = 0.032
        segment.trailing_silence_seconds = 0.076
        source_in, source_out = pipeline._source_range(segment, FPS_24)
        self.assertEqual(source_in, 0)
        self.assertIn(source_out, (104, 105))
        self.assertLessEqual(105 - source_out, 1)

    def test_large_silence_is_trimmed(self):
        pipeline, _, _, _ = self.make_pipeline(trim_silence=True, keep_pad_ms=40)
        segment = self.add_segments(pipeline, [4.40])[0]
        segment.leading_silence_seconds = 0.60
        segment.trailing_silence_seconds = 0.80
        source_in, source_out = pipeline._source_range(segment, FPS_24)
        self.assertGreater(source_in, 0)
        self.assertLess(source_out, 105)
        self.assertLessEqual(source_out, 105)

    def test_trimming_is_off_by_default(self):
        self.assertFalse(Options().trim_silence)
        pipeline, _, _, _ = self.make_pipeline()
        segment = self.add_segments(pipeline, [4.40])[0]
        segment.leading_silence_seconds = 0.60
        segment.trailing_silence_seconds = 0.80
        self.assertEqual(pipeline._source_range(segment, FPS_24), (0, 105))

    def test_unknown_duration_falls_back_to_clip_property_frames(self):
        pipeline, _, _, _ = self.make_pipeline()
        segment = self.add_segments(pipeline, [4.40])[0]
        segment.duration_seconds = None
        segment.source_frames = 105
        self.assertEqual(pipeline._source_range(segment, FPS_24), (0, 105))

    def test_nothing_known_places_a_single_frame(self):
        pipeline, _, _, _ = self.make_pipeline()
        segment = self.add_segments(pipeline, [4.40])[0]
        segment.duration_seconds = None
        segment.source_frames = None
        self.assertEqual(pipeline._source_range(segment, FPS_24), (0, 1))


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------


class TestPlacement(PipelineTestCase):
    #: 4.42, 4.46, 4.17 and 4.46 s floor to 106, 107, 100 and 107 frames, the
    #: lengths behind the observed 515 -> 621 -> 728 -> 828 -> 935 sequence.
    DURATIONS = [4.42, 4.46, 4.17, 4.46]

    def test_clips_are_contiguous_from_the_playhead(self):
        pipeline, timeline, media_pool, _ = self.make_pipeline()
        self.add_segments(pipeline, self.DURATIONS)
        pipeline._place_all()

        starts = [info["recordFrame"] for info in media_pool.calls]
        self.assertEqual(starts, [515, 621, 728, 828])
        items = timeline.tracks[-1]["items"]
        self.assertEqual([item.GetStart() for item in items], [515, 621, 728, 828])
        self.assertEqual([item.GetEnd() for item in items], [621, 728, 828, 935])
        self.assertEqual([item.GetDuration() for item in items], [106, 107, 100, 107])

    def test_end_frame_is_exclusive_in_every_call(self):
        pipeline, _, media_pool, _ = self.make_pipeline()
        self.add_segments(pipeline, self.DURATIONS)
        pipeline._place_all()
        for info, duration in zip(media_pool.calls, [106, 107, 100, 107]):
            self.assertEqual(info["startFrame"], 0)
            self.assertEqual(info["endFrame"], duration)
            self.assertEqual(info["endFrame"] - info["startFrame"], duration)
            self.assertEqual(info["mediaType"], 2)

    def test_the_manifest_records_exclusive_end_frames(self):
        pipeline, _, _, _ = self.make_pipeline()
        segments = self.add_segments(pipeline, self.DURATIONS)
        pipeline._place_all()
        for segment, duration in zip(segments, [106, 107, 100, 107]):
            self.assertEqual(segment.status, STATUS_PLACED)
            self.assertEqual(segment.start_frame, 0)
            self.assertEqual(segment.end_frame, duration)
            self.assertEqual(segment.end_frame - segment.start_frame, duration)
        self.assertEqual([s.record_frame for s in segments], [515, 621, 728, 828])
        self.assertEqual(segments[0].timeline_timecode, "00:00:21:11")

    def test_gap_frames_are_inserted_between_clips(self):
        pipeline, _, media_pool, _ = self.make_pipeline(gap_frames=6)
        self.add_segments(pipeline, self.DURATIONS)
        pipeline._place_all()
        starts = [info["recordFrame"] for info in media_pool.calls]
        self.assertEqual(starts, [515, 627, 740, 846])

    def test_a_mono_track_is_created_and_named(self):
        pipeline, timeline, _, _ = self.make_pipeline()
        self.add_segments(pipeline, [4.42])
        pipeline._place_all()
        self.assertEqual(timeline.added_tracks, [("audio", "mono")])
        self.assertEqual(timeline.tracks[-1]["name"], "AI Voice")

    def test_an_existing_ai_voice_track_is_reused(self):
        tracks = [
            {"name": "Dialogue", "items": []},
            {"name": "AI Voice", "items": []},
        ]
        pipeline, timeline, media_pool, _ = self.make_pipeline(tracks=tracks)
        self.add_segments(pipeline, [4.42])
        pipeline._place_all()
        self.assertEqual(timeline.added_tracks, [])
        self.assertEqual(media_pool.calls[0]["trackIndex"], 2)

    def test_other_tracks_are_untouched(self):
        existing = FakeTimelineItem("music", 0, 2000)
        tracks = [{"name": "Music", "items": [existing]}]
        pipeline, timeline, _, _ = self.make_pipeline(tracks=tracks)
        self.add_segments(pipeline, [4.42])
        pipeline._place_all()
        self.assertEqual(timeline.tracks[0]["items"], [existing])
        self.assertEqual(len(timeline.tracks[1]["items"]), 1)

    def test_an_overlapping_clip_aborts_placement(self):
        tracks = [{"name": "AI Voice", "items": [FakeTimelineItem("old", 500, 200)]}]
        pipeline, _, media_pool, _ = self.make_pipeline(tracks=tracks)
        self.add_segments(pipeline, self.DURATIONS)
        with self.assertRaises(PlacementAborted) as caught:
            pipeline._place_all()
        self.assertIn("already has 1 clip", caught.exception.message)
        self.assertEqual(media_pool.calls, [])

    def test_a_clip_before_the_playhead_does_not_block_placement(self):
        tracks = [{"name": "AI Voice", "items": [FakeTimelineItem("old", 0, 515)]}]
        pipeline, _, media_pool, _ = self.make_pipeline(tracks=tracks)
        self.add_segments(pipeline, [4.42])
        pipeline._place_all()
        self.assertEqual(len(media_pool.calls), 1)

    def test_a_locked_track_aborts_placement(self):
        tracks = [{"name": "AI Voice", "items": [], "locked": True}]
        pipeline, _, _, _ = self.make_pipeline(tracks=tracks)
        self.add_segments(pipeline, [4.42])
        with self.assertRaises(PlacementAborted) as caught:
            pipeline._place_all()
        self.assertIn("locked", caught.exception.message)

    def test_a_failed_append_aborts_with_a_count(self):
        pipeline, _, _, _ = self.make_pipeline(fail_append=True)
        self.add_segments(pipeline, self.DURATIONS)
        with self.assertRaises(PlacementAborted) as caught:
            pipeline._place_all()
        self.assertIn("AppendToTimeline returned None", caught.exception.message)

    def test_start_timeline_start_places_at_frame_zero(self):
        pipeline, _, media_pool, _ = self.make_pipeline(start="timeline-start")
        self.add_segments(pipeline, [4.42])
        pipeline._place_all()
        self.assertEqual(media_pool.calls[0]["recordFrame"], 0)

    def test_explicit_start_timecode(self):
        pipeline, _, media_pool, _ = self.make_pipeline(start="01:00:00:00")
        self.add_segments(pipeline, [4.42])
        pipeline._place_all()
        self.assertEqual(media_pool.calls[0]["recordFrame"], 86400)

    def test_nothing_is_placed_when_a_segment_failed(self):
        pipeline, _, media_pool, _ = self.make_pipeline()
        segments = self.add_segments(pipeline, self.DURATIONS)
        segments[2].status = "failed"
        segments[2].error = "GenerateSpeech returned None"
        pipeline._place_all()
        self.assertEqual(media_pool.calls, [])
        self.assertEqual(
            [s.status for s in segments], [STATUS_GENERATED] * 2 + ["failed", STATUS_GENERATED]
        )

    def test_resume_frame_uses_exclusive_end_frames(self):
        pipeline, _, _, _ = self.make_pipeline()
        segments = self.add_segments(pipeline, [4.42])
        segments[0].status = STATUS_PLACED
        segments[0].record_frame = 515
        segments[0].start_frame = 0
        segments[0].end_frame = 106
        self.assertEqual(pipeline._resume_frame(), 621)


# --------------------------------------------------------------------------
# GenerateSpeech settings and failure reporting
# --------------------------------------------------------------------------


class TestSpeechSettings(PipelineTestCase):
    def test_filename_is_a_prefix_per_segment(self):
        pipeline, _, _, _ = self.make_pipeline()
        segment = self.add_segments(pipeline, [4.40, 4.40, 4.40])[2]
        settings = pipeline._speech_settings(segment, add_to_timeline=False)
        self.assertEqual(settings["Filename"], "AIVoice_myscript_003")
        self.assertIs(settings["AddToTimeline"], False)
        self.assertNotIn("AudioTrack", settings)
        self.assertEqual(settings["TextInput"], "segment 3")

    def test_voice_settings_are_only_sent_when_supplied(self):
        pipeline, _, _, _ = self.make_pipeline()
        segment = self.add_segments(pipeline, [4.40])[0]
        settings = pipeline._speech_settings(segment, add_to_timeline=False)
        for key in ("VoiceModel", "Speed", "Variation", "Pitch", "GenerationID"):
            self.assertNotIn(key, settings)

    def test_supplied_voice_settings_pass_straight_through(self):
        pipeline, _, _, _ = self.make_pipeline(
            voice="Male 1", speed=50, variation=2, pitch=-50, generation_id=7
        )
        segment = self.add_segments(pipeline, [4.40])[0]
        settings = pipeline._speech_settings(segment, add_to_timeline=False)
        self.assertEqual(settings["VoiceModel"], "Male 1")
        self.assertEqual(settings["Speed"], 50)
        self.assertEqual(settings["Variation"], 2)
        self.assertEqual(settings["Pitch"], -50)
        self.assertEqual(settings["GenerationID"], 7)

    def test_custom_voice_file_implies_custom_voice(self):
        pipeline, _, _, _ = self.make_pipeline(custom_voice_file="D:\\voices\\me.wav")
        segment = self.add_segments(pipeline, [4.40])[0]
        settings = pipeline._speech_settings(segment, add_to_timeline=False)
        self.assertEqual(settings["VoiceModel"], "Custom Voice")
        self.assertEqual(settings["CustomVoiceFile"], "D:\\voices\\me.wav")

    def test_an_instant_none_blames_the_settings(self):
        pipeline, _, _, project = self.make_pipeline()
        project.speech_result = None
        segment = self.add_segments(pipeline, [4.40])[0]
        with self.assertRaises(ToolError) as caught:
            pipeline._generate_once(segment)
        self.assertIn("returned None", caught.exception.message)
        self.assertIn("voice name does not exist", caught.exception.hint)

    def test_over_long_text_is_refused_before_the_call(self):
        pipeline, _, _, project = self.make_pipeline()
        segment = self.add_segments(pipeline, [4.40])[0]
        segment.text = "x" * 351
        with self.assertRaises(ToolError) as caught:
            pipeline._generate_once(segment)
        self.assertIn("Resolve accepts at most 350", caught.exception.message)
        self.assertEqual(project.speech_calls, [])


# --------------------------------------------------------------------------
# One whole GenerateSpeech call
# --------------------------------------------------------------------------


class TestGenerateOnce(PipelineTestCase):
    def prepare(self, **overrides):
        pipeline, timeline, media_pool, project = self.make_pipeline(**overrides)
        path = self.write_generated_wav("resolve_output.wav")
        project.speech_result = FakeMediaPoolItem(path)
        segment = self.add_segments(pipeline, [0.0])[0]
        segment.duration_seconds = None
        segment.source_frames = None
        pipeline._items.clear()
        return pipeline, timeline, media_pool, project, segment, path

    def test_a_successful_call_records_everything(self):
        pipeline, _, _, project, segment, path = self.prepare()
        pipeline._generate_once(segment)

        self.assertEqual(segment.status, STATUS_GENERATED)
        self.assertEqual(segment.resolve_file_path, path)
        self.assertEqual(segment.sample_rate, 48000)
        self.assertEqual(segment.channels, 1)
        self.assertEqual(segment.sample_width, 4)
        self.assertEqual(segment.sample_format, "float")
        self.assertAlmostEqual(segment.duration_seconds, 0.5, places=3)
        self.assertEqual(segment.source_frames, 12)  # floor(0.5 x 24)
        self.assertEqual(segment.media_pool_item_unique_id, "unique-1")
        self.assertIsNotNone(segment.generation_seconds)
        self.assertIs(pipeline._items[segment.index], project.speech_result)

    def test_the_audio_is_cached_next_to_the_manifest(self):
        pipeline, _, _, _, segment, path = self.prepare()
        pipeline._generate_once(segment)
        cached = os.path.join(pipeline.options.output_dir, segment.cached_file)
        self.assertTrue(os.path.isfile(cached))
        self.assertEqual(os.path.getsize(cached), os.path.getsize(path))

    def test_the_measured_silence_is_recorded_even_though_trimming_is_off(self):
        pipeline, _, _, _, segment, _ = self.prepare()
        self.assertFalse(pipeline.options.trim_silence)
        pipeline._generate_once(segment)
        self.assertAlmostEqual(segment.leading_silence_seconds, 0.028, delta=0.003)
        self.assertAlmostEqual(segment.trailing_silence_seconds, 0.060, delta=0.003)

    def test_frames_fall_back_to_the_duration_timecode_when_the_wav_is_broken(self):
        pipeline, _, _, project, segment, path = self.prepare()
        with open(path, "wb") as handle:
            handle.write(b"RIFFnot really a wave file")
        pipeline._generate_once(segment)
        self.assertIsNone(segment.duration_seconds)
        # "Duration" is 00:00:04:09 at 24 fps.
        self.assertEqual(segment.source_frames, 105)
        self.assertEqual(pipeline._source_range(segment, FPS_24), (0, 105))

    def test_progress_lines_match_the_documented_format(self):
        pipeline, _, _, project, segment, _ = self.prepare()
        segment.status = STATUS_PENDING
        # A second, already finished segment, to exercise the (cached) line.
        cached = SegmentRecord(
            index=1, text="segment 2", char_count=9, status=STATUS_GENERATED
        )
        pipeline.manifest.segments.append(cached)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            pipeline._generate_all()
        lines = [line for line in stream.getvalue().splitlines() if line]
        self.assertRegex(lines[0], r"^Generating segment 1/2\.\.\. done \(\d+\.\ds\)$")
        self.assertEqual(lines[1], "Generating segment 2/2... (cached)")

    def test_add_to_timeline_that_places_nothing_stops_the_run(self):
        pipeline, _, _, project, segment, _ = self.prepare(placement="resolve")
        pipeline._prepare_resolve_placement()
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            with self.assertRaises(PlacementAborted) as caught:
                pipeline._generate_segment(segment, 1)
        self.assertEqual(
            stream.getvalue(),
            "Generating segment 1/1... generated, but Resolve placed nothing\n",
        )
        self.assertIn("did not place it", caught.exception.message)
        self.assertIn("--place-only", caught.exception.hint)
        # The audio is generated and cached, so a --place-only run can finish.
        self.assertEqual(segment.status, STATUS_GENERATED)
        self.assertIsNotNone(segment.cached_file)
        self.assertEqual(len(project.speech_calls), 1)
        self.assertIs(project.speech_calls[0][0]["AddToTimeline"], True)
        self.assertEqual(project.speech_calls[0][0]["AudioTrack"], 1)

    def test_add_to_timeline_is_accepted_when_a_clip_does_appear(self):
        pipeline, timeline, _, project, segment, _ = self.prepare(placement="resolve")
        pipeline._prepare_resolve_placement()
        project.place_on_track = pipeline._target_track
        pipeline._generate_once(segment)
        self.assertEqual(segment.status, STATUS_PLACED)
        self.assertEqual(segment.track_index, pipeline._target_track)
        self.assertEqual(segment.start_frame, 0)
        self.assertEqual(segment.end_frame, 12)


# --------------------------------------------------------------------------
# The command line surface that changed
# --------------------------------------------------------------------------


class TestCommandLine(unittest.TestCase):
    def parse(self, argv):
        return generate_voice.build_parser().parse_args(argv)

    def test_trim_silence_defaults_to_off(self):
        self.assertFalse(self.parse(["script.txt"]).trim_silence)

    def test_trim_silence_can_be_opted_into(self):
        self.assertTrue(self.parse(["script.txt", "--trim-silence"]).trim_silence)

    def test_no_trim_silence_no_longer_exists(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse(["script.txt", "--no-trim-silence"])

    def test_placement_defaults_to_append(self):
        self.assertEqual(self.parse(["script.txt"]).placement, "append")

    def test_placement_resolve_is_documented_as_experimental(self):
        text = generate_voice.build_parser().format_help()
        self.assertIn("EXPERIMENTAL", text)

    def test_gap_frames_defaults_to_zero(self):
        self.assertEqual(self.parse(["script.txt"]).gap_frames, 0)


if __name__ == "__main__":
    unittest.main()
