"""Tests for :mod:`resolve_tts.timecode`."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resolve_tts.timecode import (
    FrameRate,
    frames_to_seconds,
    frames_to_timecode,
    is_timecode,
    parse_frame_rate,
    seconds_to_frames,
    timecode_to_frames,
)

NON_DROP_RATES = ("23.976", "24", "25", "29.97", "30", "50", "59.94", "60")
DROP_RATES = ("29.97 DF", "59.94 DF")


class TestParseFrameRate(unittest.TestCase):
    def test_integer_rates(self):
        for text, nominal in (("24", 24), ("25", 25), ("30", 30), ("60", 60)):
            with self.subTest(text=text):
                rate = parse_frame_rate(text)
                self.assertEqual(rate.nominal, nominal)
                self.assertAlmostEqual(rate.fps, float(nominal))
                self.assertFalse(rate.drop_frame)

    def test_ntsc_rates_snap_to_exact_fractions(self):
        for text, nominal in (("23.976", 24), ("29.97", 30), ("59.94", 60)):
            with self.subTest(text=text):
                rate = parse_frame_rate(text)
                self.assertEqual(rate.nominal, nominal)
                self.assertAlmostEqual(rate.fps, nominal * 1000.0 / 1001.0, places=9)

    def test_drop_frame_suffix(self):
        rate = parse_frame_rate("29.97 DF")
        self.assertTrue(rate.drop_frame)
        self.assertEqual(rate.nominal, 30)
        self.assertEqual(rate.dropped_per_minute, 2)
        self.assertEqual(rate.separator, ";")

        rate = parse_frame_rate("59.94 DF")
        self.assertTrue(rate.drop_frame)
        self.assertEqual(rate.dropped_per_minute, 4)

    def test_numeric_input(self):
        self.assertEqual(parse_frame_rate(24).nominal, 24)
        self.assertEqual(parse_frame_rate(29.97).nominal, 30)

    def test_non_drop_suffix_is_accepted(self):
        self.assertFalse(parse_frame_rate("29.97 NDF").drop_frame)

    def test_bad_values_raise(self):
        for value in ("", "abc", "24fps", None, 0, -25, "0"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_frame_rate(value)

    def test_drop_frame_is_rejected_at_unsupported_rates(self):
        for value in ("24 DF", "25 DF", "50 DF"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_frame_rate(value)

    def test_str_round_trips(self):
        self.assertEqual(str(parse_frame_rate("25")), "25")
        self.assertEqual(str(parse_frame_rate("29.97 DF"))[-2:], "DF")


class TestIsTimecode(unittest.TestCase):
    def test_accepted(self):
        for text in ("00:00:00:00", "01:00:00:00", "23:59:59:29", "00:01:00;02", "1:02:03:04"):
            with self.subTest(text=text):
                self.assertTrue(is_timecode(text))

    def test_rejected(self):
        for text in ("playhead", "00:00:00", "00:60:00:00", "00:00:60:00", "", "abc"):
            with self.subTest(text=text):
                self.assertFalse(is_timecode(text))


class TestNonDropConversion(unittest.TestCase):
    def test_known_values(self):
        rate = parse_frame_rate("25")
        self.assertEqual(timecode_to_frames("00:00:00:00", rate), 0)
        self.assertEqual(timecode_to_frames("00:00:01:00", rate), 25)
        self.assertEqual(timecode_to_frames("00:01:00:00", rate), 1500)
        self.assertEqual(timecode_to_frames("01:00:00:00", rate), 90000)
        self.assertEqual(timecode_to_frames("01:00:00:24", rate), 90024)

    def test_hour_start_at_24(self):
        rate = parse_frame_rate("24")
        self.assertEqual(timecode_to_frames("01:00:00:00", rate), 86400)
        self.assertEqual(frames_to_timecode(86400, rate), "01:00:00:00")

    def test_23_976_counts_at_24(self):
        rate = parse_frame_rate("23.976")
        self.assertEqual(timecode_to_frames("00:00:01:00", rate), 24)
        self.assertEqual(frames_to_timecode(24, rate), "00:00:01:00")

    def test_round_trip(self):
        for text in NON_DROP_RATES:
            rate = parse_frame_rate(text)
            for frame in range(0, 400000, 3607):
                with self.subTest(rate=text, frame=frame):
                    self.assertEqual(
                        timecode_to_frames(frames_to_timecode(frame, rate), rate), frame
                    )

    def test_separator_is_a_colon(self):
        self.assertIn(":", frames_to_timecode(100, parse_frame_rate("30")))
        self.assertNotIn(";", frames_to_timecode(100, parse_frame_rate("30")))

    def test_frame_field_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            timecode_to_frames("00:00:00:25", parse_frame_rate("25"))

    def test_malformed_timecode_raises(self):
        for text in ("00:00:00", "banana", "00-00-00-00", ""):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    timecode_to_frames(text, parse_frame_rate("25"))

    def test_negative_frames_raise(self):
        with self.assertRaises(ValueError):
            frames_to_timecode(-1, parse_frame_rate("25"))


class TestDropFrameConversion(unittest.TestCase):
    def setUp(self):
        self.rate = parse_frame_rate("29.97 DF")

    def test_start_of_timeline(self):
        self.assertEqual(frames_to_timecode(0, self.rate), "00:00:00;00")
        self.assertEqual(timecode_to_frames("00:00:00;00", self.rate), 0)

    def test_first_drop_happens_at_one_minute(self):
        # 1800 frames is exactly one minute of frames; the labels ;00 and ;01
        # of minute 1 are skipped, so it lands on 00:01:00;02.
        self.assertEqual(frames_to_timecode(1799, self.rate), "00:00:59;29")
        self.assertEqual(frames_to_timecode(1800, self.rate), "00:01:00;02")
        self.assertEqual(timecode_to_frames("00:01:00;02", self.rate), 1800)

    def test_no_drop_on_the_tenth_minute(self):
        frames = timecode_to_frames("00:10:00;00", self.rate)
        self.assertEqual(frames, 17982)
        self.assertEqual(frames_to_timecode(17982, self.rate), "00:10:00;00")

    def test_one_hour_matches_the_smpte_count(self):
        # 107892 = 30 * 3600 - 2 * (60 - 6)
        self.assertEqual(timecode_to_frames("01:00:00;00", self.rate), 107892)
        self.assertEqual(frames_to_timecode(107892, self.rate), "01:00:00;00")

    def test_skipped_labels_are_rejected(self):
        for text in ("00:01:00;00", "00:01:00;01", "00:09:00;01"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    timecode_to_frames(text, self.rate)

    def test_tenth_minute_labels_are_valid(self):
        self.assertEqual(
            frames_to_timecode(timecode_to_frames("00:20:00;00", self.rate), self.rate),
            "00:20:00;00",
        )

    def test_round_trip_dense(self):
        for rate_text in DROP_RATES:
            rate = parse_frame_rate(rate_text)
            for frame in list(range(0, 4000)) + list(range(100000, 120000, 97)):
                with self.subTest(rate=rate_text, frame=frame):
                    self.assertEqual(
                        timecode_to_frames(frames_to_timecode(frame, rate), rate), frame
                    )

    def test_drop_frame_tracks_wall_clock_more_closely_than_non_drop(self):
        drop = parse_frame_rate("29.97 DF")
        non_drop = parse_frame_rate("29.97")
        one_hour_of_frames = int(round(3600 * drop.fps))
        drop_label = frames_to_timecode(one_hour_of_frames, drop)
        non_drop_label = frames_to_timecode(one_hour_of_frames, non_drop)
        self.assertTrue(drop_label.startswith("01:00:00"))
        self.assertTrue(non_drop_label.startswith("00:59:56"))

    def test_59_94_drop_frame_drops_four(self):
        rate = parse_frame_rate("59.94 DF")
        self.assertEqual(frames_to_timecode(3600, rate), "00:01:00;04")
        self.assertEqual(timecode_to_frames("00:01:00;04", rate), 3600)


class TestSecondsConversion(unittest.TestCase):
    def test_round_floor_ceil(self):
        rate = parse_frame_rate("25")
        self.assertEqual(seconds_to_frames(1.0, rate), 25)
        self.assertEqual(seconds_to_frames(1.03, rate, "floor"), 25)
        self.assertEqual(seconds_to_frames(1.03, rate, "ceil"), 26)
        self.assertEqual(seconds_to_frames(1.03, rate, "round"), 26)
        self.assertEqual(seconds_to_frames(1.01, rate, "round"), 25)

    def test_non_positive_durations_are_zero(self):
        rate = parse_frame_rate("30")
        self.assertEqual(seconds_to_frames(0.0, rate), 0)
        self.assertEqual(seconds_to_frames(-5.0, rate), 0)

    def test_ntsc_rate_is_used_not_the_nominal(self):
        rate = parse_frame_rate("29.97")
        self.assertEqual(seconds_to_frames(100.0, rate), int(round(100 * 30000 / 1001)))

    def test_bad_mode_raises(self):
        with self.assertRaises(ValueError):
            seconds_to_frames(1.0, parse_frame_rate("25"), "nearest")

    def test_frames_to_seconds_inverts(self):
        rate = parse_frame_rate("23.976")
        # Rounding to whole frames costs up to half a frame (~21 ms at 23.976).
        self.assertAlmostEqual(
            frames_to_seconds(seconds_to_frames(10.0, rate), rate), 10.0, delta=0.03
        )


class TestFrameRateDataclass(unittest.TestCase):
    def test_non_drop_rate_drops_nothing(self):
        rate = FrameRate(fps=25.0, nominal=25, drop_frame=False)
        self.assertEqual(rate.dropped_per_minute, 0)
        self.assertEqual(rate.separator, ":")


if __name__ == "__main__":
    unittest.main()
