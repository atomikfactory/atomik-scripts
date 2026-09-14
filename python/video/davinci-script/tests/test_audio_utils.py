"""Tests for :mod:`resolve_tts.audio_utils`.

Every WAV here is assembled byte by byte rather than with :mod:`wave`, because
the files Resolve's AI Speech Generator actually writes are ones :mod:`wave`
refuses to open.  The layout verified against DaVinci Resolve Studio 21.0.1 is
reproduced exactly in :class:`TestResolveGeneratedLayout`::

    JUNK (28) + fmt (40, WAVE_FORMAT_EXTENSIBLE / IEEE float) + bext (968) + data
    48000 Hz, 1 channel, 32-bit float, block align 4
"""

from __future__ import annotations

import math
import os
import shutil
import struct
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resolve_tts.audio_utils import (
    FORMAT_EXTENSIBLE,
    FORMAT_FLOAT,
    FORMAT_PCM,
    detect_silence,
    inspect_wav,
)

SAMPLE_RATE = 8000

#: The trailing 8 bytes of every KSDATAFORMAT_SUBTYPE_* GUID.
_GUID_TAIL = bytes.fromhex("800000aa00389b71")

_INT_FULL_SCALE = {1: 128.0, 2: 32768.0, 3: 8388608.0, 4: 2147483648.0}


# --------------------------------------------------------------------------
# RIFF construction helpers
# --------------------------------------------------------------------------


def chunk(chunk_id: bytes, body: bytes, declared_size: int = None) -> bytes:
    """One RIFF chunk, padded to an even length like a real writer would."""
    size = len(body) if declared_size is None else declared_size
    out = chunk_id + struct.pack("<I", size) + body
    if declared_size is None and size % 2:
        out += b"\x00"
    return out


def riff(chunks: bytes, declared_size: int = None) -> bytes:
    """Wrap already-serialised chunks in a RIFF/WAVE container."""
    body = b"WAVE" + chunks
    size = len(body) if declared_size is None else declared_size
    return b"RIFF" + struct.pack("<I", size) + body


def fmt_body(
    sample_format: str,
    bits: int,
    channels: int,
    sample_rate: int,
    extensible: bool = False,
    subformat_tag: int = None,
) -> bytes:
    """A ``fmt `` chunk body: 16 bytes, or 40 for WAVE_FORMAT_EXTENSIBLE."""
    tag = FORMAT_PCM if sample_format == "pcm" else FORMAT_FLOAT
    width = bits // 8
    block_align = width * channels
    header = struct.pack(
        "<HHIIHH",
        FORMAT_EXTENSIBLE if extensible else tag,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
    )
    if not extensible:
        return header
    guid = struct.pack("<IHH", tag if subformat_tag is None else subformat_tag, 0, 0x0010)
    return header + struct.pack("<HHI", 22, bits, 0) + guid + _GUID_TAIL


def tone_samples(
    lead_silence: float,
    tone: float,
    trail_silence: float,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
    amplitude: float = 0.5,
    noise_amplitude: float = 0.0,
    loud_channel: int = None,
) -> list:
    """Interleaved float samples: silence, a 440 Hz tone, then silence.

    ``noise_amplitude`` fills the "silent" parts with a low-level tone instead
    of digital zero, which is what the real generated clips contain (their first
    sample is non-zero, yet the first 24-32 ms sit below -50 dBFS).
    """
    values = []

    def emit(count: int, peak: float, offset: int = 0) -> None:
        for index in range(count):
            value = peak * math.sin(2.0 * math.pi * 440.0 * (index + offset) / sample_rate)
            for channel in range(channels):
                if loud_channel is not None and channel != loud_channel:
                    values.append(0.0)
                else:
                    values.append(value)

    emit(int(round(lead_silence * sample_rate)), noise_amplitude)
    emit(int(round(tone * sample_rate)), amplitude)
    emit(int(round(trail_silence * sample_rate)), noise_amplitude)
    return values


def encode(values, sample_format: str, bits: int) -> bytes:
    """Encode normalised float samples into little-endian WAV data bytes."""
    width = bits // 8
    if sample_format == "float":
        code = "<f" if width == 4 else "<d"
        return b"".join(struct.pack(code, value) for value in values)

    scale = _INT_FULL_SCALE[width]
    out = bytearray()
    for value in values:
        sample = int(value * scale)
        sample = max(int(-scale), min(int(scale) - 1, sample))
        if width == 1:
            out += bytes([sample + 128])
        elif width == 2:
            out += struct.pack("<h", sample)
        elif width == 3:
            unsigned = sample & 0xFFFFFF
            out += bytes([unsigned & 0xFF, (unsigned >> 8) & 0xFF, (unsigned >> 16) & 0xFF])
        else:
            out += struct.pack("<i", sample)
    return bytes(out)


class AudioTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="resolve_tts_audio_")
        self.addCleanup(shutil.rmtree, self.directory, True)

    def path(self, name: str) -> str:
        return os.path.join(self.directory, name)

    def write(self, name: str, payload: bytes) -> str:
        path = self.path(name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def write_wav(
        self,
        name: str,
        lead_silence: float = 0.0,
        tone: float = 0.5,
        trail_silence: float = 0.0,
        sample_format: str = "pcm",
        bits: int = 16,
        channels: int = 1,
        sample_rate: int = SAMPLE_RATE,
        extensible: bool = False,
        amplitude: float = 0.5,
        noise_amplitude: float = 0.0,
        loud_channel: int = None,
        leading_chunks: bytes = b"",
        middle_chunks: bytes = b"",
    ) -> str:
        """Write a WAV with an arbitrary chunk layout around ``fmt `` and ``data``."""
        values = tone_samples(
            lead_silence,
            tone,
            trail_silence,
            sample_rate=sample_rate,
            channels=channels,
            amplitude=amplitude,
            noise_amplitude=noise_amplitude,
            loud_channel=loud_channel,
        )
        data = encode(values, sample_format, bits)
        payload = riff(
            leading_chunks
            + chunk(b"fmt ", fmt_body(sample_format, bits, channels, sample_rate, extensible))
            + middle_chunks
            + chunk(b"data", data)
        )
        return self.write(name, payload)


# --------------------------------------------------------------------------
# The real Resolve layout
# --------------------------------------------------------------------------


class TestResolveGeneratedLayout(AudioTestCase):
    """The exact shape of a clip from Resolve Studio 21.0.1's Speech Generator."""

    def generated(self, name="generated.wav", lead=0.028, tone=4.312, trail=0.060):
        return self.write_wav(
            name,
            lead_silence=lead,
            tone=tone,
            trail_silence=trail,
            sample_format="float",
            bits=32,
            channels=1,
            sample_rate=48000,
            extensible=True,
            # Real clips are not digitally silent at either end: the first
            # sample is non-zero, it is simply far below the threshold.
            noise_amplitude=10 ** (-70.0 / 20.0),
            leading_chunks=chunk(b"JUNK", b"\x00" * 28),
            middle_chunks=chunk(b"bext", b"\x00" * 968),
        )

    def test_pythons_wave_module_cannot_open_it(self):
        # The reason this module exists at all.
        path = self.generated()
        with self.assertRaises(wave.Error):
            with wave.open(path, "rb"):
                pass

    def test_inspect_reads_the_real_layout(self):
        info = inspect_wav(self.generated())
        self.assertTrue(info.ok, info.error)
        self.assertEqual(info.sample_rate, 48000)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.bits_per_sample, 32)
        self.assertEqual(info.sample_width, 4)
        self.assertEqual(info.sample_format, "float")
        self.assertEqual(info.format_tag, FORMAT_EXTENSIBLE)
        self.assertEqual(info.block_align, 4)
        self.assertFalse(info.truncated)
        self.assertTrue(info.analyzable)
        self.assertAlmostEqual(info.duration_seconds, 4.400, places=4)
        self.assertEqual(info.frame_count, 211200)
        self.assertIn("32-bit float", info.describe())

    def test_duration_is_data_bytes_over_block_align_over_rate(self):
        info = inspect_wav(self.generated())
        self.assertEqual(
            info.duration_seconds,
            info.data_bytes / info.block_align / float(info.sample_rate),
        )

    def test_measured_padding_matches_the_live_measurements(self):
        # 24-32 ms leading and 48-76 ms trailing were measured on real clips.
        info_path = self.generated(lead=0.028, tone=4.312, trail=0.060)
        silence = detect_silence(info_path, threshold_db=-50.0)
        self.assertTrue(silence.analyzed, silence.reason)
        self.assertAlmostEqual(silence.leading_seconds, 0.028, delta=0.002)
        self.assertAlmostEqual(silence.trailing_seconds, 0.060, delta=0.002)
        self.assertGreater(silence.peak_dbfs, -10.0)

    def test_at_24_fps_the_padding_is_under_two_frames(self):
        silence = detect_silence(self.generated(), threshold_db=-50.0)
        frame = 1.0 / 24.0
        self.assertLess(silence.leading_seconds, frame)
        self.assertLess(silence.trailing_seconds, 2 * frame)

    def test_a_float_clip_can_exceed_full_scale_without_breaking(self):
        data = struct.pack("<f", 1.5) * 100 + struct.pack("<f", 0.0) * 100
        path = self.write(
            "hot.wav",
            riff(
                chunk(b"fmt ", fmt_body("float", 32, 1, 48000, extensible=True))
                + chunk(b"data", data)
            ),
        )
        info = inspect_wav(path)
        self.assertTrue(info.ok, info.error)
        silence = detect_silence(path, info=info)
        self.assertTrue(silence.analyzed)
        self.assertGreater(silence.peak_dbfs, 0.0)


# --------------------------------------------------------------------------
# Header parsing
# --------------------------------------------------------------------------


class TestInspectWav(AudioTestCase):
    def test_every_supported_integer_width(self):
        for bits in (8, 16, 24, 32):
            with self.subTest(bits=bits):
                path = self.write_wav("i%d.wav" % bits, 0.1, 0.5, 0.1, bits=bits)
                info = inspect_wav(path)
                self.assertTrue(info.ok, info.error)
                self.assertTrue(info.analyzable)
                self.assertEqual(info.sample_format, "pcm")
                self.assertEqual(info.sample_width, bits // 8)
                self.assertAlmostEqual(info.duration_seconds, 0.7, places=4)

    def test_both_float_widths(self):
        for bits in (32, 64):
            with self.subTest(bits=bits):
                path = self.write_wav(
                    "f%d.wav" % bits, 0.1, 0.5, 0.1, sample_format="float", bits=bits
                )
                info = inspect_wav(path)
                self.assertTrue(info.ok, info.error)
                self.assertTrue(info.analyzable)
                self.assertEqual(info.sample_format, "float")
                self.assertEqual(info.sample_width, bits // 8)
                self.assertAlmostEqual(info.duration_seconds, 0.7, places=4)

    def test_extensible_pcm_resolves_to_pcm(self):
        path = self.write_wav("ext_pcm.wav", 0.0, 0.25, 0.0, bits=24, extensible=True)
        info = inspect_wav(path)
        self.assertTrue(info.ok, info.error)
        self.assertEqual(info.sample_format, "pcm")
        self.assertEqual(info.format_tag, FORMAT_EXTENSIBLE)
        self.assertEqual(info.bits_per_sample, 24)

    def test_multi_channel(self):
        for channels in (2, 6):
            with self.subTest(channels=channels):
                path = self.write_wav("ch%d.wav" % channels, 0.0, 0.5, 0.0, channels=channels)
                info = inspect_wav(path)
                self.assertTrue(info.ok, info.error)
                self.assertEqual(info.channels, channels)
                self.assertEqual(info.block_align, 2 * channels)
                self.assertAlmostEqual(info.duration_seconds, 0.5, places=4)

    def test_odd_sized_chunk_is_padded_to_even(self):
        # A 5-byte chunk occupies 6 bytes; missing the pad byte would leave the
        # walker one byte out of step and it would never find 'fmt '.
        path = self.write_wav(
            "odd.wav",
            0.0,
            0.25,
            0.0,
            leading_chunks=chunk(b"LIST", b"INFOx"),
            middle_chunks=chunk(b"IARL", b"odd"),
        )
        info = inspect_wav(path)
        self.assertTrue(info.ok, info.error)
        self.assertAlmostEqual(info.duration_seconds, 0.25, places=4)

    def test_chunks_after_data_are_ignored(self):
        data = encode(tone_samples(0.0, 0.1, 0.0), "pcm", 16)
        path = self.write(
            "trailing.wav",
            riff(
                chunk(b"fmt ", fmt_body("pcm", 16, 1, SAMPLE_RATE))
                + chunk(b"data", data)
                + chunk(b"LIST", b"INFO junk")
            ),
        )
        info = inspect_wav(path)
        self.assertTrue(info.ok, info.error)
        self.assertEqual(info.frame_count, int(0.1 * SAMPLE_RATE))

    def test_truncated_data_chunk_is_reported_and_still_usable(self):
        data = encode(tone_samples(0.0, 0.5, 0.0), "pcm", 16)
        path = self.write(
            "short_data.wav",
            riff(
                chunk(b"fmt ", fmt_body("pcm", 16, 1, SAMPLE_RATE))
                # Claim twice as much audio as the file actually holds.
                + chunk(b"data", data, declared_size=len(data) * 2)
            ),
        )
        info = inspect_wav(path)
        self.assertTrue(info.ok, info.error)
        self.assertTrue(info.truncated)
        self.assertEqual(info.data_bytes, len(data))
        self.assertAlmostEqual(info.duration_seconds, 0.5, places=4)
        self.assertTrue(detect_silence(path, info=info).analyzed)

    def test_data_bytes_not_a_multiple_of_block_align(self):
        data = encode(tone_samples(0.0, 0.1, 0.0), "pcm", 16, ) + b"\x01"
        path = self.write(
            "ragged.wav",
            riff(
                chunk(b"fmt ", fmt_body("pcm", 16, 1, SAMPLE_RATE))
                + chunk(b"data", data)
            ),
        )
        info = inspect_wav(path)
        self.assertTrue(info.ok, info.error)
        self.assertEqual(info.frame_count, len(data) // 2)
        self.assertTrue(detect_silence(path, info=info).analyzed)

    def test_wrong_riff_size_field_is_ignored(self):
        data = encode(tone_samples(0.0, 0.2, 0.0), "pcm", 16)
        path = self.write(
            "badsize.wav",
            riff(
                chunk(b"fmt ", fmt_body("pcm", 16, 1, SAMPLE_RATE))
                + chunk(b"data", data),
                declared_size=4,
            ),
        )
        info = inspect_wav(path)
        self.assertTrue(info.ok, info.error)
        self.assertAlmostEqual(info.duration_seconds, 0.2, places=4)

    def test_missing_file(self):
        info = inspect_wav(self.path("nope.wav"))
        self.assertFalse(info.ok)
        self.assertIn("does not exist", info.error or "")
        self.assertFalse(info.analyzable)
        self.assertIn("unknown", info.describe())

    def test_empty_path(self):
        self.assertFalse(inspect_wav("").ok)

    def test_zero_length_file(self):
        info = inspect_wav(self.write("empty.wav", b""))
        self.assertFalse(info.ok)
        self.assertTrue(info.error)

    def test_non_riff_file(self):
        info = inspect_wav(self.write("notaudio.txt", b"this is definitely not a RIFF file"))
        self.assertFalse(info.ok)
        self.assertIn("RIFF", info.error or "")

    def test_riff_header_only(self):
        info = inspect_wav(self.write("headeronly.wav", b"RIFF" + struct.pack("<I", 4) + b"WAVE"))
        self.assertFalse(info.ok)
        self.assertIn("no fmt chunk", info.error or "")

    def test_truncated_in_the_middle_of_a_chunk_header(self):
        info = inspect_wav(
            self.write("cut.wav", riff(chunk(b"fmt ", fmt_body("pcm", 16, 1, 8000)))[:30])
        )
        self.assertFalse(info.ok)
        self.assertTrue(info.error)

    def test_no_data_chunk(self):
        info = inspect_wav(
            self.write("nodata.wav", riff(chunk(b"fmt ", fmt_body("pcm", 16, 1, 8000))))
        )
        self.assertFalse(info.ok)
        self.assertIn("no data chunk", info.error or "")

    def test_fmt_chunk_too_short(self):
        info = inspect_wav(
            self.write("shortfmt.wav", riff(chunk(b"fmt ", b"\x01\x00\x01\x00") + chunk(b"data", b"\x00" * 8)))
        )
        self.assertFalse(info.ok)
        self.assertIn("fmt chunk", info.error or "")

    def test_unsupported_format_tag(self):
        body = struct.pack("<HHIIHH", 0x0011, 1, 8000, 4000, 1, 4)  # IMA ADPCM
        info = inspect_wav(
            self.write("adpcm.wav", riff(chunk(b"fmt ", body) + chunk(b"data", b"\x00" * 16)))
        )
        self.assertFalse(info.ok)
        self.assertIn("0x0011", info.error or "")
        self.assertFalse(info.analyzable)
        self.assertIn("unknown", info.describe())

    def test_unsupported_extensible_subformat_reports_the_guid(self):
        body = fmt_body("float", 32, 1, 48000, extensible=True, subformat_tag=0x0092)
        info = inspect_wav(
            self.write("ac3.wav", riff(chunk(b"fmt ", body) + chunk(b"data", b"\x00" * 16)))
        )
        self.assertFalse(info.ok)
        self.assertIn("00000092-0000-0010-8000-00aa00389b71", info.error or "")

    def test_zero_channels_is_rejected(self):
        body = struct.pack("<HHIIHH", FORMAT_PCM, 0, 8000, 16000, 2, 16)
        info = inspect_wav(
            self.write("nochannels.wav", riff(chunk(b"fmt ", body) + chunk(b"data", b"\x00" * 16)))
        )
        self.assertFalse(info.ok)
        self.assertIn("channels", info.error or "")

    def test_zero_sample_rate_is_rejected(self):
        body = struct.pack("<HHIIHH", FORMAT_PCM, 1, 0, 0, 2, 16)
        info = inspect_wav(
            self.write("norate.wav", riff(chunk(b"fmt ", body) + chunk(b"data", b"\x00" * 16)))
        )
        self.assertFalse(info.ok)
        self.assertIn("sample rate", info.error or "")

    def test_wrong_block_align_falls_back_to_channels_times_width(self):
        body = struct.pack("<HHIIHH", FORMAT_PCM, 2, 8000, 32000, 0, 16)
        data = b"\x00\x00\x01\x00" * 40
        info = inspect_wav(
            self.write("badalign.wav", riff(chunk(b"fmt ", body) + chunk(b"data", data)))
        )
        self.assertTrue(info.ok, info.error)
        self.assertEqual(info.block_align, 4)
        self.assertEqual(info.frame_count, 40)

# --------------------------------------------------------------------------
# Silence detection
# --------------------------------------------------------------------------


class TestDetectSilence(AudioTestCase):
    def test_detects_leading_and_trailing_silence(self):
        path = self.write_wav("padded.wav", 0.30, 1.00, 0.50)
        silence = detect_silence(path, threshold_db=-50.0)
        self.assertTrue(silence.analyzed)
        self.assertAlmostEqual(silence.leading_seconds, 0.30, delta=0.005)
        self.assertAlmostEqual(silence.trailing_seconds, 0.50, delta=0.005)
        self.assertIsNotNone(silence.peak_dbfs)
        self.assertGreater(silence.peak_dbfs, -10.0)

    def test_no_padding_means_no_silence(self):
        silence = detect_silence(self.write_wav("tight.wav", 0.0, 0.75, 0.0))
        self.assertTrue(silence.analyzed)
        self.assertAlmostEqual(silence.leading_seconds, 0.0, delta=0.005)
        self.assertAlmostEqual(silence.trailing_seconds, 0.0, delta=0.005)

    def test_every_format_measures_the_same_padding(self):
        cases = [("pcm", 8), ("pcm", 16), ("pcm", 24), ("pcm", 32), ("float", 32), ("float", 64)]
        for sample_format, bits in cases:
            with self.subTest(sample_format=sample_format, bits=bits):
                path = self.write_wav(
                    "pad_%s%d.wav" % (sample_format, bits),
                    0.20,
                    0.60,
                    0.40,
                    sample_format=sample_format,
                    bits=bits,
                )
                silence = detect_silence(path, threshold_db=-50.0)
                self.assertTrue(silence.analyzed, silence.reason)
                self.assertAlmostEqual(silence.leading_seconds, 0.20, delta=0.01)
                self.assertAlmostEqual(silence.trailing_seconds, 0.40, delta=0.01)

    def test_extensible_float_with_metadata_chunks(self):
        path = self.write_wav(
            "ext_pad.wav",
            0.20,
            0.60,
            0.40,
            sample_format="float",
            bits=32,
            sample_rate=48000,
            extensible=True,
            leading_chunks=chunk(b"JUNK", b"\x00" * 28),
            middle_chunks=chunk(b"bext", b"\x00" * 968),
        )
        silence = detect_silence(path, threshold_db=-50.0)
        self.assertTrue(silence.analyzed, silence.reason)
        self.assertAlmostEqual(silence.leading_seconds, 0.20, delta=0.005)
        self.assertAlmostEqual(silence.trailing_seconds, 0.40, delta=0.005)

    def test_stereo_padding(self):
        silence = detect_silence(self.write_wav("stereo_pad.wav", 0.25, 0.50, 0.25, channels=2))
        self.assertAlmostEqual(silence.leading_seconds, 0.25, delta=0.01)
        self.assertAlmostEqual(silence.trailing_seconds, 0.25, delta=0.01)

    def test_peak_is_taken_across_channels(self):
        # Only the right channel carries the tone; the frame is still not silent.
        path = self.write_wav(
            "one_channel_loud.wav", 0.25, 0.50, 0.25, channels=2, loud_channel=1
        )
        silence = detect_silence(path, threshold_db=-50.0)
        self.assertTrue(silence.analyzed)
        self.assertAlmostEqual(silence.leading_seconds, 0.25, delta=0.01)
        self.assertAlmostEqual(silence.trailing_seconds, 0.25, delta=0.01)

    def test_low_level_noise_below_the_threshold_still_counts_as_silence(self):
        path = self.write_wav(
            "noisy_pad.wav",
            0.20,
            0.40,
            0.20,
            sample_format="float",
            bits=32,
            noise_amplitude=10 ** (-70.0 / 20.0),
        )
        silence = detect_silence(path, threshold_db=-50.0)
        self.assertAlmostEqual(silence.leading_seconds, 0.20, delta=0.005)
        self.assertAlmostEqual(silence.trailing_seconds, 0.20, delta=0.005)

    def test_entirely_silent_file(self):
        silence = detect_silence(self.write_wav("silent.wav", 1.0, 0.0, 0.0))
        self.assertTrue(silence.analyzed)
        self.assertTrue(silence.all_silent)
        self.assertEqual(silence.leading_seconds, 0.0)
        self.assertEqual(silence.trailing_seconds, 0.0)

    def test_threshold_controls_sensitivity(self):
        path = self.write_wav(
            "quiet_lead.wav",
            0.50,
            0.50,
            0.0,
            sample_format="float",
            bits=32,
            noise_amplitude=10 ** (-60.0 / 20.0),
        )
        lenient = detect_silence(path, threshold_db=-50.0)
        strict = detect_silence(path, threshold_db=-80.0)
        self.assertAlmostEqual(lenient.leading_seconds, 0.5, delta=0.01)
        self.assertAlmostEqual(strict.leading_seconds, 0.0, delta=0.01)

    def test_missing_file_is_not_analysed(self):
        silence = detect_silence(self.path("gone.wav"))
        self.assertFalse(silence.analyzed)
        self.assertFalse(silence.all_silent)
        self.assertTrue(silence.reason)

    def test_unsupported_format_is_not_analysed(self):
        body = struct.pack("<HHIIHH", 0x0011, 1, 8000, 4000, 1, 4)
        path = self.write("adpcm.wav", riff(chunk(b"fmt ", body) + chunk(b"data", b"\x00" * 16)))
        silence = detect_silence(path)
        self.assertFalse(silence.analyzed)
        self.assertTrue(silence.reason)

    def test_silence_never_exceeds_the_duration(self):
        path = self.write_wav("short.wav", 0.05, 0.05, 0.05)
        info = inspect_wav(path)
        silence = detect_silence(path, info=info)
        self.assertLessEqual(silence.leading_seconds, info.duration_seconds)
        self.assertLessEqual(silence.trailing_seconds, info.duration_seconds)
        self.assertLessEqual(
            silence.leading_seconds + silence.trailing_seconds,
            info.duration_seconds + 0.01,
        )


if __name__ == "__main__":
    unittest.main()
