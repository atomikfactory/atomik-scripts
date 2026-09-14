"""RIFF/WAVE inspection and silence detection using only the standard library.

Python's :mod:`wave` module **cannot open the files Resolve's AI Speech
Generator writes**.  Verified against DaVinci Resolve Studio 21.0.1, a
generated clip is::

    RIFF/WAVE
      JUNK  (28 bytes)
      fmt   (40 bytes, WAVE_FORMAT_EXTENSIBLE tag 0xFFFE,
             SubFormat GUID 00000003-0000-0010-8000-00aa00389b71 = IEEE float)
      bext  (968 bytes)
      data
    48000 Hz, 1 channel, 32-bit float, block align 4

and ``wave.open()`` raises
``wave.Error: unknown extended format: 00000003-0000-0010-8000-00aa00389b71``.

So this module walks the RIFF chunks itself with :mod:`struct`.  It understands
format tags 1 (PCM), 3 (IEEE float) and 0xFFFE (extensible, whose real format
is the first two bytes of the SubFormat GUID); 8/16/24/32-bit integer and
32/64-bit float samples; and any channel count.  ``audioop`` was removed in
Python 3.13 and is not used.

Nothing here raises into the pipeline: an unreadable, truncated or exotic file
comes back as an :class:`AudioInfo` with ``ok=False`` and an ``error``.
"""

from __future__ import annotations

import array
import math
import os
import struct
import sys
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "FORMAT_EXTENSIBLE",
    "FORMAT_FLOAT",
    "FORMAT_PCM",
    "AudioInfo",
    "SilenceInfo",
    "detect_silence",
    "inspect_wav",
]

#: ``wFormatTag`` values we understand.
FORMAT_PCM = 0x0001
FORMAT_FLOAT = 0x0003
FORMAT_EXTENSIBLE = 0xFFFE

#: Amplitude corresponding to 0 dBFS for each supported integer sample width.
#: Float samples are already normalised, so their full scale is 1.0.
_INT_FULL_SCALE = {1: 128.0, 2: 32768.0, 3: 8388608.0, 4: 2147483648.0}

#: Sample widths (in bytes) we can decode, per sample format.
_SUPPORTED_WIDTHS = {"pcm": (1, 2, 3, 4), "float": (4, 8)}

#: How many frames to decode per read while scanning for silence.
_READ_FRAMES = 16384

#: dBFS reported for digital silence, instead of ``-inf``.
_SILENCE_FLOOR_DB = -120.0


@dataclass
class AudioInfo:
    """Everything we can learn about a generated WAV file.

    ``ok`` is false for anything we could not parse; every other field is then
    ``None`` and :meth:`describe` says ``unknown``.  The object is still safe to
    pass around, log and store in the manifest.
    """

    path: str
    ok: bool
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    sample_width: Optional[int] = None
    """Bytes per sample, per channel."""
    bits_per_sample: Optional[int] = None
    sample_format: Optional[str] = None
    """``"pcm"`` or ``"float"`` (the resolved format, not the raw tag)."""
    format_tag: Optional[int] = None
    """The raw ``wFormatTag``; 0xFFFE for WAVE_FORMAT_EXTENSIBLE."""
    block_align: Optional[int] = None
    frame_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    data_offset: Optional[int] = None
    data_bytes: Optional[int] = None
    truncated: bool = False
    """True when the ``data`` chunk claimed more bytes than the file holds."""
    error: Optional[str] = None

    @property
    def analyzable(self) -> bool:
        """True when the sample format is one we can do arithmetic on."""
        if not self.ok or self.sample_format is None or self.sample_width is None:
            return False
        return self.sample_width in _SUPPORTED_WIDTHS.get(self.sample_format, ())

    def describe(self) -> str:
        """One-line human summary, safe to log for an unreadable file too."""
        if not self.ok:
            return "unknown (%s)" % (self.error or "unreadable")
        return "%.3fs, %d Hz, %d ch, %d-bit %s%s" % (
            self.duration_seconds or 0.0,
            self.sample_rate or 0,
            self.channels or 0,
            self.bits_per_sample or 0,
            self.sample_format or "?",
            " (truncated)" if self.truncated else "",
        )


@dataclass
class SilenceInfo:
    """Leading and trailing silence measured in a WAV file."""

    analyzed: bool
    leading_seconds: float = 0.0
    trailing_seconds: float = 0.0
    peak_dbfs: Optional[float] = None
    reason: Optional[str] = None
    """Why analysis was skipped, when ``analyzed`` is false."""

    @property
    def all_silent(self) -> bool:
        """True when no frame anywhere in the file exceeded the threshold."""
        return self.analyzed and self.reason == "silent"


# --------------------------------------------------------------------------
# RIFF walking
# --------------------------------------------------------------------------


def _guid_string(raw: bytes) -> str:
    """Format a 16-byte SubFormat GUID the way Windows prints it."""
    if len(raw) < 16:
        return raw.hex()
    first, second, third = struct.unpack_from("<IHH", raw, 0)
    return "%08x-%04x-%04x-%s-%s" % (
        first,
        second,
        third,
        raw[8:10].hex(),
        raw[10:16].hex(),
    )


def _walk_chunks(handle: Any, file_size: int) -> Iterator[Tuple[bytes, int, int, bool]]:
    """Yield ``(chunk_id, body_offset, usable_size, truncated)`` for each chunk.

    Chunk bodies are padded to an even length, which the padding byte after an
    odd ``size`` accounts for.  A chunk whose declared size runs past the end of
    the file is yielded with the bytes that actually exist and ``truncated``
    set, and iteration then stops.
    """
    position = 12
    while position + 8 <= file_size:
        handle.seek(position)
        header = handle.read(8)
        if len(header) < 8:
            return
        chunk_id, declared = struct.unpack("<4sI", header)
        body = position + 8
        available = max(0, file_size - body)
        usable = min(declared, available)
        truncated = declared > available
        yield chunk_id, body, usable, truncated
        if truncated:
            return
        position = body + declared + (declared & 1)


def _parse_fmt(raw: bytes) -> Tuple[Optional[dict], Optional[str]]:
    """Decode a ``fmt `` chunk body into a dict, or return an error string."""
    if len(raw) < 16:
        return None, "fmt chunk is only %d bytes" % len(raw)
    tag, channels, sample_rate, _byte_rate, block_align, bits = struct.unpack_from(
        "<HHIIHH", raw, 0
    )

    resolved_tag = tag
    if tag == FORMAT_EXTENSIBLE:
        # 16 header + 2 cbSize + 2 wValidBitsPerSample + 4 dwChannelMask = 24,
        # then the 16-byte SubFormat GUID whose first two bytes are the real tag.
        if len(raw) < 26:
            return None, "WAVE_FORMAT_EXTENSIBLE fmt chunk is only %d bytes" % len(raw)
        resolved_tag = struct.unpack_from("<H", raw, 24)[0]
        guid = _guid_string(raw[24:40])
    else:
        guid = None

    if resolved_tag == FORMAT_PCM:
        sample_format = "pcm"
    elif resolved_tag == FORMAT_FLOAT:
        sample_format = "float"
    else:
        return None, "unsupported format tag 0x%04X%s" % (
            resolved_tag,
            "" if guid is None else " (SubFormat %s)" % guid,
        )

    if channels < 1:
        return None, "fmt chunk reports %d channels" % channels
    if sample_rate <= 0:
        return None, "fmt chunk reports a sample rate of %d" % sample_rate
    if bits <= 0 or bits % 8:
        return None, "fmt chunk reports %d bits per sample" % bits

    width = bits // 8
    if block_align <= 0 or block_align < width * channels:
        block_align = width * channels

    return (
        {
            "format_tag": tag,
            "sample_format": sample_format,
            "channels": channels,
            "sample_rate": sample_rate,
            "bits_per_sample": bits,
            "sample_width": width,
            "block_align": block_align,
        },
        None,
    )


def inspect_wav(path: str) -> AudioInfo:
    """Read a RIFF/WAVE header and return its properties.

    Never raises: an unreadable, truncated or unsupported file comes back with
    ``ok=False`` and an ``error`` describing why.
    """
    if not path:
        return AudioInfo(path=path, ok=False, error="no file path")
    if not os.path.isfile(path):
        return AudioInfo(path=path, ok=False, error="file does not exist")

    try:
        file_size = os.path.getsize(path)
        with open(path, "rb") as handle:
            header = handle.read(12)
            if len(header) < 12:
                return AudioInfo(path=path, ok=False, error="file is only %d bytes" % len(header))
            if header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
                return AudioInfo(
                    path=path,
                    ok=False,
                    error="not a RIFF/WAVE file (starts with %r)" % header[0:4],
                )

            fmt: Optional[dict] = None
            fmt_error: Optional[str] = None
            data_offset: Optional[int] = None
            data_bytes = 0
            truncated = False

            for chunk_id, body, usable, chunk_truncated in _walk_chunks(handle, file_size):
                if chunk_id == b"fmt " and fmt is None and fmt_error is None:
                    handle.seek(body)
                    fmt, fmt_error = _parse_fmt(handle.read(usable))
                elif chunk_id == b"data" and data_offset is None:
                    data_offset = body
                    data_bytes = usable
                    truncated = chunk_truncated
                if fmt is not None and data_offset is not None:
                    # Everything we need; the rest of the file is metadata.
                    break
    except (OSError, struct.error, ValueError) as exc:
        return AudioInfo(path=path, ok=False, error="%s: %s" % (type(exc).__name__, exc))

    if fmt is None:
        return AudioInfo(path=path, ok=False, error=fmt_error or "no fmt chunk found")
    if data_offset is None:
        return AudioInfo(path=path, ok=False, error="no data chunk found")

    block_align = fmt["block_align"]
    usable_bytes = data_bytes - (data_bytes % block_align)
    frame_count = usable_bytes // block_align
    duration = frame_count / float(fmt["sample_rate"])

    return AudioInfo(
        path=path,
        ok=True,
        sample_rate=fmt["sample_rate"],
        channels=fmt["channels"],
        sample_width=fmt["sample_width"],
        bits_per_sample=fmt["bits_per_sample"],
        sample_format=fmt["sample_format"],
        format_tag=fmt["format_tag"],
        block_align=block_align,
        frame_count=frame_count,
        duration_seconds=duration,
        data_offset=data_offset,
        data_bytes=usable_bytes,
        truncated=truncated,
    )


# --------------------------------------------------------------------------
# Sample decoding
# --------------------------------------------------------------------------


def _typed_array(type_code: str, width: int, raw: bytes) -> Sequence[float]:
    """Decode ``raw`` with :mod:`array`, falling back to :mod:`struct`."""
    values = array.array(type_code)
    if values.itemsize == width:
        usable = len(raw) - (len(raw) % width)
        values.frombytes(raw[:usable])
        if sys.byteorder == "big":
            values.byteswap()
        return values
    count = len(raw) // width
    return struct.unpack("<%d%s" % (count, type_code), raw[: count * width])


def _decode_samples(raw: bytes, sample_format: str, width: int) -> Sequence[float]:
    """Decode little-endian samples to floats normalised so 0 dBFS is 1.0."""
    if sample_format == "float":
        if width == 4:
            return _typed_array("f", 4, raw)
        return _typed_array("d", 8, raw)

    if width == 1:
        # 8-bit WAV is unsigned with a midpoint of 128.
        scale = _INT_FULL_SCALE[1]
        return [(byte - 128) / scale for byte in raw]

    if width == 3:
        scale = _INT_FULL_SCALE[3]
        out: List[float] = []
        for offset in range(0, len(raw) - 2, 3):
            value = raw[offset] | (raw[offset + 1] << 8) | (raw[offset + 2] << 16)
            if value & 0x800000:
                value -= 0x1000000
            out.append(value / scale)
        return out

    scale = _INT_FULL_SCALE[width]
    type_code = "h" if width == 2 else "i"
    return [value / scale for value in _typed_array(type_code, width, raw)]


def _frame_peaks(samples: Sequence[float], channels: int) -> List[float]:
    """Collapse interleaved samples to one peak magnitude per frame."""
    if channels == 1:
        return [-value if value < 0 else value for value in samples]
    peaks: List[float] = []
    for start in range(0, len(samples) - channels + 1, channels):
        peak = 0.0
        for offset in range(channels):
            value = samples[start + offset]
            if value < 0:
                value = -value
            if value > peak:
                peak = value
        peaks.append(peak)
    return peaks


def _to_dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return _SILENCE_FLOOR_DB
    return 20.0 * math.log10(amplitude)


# --------------------------------------------------------------------------
# Silence detection
# --------------------------------------------------------------------------


def detect_silence(
    path: str,
    threshold_db: float = -50.0,
    info: Optional[AudioInfo] = None,
) -> SilenceInfo:
    """Measure leading and trailing silence in a WAV file.

    A frame counts as silent when its *peak* across all channels is at or below
    ``threshold_db`` dBFS (full scale is 1.0, for float and integer samples
    alike).  Peak rather than RMS is deliberately conservative: it under-reports
    silence, so trimming based on it can never clip the start of a word.

    Never raises; unreadable or undecodable files come back with
    ``analyzed=False`` and a ``reason``.
    """
    info = info or inspect_wav(path)
    if not info.ok:
        return SilenceInfo(analyzed=False, reason=info.error or "unreadable")
    if not info.analyzable:
        return SilenceInfo(
            analyzed=False,
            reason="unsupported sample format (%s, %s bytes per sample)"
            % (info.sample_format, info.sample_width),
        )
    if not info.frame_count:
        return SilenceInfo(analyzed=False, reason="empty file")

    assert info.sample_rate is not None and info.sample_width is not None
    assert info.channels is not None and info.block_align is not None
    assert info.sample_format is not None and info.data_offset is not None

    threshold_amplitude = 10.0 ** (threshold_db / 20.0)
    first_loud: Optional[int] = None
    last_loud: Optional[int] = None
    peak_overall = 0.0
    frame_index = 0

    try:
        with open(path, "rb") as handle:
            handle.seek(info.data_offset)
            remaining = info.data_bytes or 0
            while remaining > 0:
                wanted = min(remaining, _READ_FRAMES * info.block_align)
                raw = handle.read(wanted)
                if not raw:
                    break
                remaining -= len(raw)
                usable = len(raw) - (len(raw) % info.block_align)
                samples = _decode_samples(
                    raw[:usable], info.sample_format, info.sample_width
                )
                for peak in _frame_peaks(samples, info.channels):
                    if peak > peak_overall:
                        peak_overall = peak
                    if peak > threshold_amplitude:
                        if first_loud is None:
                            first_loud = frame_index
                        last_loud = frame_index
                    frame_index += 1
    except (OSError, struct.error, ValueError) as exc:
        return SilenceInfo(analyzed=False, reason="%s: %s" % (type(exc).__name__, exc))

    peak_dbfs = _to_dbfs(peak_overall)
    total_frames = frame_index or info.frame_count
    duration = total_frames / float(info.sample_rate)

    if first_loud is None or last_loud is None:
        return SilenceInfo(
            analyzed=True,
            leading_seconds=0.0,
            trailing_seconds=0.0,
            peak_dbfs=peak_dbfs,
            reason="silent",
        )

    leading = first_loud / float(info.sample_rate)
    trailing = max(0.0, (total_frames - (last_loud + 1)) / float(info.sample_rate))
    return SilenceInfo(
        analyzed=True,
        leading_seconds=min(leading, duration),
        trailing_seconds=min(trailing, duration),
        peak_dbfs=peak_dbfs,
    )
