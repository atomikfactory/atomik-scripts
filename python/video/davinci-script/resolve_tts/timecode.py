"""SMPTE timecode <-> frame conversion, including drop-frame rates.

Resolve reports its timeline rate through ``Timeline.GetSetting("timelineFrameRate")``
as a string such as ``"24"``, ``"23.976"``, ``"29.97"`` or ``"29.97 DF"`` -- the
``DF`` suffix meaning drop-frame.  Playhead positions come back from
``Timeline.GetCurrentTimecode()`` as ``HH:MM:SS:FF`` (non-drop) or
``HH:MM:SS;FF`` (drop-frame).

Drop-frame timecode never drops *frames*, only *labels*: at the start of every
minute except every tenth minute, the first ``nominal // 15`` frame numbers are
skipped (2 at 29.97, 4 at 59.94) so that the clock keeps pace with wall time.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Union

__all__ = [
    "FrameRate",
    "frames_to_timecode",
    "is_timecode",
    "parse_frame_rate",
    "seconds_to_frames",
    "timecode_to_frames",
]

_TIMECODE_RE = re.compile(r"^(\d{1,3}):([0-5]\d):([0-5]\d)[:;.](\d{1,3})$")
_RATE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(df|drop|nd|ndf)?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class FrameRate:
    """A timeline frame rate.

    Attributes:
        fps: The true rate in frames per second (``30000 / 1001`` for 29.97).
        nominal: The integer rate timecode labels count at (30 for 29.97).
        drop_frame: Whether drop-frame labelling is in effect.
    """

    fps: float
    nominal: int
    drop_frame: bool

    @property
    def dropped_per_minute(self) -> int:
        """Frame labels skipped at the top of a non-tenth minute (0 when non-drop)."""
        return self.nominal // 15 if self.drop_frame else 0

    @property
    def separator(self) -> str:
        """``';'`` for drop-frame timecode, ``':'`` otherwise."""
        return ";" if self.drop_frame else ":"

    def __str__(self) -> str:
        rate = ("%f" % self.fps).rstrip("0").rstrip(".")
        return rate + (" DF" if self.drop_frame else "")


def parse_frame_rate(value: Union[str, float, int, None]) -> FrameRate:
    """Build a :class:`FrameRate` from a Resolve setting string or a number.

    Accepts ``"29.97 DF"``, ``"29.97"``, ``29.97``, ``"24"`` and ``24``.

    Raises:
        ValueError: The value is unparseable, non-positive, or asks for
            drop-frame at a rate where drop-frame is undefined.
    """
    if value is None:
        raise ValueError("frame rate is missing")

    drop = False
    if isinstance(value, str):
        match = _RATE_RE.match(value)
        if not match:
            raise ValueError("unrecognised frame rate %r" % (value,))
        fps = float(match.group(1))
        suffix = (match.group(2) or "").lower()
        drop = suffix in ("df", "drop")
    else:
        fps = float(value)

    if fps <= 0:
        raise ValueError("frame rate must be positive, got %r" % (value,))

    nominal = int(round(fps))
    if nominal <= 0:
        raise ValueError("frame rate must be positive, got %r" % (value,))

    # Snap the NTSC pulldown rates onto their exact rational value so long
    # durations do not drift.
    pulldown = nominal * 1000.0 / 1001.0
    if abs(fps - pulldown) < 0.02:
        fps = pulldown

    if drop and nominal % 30 != 0:
        raise ValueError(
            "drop-frame timecode is only defined for 29.97/59.94/119.88, not %r"
            % (value,)
        )
    return FrameRate(fps=fps, nominal=nominal, drop_frame=drop)


def is_timecode(text: str) -> bool:
    """True when ``text`` looks like ``HH:MM:SS:FF`` or ``HH:MM:SS;FF``."""
    return bool(_TIMECODE_RE.match(text.strip()))


def timecode_to_frames(timecode: str, frame_rate: FrameRate) -> int:
    """Convert a timecode string to an absolute frame count.

    Raises:
        ValueError: The string is malformed, the frame field exceeds the frame
            rate, or the label is one that drop-frame skips.
    """
    match = _TIMECODE_RE.match(timecode.strip())
    if not match:
        raise ValueError(
            "invalid timecode %r (expected HH:MM:SS:FF or HH:MM:SS;FF)" % (timecode,)
        )
    hours, minutes, seconds, frames = (int(g) for g in match.groups())
    nominal = frame_rate.nominal
    if frames >= nominal:
        raise ValueError(
            "timecode %r has frame %d but the rate is %d fps"
            % (timecode, frames, nominal)
        )

    total = ((hours * 60 + minutes) * 60 + seconds) * nominal + frames
    dropped = frame_rate.dropped_per_minute
    if dropped:
        total_minutes = hours * 60 + minutes
        if seconds == 0 and total_minutes % 10 != 0 and frames < dropped:
            raise ValueError(
                "timecode %r does not exist at %s (drop-frame skips it)"
                % (timecode, frame_rate)
            )
        total -= dropped * (total_minutes - total_minutes // 10)
    return total


def frames_to_timecode(frames: int, frame_rate: FrameRate) -> str:
    """Convert an absolute frame count to a timecode string.

    Raises:
        ValueError: ``frames`` is negative.
    """
    if frames < 0:
        raise ValueError("frame count must not be negative, got %d" % (frames,))

    nominal = frame_rate.nominal
    dropped = frame_rate.dropped_per_minute
    if dropped:
        frames_per_10_minutes = nominal * 600 - dropped * 9
        frames_per_minute = nominal * 60 - dropped
        tens, remainder = divmod(frames, frames_per_10_minutes)
        frames += dropped * 9 * tens
        if remainder >= dropped:
            frames += dropped * ((remainder - dropped) // frames_per_minute)

    frame_field = frames % nominal
    total_seconds = frames // nominal
    return "%02d:%02d:%02d%s%02d" % (
        total_seconds // 3600,
        (total_seconds // 60) % 60,
        total_seconds % 60,
        frame_rate.separator,
        frame_field,
    )


def seconds_to_frames(seconds: float, frame_rate: FrameRate, mode: str = "round") -> int:
    """Convert a duration in seconds to whole frames at the true rate.

    Args:
        seconds: Duration to convert; negative values clamp to zero.
        frame_rate: The rate to convert at (uses ``fps``, not ``nominal``).
        mode: ``"round"``, ``"floor"`` or ``"ceil"``.

    Raises:
        ValueError: ``mode`` is not one of the three accepted values.
    """
    if seconds <= 0:
        return 0
    exact = seconds * frame_rate.fps
    if mode == "round":
        return int(round(exact))
    if mode == "floor":
        return int(math.floor(exact))
    if mode == "ceil":
        return int(math.ceil(exact))
    raise ValueError("mode must be 'round', 'floor' or 'ceil', got %r" % (mode,))


def frames_to_seconds(frames: int, frame_rate: FrameRate) -> float:
    """Convert whole frames back to seconds at the true rate."""
    return frames / frame_rate.fps
