from pathlib import Path

import numpy as np
import pytest

from local_tts import audio
from local_tts.audio import AudioClip
from local_tts.errors import InputError, UserError
from tests.conftest import make_tone


def test_to_mono_handles_layouts():
    stereo_nc = np.ones((100, 2), dtype=np.float32)
    stereo_cn = np.ones((2, 100), dtype=np.float32)
    assert audio.to_mono(stereo_nc).shape == (100,)
    assert audio.to_mono(stereo_cn).shape == (100,)
    assert audio.to_mono(np.ones(50, dtype=np.float32)).shape == (50,)


def test_audioclip_converts_to_mono_float32():
    clip = AudioClip(np.ones((10, 2), dtype=np.float64), 16000)
    assert clip.samples.dtype == np.float32 and clip.samples.shape == (10,)
    assert clip.duration == pytest.approx(10 / 16000)


def test_save_and_load_roundtrip(tmp_path: Path):
    clip = make_tone(0.5, sr=22050)
    for fmt, depth in (("wav", 16), ("wav", 24), ("wav", 32), ("flac", 16), ("ogg", 16)):
        out = audio.save_audio(clip, tmp_path / f"t_{depth}.{fmt}", fmt=fmt, bit_depth=depth)
        assert out.exists() and out.suffix == f".{fmt}"
        back = audio.load_audio(out)
        assert back.sample_rate == 22050
        assert abs(back.duration - clip.duration) < 0.05


def test_save_audio_fixes_extension_and_rejects_bad_format(tmp_path: Path):
    clip = make_tone(0.1)
    out = audio.save_audio(clip, tmp_path / "x.txt", fmt="wav")
    assert out.suffix == ".wav"
    with pytest.raises(UserError):
        audio.save_audio(clip, tmp_path / "x.xyz", fmt="xyz")
    with pytest.raises(UserError):
        audio.save_audio(clip, tmp_path / "x.wav", bit_depth=8)


def test_load_audio_errors(tmp_path: Path):
    with pytest.raises(InputError):
        audio.load_audio(tmp_path / "missing.wav")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(InputError):
        audio.load_audio(bad)


def test_trim_silence_keeps_padding():
    sr = 1000
    sig = np.concatenate([np.zeros(500), np.ones(1000) * 0.5, np.zeros(500)]).astype(np.float32)
    clip = AudioClip(sig, sr)
    trimmed = audio.trim_silence(clip, threshold_db=-40, pad_ms=100)
    # 1000 loud samples + 100 ms padding on both sides = 1200
    assert len(trimmed.samples) == 1200
    silent = audio.trim_silence(AudioClip(np.zeros(100, dtype=np.float32), sr))
    assert len(silent.samples) == 0


def test_peak_normalize_and_fade():
    clip = make_tone(0.2, amplitude=0.25)
    norm = audio.peak_normalize(clip, -1.0)
    assert norm.peak == pytest.approx(10 ** (-1 / 20), abs=1e-3)
    assert audio.peak_normalize(AudioClip(np.zeros(10, dtype=np.float32), 100)).peak == 0
    faded = audio.apply_fade(AudioClip(np.ones(1000, dtype=np.float32), 1000), ms=10)
    assert faded.samples[0] == 0 and faded.samples[-1] == 0 and faded.samples[500] == 1


def test_concatenate_with_gaps_and_mixed_rates():
    a = make_tone(1.0, sr=24000)
    b = make_tone(1.0, sr=24000)
    joined = audio.concatenate([a, b], gaps=[0.5, 0.0])
    assert joined.duration == pytest.approx(2.5, abs=1e-3)
    # A clip at another rate is resampled to the first clip's rate.
    c = make_tone(1.0, sr=48000)
    joined2 = audio.concatenate([a, c], gaps=[0.0, 0.0])
    assert joined2.sample_rate == 24000 and joined2.duration == pytest.approx(2.0, abs=0.01)
    with pytest.raises(ValueError):
        audio.concatenate([])
    with pytest.raises(ValueError):
        audio.concatenate([a], gaps=[0.0, 0.0])


def test_resample_changes_rate():
    clip = make_tone(0.5, sr=24000)
    out = audio.resample(clip, 48000)
    assert out.sample_rate == 48000 and len(out.samples) == pytest.approx(24000, abs=50)
    assert audio.resample(clip, 24000) is clip


def test_silence_and_truncate():
    assert audio.silence(0.5, 1000).duration == 0.5
    clip = make_tone(2.0, sr=1000)
    assert audio.truncate(clip, 1.0).duration == 1.0
    assert audio.truncate(clip, 5.0) is clip


def test_time_stretch_validates_speed():
    clip = make_tone(0.5)
    assert audio.time_stretch(clip, 1.0) is clip
    with pytest.raises(UserError):
        audio.time_stretch(clip, 3.0)


def test_file_fingerprint_is_stable(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"abc")
    assert audio.file_fingerprint(p) == audio.file_fingerprint(p)
    assert audio.file_fingerprint(p) != audio.file_fingerprint(p, "x")
