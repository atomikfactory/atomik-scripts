"""Shared fixtures.  No test here needs torch or a model download."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_tts.audio import AudioClip  # noqa: E402
from local_tts.engines import _REGISTRY  # noqa: E402
from local_tts.engines.base import ModelInfo, PresetVoice, TTSEngine, VoiceRequest  # noqa: E402


def make_tone(seconds: float, sr: int = 24000, freq: float = 220.0, amplitude: float = 0.5) -> AudioClip:
    t = np.arange(int(seconds * sr)) / sr
    return AudioClip((amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)


class FakeEngine(TTSEngine):
    """Deterministic stand-in for a real model: duration proportional to text length."""

    info = ModelInfo(
        id="fake",
        name="Fake engine",
        family="tests",
        description="test double",
        license="MIT",
        license_url="",
        homepage="",
        languages=("en",),
        supports_cloning=True,
        has_preset_voices=True,
        has_default_voice=True,
        native_speed=False,
        cpu_supported=True,
        cpu_note="",
        approx_vram_gb=0,
        download_gb=0,
        sample_rate=24000,
        max_chunk_chars=120,
        min_reference_seconds=1.0,
        requirements_file="requirements.txt",
    )
    calls: list[str] = []
    fail_on: set[str] = set()
    hallucinate_first: set[str] = set()  # runaway output on the first attempt only
    hallucinate_always: set[str] = set()  # runaway output on every attempt

    def load(self) -> None:
        self._loaded = True

    def list_voices(self) -> list[PresetVoice]:
        return [PresetVoice("default", "Default", "en"), PresetVoice("alt", "Alt", "en")]

    def set_voice(self, voice: VoiceRequest) -> str:
        self.voice = voice
        if voice.preset:
            self.resolve_preset(voice.preset)
        return "fake voice"

    def synthesize(self, text: str, *, seed: int | None = None) -> AudioClip:
        type(self).calls.append(text)
        if text in type(self).fail_on:
            raise RuntimeError("synthetic failure")
        seconds = len(text) / 14.0
        first_call = type(self).calls.count(text) == 1
        if text in type(self).hallucinate_always or (text in type(self).hallucinate_first and first_call):
            seconds *= 10  # runaway generation
        return make_tone(max(seconds, 0.3), self.info.sample_rate)


@pytest.fixture(autouse=True)
def _register_fake_engine(monkeypatch):
    monkeypatch.setitem(_REGISTRY, "fake", ("tests.conftest", "FakeEngine"))
    FakeEngine.calls = []
    FakeEngine.fail_on = set()
    FakeEngine.hallucinate_first = set()
    FakeEngine.hallucinate_always = set()
    yield


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    """Pipeline tests must not depend on torch: pretend resolve_device() found a CPU."""
    from local_tts import pipeline
    from local_tts.devices import DeviceInfo

    monkeypatch.setattr(pipeline, "resolve_device", lambda *a, **k: DeviceInfo(device="cpu", name="CPU"))
    yield


@pytest.fixture
def project(tmp_path: Path) -> dict[str, Path]:
    dirs = {name: tmp_path / name for name in ("voices", "output", "cache", "models", "input")}
    for d in dirs.values():
        d.mkdir()
    return dirs


@pytest.fixture
def reference_wav(tmp_path: Path) -> Path:
    """A 6-second 'recording' with silence around it."""
    sr = 44100
    tone = make_tone(6.0, sr=sr, freq=180.0, amplitude=0.4)
    padded = np.concatenate([np.zeros(sr, dtype=np.float32), tone.samples, np.zeros(sr, dtype=np.float32)])
    stereo = np.stack([padded, padded * 0.8], axis=1)
    path = tmp_path / "ref.wav"
    import soundfile as sf

    sf.write(str(path), stereo, sr, subtype="PCM_16")
    return path


@pytest.fixture
def script_file(tmp_path: Path) -> Path:
    path = tmp_path / "script.txt"
    path.write_text(
        "First paragraph, sentence one. Sentence two is here!\n\n"
        "Second paragraph has Dr. Who and 3.5 percent. Another sentence follows?\n",
        encoding="utf-8",
    )
    return path
