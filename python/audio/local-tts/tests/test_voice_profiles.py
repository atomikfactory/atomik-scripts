import json

import numpy as np
import pytest
import soundfile as sf

from local_tts import audio
from local_tts import voice_profiles as vp
from local_tts.errors import VoiceError


def test_create_and_list_profile(project, reference_wav):
    prof = vp.create_profile(project["voices"], "my_voice", reference_wav, transcript="hello there", language="en")
    assert prof.name == "my_voice"
    assert prof.reference_audio == project["voices"] / "my_voice" / "reference.wav"
    assert prof.transcript == "hello there"
    assert prof.language == "en"
    meta = json.loads((prof.path / "profile.json").read_text())
    assert meta["language"] == "en" and meta["source_sample_rate"] == 44100

    listed = vp.list_profiles(project["voices"])
    assert [p.name for p in listed] == ["my_voice"]
    assert vp.load_profile(project["voices"], "my_voice").is_valid


def test_create_profile_validation(project, reference_wav, tmp_path):
    with pytest.raises(VoiceError):
        vp.create_profile(project["voices"], "bad name!", reference_wav)
    with pytest.raises(VoiceError):
        vp.create_profile(project["voices"], "ok", tmp_path / "missing.wav")
    short = tmp_path / "short.wav"
    sf.write(str(short), np.ones(4410, dtype=np.float32) * 0.3, 44100)
    with pytest.raises(VoiceError, match="too short"):
        vp.create_profile(project["voices"], "ok", short)
    silent = tmp_path / "silent.wav"
    sf.write(str(silent), np.zeros(44100 * 4, dtype=np.float32), 44100)
    with pytest.raises(VoiceError, match="silent"):
        vp.create_profile(project["voices"], "ok", silent)


def test_create_profile_overwrite(project, reference_wav):
    vp.create_profile(project["voices"], "v", reference_wav)
    with pytest.raises(VoiceError, match="already exists"):
        vp.create_profile(project["voices"], "v", reference_wav)
    prof = vp.create_profile(project["voices"], "v", reference_wav, overwrite=True, transcript="t")
    assert prof.transcript == "t"


def test_load_profile_errors(project):
    with pytest.raises(VoiceError, match="not found"):
        vp.load_profile(project["voices"], "ghost")
    (project["voices"] / "empty").mkdir()
    with pytest.raises(VoiceError, match="no reference"):
        vp.load_profile(project["voices"], "empty")
    listed = vp.list_profiles(project["voices"])
    assert len(listed) == 1 and not listed[0].is_valid


def test_profile_discovery_prefers_reference_stem(project, reference_wav):
    folder = project["voices"] / "n"
    folder.mkdir()
    import shutil

    shutil.copy(reference_wav, folder / "zzz.wav")
    shutil.copy(reference_wav, folder / "reference.flac")
    (folder / "transcript.txt").write_text("words", encoding="utf-8")
    prof = vp.load_profile(project["voices"], "n")
    assert prof.reference_audio.name == "reference.flac"
    assert prof.transcript == "words"
    assert prof.cache_dir == folder / ".cache"


def test_hidden_folders_ignored(project):
    (project["voices"] / ".cache").mkdir()
    (project["voices"] / "_drafts").mkdir()
    assert vp.list_profiles(project["voices"]) == []
    assert vp.list_profiles(project["voices"] / "does-not-exist") == []


def test_prepare_reference_trims_mono_and_caches(project, reference_wav):
    out = vp.prepare_reference(reference_wav, project["cache"], max_seconds=30)
    clip = audio.load_audio(out)
    # 6 s tone + 1 s silence each side -> trimmed to ~6.3 s, mono, original sample rate kept.
    assert 6.0 <= clip.duration <= 6.5
    assert clip.sample_rate == 44100
    assert out.parent == project["cache"]
    again = vp.prepare_reference(reference_wav, project["cache"], max_seconds=30)
    assert again == out  # cached

    capped = vp.prepare_reference(reference_wav, project["cache"], max_seconds=4)
    assert audio.load_audio(capped).duration == pytest.approx(4.0, abs=0.01)

    raw = vp.prepare_reference(reference_wav, project["cache"], preprocess=False)
    assert raw == reference_wav  # WAV input is passed through untouched


def test_prepare_reference_respects_model_minimum(project, reference_wav):
    with pytest.raises(VoiceError, match="too short"):
        vp.prepare_reference(reference_wav, project["cache"], min_seconds=10.0)
