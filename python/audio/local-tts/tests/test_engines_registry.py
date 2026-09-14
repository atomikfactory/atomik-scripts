"""Registry and engine metadata checks - importing engine modules must not need torch."""

import sys

import pytest

from local_tts.engines import get_engine_class, list_model_infos, model_ids, resolve_model_id
from local_tts.engines.base import VoiceRequest
from local_tts.errors import ModelError


def test_all_models_expose_consistent_info():
    infos = list_model_infos()
    assert len(infos) >= 8 and len({i.id for i in infos}) == len(infos)
    assert [i.id for i in infos] == model_ids()
    for info in infos:
        assert info.id and info.name and info.license and info.requirements_file
        assert info.sample_rate > 0 and info.max_chunk_chars >= 100
        assert info.languages
        cls = get_engine_class(info.id)
        assert cls.info is info
        # list_voices() works without loading anything.
        voices = cls("cpu").list_voices()
        if info.has_preset_voices:
            assert voices


def test_aliases_and_unknown():
    assert resolve_model_id("turbo") == "chatterbox-turbo"
    assert resolve_model_id("QWEN3") == "qwen3-tts"
    assert resolve_model_id("Kokoro") == "kokoro"
    with pytest.raises(ModelError):
        resolve_model_id("does-not-exist")


def test_default_model_has_default_voice_and_cloning():
    info = get_engine_class("chatterbox-turbo").info
    assert info.supports_cloning and info.has_default_voice and "en" in info.languages


def test_kokoro_voice_table_is_well_formed():
    from local_tts.engines.kokoro import LANG_CODES, PRESET_VOICES

    ids = [v.id for v in PRESET_VOICES]
    assert len(ids) == len(set(ids))
    assert "af_heart" in ids
    for v in PRESET_VOICES:
        assert v.id[0] in LANG_CODES and v.id[1] in "fm"


def test_voice_request_fingerprint(tmp_path):
    a = VoiceRequest(language="en", options={"exaggeration": 0.5})
    b = VoiceRequest(language="en", options={"exaggeration": 0.5})
    c = VoiceRequest(language="de", options={"exaggeration": 0.5})
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()
    ref = tmp_path / "r.wav"
    ref.write_bytes(b"123")
    d = VoiceRequest(reference_audio=ref)
    assert d.fingerprint() != a.fingerprint()


def test_engine_modules_do_not_import_torch_at_import_time():
    # Importing registry + all engine modules happened above; torch must not have been pulled in by them.
    for mod in ("local_tts.engines.chatterbox", "local_tts.engines.kokoro", "local_tts.engines.qwen3_tts"):
        assert mod in sys.modules
    # If torch is importable in this environment it may already be loaded by other tests,
    # so only check the engine modules' own globals.
    import local_tts.engines.chatterbox as cb

    assert "torch" not in vars(cb)


def test_missing_dependency_message_when_package_absent(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec

    def fake_find_spec(name, *a, **k):
        if name == "chatterbox":
            return None
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    cls = get_engine_class("chatterbox-turbo")
    assert "chatterbox" in (cls.missing_dependency() or "")
