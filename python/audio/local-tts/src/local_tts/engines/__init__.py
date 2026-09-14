"""Engine registry: model id -> engine class (loaded lazily)."""

from __future__ import annotations

import importlib

from ..errors import ModelError
from .base import ModelInfo, PresetVoice, TTSEngine, VoiceRequest

# model id -> (module, class).  Order = display order in --list-models.
_REGISTRY: dict[str, tuple[str, str]] = {
    "chatterbox-turbo": ("local_tts.engines.chatterbox", "ChatterboxTurboEngine"),
    "chatterbox": ("local_tts.engines.chatterbox", "ChatterboxEngine"),
    "chatterbox-multilingual": ("local_tts.engines.chatterbox", "ChatterboxMultilingualEngine"),
    "chatterbox-nano": ("local_tts.engines.chatterbox", "ChatterboxNanoEngine"),
    "kokoro": ("local_tts.engines.kokoro", "KokoroEngine"),
    "qwen3-tts": ("local_tts.engines.qwen3_tts", "Qwen3TTSEngine"),
    "qwen3-tts-1.7b": ("local_tts.engines.qwen3_tts", "Qwen3TTSLargeEngine"),
    "qwen3-tts-voices": ("local_tts.engines.qwen3_tts", "Qwen3TTSCustomVoiceEngine"),
}

_ALIASES = {
    "turbo": "chatterbox-turbo",
    "chatterbox-original": "chatterbox",
    "chatterbox-en": "chatterbox",
    "multilingual": "chatterbox-multilingual",
    "chatterbox-mtl": "chatterbox-multilingual",
    "nano": "chatterbox-nano",
    "qwen": "qwen3-tts",
    "qwen3": "qwen3-tts",
    "qwen3-tts-0.6b": "qwen3-tts",
}


def model_ids() -> list[str]:
    return list(_REGISTRY)


def resolve_model_id(name: str) -> str:
    key = (name or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _REGISTRY:
        raise ModelError(
            f"Unknown model '{name}'.",
            hints=["Available models: " + ", ".join(model_ids()), "Run: python tts.py --list-models"],
        )
    return key


def get_engine_class(name: str) -> type[TTSEngine]:
    model_id = resolve_model_id(name)
    module_name, class_name = _REGISTRY[model_id]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def list_model_infos() -> list[ModelInfo]:
    return [get_engine_class(mid).info for mid in model_ids()]


__all__ = [
    "ModelInfo",
    "PresetVoice",
    "TTSEngine",
    "VoiceRequest",
    "get_engine_class",
    "list_model_infos",
    "model_ids",
    "resolve_model_id",
]
