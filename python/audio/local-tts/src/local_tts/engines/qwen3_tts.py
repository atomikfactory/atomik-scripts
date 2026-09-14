"""Qwen3-TTS engines (Alibaba Qwen team, Apache-2.0).

* ``qwen3-tts``         0.6B-Base  - 3-second zero-shot voice cloning, 10 languages.
* ``qwen3-tts-1.7b``    1.7B-Base  - higher quality clone, needs the whole 8 GB.
* ``qwen3-tts-voices``  0.6B-CustomVoice - 9 preset speakers, optional style
                        instruction (``--instruct "calm, slow"``), no cloning.

Cloning works in two modes.  With a transcript of the reference recording
(``reference.txt`` in the profile or ``--voice-text``) the model uses
in-context learning and clones prosody as well as timbre.  Without a
transcript only the speaker embedding is used (``x_vector_only_mode``), which
still works but is noticeably less faithful.

IMPORTANT: ``qwen-tts`` pins ``transformers==4.57.x`` while ``chatterbox-tts``
pins ``transformers==5.x``.  The two cannot share one virtual environment; see
README.md -> Alternative Models -> Qwen3-TTS.

Package: https://github.com/QwenLM/Qwen3-TTS  (pip install qwen-tts)
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..audio import AudioClip
from ..errors import ModelError, VoiceError
from .base import ModelInfo, PresetVoice, TTSEngine, VoiceRequest, check_import

log = logging.getLogger(__name__)

_LICENSE = "Apache-2.0"
_LICENSE_URL = "https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE"
_HOMEPAGE = "https://github.com/QwenLM/Qwen3-TTS"
_REQ = "requirements-qwen3.txt"

LANGUAGE_NAMES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "de": "German",
    "fr": "French", "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian",
}

CUSTOM_VOICES = [
    PresetVoice("Ryan", "Ryan", "en", "English, male, energetic"),
    PresetVoice("Aiden", "Aiden", "en", "English, male, relaxed"),
    PresetVoice("Vivian", "Vivian", "zh", "Chinese, female, bright"),
    PresetVoice("Serena", "Serena", "zh", "Chinese, female, warm"),
    PresetVoice("Uncle_Fu", "Uncle Fu", "zh", "Chinese, male, deep"),
    PresetVoice("Dylan", "Dylan", "zh", "Chinese (Beijing dialect), male"),
    PresetVoice("Eric", "Eric", "zh", "Chinese (Sichuan dialect), male"),
    PresetVoice("Ono_Anna", "Ono Anna", "ja", "Japanese, female"),
    PresetVoice("Sohee", "Sohee", "ko", "Korean, female"),
]


def _language_name(code: str | None) -> str:
    if not code:
        return "Auto"
    return LANGUAGE_NAMES.get(code.lower().split("-")[0], "Auto")


class _QwenBase(TTSEngine):
    repo_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    mode: str = "clone"  # clone | custom

    def __init__(self, device: str, models_dir: Path | None = None) -> None:
        super().__init__(device, models_dir)
        self.model = None
        self._prompt = None
        self._speaker: str | None = None

    @classmethod
    def missing_dependency(cls) -> str | None:
        return check_import("torch", "qwen_tts")

    def load(self) -> None:
        self.require_dependencies()
        import torch
        from qwen_tts import Qwen3TTSModel

        is_cuda = self.device.startswith("cuda")
        dtype = torch.bfloat16 if is_cuda else torch.float32
        kwargs = {"device_map": self.device if is_cuda else "cpu", "dtype": dtype}
        if is_cuda and check_import("flash_attn") is None:
            kwargs["attn_implementation"] = "flash_attention_2"
        else:
            kwargs["attn_implementation"] = "sdpa"
        try:
            self.model = Qwen3TTSModel.from_pretrained(self.repo_id, **kwargs)
        except (OSError, ValueError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower():
                raise
            raise ModelError(
                f"Failed to load {self.info.name}: {exc}",
                hints=["First run downloads the weights; retry if the download was interrupted.",
                       "qwen-tts needs its own virtual environment (transformers 4.57.x). See README."],
            ) from exc
        self._loaded = True

    def unload(self) -> None:
        self.model = None
        self._prompt = None
        super().unload()
        from ..devices import free_cuda_memory

        free_cuda_memory()

    # ---------------------------------------------------------------- voices #
    def list_voices(self) -> list[PresetVoice]:
        return CUSTOM_VOICES if self.mode == "custom" else []

    def set_voice(self, voice: VoiceRequest) -> str:
        if self.model is None:
            raise ModelError("Model must be loaded before setting a voice.")
        self.voice = voice
        if self.mode == "custom":
            if voice.reference_audio is not None:
                raise VoiceError(
                    f"{self.info.name} uses preset speakers and cannot clone voices.",
                    hints=["Use --model qwen3-tts for cloning, or pick a speaker with --voice Ryan"],
                )
            preset = self.resolve_preset(voice.preset or "Ryan")
            self._speaker = preset.id
            return f"preset speaker '{preset.id}'"

        if voice.reference_audio is None:
            raise VoiceError(
                f"{self.info.name} has no built-in voice; a reference recording is required.",
                hints=["Add --voice-sample path/to/recording.wav or --voice-profile NAME",
                       "Or use --model qwen3-tts-voices for preset speakers."],
            )
        import torch

        cache = self.embedding_cache_path(voice)
        prompt = None
        if cache and cache.exists():
            try:
                prompt = torch.load(cache, map_location=self.device, weights_only=False)
            except Exception as exc:  # cache is only an optimisation
                log.debug("Ignoring unusable prompt cache %s: %s", cache, exc)
        if prompt is None:
            xvec_only = not bool(voice.reference_text)
            if xvec_only:
                log.warning(
                    "No transcript for the reference recording - using speaker-embedding-only cloning. "
                    "Add reference.txt to the voice profile (or --voice-text) for a closer clone."
                )
            try:
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=str(voice.reference_audio),
                    ref_text=voice.reference_text or None,
                    x_vector_only_mode=xvec_only,
                )
            except (ValueError, RuntimeError, OSError) as exc:
                if "out of memory" in str(exc).lower():
                    raise
                raise VoiceError(f"Could not build a voice prompt from the reference audio: {exc}") from exc
            if cache is not None:
                try:
                    torch.save(prompt, cache)
                except Exception as exc:  # pragma: no cover
                    log.debug("Could not cache voice prompt: %s", exc)
        self._prompt = prompt
        return f"cloned from {voice.label or voice.reference_audio.name}"

    # ------------------------------------------------------------- synthesis #
    def synthesize(self, text: str, *, seed: int | None = None) -> AudioClip:
        if self.model is None:
            raise ModelError("Model is not loaded.")
        import numpy as np
        import torch

        opts = self.voice.options if self.voice else {}
        language = _language_name(self.voice.language if self.voice else "en")
        self.seed_everything(seed)
        with self.quiet_progress_bars(), torch.inference_mode():
            if self.mode == "custom":
                wavs, sr = self.model.generate_custom_voice(
                    text=text, speaker=self._speaker, language=language, instruct=opts.get("instruct") or None,
                )
            else:
                wavs, sr = self.model.generate_voice_clone(text=text, language=language, voice_clone_prompt=self._prompt)
        if not wavs:
            raise ModelError("Qwen3-TTS returned no audio for this chunk.")
        return AudioClip(np.asarray(wavs[0], dtype=np.float32), int(sr))


class Qwen3TTSEngine(_QwenBase):
    repo_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    mode = "clone"
    info = ModelInfo(
        id="qwen3-tts",
        name="Qwen3-TTS 0.6B (Base)",
        family="Qwen3-TTS (Alibaba)",
        description="Strong alternative cloning model; 10 languages; best with a transcript of the reference. "
                    "Needs its own virtual environment (transformers version conflict with Chatterbox).",
        license=_LICENSE,
        license_url=_LICENSE_URL,
        homepage=_HOMEPAGE,
        languages=tuple(LANGUAGE_NAMES),
        supports_cloning=True,
        has_preset_voices=False,
        has_default_voice=False,
        native_speed=False,
        cpu_supported=True,
        cpu_note="runs in float32 on CPU, slow",
        approx_vram_gb=4.5,
        download_gb=2.5,
        sample_rate=24000,
        max_chunk_chars=350,
        min_reference_seconds=3.0,
        requirements_file=_REQ,
    )


class Qwen3TTSLargeEngine(_QwenBase):
    repo_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    mode = "clone"
    info = ModelInfo(
        id="qwen3-tts-1.7b",
        name="Qwen3-TTS 1.7B (Base)",
        family="Qwen3-TTS (Alibaba)",
        description="Larger Qwen3-TTS clone model. Fits an 8 GB card in bf16 with little headroom; "
                    "close other GPU apps. Needs its own virtual environment.",
        license=_LICENSE,
        license_url=_LICENSE_URL,
        homepage=_HOMEPAGE,
        languages=tuple(LANGUAGE_NAMES),
        supports_cloning=True,
        has_preset_voices=False,
        has_default_voice=False,
        native_speed=False,
        cpu_supported=True,
        cpu_note="runs in float32 on CPU, very slow, needs ~8 GB RAM",
        approx_vram_gb=7.0,
        download_gb=4.6,
        sample_rate=24000,
        max_chunk_chars=350,
        min_reference_seconds=3.0,
        requirements_file=_REQ,
    )


class Qwen3TTSCustomVoiceEngine(_QwenBase):
    repo_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    mode = "custom"
    info = ModelInfo(
        id="qwen3-tts-voices",
        name="Qwen3-TTS 0.6B (CustomVoice)",
        family="Qwen3-TTS (Alibaba)",
        description="9 preset speakers (2 English) with optional style instructions via --instruct. No cloning. "
                    "Needs its own virtual environment.",
        license=_LICENSE,
        license_url=_LICENSE_URL,
        homepage=_HOMEPAGE,
        languages=tuple(LANGUAGE_NAMES),
        supports_cloning=False,
        has_preset_voices=True,
        has_default_voice=True,
        native_speed=False,
        cpu_supported=True,
        cpu_note="runs in float32 on CPU, slow",
        approx_vram_gb=4.5,
        download_gb=2.5,
        sample_rate=24000,
        max_chunk_chars=350,
        min_reference_seconds=0.0,
        requirements_file=_REQ,
    )
