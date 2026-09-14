"""Kokoro-82M engine (Apache-2.0).

Kokoro is a tiny (82M) model with a set of fixed, high-quality voices.  It
does **not** clone voices, but it is fast enough for real work on a CPU,
supports native speed control and 9 languages, and downloads only ~330 MB.

Package: https://github.com/hexgrad/kokoro  (pip install kokoro)
Voices:  https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..audio import AudioClip
from ..errors import ModelError, VoiceError
from .base import ModelInfo, PresetVoice, TTSEngine, VoiceRequest, check_import

log = logging.getLogger(__name__)

REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000
DEFAULT_VOICE_ID = "af_heart"

# Kokoro lang_code letter -> language.  The first letter of a voice id is its lang code.
LANG_CODES = {
    "a": ("en", "American English"),
    "b": ("en", "British English"),
    "e": ("es", "Spanish"),
    "f": ("fr", "French"),
    "h": ("hi", "Hindi"),
    "i": ("it", "Italian"),
    "j": ("ja", "Japanese"),
    "p": ("pt", "Brazilian Portuguese"),
    "z": ("zh", "Mandarin Chinese"),
}
LANGUAGE_TO_CODE = {
    "en": "a", "en-us": "a", "en-gb": "b", "en-uk": "b", "es": "e", "fr": "f", "hi": "h",
    "it": "i", "ja": "j", "pt": "p", "pt-br": "p", "zh": "z", "cmn": "z",
}

# (voice id, quality grade from VOICES.md where published)
_VOICES: list[tuple[str, str]] = [
    ("af_heart", "A"), ("af_bella", "A-"), ("af_nicole", "B-"), ("af_aoede", "C+"), ("af_kore", "C+"),
    ("af_sarah", "C+"), ("af_alloy", "C"), ("af_nova", "C"), ("af_sky", "C-"), ("af_jessica", "D"), ("af_river", "D"),
    ("am_fenrir", "C+"), ("am_michael", "C+"), ("am_puck", "C+"), ("am_echo", "D"), ("am_eric", "D"),
    ("am_liam", "D"), ("am_onyx", "D"), ("am_santa", "D-"), ("am_adam", "F+"),
    ("bf_emma", "B-"), ("bf_isabella", "C"), ("bf_alice", "D"), ("bf_lily", "D"),
    ("bm_fable", "C"), ("bm_george", "C"), ("bm_lewis", "D+"), ("bm_daniel", "D"),
    ("ef_dora", ""), ("em_alex", ""), ("em_santa", ""),
    ("ff_siwis", "B-"),
    ("hf_alpha", "C"), ("hf_beta", "C"), ("hm_omega", "C"), ("hm_psi", "C"),
    ("if_sara", "C"), ("im_nicola", "C"),
    ("jf_alpha", "C+"), ("jf_gongitsune", "C"), ("jf_nezumi", "C-"), ("jf_tebukuro", "C"), ("jm_kumo", "C-"),
    ("pf_dora", ""), ("pm_alex", ""), ("pm_santa", ""),
    ("zf_xiaobei", "D"), ("zf_xiaoni", "D"), ("zf_xiaoxiao", "D"), ("zf_xiaoyi", "D"),
    ("zm_yunjian", "D"), ("zm_yunxi", "D"), ("zm_yunxia", "D"), ("zm_yunyang", "D"),
]


def _preset(voice_id: str, grade: str) -> PresetVoice:
    lang_code = voice_id[0]
    gender = "female" if voice_id[1] == "f" else "male"
    iso, lang_name = LANG_CODES.get(lang_code, ("?", "Unknown"))
    desc = f"{lang_name}, {gender}" + (f", grade {grade}" if grade else "")
    return PresetVoice(id=voice_id, name=voice_id, language=iso, description=desc)


PRESET_VOICES = [_preset(v, g) for v, g in _VOICES]


class KokoroEngine(TTSEngine):
    info = ModelInfo(
        id="kokoro",
        name="Kokoro-82M",
        family="Kokoro (hexgrad)",
        description="Tiny, fast, CPU-friendly model with ~50 fixed voices (US/UK English, es, fr, hi, it, ja, pt, zh). "
                    "No voice cloning. Native --speed control. Great when you just need a clean narrator quickly.",
        license="Apache-2.0",
        license_url="https://huggingface.co/hexgrad/Kokoro-82M",
        homepage="https://github.com/hexgrad/kokoro",
        languages=tuple(sorted({iso for iso, _ in LANG_CODES.values()})),
        supports_cloning=False,
        has_preset_voices=True,
        has_default_voice=True,
        native_speed=True,
        cpu_supported=True,
        cpu_note="fast on CPU (faster than real time)",
        approx_vram_gb=1.0,
        download_gb=0.35,
        sample_rate=SAMPLE_RATE,
        max_chunk_chars=350,
        min_reference_seconds=0.0,
        requirements_file="requirements-kokoro.txt",
        notes="Japanese/Chinese need extra packages: pip install misaki[ja] / misaki[zh]. "
              "English out-of-dictionary words need espeak-ng (bundled via espeakng-loader on most systems).",
    )

    def __init__(self, device: str, models_dir: Path | None = None) -> None:
        super().__init__(device, models_dir)
        self._pipelines: dict[str, object] = {}
        self._voice_id = DEFAULT_VOICE_ID
        self._lang_code = "a"
        self._speed = 1.0

    @classmethod
    def missing_dependency(cls) -> str | None:
        return check_import("torch", "kokoro")

    def load(self) -> None:
        self.require_dependencies()
        # Model weights are fetched lazily per language pipeline; warm up the default.
        self._pipeline(self._lang_code)
        self._loaded = True

    def _pipeline(self, lang_code: str):
        if lang_code in self._pipelines:
            return self._pipelines[lang_code]
        from kokoro import KPipeline

        try:
            with self.quiet_progress_bars():
                pipeline = KPipeline(lang_code=lang_code, repo_id=REPO_ID, device=self.device)
        except ImportError as exc:
            extra = {"j": "misaki[ja]", "z": "misaki[zh]"}.get(lang_code, "misaki[en]")
            raise ModelError(f"Kokoro language pack missing: {exc}", hints=[f"pip install \"{extra}\""]) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelError(f"Failed to load Kokoro: {exc}", hints=["Check your connection for the first download."]) from exc
        self._pipelines[lang_code] = pipeline
        return pipeline

    def unload(self) -> None:
        self._pipelines.clear()
        super().unload()
        from ..devices import free_cuda_memory

        free_cuda_memory()

    # ---------------------------------------------------------------- voices #
    def list_voices(self) -> list[PresetVoice]:
        return PRESET_VOICES

    def set_voice(self, voice: VoiceRequest) -> str:
        self.voice = voice
        if voice.reference_audio is not None:
            raise VoiceError(
                "Kokoro cannot clone voices - it only has fixed preset voices.",
                hints=["Use --model chatterbox-turbo (default) for voice cloning, or pick a preset with --voice af_heart"],
            )
        voice_id = voice.preset or DEFAULT_VOICE_ID
        # Voice blending "af_heart,af_bella" is supported by Kokoro: validate every part.
        parts = [p.strip() for p in voice_id.split(",") if p.strip()]
        for part in parts:
            self.resolve_preset(part)
        first = parts[0]
        lang_code = first[0] if first[0] in LANG_CODES else LANGUAGE_TO_CODE.get((voice.language or "en").lower(), "a")
        self._voice_id = ",".join(parts)
        self._lang_code = lang_code
        self._speed = float(voice.options.get("speed", 1.0) or 1.0)
        if not 0.5 <= self._speed <= 2.0:
            raise VoiceError("--speed must be between 0.5 and 2.0.")
        self._pipeline(lang_code)
        return f"preset '{self._voice_id}' ({LANG_CODES[lang_code][1]})"

    # ------------------------------------------------------------- synthesis #
    def synthesize(self, text: str, *, seed: int | None = None) -> AudioClip:
        import numpy as np
        import torch

        pipeline = self._pipeline(self._lang_code)
        parts: list[np.ndarray] = []
        with self.quiet_progress_bars(), torch.inference_mode():
            for _graphemes, _phonemes, audio in pipeline(text, voice=self._voice_id, speed=self._speed, split_pattern=None):
                if audio is None:
                    continue
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().float().numpy()
                parts.append(np.asarray(audio, dtype=np.float32))
        if not parts:
            raise ModelError("Kokoro produced no audio for this chunk.")
        return AudioClip(np.concatenate(parts), SAMPLE_RATE)
