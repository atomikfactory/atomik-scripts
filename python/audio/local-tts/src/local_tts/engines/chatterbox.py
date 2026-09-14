"""Chatterbox engines (Resemble AI, MIT license).

Four variants share one implementation:

* ``chatterbox-turbo``        350M, English, fastest, default.  1-step decoder,
                              supports ``[laugh]``/``[chuckle]``/``[cough]`` tags.
* ``chatterbox``              500M, English, original model with ``exaggeration`` and
                              ``cfg_weight`` controls.
* ``chatterbox-multilingual`` 500M, 23 languages (v3 weights).
* ``chatterbox-nano``         110M, English, designed for CPU.

All of them do zero-shot voice cloning from a short reference clip and ship a
built-in default voice (``conds.pt``), so they work without any reference.
Every output is watermarked by Resemble's Perth watermarker (part of the model
package, cannot be disabled here).

Package: https://github.com/resemble-ai/chatterbox  (pip install chatterbox-tts)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from ..audio import AudioClip
from ..errors import ModelError, VoiceError
from .base import ModelInfo, PresetVoice, TTSEngine, VoiceRequest, check_import

log = logging.getLogger(__name__)

_LICENSE = "MIT"
_LICENSE_URL = "https://github.com/resemble-ai/chatterbox/blob/master/LICENSE"
_HOMEPAGE = "https://github.com/resemble-ai/chatterbox"
_REQ = "requirements-chatterbox.txt"

MULTILINGUAL_LANGUAGES = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
)

_GIT_INSTALL_HINT = ("pip install --upgrade \"chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git\" "
                     "(the GitHub version adds Nano and the Multilingual v3 weights)")


def _accepts(func, parameter: str) -> bool:
    """True if *func* has a keyword parameter called *parameter* (API differs between releases)."""
    import inspect

    try:
        return parameter in inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


DEFAULT_VOICE = PresetVoice(
    id="default", name="Built-in voice", language="en",
    description="Neutral voice shipped with the model (conds.pt). Add --voice-sample to clone a voice.",
)


class _ChatterboxBase(TTSEngine):
    variant: str = "turbo"  # turbo | nano | original | multilingual

    def __init__(self, device: str, models_dir: Path | None = None) -> None:
        super().__init__(device, models_dir)
        self.model = None
        self._default_conds = None

    @classmethod
    def missing_dependency(cls) -> str | None:
        return check_import("torch", "chatterbox")

    # ------------------------------------------------------------------ load #
    def load(self) -> None:
        self.require_dependencies()
        import torch  # noqa: F401  (ensures a clear error before chatterbox imports)

        try:
            if self.variant in ("turbo", "nano"):
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                if self.variant == "nano":
                    if not _accepts(ChatterboxTurboTTS.from_pretrained, "nano"):
                        raise ModelError(
                            "chatterbox-nano needs a newer chatterbox-tts than the PyPI release installed here.",
                            hints=[_GIT_INSTALL_HINT, "Or use --model chatterbox-turbo / --model kokoro instead."],
                        )
                    self.model = ChatterboxTurboTTS.from_pretrained(device=self.device, nano=True)
                else:
                    self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            elif self.variant == "original":
                from chatterbox.tts import ChatterboxTTS

                self.model = ChatterboxTTS.from_pretrained(device=self.device)
            elif self.variant == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS

                if _accepts(ChatterboxMultilingualTTS.from_pretrained, "t3_model"):
                    self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device, t3_model="v3")
                else:
                    log.info("Installed chatterbox-tts ships the multilingual v2 weights; for v3 install from GitHub "
                             "(see README -> Model Selection).")
                    self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            else:  # pragma: no cover
                raise ModelError(f"Unknown Chatterbox variant '{self.variant}'")
        except (OSError, ValueError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower():
                raise
            raise ModelError(
                f"Failed to load {self.info.name}: {exc}",
                hints=[
                    "If this is the first run, the download may have been interrupted - just run again.",
                    "Check your internet connection or set HF_HUB_OFFLINE=1 once the weights are cached.",
                    "Weights are cached under the models directory (default: models/).",
                ],
            ) from exc
        self._default_conds = getattr(self.model, "conds", None)
        self._loaded = True

    def unload(self) -> None:
        self.model = None
        self._default_conds = None
        super().unload()
        from ..devices import free_cuda_memory

        free_cuda_memory()

    # ---------------------------------------------------------------- voices #
    def list_voices(self) -> list[PresetVoice]:
        return [DEFAULT_VOICE]

    def _conditionals_class(self):
        module = sys.modules.get(type(self.model).__module__)
        cls = getattr(module, "Conditionals", None)
        if cls is None:
            from chatterbox.tts import Conditionals as cls  # type: ignore[no-redef]
        return cls

    def _load_cached_conds(self, path: Path):
        import torch

        try:
            conds_cls = self._conditionals_class()
            conds = conds_cls.load(path, map_location=torch.device(self.device))
            return conds.to(self.device)
        except Exception as exc:  # cache is only an optimisation
            log.debug("Ignoring unusable embedding cache %s: %s", path, exc)
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def set_voice(self, voice: VoiceRequest) -> str:
        if self.model is None:
            raise ModelError("Model must be loaded before setting a voice.")
        self.voice = voice
        if self.variant == "multilingual":
            lang = (voice.language or "en").lower()
            if lang not in MULTILINGUAL_LANGUAGES:
                raise VoiceError(
                    f"Language '{voice.language}' is not supported by {self.info.name}.",
                    hints=["Supported: " + ", ".join(MULTILINGUAL_LANGUAGES)],
                )

        if voice.reference_audio is not None:
            exaggeration = float(voice.options.get("exaggeration", 0.5))
            cache = self.embedding_cache_path(voice)
            conds = self._load_cached_conds(cache) if cache and cache.exists() else None
            if conds is not None:
                self.model.conds = conds
                log.debug("Loaded cached speaker conditionals from %s", cache)
            else:
                try:
                    if self.variant in ("turbo", "nano"):
                        self.model.prepare_conditionals(str(voice.reference_audio), exaggeration=exaggeration, norm_loudness=True)
                    else:
                        self.model.prepare_conditionals(str(voice.reference_audio), exaggeration=exaggeration)
                except AssertionError as exc:
                    raise VoiceError(f"Reference audio rejected by the model: {exc}") from exc
                if cache is not None:
                    try:
                        self.model.conds.save(cache)
                    except Exception as exc:  # pragma: no cover
                        log.debug("Could not cache speaker conditionals: %s", exc)
            return f"cloned from {voice.label or voice.reference_audio.name}"

        if voice.preset and voice.preset.lower() != DEFAULT_VOICE.id:
            self.resolve_preset(voice.preset)  # raises with the list of valid voices
        if self._default_conds is None:
            raise VoiceError(
                f"{self.info.name} has no built-in voice; a reference recording is required.",
                hints=["Add --voice-sample path/to/recording.wav or --voice-profile NAME"],
            )
        self.model.conds = self._default_conds
        return "built-in default voice"

    # ------------------------------------------------------------- synthesis #
    def synthesize(self, text: str, *, seed: int | None = None) -> AudioClip:
        if self.model is None:
            raise ModelError("Model is not loaded.")
        import torch

        opts = self.voice.options if self.voice else {}
        kwargs: dict = {"temperature": float(opts.get("temperature", 0.8))}
        if self.variant in ("original", "multilingual"):
            kwargs["exaggeration"] = float(opts.get("exaggeration", 0.5))
            kwargs["cfg_weight"] = float(opts.get("cfg_weight", 0.5))
        if self.variant == "multilingual":
            kwargs["language_id"] = (self.voice.language if self.voice else "en").lower()

        self.seed_everything(seed)
        with self.quiet_progress_bars(), torch.inference_mode():
            wav = self.model.generate(text, **kwargs)
        samples = wav.squeeze().detach().cpu().float().numpy()
        return AudioClip(samples, int(self.model.sr))


class ChatterboxTurboEngine(_ChatterboxBase):
    variant = "turbo"
    info = ModelInfo(
        id="chatterbox-turbo",
        name="Chatterbox Turbo",
        family="Chatterbox (Resemble AI)",
        description="Best default for English voice-overs: fast, expressive, excellent zero-shot cloning, "
                    "supports [laugh] [chuckle] [cough] tags. 350M parameters, 1-step decoder.",
        license=_LICENSE,
        license_url=_LICENSE_URL,
        homepage=_HOMEPAGE,
        languages=("en",),
        supports_cloning=True,
        has_preset_voices=False,
        has_default_voice=True,
        native_speed=False,
        cpu_supported=True,
        cpu_note="works on CPU but several times slower than real time",
        approx_vram_gb=3.5,
        download_gb=4.0,
        sample_rate=24000,
        max_chunk_chars=300,
        min_reference_seconds=5.5,  # the model asserts > 5 s
        requirements_file=_REQ,
        notes="Reference clip: 10-20 s of clean speech. exaggeration/cfg_weight are not supported by Turbo.",
    )


class ChatterboxNanoEngine(_ChatterboxBase):
    variant = "nano"
    info = ModelInfo(
        id="chatterbox-nano",
        name="Chatterbox Nano",
        family="Chatterbox (Resemble AI)",
        description="110M-parameter version of Turbo made for CPU / low-VRAM machines. Lower quality than Turbo. "
                    "Requires chatterbox-tts installed from GitHub (not in the 0.1.7 PyPI release).",
        license=_LICENSE,
        license_url=_LICENSE_URL,
        homepage=_HOMEPAGE,
        languages=("en",),
        supports_cloning=True,
        has_preset_voices=False,
        has_default_voice=True,
        native_speed=False,
        cpu_supported=True,
        cpu_note="designed for CPU (about 3x faster than real time on 8 cores)",
        approx_vram_gb=2.5,
        download_gb=3.0,
        sample_rate=24000,
        max_chunk_chars=300,
        min_reference_seconds=5.5,
        requirements_file=_REQ,
    )


class ChatterboxEngine(_ChatterboxBase):
    variant = "original"
    info = ModelInfo(
        id="chatterbox",
        name="Chatterbox (original)",
        family="Chatterbox (Resemble AI)",
        description="Original 500M English model. Slower than Turbo but offers --exaggeration and --cfg-weight "
                    "for emotion/pacing control.",
        license=_LICENSE,
        license_url=_LICENSE_URL,
        homepage=_HOMEPAGE,
        languages=("en",),
        supports_cloning=True,
        has_preset_voices=False,
        has_default_voice=True,
        native_speed=False,
        cpu_supported=True,
        cpu_note="very slow on CPU",
        approx_vram_gb=4.5,
        download_gb=3.3,
        sample_rate=24000,
        max_chunk_chars=300,
        min_reference_seconds=3.0,
        requirements_file=_REQ,
        notes="Defaults exaggeration=0.5 cfg_weight=0.5. Expressive: exaggeration 0.7, cfg_weight 0.3.",
    )


class ChatterboxMultilingualEngine(_ChatterboxBase):
    variant = "multilingual"
    info = ModelInfo(
        id="chatterbox-multilingual",
        name="Chatterbox Multilingual",
        family="Chatterbox (Resemble AI)",
        description="500M model speaking 23 languages with zero-shot cloning. Use --language XX. "
                    "Pick this for anything that is not English. (PyPI 0.1.7 = v2 weights; GitHub version = v3.)",
        license=_LICENSE,
        license_url=_LICENSE_URL,
        homepage=_HOMEPAGE,
        languages=MULTILINGUAL_LANGUAGES,
        supports_cloning=True,
        has_preset_voices=False,
        has_default_voice=True,
        native_speed=False,
        cpu_supported=True,
        cpu_note="very slow on CPU",
        approx_vram_gb=4.5,
        download_gb=3.3,
        sample_rate=24000,
        max_chunk_chars=300,
        min_reference_seconds=3.0,
        requirements_file=_REQ,
        notes="The reference clip should be in the same language as --language, or the accent transfers.",
    )
