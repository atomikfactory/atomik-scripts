"""Command-line interface (``python tts.py ...``)."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
import warnings
from pathlib import Path

from . import __version__
from .audio import SUPPORTED_OUTPUT_FORMATS
from .config import DEFAULT_OUTPUT_NAME, Settings, configure_model_cache
from .errors import InputError, UserError, VoiceError
from .text_processing import chunk_text, describe_chunks, load_script

EXAMPLES = """\
examples:
  python tts.py --input input/script.txt
  python tts.py --input input/script.txt --output output/episode1.wav
  python tts.py --input input/script.txt --voice-sample recordings/me.wav
  python tts.py --save-voice-profile my_voice --voice-sample recordings/me.wav --voice-text "exact transcript"
  python tts.py --input input/script.txt --voice-profile my_voice
  python tts.py --input input/script.txt --model kokoro --voice af_heart --speed 1.1
  python tts.py --input input/script.txt --model chatterbox-multilingual --language de --voice-profile narrator
  python tts.py --list-models
  python tts.py --list-voices [--model kokoro]
  python tts.py --input input/script.txt --dry-run      # show how the text will be chunked
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tts.py",
        description="Local text-to-speech and voice cloning for long-form voice-overs.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"local-tts {__version__}")

    io = p.add_argument_group("input / output")
    io.add_argument("-i", "--input", metavar="FILE", help="script text file (.txt)")
    io.add_argument("--text", metavar="TEXT", help="synthesize this text instead of a file (quick tests)")
    io.add_argument("-o", "--output", metavar="PATH", help="output file or directory (default: output/<script name>.wav)")
    io.add_argument("--format", choices=SUPPORTED_OUTPUT_FORMATS, default=None, help="output format (default: from --output extension, else wav)")
    io.add_argument("--bit-depth", type=int, choices=(16, 24, 32), default=16, help="WAV/FLAC bit depth (default: 16)")
    io.add_argument("--sample-rate", type=int, metavar="HZ", default=None, help="resample the final file, e.g. 48000 for video editing (default: model native, 24000)")
    io.add_argument("--manifest", action="store_true", help="also write <output>.json with per-chunk timestamps")

    model = p.add_argument_group("model")
    model.add_argument("-m", "--model", metavar="ID", default=None, help="model id (see --list-models; default: chatterbox-turbo)")
    model.add_argument("--device", default=None, help="auto | cuda | cuda:N | cpu (default: auto)")
    model.add_argument("-l", "--language", metavar="CODE", default=None, help="language code, e.g. en, de, fr (default: en or the profile's language)")
    model.add_argument("--models-dir", metavar="DIR", default=None, help="where model weights are downloaded/cached (default: models/)")

    voice = p.add_argument_group("voice")
    voice.add_argument("--voice", metavar="NAME", help="preset voice of the model (e.g. af_heart for kokoro) or a voice profile name")
    voice.add_argument("--voice-sample", metavar="FILE", help="reference recording to clone (WAV/FLAC/MP3/OGG/M4A...)")
    voice.add_argument("--voice-text", metavar="TEXT", help="exact transcript of the reference recording (improves Qwen3-TTS cloning)")
    voice.add_argument("--voice-profile", metavar="NAME", help="reuse a saved profile from voices/<NAME>/")
    voice.add_argument("--save-voice-profile", metavar="NAME", help="save --voice-sample (+ --voice-text) as voices/<NAME>/ for reuse")
    voice.add_argument("--overwrite", action="store_true", help="allow --save-voice-profile to replace an existing profile")
    voice.add_argument("--no-preprocess-reference", action="store_true", help="feed the reference recording to the model untouched (no mono/trim/cap)")
    voice.add_argument("--max-reference-seconds", type=float, default=30.0, metavar="SEC", help="use at most this much of the reference (default: 30)")

    style = p.add_argument_group("delivery")
    style.add_argument("--speed", type=float, default=1.0, help="speaking rate 0.5-2.0 (native for kokoro, ffmpeg post-process otherwise)")
    style.add_argument("--exaggeration", type=float, default=None, help="chatterbox / chatterbox-multilingual emotion intensity (default 0.5)")
    style.add_argument("--cfg-weight", type=float, default=None, help="chatterbox / chatterbox-multilingual pacing (default 0.5; lower = faster, more expressive)")
    style.add_argument("--temperature", type=float, default=None, help="sampling temperature for Chatterbox models (default 0.8)")
    style.add_argument("--instruct", metavar="TEXT", help="style instruction for qwen3-tts-voices, e.g. \"calm and slow\"")
    style.add_argument("--seed", type=int, default=None, help="random seed for reproducible output")

    chunking = p.add_argument_group("long-form processing")
    chunking.add_argument("--max-chunk-chars", type=int, default=None, metavar="N", help="max characters per generation (default: model specific, ~300)")
    chunking.add_argument("--newline-is-break", action="store_true", help="treat every line break as a paragraph break (default: only blank lines)")
    chunking.add_argument("--sentence-gap", type=float, default=0.25, metavar="SEC", help="silence between chunks inside a paragraph (default 0.25)")
    chunking.add_argument("--paragraph-gap", type=float, default=0.6, metavar="SEC", help="silence between paragraphs (default 0.6)")
    chunking.add_argument("--no-trim", action="store_true", help="keep the model's own leading/trailing silence on every chunk")
    chunking.add_argument("--no-normalize", action="store_true", help="skip peak normalisation of the final file")
    chunking.add_argument("--retries", type=int, default=2, help="regeneration attempts for chunks that fail or look wrong (default 2)")
    chunking.add_argument("--no-cache", action="store_true", help="do not reuse/save per-chunk audio in .cache/ (disables resume)")
    chunking.add_argument("--clear-cache", action="store_true", help="delete cached chunk audio before starting")
    chunking.add_argument("--dry-run", action="store_true", help="only show how the script will be split into chunks")

    listing = p.add_argument_group("information")
    listing.add_argument("--list-models", action="store_true", help="show supported models and whether they are installed")
    listing.add_argument("--list-voices", action="store_true", help="show voice profiles and the preset voices of --model")
    listing.add_argument("-v", "--verbose", action="store_true", help="more progress details")
    listing.add_argument("--debug", action="store_true", help="full tracebacks and library logs")
    listing.add_argument("-q", "--quiet", action="store_true", help="only print errors and the final output path")
    return p


# --------------------------------------------------------------------------- #
class _OwnMessagesOnly(logging.Filter):
    """Let our own log records through; third-party libraries only at ERROR or above."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith("local_tts") or record.levelno >= logging.ERROR


def _setup_logging(args: argparse.Namespace) -> None:
    level = logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", stream=sys.stderr)
    # numba (used by librosa / the watermarker) dumps compiler IR at DEBUG level - never useful here.
    logging.getLogger("numba").setLevel(logging.WARNING)
    if not args.debug:
        # Third-party libraries are very chatty (HTTP requests, deprecation notes...); users only need our messages.
        for handler in logging.getLogger().handlers:
            handler.addFilter(_OwnMessagesOnly())
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(argv: list[str] | None = None) -> int:
    # Windows consoles often use a legacy code page; never crash on a non-ASCII script preview.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, AttributeError):  # pragma: no cover
                pass
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args)

    settings = Settings.from_env()
    if args.models_dir:
        settings.models_dir = Path(args.models_dir).expanduser()
    if args.model:
        settings.model = args.model
    if args.device:
        settings.device = args.device
    configure_model_cache(settings.models_dir)

    try:
        return _run(args, settings, parser)
    except UserError as exc:
        print(exc.format(), file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return exc.exit_code
    except KeyboardInterrupt:
        print("\nInterrupted. Finished chunks are cached - run the same command again to resume.", file=sys.stderr)
        return 130
    except Exception as exc:  # unexpected -> bug
        if args.debug:
            raise
        print(f"Unexpected error: {type(exc).__name__}: {exc}\nRun again with --debug for the full traceback.", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace, settings: Settings, parser: argparse.ArgumentParser) -> int:
    from .engines import get_engine_class, resolve_model_id

    if args.list_models:
        print_models(settings.model)
        return 0

    model_id = resolve_model_id(settings.model)
    engine_cls = get_engine_class(model_id)
    info = engine_cls.info

    if args.list_voices:
        print_voices(engine_cls, settings)
        return 0

    if args.save_voice_profile:
        if not args.voice_sample:
            raise VoiceError("--save-voice-profile needs --voice-sample FILE.")
        from .voice_profiles import create_profile

        profile = create_profile(
            settings.voices_dir, args.save_voice_profile, Path(args.voice_sample),
            transcript=args.voice_text, language=args.language, overwrite=args.overwrite,
        )
        print(f"Saved voice profile '{profile.name}' -> {profile.path}")
        if not args.input and not args.text:
            print(f"Use it with: python tts.py --input input/script.txt --voice-profile {profile.name}")
            return 0
        args.voice_profile = profile.name
        args.voice_sample = None

    if not args.input and not args.text:
        parser.print_usage(sys.stderr)
        raise InputError("Nothing to do. Give a script with --input FILE (or --text \"...\"), or use --list-models / --list-voices.")
    if args.input and args.text:
        raise InputError("Use either --input or --text, not both.")

    text = load_script(args.input) if args.input else args.text
    max_chars = args.max_chunk_chars or info.max_chunk_chars

    if args.dry_run:
        chunks = chunk_text(text, max_chars, newline_is_break=args.newline_is_break)
        print(f"Model: {info.id} (max {max_chars} chars per chunk)")
        print(describe_chunks(chunks))
        return 0

    from .pipeline import GenerationOptions, Reporter, clear_chunk_cache, generate_voiceover

    voice = build_voice_request(args, settings, engine_cls)
    output_path = resolve_output_path(args, settings)
    if args.clear_cache:
        n = clear_chunk_cache(settings.cache_dir)
        print(f"Cleared {n} cached chunk(s).")

    options = GenerationOptions(
        model_id=model_id,
        output_path=output_path,
        voice=voice,
        device=settings.device,
        models_dir=settings.models_dir,
        cache_dir=None if args.no_cache else settings.cache_dir,
        max_chunk_chars=max_chars,
        newline_is_break=args.newline_is_break,
        sentence_gap=max(0.0, args.sentence_gap),
        paragraph_gap=max(0.0, args.paragraph_gap),
        trim_chunks=not args.no_trim,
        normalize=not args.no_normalize,
        speed=args.speed,
        seed=args.seed,
        retries=max(0, args.retries),
        output_format=args.format or (output_path.suffix.lstrip(".").lower() or "wav"),
        bit_depth=args.bit_depth,
        output_sample_rate=args.sample_rate,
        write_manifest=args.manifest,
    )
    if not 0.5 <= options.speed <= 2.0:
        raise UserError("--speed must be between 0.5 and 2.0.")

    result = generate_voiceover(text, options, Reporter(quiet=args.quiet, verbose=args.verbose))

    if args.quiet:
        print(result.output_path)
    else:
        print("Done!")
        print(f"Output: {result.output_path}")
        mins, secs = divmod(result.duration, 60)
        speedup = result.duration / result.elapsed if result.elapsed > 0 else 0
        print(f"Length: {int(mins)}m {secs:04.1f}s @ {result.sample_rate} Hz | chunks: {len(result.chunks)} "
              f"({result.cached_chunks} from cache) | took {result.elapsed:.0f}s ({speedup:.1f}x real time) on {result.device.name}")
        if result.manifest_path:
            print(f"Manifest: {result.manifest_path}")
        warnings_ = [c for c in result.chunks if c.warning]
        if warnings_:
            print(f"Note: {len(warnings_)} chunk(s) had quality warnings (listed above). Listen to them and re-run with --seed N or --clear-cache if needed.")
    return 0


# --------------------------------------------------------------------------- #
def build_voice_request(args: argparse.Namespace, settings: Settings, engine_cls):
    """Turn --voice / --voice-sample / --voice-profile into a VoiceRequest."""
    from .engines import VoiceRequest
    from .voice_profiles import list_profiles, load_profile, prepare_reference

    info = engine_cls.info
    chosen = [n for n, v in (("--voice-sample", args.voice_sample), ("--voice-profile", args.voice_profile)) if v]
    if len(chosen) > 1:
        raise VoiceError(f"Use only one of {', '.join(chosen)}.")

    reference: Path | None = None
    transcript: str | None = args.voice_text
    preset: str | None = None
    language = args.language
    cache_dir: Path | None = None
    label = ""
    options: dict = {}

    if args.voice_profile:
        profile = load_profile(settings.voices_dir, args.voice_profile)
        reference = profile.reference_audio
        transcript = transcript or profile.transcript
        language = language or profile.language
        cache_dir = profile.cache_dir
        label = f"profile '{profile.name}'"
        options.update(profile.options)
    elif args.voice_sample:
        reference = Path(args.voice_sample).expanduser()
        if not reference.is_file():
            raise VoiceError(f"Voice sample not found: {reference}")
        cache_dir = settings.cache_dir / "voices"
        label = reference.name

    if args.voice:
        preset_ids = {v.id.lower() for v in engine_cls("cpu").list_voices()} if info.has_preset_voices or not info.supports_cloning else {"default"}
        if args.voice.lower() in preset_ids or (args.voice.lower() == "default" and info.has_default_voice):
            if reference is not None:
                raise VoiceError("--voice (preset) and --voice-sample/--voice-profile cannot be combined.")
            preset = args.voice
        else:
            profile_names = {p.name for p in list_profiles(settings.voices_dir) if p.is_valid}
            if args.voice in profile_names and reference is None:
                # Friendly shortcut: --voice my_voice == --voice-profile my_voice
                profile = load_profile(settings.voices_dir, args.voice)
                reference, transcript = profile.reference_audio, transcript or profile.transcript
                language, cache_dir, label = language or profile.language, profile.cache_dir, f"profile '{profile.name}'"
                options.update(profile.options)
            else:
                hints = []
                if info.has_preset_voices:
                    hints.append(f"Preset voices: python tts.py --list-voices --model {info.id}")
                if profile_names:
                    hints.append("Voice profiles: " + ", ".join(sorted(profile_names)))
                elif info.supports_cloning:
                    hints.append("Create a profile: python tts.py --save-voice-profile NAME --voice-sample recording.wav")
                raise VoiceError(f"Unknown voice '{args.voice}' for model '{info.id}'.", hints=hints)

    if reference is not None:
        if not info.supports_cloning:
            raise VoiceError(
                f"{info.name} does not support voice cloning.",
                hints=["Use --model chatterbox-turbo (default) or another cloning model from --list-models."],
            )
        reference = prepare_reference(
            reference, cache_dir or settings.cache_dir / "voices",
            preprocess=not args.no_preprocess_reference,
            max_seconds=args.max_reference_seconds,
            min_seconds=info.min_reference_seconds,
        )

    # Only pass options the user actually set so model defaults apply otherwise.
    for key in ("exaggeration", "cfg_weight", "temperature", "instruct"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    if info.native_speed and abs(args.speed - 1.0) > 1e-3:
        options["speed"] = args.speed

    return VoiceRequest(
        reference_audio=reference,
        reference_text=transcript,
        preset=preset,
        language=(language or settings.language).lower(),
        cache_dir=cache_dir,
        options=options,
        label=label,
    )


def resolve_output_path(args: argparse.Namespace, settings: Settings) -> Path:
    fmt = args.format or "wav"
    default_name = (Path(args.input).stem if args.input else Path(DEFAULT_OUTPUT_NAME).stem) + f".{fmt}"
    if not args.output:
        return settings.output_dir / default_name
    out = Path(args.output).expanduser()
    if out.is_dir() or str(args.output).endswith(("/", "\\")) or not out.suffix:
        return out / default_name
    if args.format and out.suffix.lstrip(".").lower() != args.format:
        out = out.with_suffix(f".{args.format}")
    return out


# --------------------------------------------------------------------------- #
def print_models(default_model: str) -> None:
    from .engines import get_engine_class, list_model_infos

    print("Supported models (* = default):\n")
    for info in list_model_infos():
        problem = get_engine_class(info.id).missing_dependency()
        status = "installed" if problem is None else f"not installed -> pip install -r {info.requirements_file}"
        marker = "*" if info.id == default_model else " "
        clone = "yes" if info.supports_cloning else "no"
        presets = "yes" if info.has_preset_voices else "no"
        langs = ", ".join(info.languages) if len(info.languages) <= 6 else f"{len(info.languages)} languages"
        print(f"{marker} {info.id}")
        print(f"    {info.name} - {info.license} - {status}")
        print(f"    {info.description}")
        print(f"    cloning: {clone} | preset voices: {presets} | languages: {langs} | ~{info.approx_vram_gb:g} GB VRAM | "
              f"download ~{info.download_gb:g} GB | CPU: {info.cpu_note}")
        if info.notes:
            print(f"    note: {info.notes}")
        print()


def print_voices(engine_cls, settings: Settings) -> None:
    from .voice_profiles import list_profiles

    info = engine_cls.info
    profiles = list_profiles(settings.voices_dir)
    print(f"Voice profiles in {settings.voices_dir}:")
    if not profiles:
        print("  (none) - create one: python tts.py --save-voice-profile NAME --voice-sample recording.wav")
    for p in profiles:
        if p.is_valid:
            extra = []
            if p.language:
                extra.append(f"language {p.language}")
            extra.append("transcript" if p.transcript else "no transcript")
            if p.description:
                extra.append(p.description)
            print(f"  {p.name:<24} {p.reference_audio.name} ({', '.join(extra)})")
        else:
            print(f"  {p.name:<24} INVALID - no reference audio found in {p.path}")
    if profiles and not info.supports_cloning:
        print(f"  (note: {info.name} cannot use profiles - it has no voice cloning)")

    print(f"\nPreset voices for model '{info.id}':")
    voices = engine_cls("cpu").list_voices()
    if not voices:
        print("  (none - this model needs a reference recording: --voice-sample or --voice-profile)")
    for v in voices:
        print(f"  {v.id:<16} {v.description}")
    if info.id != "kokoro":
        print("\nTip: python tts.py --list-voices --model kokoro   shows ~50 preset voices in 9 languages.")
