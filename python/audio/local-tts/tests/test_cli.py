import os
from pathlib import Path

import pytest

from local_tts import cli
from local_tts.config import Settings, load_dotenv


@pytest.fixture
def env_settings(project, monkeypatch):
    """Point the CLI at temporary folders through environment variables."""
    monkeypatch.setenv("LOCAL_TTS_VOICES_DIR", str(project["voices"]))
    monkeypatch.setenv("LOCAL_TTS_OUTPUT_DIR", str(project["output"]))
    monkeypatch.setenv("LOCAL_TTS_CACHE_DIR", str(project["cache"]))
    monkeypatch.setenv("LOCAL_TTS_MODELS_DIR", str(project["models"]))
    monkeypatch.setenv("LOCAL_TTS_MODEL", "fake")
    monkeypatch.delenv("HF_HOME", raising=False)
    return project


# ------------------------------------------------------------------ parser #
def test_parser_defaults():
    args = cli.build_parser().parse_args(["--input", "x.txt"])
    assert args.input == "x.txt" and args.model is None and args.speed == 1.0
    assert args.bit_depth == 16 and args.retries == 2 and not args.dry_run


def test_parser_rejects_bad_format():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--input", "x.txt", "--format", "aiff"])


def test_help_lists_examples(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--voice-profile" in out and "--list-models" in out and "examples:" in out


# ----------------------------------------------------------------- listing #
def test_list_models(env_settings, capsys):
    assert cli.main(["--list-models"]) == 0
    out = capsys.readouterr().out
    assert "chatterbox-turbo" in out and "kokoro" in out and "qwen3-tts" in out
    assert "MIT" in out and "Apache-2.0" in out


def test_list_voices_for_model(env_settings, capsys):
    assert cli.main(["--list-voices", "--model", "kokoro"]) == 0
    out = capsys.readouterr().out
    assert "af_heart" in out and "Voice profiles" in out


def test_unknown_model_is_user_error(env_settings, capsys):
    assert cli.main(["--list-voices", "--model", "nope"]) == 2
    err = capsys.readouterr().err
    assert "Unknown model 'nope'" in err and "Available models" in err


# ------------------------------------------------------------- input paths #
def test_no_input_gives_usage_error(env_settings, capsys):
    assert cli.main([]) == 1
    assert "Nothing to do" in capsys.readouterr().err


def test_missing_input_file(env_settings, capsys):
    assert cli.main(["--input", "does-not-exist.txt"]) == 1
    assert "Input file not found" in capsys.readouterr().err


def test_dry_run(env_settings, script_file, capsys):
    assert cli.main(["--input", str(script_file), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "chunk(s)" in out and "Dr. Who" in out


def test_generate_via_cli_default_output(env_settings, script_file, capsys):
    assert cli.main(["--input", str(script_file)]) == 0
    out = capsys.readouterr().out
    expected = env_settings["output"] / "script.wav"
    assert expected.exists()
    assert "Done!" in out and str(expected) in out


def test_generate_via_cli_custom_output_and_quiet(env_settings, script_file, capsys):
    target = env_settings["output"] / "sub" / "final.flac"
    assert cli.main(["--input", str(script_file), "--output", str(target), "--quiet", "--no-cache"]) == 0
    assert target.exists()
    assert capsys.readouterr().out.strip() == str(target)


def test_output_directory_argument(env_settings, script_file):
    outdir = env_settings["output"] / "dir"
    outdir.mkdir()
    assert cli.main(["--input", str(script_file), "--output", str(outdir), "--format", "ogg"]) == 0
    assert (outdir / "script.ogg").exists()


def test_text_argument(env_settings, capsys):
    assert cli.main(["--text", "Just a quick test sentence.", "--no-cache"]) == 0
    assert (env_settings["output"] / "voiceover.wav").exists()


def test_input_and_text_conflict(env_settings, script_file, capsys):
    assert cli.main(["--input", str(script_file), "--text", "x"]) == 1
    assert "not both" in capsys.readouterr().err


# ------------------------------------------------------------------ voices #
def test_save_and_use_voice_profile(env_settings, reference_wav, script_file, capsys):
    rc = cli.main(["--save-voice-profile", "narrator", "--voice-sample", str(reference_wav), "--voice-text", "hello"])
    assert rc == 0
    assert (env_settings["voices"] / "narrator" / "reference.wav").exists()
    assert (env_settings["voices"] / "narrator" / "reference.txt").read_text().strip() == "hello"
    capsys.readouterr()

    assert cli.main(["--input", str(script_file), "--voice-profile", "narrator", "--no-cache"]) == 0
    # pre-processed reference lives in the profile's cache folder
    assert list((env_settings["voices"] / "narrator" / ".cache").glob("ref_*.wav"))
    capsys.readouterr()

    # --voice NAME falls back to a profile of that name
    assert cli.main(["--input", str(script_file), "--voice", "narrator", "--no-cache"]) == 0


def test_voice_sample_direct(env_settings, reference_wav, script_file):
    assert cli.main(["--input", str(script_file), "--voice-sample", str(reference_wav), "--no-cache"]) == 0
    assert list((env_settings["cache"] / "voices").glob("ref_*.wav"))


def test_unknown_voice_and_profile_errors(env_settings, script_file, capsys):
    assert cli.main(["--input", str(script_file), "--voice", "ghost"]) == 1
    assert "Unknown voice 'ghost'" in capsys.readouterr().err
    assert cli.main(["--input", str(script_file), "--voice-profile", "ghost"]) == 1
    assert "not found" in capsys.readouterr().err
    assert cli.main(["--input", str(script_file), "--voice-sample", "missing.wav"]) == 1
    assert "not found" in capsys.readouterr().err


def test_preset_voice_and_conflicts(env_settings, reference_wav, script_file, capsys):
    assert cli.main(["--input", str(script_file), "--voice", "alt", "--no-cache"]) == 0
    capsys.readouterr()
    assert cli.main(["--input", str(script_file), "--voice", "alt", "--voice-sample", str(reference_wav)]) == 1
    assert "cannot be combined" in capsys.readouterr().err
    assert cli.main(["--input", str(script_file), "--voice-sample", str(reference_wav), "--voice-profile", "x"]) == 1
    assert "only one of" in capsys.readouterr().err


def test_cloning_rejected_for_preset_only_model(env_settings, reference_wav, script_file, capsys):
    rc = cli.main(["--input", str(script_file), "--model", "kokoro", "--voice-sample", str(reference_wav)])
    assert rc == 1
    assert "does not support voice cloning" in capsys.readouterr().err


def test_save_profile_requires_sample(env_settings, capsys):
    assert cli.main(["--save-voice-profile", "x"]) == 1
    assert "needs --voice-sample" in capsys.readouterr().err


def test_speed_validation(env_settings, script_file, capsys):
    assert cli.main(["--input", str(script_file), "--speed", "5"]) == 1
    assert "--speed" in capsys.readouterr().err


# ------------------------------------------------------------------ config #
def test_dotenv_loading(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('LOCAL_TTS_MODEL="kokoro"\n# comment\nexport LOCAL_TTS_DEVICE=cpu\nBAD LINE\n', encoding="utf-8")
    monkeypatch.delenv("LOCAL_TTS_MODEL", raising=False)
    monkeypatch.setenv("LOCAL_TTS_DEVICE", "cuda")
    values = load_dotenv(env)
    assert values == {"LOCAL_TTS_MODEL": "kokoro", "LOCAL_TTS_DEVICE": "cpu"}
    assert os.environ["LOCAL_TTS_MODEL"] == "kokoro"
    assert os.environ["LOCAL_TTS_DEVICE"] == "cuda"  # existing env wins
    assert load_dotenv(tmp_path / "missing.env") == {}


def test_settings_from_env_relative_paths(monkeypatch):
    monkeypatch.setenv("LOCAL_TTS_OUTPUT_DIR", "renders")
    s = Settings.from_env(dotenv=False)
    assert s.output_dir == s.project_root / "renders"
    assert s.output_dir.is_absolute()


def test_resolve_output_path_variants(tmp_path):
    settings = Settings(output_dir=tmp_path)
    ns = cli.build_parser().parse_args(["--input", "input/my script.txt"])
    assert cli.resolve_output_path(ns, settings) == tmp_path / "my script.wav"
    ns = cli.build_parser().parse_args(["--input", "a.txt", "--format", "mp3"])
    assert cli.resolve_output_path(ns, settings).name == "a.mp3"
    ns = cli.build_parser().parse_args(["--input", "a.txt", "--output", "out/final.wav", "--format", "flac"])
    assert cli.resolve_output_path(ns, settings) == Path("out/final.flac")
    ns = cli.build_parser().parse_args(["--text", "hi"])
    assert cli.resolve_output_path(ns, settings) == tmp_path / "voiceover.wav"
