
import pytest

from local_tts import audio, pipeline
from local_tts.engines import VoiceRequest
from local_tts.errors import GenerationError, OutOfMemoryError
from local_tts.pipeline import GenerationOptions, Reporter, generate_voiceover, looks_suspicious
from local_tts.text_processing import chunk_text
from tests.conftest import FakeEngine, make_tone

TEXT = ("First paragraph, sentence one. Sentence two is a bit longer than the first one!\n\n"
        "Second paragraph. It also has two sentences that need generating.")


def _options(project, **kw) -> GenerationOptions:
    base = dict(
        model_id="fake",
        output_path=project["output"] / "vo.wav",
        voice=VoiceRequest(),
        cache_dir=project["cache"],
        models_dir=project["models"],
    )
    base.update(kw)
    return GenerationOptions(**base)


def test_generate_voiceover_end_to_end(project):
    result = generate_voiceover(TEXT, _options(project), Reporter(quiet=True))
    assert result.output_path.exists()
    chunks = chunk_text(TEXT, FakeEngine.info.max_chunk_chars)
    assert len(result.chunks) == len(chunks) >= 2
    assert result.cached_chunks == 0
    # Stitched length = trimmed chunks + gaps (paragraph gap after paragraph 1).
    total_clips = sum(c.clip.duration for c in result.chunks)
    assert result.duration > total_clips  # gaps were added
    assert result.chunks[0].start == 0.0
    assert all(result.chunks[i].end <= result.chunks[i + 1].start for i in range(len(result.chunks) - 1))
    saved = audio.load_audio(result.output_path)
    assert saved.sample_rate == 24000
    assert saved.peak == pytest.approx(10 ** (-1 / 20), abs=0.01)  # peak normalised to -1 dBFS


def test_cache_resume_reuses_chunks(project):
    generate_voiceover(TEXT, _options(project), Reporter(quiet=True))
    first_calls = len(FakeEngine.calls)
    FakeEngine.calls = []
    result = generate_voiceover(TEXT, _options(project), Reporter(quiet=True))
    assert FakeEngine.calls == []  # nothing regenerated
    assert result.cached_chunks == first_calls
    assert pipeline.clear_chunk_cache(project["cache"]) == first_calls
    generate_voiceover(TEXT, _options(project, cache_dir=None), Reporter(quiet=True))
    assert not (project["cache"] / "chunks").exists() or not list((project["cache"] / "chunks").glob("*.wav"))


def test_cache_key_changes_with_voice_and_seed(project):
    k1 = pipeline.chunk_cache_key("fake", "v1", "hello", None)
    assert k1 == pipeline.chunk_cache_key("fake", "v1", "hello", None)
    assert k1 != pipeline.chunk_cache_key("fake", "v2", "hello", None)
    assert k1 != pipeline.chunk_cache_key("fake", "v1", "hello", 1)
    assert k1 != pipeline.chunk_cache_key("other", "v1", "hello", None)


def test_retry_on_failure_then_error(project):
    chunks = chunk_text(TEXT, FakeEngine.info.max_chunk_chars)
    FakeEngine.fail_on = {chunks[0].text}
    with pytest.raises(GenerationError, match="failed after 3 attempt"):
        generate_voiceover(TEXT, _options(project, retries=2), Reporter(quiet=True))
    assert FakeEngine.calls.count(chunks[0].text) == 3


def test_hallucination_triggers_retry_and_keeps_best(project):
    chunks = chunk_text(TEXT, FakeEngine.info.max_chunk_chars)
    FakeEngine.hallucinate_first = {chunks[1].text}
    result = generate_voiceover(TEXT, _options(project, retries=1, seed=1), Reporter(quiet=True))
    second = result.chunks[1]
    assert second.attempts == 2
    assert second.warning is None  # the retry produced a sane length
    assert FakeEngine.calls.count(chunks[1].text) == 2


def test_hallucination_all_attempts_keeps_best_with_warning(project):
    chunks = chunk_text(TEXT, FakeEngine.info.max_chunk_chars)
    FakeEngine.hallucinate_always = {chunks[0].text}
    result = generate_voiceover(TEXT, _options(project, retries=1, seed=100), Reporter(quiet=True))
    assert result.chunks[0].warning and "kept best" in result.chunks[0].warning


def test_oom_is_translated(project, monkeypatch):
    def boom(self, text, *, seed=None):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(FakeEngine, "synthesize", boom)
    with pytest.raises(OutOfMemoryError) as exc:
        generate_voiceover(TEXT, _options(project), Reporter(quiet=True))
    assert any("--device cpu" in h for h in exc.value.hints)


def test_looks_suspicious():
    text = "x" * 140  # ~10 s expected
    assert looks_suspicious(make_tone(10.0), text) is None
    assert "hallucination" in looks_suspicious(make_tone(40.0), text)
    assert "truncation" in looks_suspicious(make_tone(1.0), text)
    assert looks_suspicious(make_tone(0.05), text) is not None
    assert looks_suspicious(make_tone(0.4), "Hi.") is None


def test_output_formats_sample_rate_and_manifest(project):
    opts = _options(project, output_path=project["output"] / "vo.flac", output_format="flac",
                    output_sample_rate=48000, write_manifest=True, bit_depth=24)
    result = generate_voiceover(TEXT, opts, Reporter(quiet=True))
    assert result.output_path.suffix == ".flac"
    assert result.sample_rate == 48000
    assert result.manifest_path and result.manifest_path.exists()
    import json

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["sample_rate"] == 48000 and len(manifest["chunks"]) == len(result.chunks)
    assert manifest["chunks"][0]["start"] == 0.0


def test_gaps_and_trim_options(project):
    a = generate_voiceover(TEXT, _options(project, cache_dir=None, sentence_gap=0.0, paragraph_gap=0.0, trim_chunks=False,
                                          normalize=False), Reporter(quiet=True))
    b = generate_voiceover(TEXT, _options(project, cache_dir=None, sentence_gap=1.0, paragraph_gap=2.0, trim_chunks=False,
                                          normalize=False), Reporter(quiet=True))
    n_chunks = len(a.chunks)
    n_para_ends = sum(1 for c in a.chunks if c.chunk.ends_paragraph) - 1  # last chunk has no gap
    expected_extra = (n_chunks - 1 - n_para_ends) * 1.0 + n_para_ends * 2.0
    assert b.duration - a.duration == pytest.approx(expected_extra, abs=0.01)


def test_reporter_prints_progress(project, capsys):
    generate_voiceover("Hello world. Second sentence.", _options(project, cache_dir=None), Reporter())
    out = capsys.readouterr().out
    assert "Loading model" in out and "Combining audio" in out and "[1/" in out
