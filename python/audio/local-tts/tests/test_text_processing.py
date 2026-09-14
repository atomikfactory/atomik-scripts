from pathlib import Path

import pytest

from local_tts.errors import InputError
from local_tts.text_processing import (
    chunk_text,
    describe_chunks,
    load_script,
    normalize_text,
    split_long_sentence,
    split_paragraphs,
    split_sentences,
)


# ----------------------------------------------------------------- loading #
def test_load_script_utf8_bom(tmp_path: Path):
    p = tmp_path / "s.txt"
    p.write_bytes("﻿Hello world.".encode())
    assert load_script(p) == "Hello world."


def test_load_script_cp1252_fallback(tmp_path: Path):
    p = tmp_path / "s.txt"
    p.write_bytes(b"caf\xe9 \x93quoted\x94")  # cp1252 bytes: é and curly quotes
    text = load_script(p)
    assert "café" in text and "“quoted”" in text


def test_load_script_missing_and_empty(tmp_path: Path):
    with pytest.raises(InputError):
        load_script(tmp_path / "nope.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(InputError):
        load_script(empty)
    with pytest.raises(InputError):
        load_script(tmp_path)


# ------------------------------------------------------------- normalizing #
def test_normalize_strips_markdown_and_quotes():
    text = "# Title\r\n\r\n- item **bold** “quoted” ‘single’\n> quote\n1. numbered"
    out = normalize_text(text)
    assert out.splitlines()[0] == "Title"
    assert "**" not in out and "#" not in out
    assert '"quoted"' in out and "'single'" in out
    assert "- item" not in out and "1. numbered" not in out
    assert "numbered" in out and "quote" in out


def test_normalize_collapses_blank_lines_and_nbsp():
    out = normalize_text("a\n\n\n\n\nb c")
    assert out == "a\n\nb c"


# -------------------------------------------------------------- paragraphs #
def test_split_paragraphs_joins_soft_wraps():
    text = "line one\nline two\n\nsecond para"
    assert split_paragraphs(text) == ["line one line two", "second para"]
    assert split_paragraphs(text, newline_is_break=True) == ["line one", "line two", "second para"]


# --------------------------------------------------------------- sentences #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello there. How are you? Fine!", ["Hello there.", "How are you?", "Fine!"]),
        ("Dr. Smith met Mr. Jones at 4 p.m. on Tuesday. They talked.", ["Dr. Smith met Mr. Jones at 4 p.m. on Tuesday.", "They talked."]),
        ("It costs 3.5 percent, i.e. a lot. Really.", ["It costs 3.5 percent, i.e. a lot.", "Really."]),
        ('She said "stop." Then she left.', ['She said "stop."', "Then she left."]),
        ("J. K. Rowling wrote it. Yes.", ["J. K. Rowling wrote it.", "Yes."]),
        ("Wait... what? Nothing... never mind.", ["Wait... what?", "Nothing... never mind."]),
        ("See https://example.org/a.b.c for details. Done.", ["See https://example.org/a.b.c for details.", "Done."]),
        ("No punctuation at all", ["No punctuation at all"]),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


# ---------------------------------------------------------------- chunking #
def test_split_long_sentence_prefers_clauses():
    s = "This is a long clause that keeps going, and then another clause continues here; finally it ends with more words."
    pieces = split_long_sentence(s, 60)
    assert all(len(p) <= 60 for p in pieces)
    assert " ".join(pieces).replace("  ", " ") == s
    assert pieces[0].endswith(",")


def test_split_long_sentence_hard_cut_without_spaces():
    s = "x" * 150
    pieces = split_long_sentence(s, 50)
    assert pieces == ["x" * 50, "x" * 50, "x" * 50]


def test_chunk_text_respects_limit_and_paragraphs():
    text = ("Sentence number one is here. Sentence number two is here. Sentence number three is here.\n\n"
            "A new paragraph starts. It has two sentences.")
    chunks = chunk_text(text, max_chars=60)
    assert all(len(c.text) <= 60 for c in chunks)
    # Paragraph boundaries are preserved.
    assert {c.paragraph for c in chunks} == {0, 1}
    ends = [c for c in chunks if c.ends_paragraph]
    assert len(ends) == 2
    assert chunks[-1].ends_paragraph
    # Order and indices are stable.
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # No text was lost.
    joined = " ".join(c.text for c in chunks)
    for word in ("one", "three", "paragraph", "two sentences."):
        assert word in joined


def test_chunk_text_balances_sizes():
    sentences = ["This sentence has about forty characters."] * 5  # 42 chars each
    text = " ".join(sentences)
    chunks = chunk_text(text, max_chars=100)
    sizes = [len(c.text) for c in chunks]
    assert len(chunks) == 3
    assert max(sizes) - min(sizes) <= 45  # no tiny trailing chunk


def test_chunk_text_single_short_line():
    chunks = chunk_text("Hello.", max_chars=300)
    assert len(chunks) == 1 and chunks[0].text == "Hello." and chunks[0].ends_paragraph


def test_chunk_text_rejects_tiny_limit_and_empty():
    with pytest.raises(InputError):
        chunk_text("Hello.", max_chars=5)
    with pytest.raises(InputError):
        chunk_text("   \n\n  ", max_chars=100)


def test_describe_chunks_lists_every_chunk():
    chunks = chunk_text("One. Two. Three.", max_chars=300)
    out = describe_chunks(chunks)
    assert "1 chunk(s)" in out and "One. Two. Three." in out
