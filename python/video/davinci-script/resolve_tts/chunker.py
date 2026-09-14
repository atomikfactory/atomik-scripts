"""Natural-language chunking of a narration script.

Resolve's ``GenerateSpeech`` accepts at most 350 characters of ``TextInput``.
This module splits arbitrarily long text into ordered chunks that never exceed
a configured limit, preferring the most natural break available:

    paragraph  ->  sentence  ->  clause  ->  word  ->  hard split

A hard split (a cut inside a word) only ever happens when a single "word" --
typically a very long URL -- is itself longer than ``max_chars``.

The implementation works on *spans* (index pairs) into the whitespace-collapsed
paragraph.  Every emitted chunk is therefore a contiguous substring of its
paragraph, which makes the "no characters invented, deleted or reordered"
guarantee structural rather than something we have to test our way into.

This module is pure: it never imports or touches DaVinci Resolve.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

__all__ = [
    "Chunk",
    "DEFAULT_MAX_CHARS",
    "RESOLVE_TEXT_LIMIT",
    "collapse_whitespace",
    "normalize_text",
    "split_paragraphs",
    "split_sentences",
    "split_text",
]

#: Hard cap enforced by Resolve on a single ``GenerateSpeech`` call.
RESOLVE_TEXT_LIMIT = 350

#: Default safety limit; leaves headroom under :data:`RESOLVE_TEXT_LIMIT`.
DEFAULT_MAX_CHARS = 300

_BOM = "﻿"

Span = Tuple[int, int]
_Splitter = Callable[[Span], List[Span]]


@dataclass(frozen=True)
class Chunk:
    """One unit of text destined for a single ``GenerateSpeech`` call."""

    index: int
    """Zero-based position in the ordered chunk list."""

    text: str
    """The chunk text, stripped, never longer than ``max_chars``."""

    char_count: int
    """``len(text)`` -- stored so manifests stay self-describing."""

    paragraph_index: int
    """Zero-based index of the source paragraph this chunk came from."""

    def __post_init__(self) -> None:
        if self.char_count != len(self.text):
            raise ValueError("char_count does not match text length")


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def normalize_text(text: str, nfc: bool = True) -> str:
    """Strip a UTF-8 BOM, normalise line endings and optionally apply NFC.

    Nothing else is changed: punctuation, quotes, dashes, contractions,
    numbers and non-ASCII characters are preserved verbatim.
    """
    text = text.replace(_BOM, "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if nfc:
        text = unicodedata.normalize("NFC", text)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse every run of whitespace to one space and strip the ends."""
    return " ".join(text.split())


# --------------------------------------------------------------------------
# Paragraph detection
# --------------------------------------------------------------------------

# A bullet or numbered list marker at the start of a line.  Each such line
# begins its own paragraph so the voice gets a natural pause between items.
_BULLET_RE = re.compile(
    r"^[ \t]*(?:[-*•‣▪◦·⁃]|\(?\d{1,3}[.)])[ \t]+(?=\S)"
)

_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")


def _is_bullet_line(line: str) -> bool:
    return bool(_BULLET_RE.match(line))


def split_paragraphs(text: str) -> List[str]:
    """Split normalised text into whitespace-collapsed paragraphs.

    Blank lines separate paragraphs.  A bullet or numbered list line starts a
    new paragraph; unmarked lines that follow it are treated as its wrapped
    continuation.
    """
    paragraphs: List[str] = []
    for block in _BLANK_LINE_RE.split(text):
        current: List[str] = []
        for line in block.split("\n"):
            if not line.strip():
                continue
            if _is_bullet_line(line) and current:
                paragraphs.append(collapse_whitespace(" ".join(current)))
                current = []
            current.append(line)
        if current:
            paragraphs.append(collapse_whitespace(" ".join(current)))
    return [p for p in paragraphs if p]


# --------------------------------------------------------------------------
# Protected spans (URLs and e-mail addresses are never cut into)
# --------------------------------------------------------------------------

_PROTECT_RE = re.compile(
    # A URL, minus any sentence punctuation that merely trails it.
    r"(?:(?:https?|ftp|file)://|www\.)[^\s]*[^\s.,;:!?)\]}'\"]"
    # An e-mail address; the domain may not end on a dot.
    r"|[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+",
    re.IGNORECASE,
)


def _protected_spans(text: str) -> List[Span]:
    return [m.span() for m in _PROTECT_RE.finditer(text)]


def _in_spans(pos: int, spans: Sequence[Span]) -> bool:
    for start, end in spans:
        if start <= pos < end:
            return True
    return False


# --------------------------------------------------------------------------
# Sentence detection
# --------------------------------------------------------------------------

_LATIN_TERMINATORS = ".!?…"
_CJK_TERMINATORS = "。！？"
_CLOSERS = "\"'’”)]}»›"
_OPENERS = "\"'‘“([{«‹"

# Tokens that end in a period without ending a sentence.  Deliberately
# conservative: an entry that is also a common English word (for example
# "no", "sun" or "sat") would swallow real sentence breaks, so those are
# excluded even though they are legitimate abbreviations.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "mx", "dr", "prof", "sr", "jr", "st", "messrs",
        "mme", "mlle", "rev", "hon", "capt", "lt", "sgt", "col", "gen",
        "vs", "etc", "e.g", "i.e", "cf", "viz", "ibid", "al", "esp",
        "u.s", "u.k", "u.s.a", "u.k.", "e.u", "a.m", "p.m", "p.s",
        "ph.d", "m.d", "b.a", "m.a", "b.sc", "m.sc",
        "fig", "figs", "vol", "vols", "chap", "pp", "no.",
        "inc", "ltd", "corp", "dept", "approx", "est.",
        "jan", "feb", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
        "nov", "dec",
    }
)


def _is_sentence_break(
    par: str,
    dot_start: int,
    dot_end: int,
    boundary: int,
    protected: Sequence[Span],
) -> bool:
    """Decide whether the terminator run at ``par[dot_start:dot_end]`` ends a sentence.

    ``boundary`` is the index just past any closing quotes or brackets that
    trail the terminator; the caller has already checked that a space follows.
    """
    if _in_spans(dot_start, protected):
        return False

    run = par[dot_start:dot_end]
    if "…" in run:
        return False  # ellipsis character
    if run.count(".") >= 2:
        return False  # "..." style ellipsis

    if run == ".":
        # The token immediately before the period.
        start = dot_start
        while start > 0 and not par[start - 1].isspace():
            start -= 1
        token = par[start:dot_start]
        if token:
            bare = token.lstrip(_OPENERS)
            if len(bare) == 1 and bare.isalpha() and bare.isupper():
                return False  # an initial such as "J." in "J. R. R. Tolkien"
            if bare.lower() in _ABBREVIATIONS:
                return False
        # A lower-case word after a period is far more often a missed
        # abbreviation than a new sentence.
        nxt = boundary
        while nxt < len(par) and par[nxt] == " ":
            nxt += 1
        if nxt < len(par) and par[nxt].islower():
            return False
    return True


def _sentence_spans(par: str) -> List[Span]:
    """Return ordered, non-overlapping sentence spans covering ``par``."""
    n = len(par)
    if n == 0:
        return []
    protected = _protected_spans(par)
    spans: List[Span] = []
    start = 0
    i = 0
    while i < n:
        ch = par[i]

        if ch in _CJK_TERMINATORS:
            end = i + 1
            while end < n and par[end] in _CLOSERS:
                end += 1
            if end < n and not _in_spans(i, protected):
                spans.append((start, end))
                nxt = end
                while nxt < n and par[nxt] == " ":
                    nxt += 1
                start = nxt
                i = nxt
                continue
            i = end
            continue

        if ch in _LATIN_TERMINATORS:
            run_end = i
            while run_end < n and par[run_end] in _LATIN_TERMINATORS:
                run_end += 1
            boundary = run_end
            while boundary < n and par[boundary] in _CLOSERS:
                boundary += 1
            if boundary >= n:
                i = boundary
                continue
            if par[boundary] != " ":
                i = run_end
                continue
            if _is_sentence_break(par, i, run_end, boundary, protected):
                spans.append((start, boundary))
                nxt = boundary
                while nxt < n and par[nxt] == " ":
                    nxt += 1
                start = nxt
                i = nxt
                continue
            i = run_end
            continue

        i += 1

    if start < n:
        spans.append((start, n))
    return spans


def split_sentences(paragraph: str) -> List[str]:
    """Split one whitespace-collapsed paragraph into sentences."""
    return [paragraph[s:e] for s, e in _sentence_spans(paragraph)]


# --------------------------------------------------------------------------
# Clause / word / hard splitting
# --------------------------------------------------------------------------

_CLAUSE_CHARS = ",;:—–)]}"
_QUOTE_CLOSERS = "\"'’”"


def _is_clause_break(par: str, pos: int, lo: int) -> bool:
    """True when the space at ``par[pos]`` is a reasonable clause boundary."""
    j = pos - 1
    while j >= lo and par[j] in _QUOTE_CLOSERS:
        j -= 1
    if j < lo:
        return False
    ch = par[j]
    if ch in _CLAUSE_CHARS:
        return True
    if ch == "-":
        k = j
        while k >= lo and par[k] == "-":
            k -= 1
        # A standalone "-" or "--" used as a dash, not a hyphenated word.
        return k < lo or par[k].isspace()
    return False


def _clause_spans(par: str, span: Span) -> List[Span]:
    """Split a span at clause boundaries; returns ``[span]`` when there are none."""
    lo, hi = span
    protected = _protected_spans(par[lo:hi])
    cuts: List[int] = []
    for pos in range(lo + 1, hi - 1):
        if par[pos] != " ":
            continue
        if _in_spans(pos - 1 - lo, protected):
            continue
        after = par[pos + 1]
        if _is_clause_break(par, pos, lo) or after in _OPENERS:
            cuts.append(pos)
    if not cuts:
        return [span]

    out: List[Span] = []
    prev = lo
    for cut in cuts:
        if cut > prev:
            out.append((prev, cut))
        prev = cut + 1
    if prev < hi:
        out.append((prev, hi))
    return out or [span]


def _hard_spans(lo: int, hi: int, max_chars: int) -> List[Span]:
    """Cut blindly every ``max_chars`` characters -- the last resort."""
    return [(pos, min(pos + max_chars, hi)) for pos in range(lo, hi, max_chars)]


def _word_spans(par: str, span: Span, max_chars: int) -> List[Span]:
    """Pack whole words; hard-split only words longer than ``max_chars``."""
    lo, hi = span
    out: List[Span] = []
    open_index = -1
    i = lo
    while i < hi:
        if par[i] == " ":
            i += 1
            continue
        j = i
        while j < hi and par[j] != " ":
            j += 1
        if j - i > max_chars:
            out.extend(_hard_spans(i, j, max_chars))
            open_index = -1
        elif open_index >= 0 and j - out[open_index][0] <= max_chars:
            out[open_index] = (out[open_index][0], j)
        else:
            out.append((i, j))
            open_index = len(out) - 1
        i = j
    return out


def _pack(
    spans: Sequence[Span], max_chars: int, splitter: _Splitter
) -> List[Span]:
    """Greedily merge consecutive spans while they still fit.

    Merging two adjacent spans keeps the separator characters between them, so
    the merged span remains a contiguous substring of the paragraph.  A span
    that is oversized on its own is handed to ``splitter`` and closes the
    current run.
    """
    out: List[Span] = []
    open_index = -1
    for start, end in spans:
        if end - start > max_chars:
            out.extend(splitter((start, end)))
            open_index = -1
            continue
        if open_index >= 0 and end - out[open_index][0] <= max_chars:
            out[open_index] = (out[open_index][0], end)
            continue
        out.append((start, end))
        open_index = len(out) - 1
    return out


def _paragraph_spans(par: str, max_chars: int) -> List[Span]:
    """Split one whitespace-collapsed paragraph into fitting spans."""
    n = len(par)
    if n == 0:
        return []
    if n <= max_chars:
        return [(0, n)]

    def split_oversized(span: Span) -> List[Span]:
        clauses = _clause_spans(par, span)
        if len(clauses) <= 1:
            return _word_spans(par, span, max_chars)
        return _pack(
            clauses, max_chars, lambda sp: _word_spans(par, sp, max_chars)
        )

    return _pack(_sentence_spans(par), max_chars, split_oversized)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def _merge_across_paragraphs(
    pieces: List[Tuple[int, str]], max_chars: int
) -> List[Tuple[int, str]]:
    """Greedily join short consecutive pieces that came from different paragraphs."""
    merged: List[Tuple[int, str]] = []
    for paragraph_index, text in pieces:
        if merged and len(merged[-1][1]) + 1 + len(text) <= max_chars:
            prev_index, prev_text = merged[-1]
            merged[-1] = (prev_index, prev_text + " " + text)
            continue
        merged.append((paragraph_index, text))
    return merged


def split_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    merge_paragraphs: bool = False,
    normalize: bool = True,
) -> List[Chunk]:
    """Split ``text`` into ordered :class:`Chunk` objects of at most ``max_chars``.

    Args:
        text: The full narration script.
        max_chars: Inclusive upper bound on every chunk's character count.
        merge_paragraphs: When true, short consecutive paragraphs may share a
            chunk.  Off by default because a paragraph break is a natural
            pause in the delivered voice-over.
        normalize: Apply :func:`normalize_text` first (BOM, CRLF, NFC).

    Returns:
        Chunks in script order.  Every chunk is non-empty, stripped, and at
        most ``max_chars`` characters long.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    source = normalize_text(text) if normalize else text
    pieces: List[Tuple[int, str]] = []
    for paragraph_index, paragraph in enumerate(split_paragraphs(source)):
        for start, end in _paragraph_spans(paragraph, max_chars):
            chunk_text = paragraph[start:end].strip()
            if chunk_text:
                pieces.append((paragraph_index, chunk_text))

    if merge_paragraphs:
        pieces = _merge_across_paragraphs(pieces, max_chars)

    return [
        Chunk(
            index=index,
            text=chunk_text,
            char_count=len(chunk_text),
            paragraph_index=paragraph_index,
        )
        for index, (paragraph_index, chunk_text) in enumerate(pieces)
    ]
