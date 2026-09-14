"""Script loading, cleanup and sentence-aware chunking.

The pipeline is::

    script.txt -> load_script() -> normalize_text() -> split_paragraphs()
              -> split_sentences() -> chunk_text() -> [TextChunk, ...]

Chunks never cross a paragraph boundary, never cut a sentence unless the
sentence alone is longer than the limit, and are balanced in size so that a
paragraph does not end with a two-word fragment (very short inputs make most
TTS models hallucinate).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import InputError

log = logging.getLogger(__name__)

# Rough speaking rate used for ETA / sanity checks: ~150 words/min ≈ 14 chars/s.
CHARS_PER_SECOND = 14.0

# Words that end with a period but do not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "inc", "ltd", "co",
    "corp", "no", "fig", "approx", "dept", "est", "gen", "gov", "lt", "col", "sgt", "capt", "rev",
    "hon", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "vol", "ch", "pp", "ed", "eds", "al", "e.g", "i.e", "u.s", "u.k", "a.m", "p.m", "ph.d", "b.a", "m.a",
}

_SENTENCE_END_RE = re.compile(r"([.!?…]+)([\"'”’)\]]*)(\s+)(?=\S)")
_CLAUSE_SEPARATORS = ("; ", ": ", " — ", " – ", ", ", " - ")
_TERMINAL_PUNCT = ".!?…\"'”’)]"

_QUOTE_MAP = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "‹": "'", "›": "'",
    " ": " ", " ": " ", " ": " ", "​": "", "﻿": "", "\t": " ",
}
_QUOTE_TRANS = str.maketrans(_QUOTE_MAP)

_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•‣◦▪]|\d{1,3}[.)]|[a-zA-Z][.)])\s+")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?")


@dataclass(frozen=True)
class TextChunk:
    """One unit of text sent to the TTS model."""

    index: int  # 0-based position in the script
    text: str
    paragraph: int  # 0-based paragraph number
    ends_paragraph: bool  # True -> use the (longer) paragraph pause after it

    @property
    def estimated_seconds(self) -> float:
        return estimate_seconds(self.text)


def estimate_seconds(text: str) -> float:
    """Very rough spoken duration estimate for *text*."""
    return max(0.5, len(text) / CHARS_PER_SECOND)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_script(path: str | Path) -> str:
    """Read a text file, tolerating BOMs and the common Windows encodings."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"Input file not found: {p}", hints=["Check the path, e.g. --input input/script.txt"])
    if p.is_dir():
        raise InputError(f"Input path is a directory, not a text file: {p}")
    if p.suffix.lower() not in {".txt", ".md", ".text", ""}:
        log.warning("Input '%s' does not look like a .txt file; reading it as plain text anyway.", p.name)
    raw = p.read_bytes()
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        log.warning("Input is not valid UTF-8; decoding as cp1252 (Windows-1252).")
        text = raw.decode("cp1252", errors="replace")
    if not text.strip():
        raise InputError(f"Input file is empty: {p}")
    return text


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """Normalise newlines/whitespace/quotes and strip light Markdown markup.

    The result is plain prose.  Model-specific punctuation handling (e.g. how
    an em dash is spoken) is left to each engine.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_QUOTE_TRANS)
    text = text.replace("**", "").replace("`", "")

    lines = []
    for line in text.split("\n"):
        line = _MD_HEADING_RE.sub("", line)
        line = _BLOCKQUOTE_RE.sub("", line)
        line = _LIST_MARKER_RE.sub("", line)
        line = re.sub(r"[ ]{2,}", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    # Collapse 3+ newlines to exactly one blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str, *, newline_is_break: bool = False) -> list[str]:
    """Split on blank lines (or on every newline when *newline_is_break*)."""
    if newline_is_break:
        parts = text.split("\n")
    else:
        parts = re.split(r"\n\s*\n", text)
        # Single newlines inside a paragraph are soft wraps -> join with a space.
        parts = [" ".join(p.split("\n")) for p in parts]
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- #
# Sentences
# --------------------------------------------------------------------------- #
def _is_abbreviation(word: str) -> bool:
    w = word.lower().strip("\"'([“‘")
    w = w.rstrip(".")
    if not w:
        return False
    if w in _ABBREVIATIONS:
        return True
    # Single-letter initials: "J. K. Rowling", "E. coli"
    if len(w) == 1 and w.isalpha():
        return True
    # Dotted acronyms like "u.s.a"
    if "." in w and all(len(part) <= 2 for part in w.split(".") if part):
        return True
    return False


def split_sentences(paragraph: str) -> list[str]:
    """Split *paragraph* into sentences with basic abbreviation handling."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    sentences: list[str] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(paragraph):
        punct, closers, _ws = m.group(1), m.group(2), m.group(3)
        end = m.start() + len(punct) + len(closers)
        next_char = paragraph[m.end()] if m.end() < len(paragraph) else ""
        before = paragraph[start:m.start()]
        last_word = before.split()[-1] if before.split() else ""

        if punct.startswith(".") and len(punct) == 1:
            if _is_abbreviation(last_word):
                continue
            # "approx. 5 items" / "e.g. this" -> next word lowercase, not a boundary.
            if next_char.islower():
                continue
        elif (punct.startswith(".") and len(punct) > 1) or punct.startswith("…"):
            # Ellipsis: only a boundary when followed by a capital / quote / digit.
            if next_char.islower():
                continue
        candidate = paragraph[start:end].strip()
        if candidate:
            sentences.append(candidate)
        start = m.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Break a sentence longer than *max_chars* at clause boundaries, then words."""
    sentence = sentence.strip()
    if len(sentence) <= max_chars:
        return [sentence]

    pieces: list[str] = []
    remaining = sentence
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        cut = -1
        # Prefer the right-most clause separator inside the window.
        for sep in _CLAUSE_SEPARATORS:
            idx = window.rfind(sep)
            if idx > max_chars // 4:  # avoid absurdly small heads
                cut = max(cut, idx + len(sep.rstrip()))
        if cut == -1:
            idx = window.rfind(" ")
            cut = idx if idx > 0 else max_chars  # hard cut as a last resort
        head, remaining = remaining[:cut].strip(), remaining[cut:].strip()
        if head:
            pieces.append(head)
    if remaining:
        pieces.append(remaining)
    return pieces


def _joined_len(units: list[str]) -> int:
    return sum(len(u) for u in units) + max(0, len(units) - 1)


def _pack_balanced(units: list[str], max_chars: int) -> list[str]:
    """Greedy packing towards a balanced target size (never above *max_chars*).

    A unit is appended when the result stays within *max_chars* and gets the
    chunk closer to the target size.  A small trailing chunk then pulls units
    back from its predecessor so a paragraph never ends in a tiny fragment.
    """
    if not units:
        return []
    total = _joined_len(units)
    n_target = max(1, math.ceil(total / max_chars))
    target = math.ceil(total / n_target)

    groups: list[list[str]] = [[units[0]]]
    for unit in units[1:]:
        current_len = _joined_len(groups[-1])
        candidate_len = current_len + 1 + len(unit)
        closer = abs(candidate_len - target) <= abs(current_len - target)
        if candidate_len <= max_chars and (candidate_len <= target or closer):
            groups[-1].append(unit)
        else:
            groups.append([unit])

    # Rebalance: move units from the second-to-last group into a small last group.
    while len(groups) >= 2 and len(groups[-2]) >= 2:
        last, prev = groups[-1], groups[-2]
        last_len, prev_len = _joined_len(last), _joined_len(prev)
        moved = prev[-1]
        new_last, new_prev = last_len + 1 + len(moved), prev_len - 1 - len(moved)
        if new_last <= max_chars and abs(new_last - new_prev) < abs(last_len - prev_len):
            last.insert(0, prev.pop())
        else:
            break
    return [" ".join(g) for g in groups]


def chunk_text(
    text: str,
    max_chars: int = 300,
    *,
    newline_is_break: bool = False,
    normalize: bool = True,
) -> list[TextChunk]:
    """Turn a whole script into ordered, model-sized chunks."""
    if max_chars < 20:
        raise InputError("--max-chunk-chars must be at least 20.")
    if normalize:
        text = normalize_text(text)
    paragraphs = split_paragraphs(text, newline_is_break=newline_is_break)
    if not paragraphs:
        raise InputError("The script contains no readable text.")

    chunks: list[TextChunk] = []
    for p_idx, paragraph in enumerate(paragraphs):
        units: list[str] = []
        for sentence in split_sentences(paragraph):
            units.extend(split_long_sentence(sentence, max_chars))
        packed = _pack_balanced(units, max_chars)
        for j, piece in enumerate(packed):
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    text=piece,
                    paragraph=p_idx,
                    ends_paragraph=(j == len(packed) - 1),
                )
            )
    return chunks


def describe_chunks(chunks: list[TextChunk]) -> str:
    """Human-readable preview used by ``--dry-run``."""
    lines = [
        f"{len(chunks)} chunk(s), ~{sum(c.estimated_seconds for c in chunks) / 60:.1f} min of speech (rough estimate)",
        "(P = last chunk of a paragraph -> longer pause follows)",
    ]
    for c in chunks:
        marker = "P" if c.ends_paragraph else " "
        preview = c.text if len(c.text) <= 110 else c.text[:107] + "..."
        lines.append(f"[{c.index + 1:>3}] {marker} ({len(c.text):>3} chars) {preview}")
    return "\n".join(lines)
