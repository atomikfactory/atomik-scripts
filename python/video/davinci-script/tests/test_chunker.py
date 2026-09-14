"""Tests for :mod:`resolve_tts.chunker`.

The chunker is the part of this tool that has to be right: a chunk that is one
character too long is silently rejected by Resolve, and a chunk that drops or
duplicates text corrupts the narration.  These tests therefore lean heavily on
invariants that must hold for *every* input, not just the hand-written ones.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resolve_tts import chunker
from resolve_tts.chunker import (
    Chunk,
    collapse_whitespace,
    normalize_text,
    split_paragraphs,
    split_sentences,
    split_text,
)


def squeeze(text: str) -> str:
    """All whitespace removed -- the strictest 'same characters' comparison."""
    return "".join(text.split())


class ChunkerTestCase(unittest.TestCase):
    """Shared invariant assertions."""

    def assert_invariants(self, chunks, source, max_chars):
        """Assert the guarantees that hold for every input and every limit."""
        self.assertTrue(chunks, "the chunker produced no chunks")
        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)
            self.assertLessEqual(
                len(chunk.text),
                max_chars,
                "chunk %d is %d chars: %r" % (chunk.index, len(chunk.text), chunk.text),
            )
            self.assertTrue(chunk.text, "empty chunk at index %d" % chunk.index)
            self.assertEqual(chunk.text, chunk.text.strip(), "chunk is not stripped")
            self.assertEqual(chunk.char_count, len(chunk.text))
            self.assertGreaterEqual(chunk.paragraph_index, 0)

        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))
        self.assertEqual(
            [c.paragraph_index for c in chunks],
            sorted(c.paragraph_index for c in chunks),
            "paragraph order was not preserved",
        )
        self.assertEqual(
            squeeze("".join(c.text for c in chunks)),
            squeeze(normalize_text(source)),
            "characters were added, dropped or reordered",
        )

    def assert_no_word_was_split(self, chunks, source):
        """Rejoining with single spaces must rebuild the collapsed source."""
        self.assertEqual(
            " ".join(c.text for c in chunks),
            collapse_whitespace(normalize_text(source)),
        )

    def assert_chunks_are_substrings(self, chunks, source):
        """Each chunk must appear verbatim in its whitespace-collapsed paragraph."""
        paragraphs = split_paragraphs(normalize_text(source))
        for chunk in chunks:
            paragraph = paragraphs[chunk.paragraph_index]
            self.assertIn(chunk.text, paragraph)


class TestParagraphSplitting(ChunkerTestCase):
    def test_blank_lines_separate_paragraphs(self):
        text = "First paragraph line one.\nStill paragraph one.\n\n\nSecond paragraph."
        self.assertEqual(
            split_paragraphs(text),
            ["First paragraph line one. Still paragraph one.", "Second paragraph."],
        )

    def test_whitespace_only_lines_are_dropped(self):
        text = "Alpha.\n   \n\t\nBravo."
        self.assertEqual(split_paragraphs(text), ["Alpha.", "Bravo."])

    def test_line_breaks_inside_a_paragraph_become_single_spaces(self):
        text = "One\n   two\t\tthree\nfour"
        self.assertEqual(split_paragraphs(text), ["One two three four"])

    def test_bullets_become_their_own_paragraphs(self):
        text = (
            "Here is the list:\n"
            "- First item\n"
            "* Second item\n"
            "• Third item\n"
            "1. Fourth item\n"
            "2) Fifth item\n"
        )
        self.assertEqual(
            split_paragraphs(text),
            [
                "Here is the list:",
                "- First item",
                "* Second item",
                "• Third item",
                "1. Fourth item",
                "2) Fifth item",
            ],
        )

    def test_wrapped_bullet_continuation_stays_with_its_bullet(self):
        text = "- A bullet that wraps\n  onto a second line\n- Another bullet"
        self.assertEqual(
            split_paragraphs(text),
            ["- A bullet that wraps onto a second line", "- Another bullet"],
        )

    def test_bullet_chunks_keep_their_marker(self):
        text = "- Alpha item\n- Bravo item"
        chunks = split_text(text, max_chars=50)
        self.assertEqual([c.text for c in chunks], ["- Alpha item", "- Bravo item"])

    def test_paragraphs_are_not_merged_by_default(self):
        text = "Alpha.\n\nBravo.\n\nCharlie."
        chunks = split_text(text, max_chars=300)
        self.assertEqual([c.text for c in chunks], ["Alpha.", "Bravo.", "Charlie."])
        self.assertEqual([c.paragraph_index for c in chunks], [0, 1, 2])

    def test_merge_paragraphs_packs_short_paragraphs(self):
        text = "Alpha.\n\nBravo.\n\nCharlie."
        chunks = split_text(text, max_chars=300, merge_paragraphs=True)
        self.assertEqual([c.text for c in chunks], ["Alpha. Bravo. Charlie."])
        self.assert_invariants(chunks, text, 300)

    def test_merge_paragraphs_still_respects_the_limit(self):
        text = "\n\n".join(["Paragraph number %d is here." % n for n in range(20)])
        chunks = split_text(text, max_chars=60, merge_paragraphs=True)
        self.assert_invariants(chunks, text, 60)


class TestSentenceSplitting(ChunkerTestCase):
    def test_simple_sentences(self):
        self.assertEqual(
            split_sentences("One fish. Two fish! Red fish? Blue fish."),
            ["One fish.", "Two fish!", "Red fish?", "Blue fish."],
        )

    def test_quotes_after_terminal_punctuation(self):
        self.assertEqual(
            split_sentences('He said "Stop." Then he left.'),
            ['He said "Stop."', "Then he left."],
        )
        self.assertEqual(
            split_sentences('(She replied "Never.") The door closed.'),
            ['(She replied "Never.")', "The door closed."],
        )
        self.assertEqual(
            split_sentences("‘Who goes there?’ The sentry waited."),
            ["‘Who goes there?’", "The sentry waited."],
        )

    def test_abbreviations_are_not_sentence_ends(self):
        for text in (
            "Mr. Smith arrived early.",
            "Mrs. Patel and Dr. Lee agreed.",
            "We met on St. James Street today.",
            "Compare cats vs. dogs here.",
            "Use a tool, e.g. Resolve, for this.",
            "That is, i.e. the point exactly.",
            "Bring pens, paper, etc. Then begin.",
            "The U.S. Army marched north.",
            "We start at 9 a.m. Sharp arrivals only.",
            "It was J. R. R. Tolkien who wrote it.",
        ):
            with self.subTest(text=text):
                self.assertEqual(split_sentences(text), [text])

    def test_decimals_are_not_sentence_ends(self):
        text = "Pi is 3.14159 and e is 2.71828 exactly."
        self.assertEqual(split_sentences(text), [text])

    def test_ellipses_are_not_sentence_ends(self):
        text = "She paused... then continued walking."
        self.assertEqual(split_sentences(text), [text])
        unicode_text = "She paused… then continued walking."
        self.assertEqual(split_sentences(unicode_text), [unicode_text])

    def test_urls_and_emails_are_never_split(self):
        text = "Visit https://example.com/a.b/c. Then email a.b@example.com. Done."
        self.assertEqual(
            split_sentences(text),
            [
                "Visit https://example.com/a.b/c.",
                "Then email a.b@example.com.",
                "Done.",
            ],
        )

    def test_url_is_kept_whole_when_it_fits(self):
        url = "https://example.org/reports/2024/post-production-economics.pdf"
        text = (
            "The full report, with all of its many caveats and appendices, "
            "lives at " + url + " and it is free to read."
        )
        chunks = split_text(text, max_chars=90)
        self.assertTrue(
            any(url in chunk.text for chunk in chunks),
            "the URL was split across chunks",
        )
        self.assert_invariants(chunks, text, 90)

    def test_text_without_terminal_punctuation_is_one_sentence(self):
        text = "This narration simply stops without any final punctuation"
        self.assertEqual(split_sentences(text), [text])

    def test_cjk_terminators(self):
        text = "今日は晴れ。明日は雨。"
        self.assertEqual(
            split_sentences(text),
            ["今日は晴れ。", "明日は雨。"],
        )


class TestChunkSizes(ChunkerTestCase):
    def test_short_paragraph_is_one_chunk(self):
        text = (
            "This opening paragraph is deliberately short, running to roughly one "
            "hundred characters in total."
        )
        self.assertLessEqual(len(text), 120)
        chunks = split_text(text, max_chars=300)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assert_invariants(chunks, text, 300)

    def test_medium_paragraph_splits_on_sentences(self):
        sentence = "The colourist adjusted the midtones until the shot finally settled."
        text = " ".join([sentence] * 7)
        self.assertGreater(len(text), 450)
        chunks = split_text(text, max_chars=300)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.text.endswith("."), chunk.text)
        self.assert_invariants(chunks, text, 300)
        self.assert_no_word_was_split(chunks, text)
        self.assert_chunks_are_substrings(chunks, text)

    def test_two_thousand_character_paragraph(self):
        sentences = [
            "Sentence number %d explains one more detail about the grade." % n
            for n in range(34)
        ]
        text = " ".join(sentences)
        self.assertGreater(len(text), 2000)
        chunks = split_text(text, max_chars=300)
        self.assert_invariants(chunks, text, 300)
        self.assert_no_word_was_split(chunks, text)
        self.assertGreaterEqual(len(chunks), 7)

    def test_packing_is_greedy(self):
        sentence = "Ten chars."
        text = " ".join([sentence] * 12)
        chunks = split_text(text, max_chars=50)
        # 50 chars fits four "Ten chars." (44 chars) but not five (55).
        self.assertEqual(chunks[0].text, " ".join([sentence] * 4))
        self.assert_invariants(chunks, text, 50)


class TestOversizedSentences(ChunkerTestCase):
    def test_long_sentence_with_commas_splits_on_clauses(self):
        clauses = [
            "the first consideration is colour temperature",
            "the second is the shape of the highlight roll-off",
            "the third is how the skin tones sit against the background",
            "the fourth is whether the whole thing survives a broadcast-safe pass",
            "and the fifth is whether anybody watching will ever notice",
        ]
        text = "When you grade a scene like this one, " + ", ".join(clauses)
        self.assertGreater(len(text), 300)
        self.assertNotIn(".", text)
        chunks = split_text(text, max_chars=120)
        self.assertGreater(len(chunks), 2)
        self.assert_invariants(chunks, text, 120)
        self.assert_no_word_was_split(chunks, text)
        # Most breaks should land after a comma.
        comma_endings = sum(1 for c in chunks[:-1] if c.text.endswith(","))
        self.assertGreaterEqual(comma_endings, len(chunks) - 2)

    def test_semicolons_and_dashes_are_clause_boundaries(self):
        text = (
            "The room was silent; the console hummed — nobody spoke, nobody "
            "moved, nobody dared touch the trackball; the render bar crept "
            "forward one percent at a time - and then it stopped completely"
        )
        chunks = split_text(text, max_chars=90)
        self.assert_invariants(chunks, text, 90)
        self.assert_no_word_was_split(chunks, text)

    def test_seven_hundred_character_sentence_without_punctuation(self):
        words = ["word%02d" % n for n in range(100)]
        text = " ".join(words)
        self.assertGreater(len(text), 690)
        self.assertLess(len(text), 710)
        chunks = split_text(text, max_chars=100)
        self.assert_invariants(chunks, text, 100)
        self.assert_no_word_was_split(chunks, text)
        for chunk in chunks:
            for word in chunk.text.split(" "):
                self.assertIn(word, words)

    def test_a_single_oversized_word_is_hard_split(self):
        long_word = "https://example.org/" + ("segment" * 55)
        self.assertGreater(len(long_word), 400)
        text = "Read it here: " + long_word + " and then stop."
        chunks = split_text(text, max_chars=100)
        self.assert_invariants(chunks, text, 100)
        rebuilt = "".join(c.text for c in chunks)
        self.assertIn(squeeze(long_word), squeeze(rebuilt))

    def test_hard_split_is_the_only_intra_word_split(self):
        long_word = "A" * 400
        text = "Before " + long_word + " after"
        chunks = split_text(text, max_chars=100)
        self.assert_invariants(chunks, text, 100)
        pieces = [c.text for c in chunks]
        self.assertEqual(pieces[0], "Before")
        self.assertEqual(pieces[-1], "after")
        middle = "".join(pieces[1:-1])
        self.assertEqual(middle, long_word)
        self.assertEqual(len(pieces[1]), 100)

    def test_word_longer_than_limit_alone(self):
        text = "Z" * 250
        chunks = split_text(text, max_chars=50)
        self.assertEqual(len(chunks), 5)
        self.assert_invariants(chunks, text, 50)


class TestTextVariety(ChunkerTestCase):
    def test_punctuation_heavy_text(self):
        text = (
            "\"Wait!\" she said. \"Wait -- did you (really) mean it?\" He shrugged; "
            "she frowned: the silence stretched... and stretched. \"Fine,\" he "
            "muttered, \"have it your way!\" [Exit, stage left.] The end?"
        )
        for limit in (50, 80, 150, 300):
            with self.subTest(max_chars=limit):
                chunks = split_text(text, max_chars=limit)
                self.assert_invariants(chunks, text, limit)
                self.assert_no_word_was_split(chunks, text)

    def test_unicode_text(self):
        text = (
            "Le réalisateur a dit « c'est fini » — et puis, "
            "rien. 今日はとても良い天気"
            "です。你好，世界。 She smiled "
            "\U0001f642 and said “that’s enough” – quietly, "
            "firmly, finally… naïve café résumé."
        )
        for limit in (50, 90, 300):
            with self.subTest(max_chars=limit):
                chunks = split_text(text, max_chars=limit)
                self.assert_invariants(chunks, text, limit)

    def test_numbers_and_measurements(self):
        text = (
            "The file was 1.44 GB, encoded at 23.976 fps, delivered on 3.5 inch "
            "media for $1,299.99. Version 2.0 shipped on 4.7.2024. It worked."
        )
        chunks = split_text(text, max_chars=300)
        self.assertEqual(len(chunks), 1)
        self.assert_invariants(chunks, text, 300)

    def test_text_ending_without_punctuation(self):
        text = "The final line of this narration has no full stop at the end"
        chunks = split_text(text, max_chars=300)
        self.assertEqual([c.text for c in chunks], [text])

    def test_bom_and_crlf_are_normalised(self):
        text = "﻿First line.\r\n\r\nSecond line.\r\n"
        chunks = split_text(text, max_chars=300)
        self.assertEqual([c.text for c in chunks], ["First line.", "Second line."])

    def test_empty_and_whitespace_only_input(self):
        self.assertEqual(split_text(""), [])
        self.assertEqual(split_text("   \n\n\t  \n"), [])

    def test_normalize_text_strips_bom_and_normalises_newlines(self):
        self.assertEqual(normalize_text("﻿a\r\nb\rc"), "a\nb\nc")

    def test_nfc_normalisation_is_applied(self):
        decomposed = "café"
        precomposed = "café"
        self.assertNotEqual(decomposed, precomposed)
        self.assertEqual(normalize_text(decomposed), precomposed)


class TestLimits(ChunkerTestCase):
    SOURCE = (
        "Post-production changed quietly. The tools got cheaper, then they got "
        "better, and eventually they got free. Nobody announced it; it simply "
        "happened, one release at a time, until the old assumptions stopped "
        "being true.\n\n"
        "\"That is either the best news in the history of cinema,\" she said, "
        "\"or the worst.\" Dr. Marsh has been arguing both sides for years, and "
        "she is not finished yet."
    )

    def test_edge_limits(self):
        for limit in (50, 51, 99, 100, 200, 299, 300, 330, 349, 350):
            with self.subTest(max_chars=limit):
                chunks = split_text(self.SOURCE, max_chars=limit)
                self.assert_invariants(chunks, self.SOURCE, limit)
                self.assert_chunks_are_substrings(chunks, self.SOURCE)

    def test_smaller_limits_never_produce_fewer_chunks(self):
        counts = [len(split_text(self.SOURCE, max_chars=n)) for n in (50, 100, 200, 350)]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_invalid_limit_raises(self):
        with self.assertRaises(ValueError):
            split_text("hello", max_chars=0)

    def test_default_limit_is_under_the_resolve_cap(self):
        self.assertLessEqual(chunker.DEFAULT_MAX_CHARS, chunker.RESOLVE_TEXT_LIMIT)
        self.assertEqual(chunker.RESOLVE_TEXT_LIMIT, 350)

    def test_determinism(self):
        first = split_text(self.SOURCE, max_chars=120)
        second = split_text(self.SOURCE, max_chars=120)
        self.assertEqual([c.text for c in first], [c.text for c in second])
        self.assertEqual(
            [c.paragraph_index for c in first], [c.paragraph_index for c in second]
        )


class TestRandomisedProperties(ChunkerTestCase):
    WORDS = [
        "grade", "shot", "timeline", "colour", "frame", "render", "audio",
        "narration", "sequence", "deliverable", "codec", "conform", "master",
        "café", "résumé", "今日", "voice-over",
        "Dr.", "e.g.", "3.14", "U.S.", "https://example.com/x", "a@b.example",
    ]

    def _random_document(self, rng: random.Random) -> str:
        paragraphs = []
        for _ in range(rng.randint(1, 6)):
            sentences = []
            for _ in range(rng.randint(1, 8)):
                words = [rng.choice(self.WORDS) for _ in range(rng.randint(1, 40))]
                body = " ".join(words)
                if rng.random() < 0.3:
                    body = body.replace(" ", ", ", 1)
                sentences.append(body + rng.choice([".", "!", "?", ""]))
            paragraph = " ".join(sentences)
            if rng.random() < 0.15:
                paragraph = "- " + paragraph
            if rng.random() < 0.1:
                paragraph += " " + "X" * rng.randint(60, 500)
            paragraphs.append(paragraph)
        return "\n\n".join(paragraphs)

    def test_invariants_hold_for_random_documents(self):
        rng = random.Random(20240906)
        for iteration in range(200):
            text = self._random_document(rng)
            limit = rng.choice([50, 73, 120, 200, 300, 350])
            merge = rng.random() < 0.3
            with self.subTest(iteration=iteration, max_chars=limit, merge=merge):
                chunks = split_text(text, max_chars=limit, merge_paragraphs=merge)
                if not chunks:
                    self.assertFalse(text.strip())
                    continue
                self.assert_invariants(chunks, text, limit)

    def test_random_documents_are_deterministic(self):
        rng = random.Random(7)
        for _ in range(25):
            text = self._random_document(rng)
            a = [c.text for c in split_text(text, max_chars=137)]
            b = [c.text for c in split_text(text, max_chars=137)]
            self.assertEqual(a, b)


class TestExampleScript(ChunkerTestCase):
    def test_bundled_example_script(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "example_script.txt",
        )
        if not os.path.isfile(path):
            self.skipTest("example_script.txt is not present")
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        for limit in (50, 200, 300, 350):
            with self.subTest(max_chars=limit):
                chunks = split_text(text, max_chars=limit)
                self.assert_invariants(chunks, text, limit)
                self.assert_chunks_are_substrings(chunks, text)


if __name__ == "__main__":
    unittest.main()
