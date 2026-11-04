"""Tests for the text-preprocessing helpers.

These verify the deliberate design decision to *preserve* casing and punctuation
(which carry stylistic clickbait signal) while still cleaning whitespace and
handling Unicode and intra-word symbols correctly.
"""

from __future__ import annotations

import pytest

from src import preprocessing as pp


class TestClean:
    def test_collapses_and_trims_whitespace(self):
        assert pp.clean("  Hello    world  ") == "Hello world"

    def test_collapses_tabs_and_newlines(self):
        assert pp.clean("a\t\tb\n c") == "a b c"

    def test_preserves_case_and_punctuation(self):
        # Casing and "!"/"?" are clickbait cues and must survive cleaning.
        assert pp.clean("You WON'T Believe This!") == "You WON'T Believe This!"

    def test_canonicalizes_single_quotes_without_dropping_punctuation(self):
        assert pp.clean("  \u2018You WON\u2019T believe THIS!\u2019  ") == "'You WON'T believe THIS!'"

    def test_accepts_non_string_input(self):
        # Labels/ids occasionally arrive as numbers; clean must not crash.
        assert pp.clean(123) == "123"

    def test_empty_string(self):
        assert pp.clean("") == ""
        assert pp.clean("     ") == ""


class TestTokenize:
    def test_basic_split(self):
        assert pp.tokenize("The quick brown fox") == ["The", "quick", "brown", "fox"]

    def test_keeps_apostrophes_and_hyphens_inside_words(self):
        assert pp.tokenize("you'll love game-changing tips") == [
            "you'll",
            "love",
            "game-changing",
            "tips",
        ]

    @pytest.mark.parametrize("apostrophe", ["\u2018", "\u2019", "\u201b", "\u02bc", "\uff07"])
    def test_unicode_contractions_match_ascii_tokens(self, apostrophe):
        text = f"You won{apostrophe}t believe this! You{apostrophe}ll love it."
        assert pp.tokenize(text) == pp.tokenize("You won't believe this! You'll love it.")

    def test_strips_surrounding_punctuation(self):
        assert pp.tokenize("Wow! Amazing, really?") == ["Wow", "Amazing", "really"]

    def test_keeps_digits_as_tokens(self):
        assert pp.tokenize("17 Things") == ["17", "Things"]

    def test_empty_input_yields_no_tokens(self):
        assert pp.tokenize("") == []
        assert pp.tokenize("   ") == []


class TestTokenizeLower:
    def test_lowercases(self):
        assert pp.tokenize_lower("You WON'T Believe") == ["you", "won't", "believe"]

    def test_consistent_with_tokenize(self):
        text = "Game-Changing TIPS You'll Love"
        assert pp.tokenize_lower(text) == [t.lower() for t in pp.tokenize(text)]


class TestUnicode:
    def test_nfkc_normalization_of_compatibility_chars(self):
        # The fullwidth digit "１" (U+FF11) normalizes to ASCII "1" under NFKC.
        assert pp.normalize_unicode("１") == "1"

    @pytest.mark.parametrize("apostrophe", ["\u2018", "\u2019", "\u201b", "\u02bc", "\uff07"])
    def test_canonicalizes_apostrophes(self, apostrophe):
        assert pp.normalize_unicode(f"won{apostrophe}t") == "won't"

    def test_normalized_text_lower(self):
        assert pp.normalized_text_lower("  HELLO World  ") == "hello world"

    def test_smart_contractions_match_ascii_in_lowercase_views(self):
        text = "You WON\u2019T believe this!"
        assert pp.tokenize_lower(text) == ["you", "won't", "believe", "this"]
        assert pp.normalized_text_lower(text) == "you won't believe this!"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
