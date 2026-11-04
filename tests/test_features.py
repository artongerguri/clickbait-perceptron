"""Tests for the engineered feature extractor.

The extractor is the interpretable core of the project, so these tests pin down
(a) its structural contract — stable column order, name/width agreement, matrix
shape — and (b) the semantics of individual linguistic cues on headlines whose
expected values can be reasoned about by hand. Edge cases (empty / whitespace /
punctuation-only headlines) must never raise and must yield finite numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features import FeatureExtractor

EXPECTED_N_FEATURES = 33


@pytest.fixture(scope="module")
def fx():
    return FeatureExtractor()


def value(fx: FeatureExtractor, headline: str, name: str) -> float:
    """Convenience: single feature value for a single headline."""
    return fx.extract_one(headline)[name]


# --------------------------------------------------------------------------- #
# Structural contract                                                         #
# --------------------------------------------------------------------------- #
class TestContract:
    def test_feature_count(self, fx):
        assert fx.n_features == EXPECTED_N_FEATURES
        assert len(fx.feature_names) == EXPECTED_N_FEATURES

    def test_feature_names_unique(self, fx):
        assert len(set(fx.feature_names)) == fx.n_features

    def test_transform_shape(self, fx):
        X = fx.transform(["a headline", "another one", "third"])
        assert X.shape == (3, EXPECTED_N_FEATURES)
        assert X.dtype == np.float64

    @pytest.mark.parametrize("headlines", [[], iter(())])
    def test_empty_transform_keeps_feature_width(self, fx, headlines):
        X = fx.transform(headlines)
        assert X.shape == (0, EXPECTED_N_FEATURES)
        assert X.dtype == np.float64

    def test_extract_one_keys_match_names(self, fx):
        d = fx.extract_one("Some Headline")
        assert list(d.keys()) == fx.feature_names

    def test_extract_one_matches_transform_row(self, fx):
        h = "You Won't Believe These 7 Amazing Tricks!"
        row_dict = fx.extract_one(h)
        row_mat = fx.transform([h])[0]
        np.testing.assert_allclose(
            [row_dict[n] for n in fx.feature_names], row_mat
        )

    def test_call_alias_is_transform(self, fx):
        heads = ["one headline", "two headline"]
        np.testing.assert_array_equal(fx(heads), fx.transform(heads))

    def test_all_values_finite(self, fx):
        X = fx.transform(
            ["17 Things!!!", "Are you serious?", "", "   ", "!!!", "2024"]
        )
        assert np.all(np.isfinite(X))


# --------------------------------------------------------------------------- #
# Numeric / listicle cues                                                     #
# --------------------------------------------------------------------------- #
class TestNumericCues:
    def test_starts_with_number(self, fx):
        assert value(fx, "17 Things Only Dog Lovers Know", "starts_with_number") == 1.0
        assert value(fx, "The 17 Things You Missed", "starts_with_number") == 0.0

    def test_contains_number(self, fx):
        assert value(fx, "Top 5 tips", "contains_number") == 1.0
        assert value(fx, "No digits here", "contains_number") == 0.0

    def test_number_token_count(self, fx):
        assert value(fx, "7 and 8 and 9", "number_token_count") == 3.0

    def test_contains_year(self, fx):
        assert value(fx, "Best films of 2024", "contains_year") == 1.0
        assert value(fx, "Best films of 1850", "contains_year") == 0.0  # not 19xx/20xx
        assert value(fx, "Best films ever", "contains_year") == 0.0


# --------------------------------------------------------------------------- #
# Reader address / forward reference                                          #
# --------------------------------------------------------------------------- #
class TestAddressAndReference:
    def test_second_person(self, fx):
        d = fx.extract_one("What Your Zodiac Says About You")
        assert d["second_person_count"] == 2.0  # "Your", "You"
        assert d["has_second_person"] == 1.0

    def test_no_second_person(self, fx):
        d = fx.extract_one("Senate Approves Budget Deal")
        assert d["second_person_count"] == 0.0
        assert d["has_second_person"] == 0.0

    def test_first_person(self, fx):
        assert value(fx, "Why I Quit My Job", "first_person_count") == 2.0  # I, My

    def test_demonstrative_and_forward_reference(self, fx):
        d = fx.extract_one("This Is Why Dogs Are Great")
        assert d["demonstrative_count"] == 1.0
        assert d["has_forward_reference"] == 1.0


# --------------------------------------------------------------------------- #
# Clickbait phrase lexicon                                                    #
# --------------------------------------------------------------------------- #
class TestClickbaitPhrases:
    def test_single_phrase(self, fx):
        d = fx.extract_one("This Will Make You Cry")
        assert d["clickbait_phrase_count"] >= 1.0
        assert d["has_clickbait_phrase"] == 1.0

    def test_no_phrase(self, fx):
        d = fx.extract_one("Senate Approves Budget Deal After Talks")
        assert d["clickbait_phrase_count"] == 0.0
        assert d["has_clickbait_phrase"] == 0.0

    @pytest.mark.parametrize("headline", [
        "Every Day Brings Several Changes To Everything",
        "There's A New Budget Deal",
        "There Is A New Budget Deal",
        "There Are New Budget Measures",
        "Unranked Teams Win Again",
        "Whichever Candidate Wins The Election",
    ])
    def test_phrases_do_not_match_inside_other_words(self, fx, headline):
        d = fx.extract_one(headline)
        assert d["clickbait_phrase_count"] == 0.0
        assert d["has_clickbait_phrase"] == 0.0

    @pytest.mark.parametrize("headline", [
        '"Ever!"', "Here's A Dog", "The Teams, Ranked", "Which Dog Are You?",
    ])
    def test_phrases_match_whole_words_at_punctuation(self, fx, headline):
        d = fx.extract_one(headline)
        assert d["clickbait_phrase_count"] == 1.0
        assert d["has_clickbait_phrase"] == 1.0

    def test_distinct_phrases_count_once_each(self, fx):
        assert value(fx, "Ever, ever, ever", "clickbait_phrase_count") == 1.0
        # Overlapping lexicon entries are still distinct cues.
        assert value(fx, "Reasons why", "clickbait_phrase_count") == 2.0

    @pytest.mark.parametrize("headline", ["Which", "Which?"])
    def test_which_trigger_requires_trailing_space(self, fx, headline):
        assert value(fx, headline, "clickbait_phrase_count") == 0.0

    def test_smart_apostrophes_match_the_same_phrases(self, fx):
        plain = fx.extract_one("You won't believe what happened")
        smart = fx.extract_one("You won’t believe what happened")
        assert smart["clickbait_phrase_count"] == plain["clickbait_phrase_count"] == 2.0
        assert smart["has_clickbait_phrase"] == plain["has_clickbait_phrase"] == 1.0


# --------------------------------------------------------------------------- #
# Interrogative structure                                                     #
# --------------------------------------------------------------------------- #
class TestInterrogative:
    def test_is_question(self, fx):
        assert value(fx, "Are You Smart Enough?", "is_question") == 1.0
        assert value(fx, "You Are Smart Enough", "is_question") == 0.0

    def test_starts_with_question_word(self, fx):
        assert value(fx, "Which Dog Are You?", "starts_with_question_word") == 1.0
        assert value(fx, "Dogs Are Great", "starts_with_question_word") == 0.0

    def test_starts_with_auxiliary(self, fx):
        assert value(fx, "Can You Guess The Answer?", "starts_with_auxiliary") == 1.0
        assert value(fx, "People Guess The Answer", "starts_with_auxiliary") == 0.0


# --------------------------------------------------------------------------- #
# Hyperbole                                                                   #
# --------------------------------------------------------------------------- #
class TestHyperbole:
    def test_superlative_lexicon(self, fx):
        assert value(fx, "The Best Day Ever", "superlative_count") == 2.0  # best, ever

    @pytest.mark.parametrize("word", ["tallest", "fastest", "largest", "happiest"])
    def test_regular_superlatives(self, fx, word):
        assert value(fx, f"The {word} example", "superlative_count") == 1.0

    @pytest.mark.parametrize("word", ["forest", "arrest", "protest", "honest", "guest"])
    def test_est_suffix_is_not_enough_for_a_superlative(self, fx, word):
        assert value(fx, word, "superlative_count") == 0.0

    def test_superlatives_count_each_occurrence_once(self, fx):
        assert value(fx, "The Best And Best And Tallest", "superlative_count") == 3.0

    def test_intensifier(self, fx):
        assert value(fx, "This Is Literally Insane", "intensifier_count") == 1.0


# --------------------------------------------------------------------------- #
# Punctuation / style                                                         #
# --------------------------------------------------------------------------- #
class TestPunctuation:
    def test_exclamation_count(self, fx):
        assert value(fx, "Wow! Amazing!!", "exclamation_count") == 3.0

    def test_question_mark_count(self, fx):
        assert value(fx, "Really?? Are you sure?", "question_mark_count") == 3.0

    def test_has_ellipsis(self, fx):
        assert value(fx, "You Won't Believe What Happened...", "has_ellipsis") == 1.0
        assert value(fx, "A normal headline", "has_ellipsis") == 0.0

    def test_double_quote_count(self, fx):
        assert value(fx, 'He Said "Hello" Loudly', "double_quote_count") == 2.0

    def test_has_colon(self, fx):
        assert value(fx, "Breaking: Market Falls", "has_colon") == 1.0
        assert value(fx, "Market Falls Sharply", "has_colon") == 0.0


# --------------------------------------------------------------------------- #
# Length                                                                      #
# --------------------------------------------------------------------------- #
class TestLength:
    def test_word_and_char_count(self, fx):
        d = fx.extract_one("Three word headline")
        assert d["word_count"] == 3.0
        assert d["char_count"] == len("Three word headline")

    def test_avg_word_length(self, fx):
        # "ab cde" -> mean(2, 3) = 2.5
        assert value(fx, "ab cde", "avg_word_length") == 2.5

    def test_long_word_ratio(self, fx):
        # "extraordinary cat" -> 1 of 2 tokens has length > 6
        assert value(fx, "extraordinary cat", "long_word_ratio") == 0.5


# --------------------------------------------------------------------------- #
# Capitalization                                                              #
# --------------------------------------------------------------------------- #
class TestCapitalization:
    def test_title_case_ratio_all_caps_words(self, fx):
        # Every alphabetic word capitalized -> ratio 1.0 (BuzzFeed style).
        assert value(fx, "You Won't Believe This Dog", "title_case_ratio") == 1.0

    def test_title_case_ratio_sentence_case(self, fx):
        # 7 of 8 alphabetic words capitalized -> 0.875 (news-agency style).
        r = value(fx, "Senate Approves Budget Deal After Months of Negotiation",
                  "title_case_ratio")
        assert r == pytest.approx(0.875)

    def test_all_caps_word_count(self, fx):
        assert value(fx, "BREAKING NEWS today", "all_caps_word_count") == 2.0


# --------------------------------------------------------------------------- #
# Informal register / affect / function words                                #
# --------------------------------------------------------------------------- #
class TestRegisterAffectFunction:
    def test_slang_count(self, fx):
        assert value(fx, "This Is Cute AF OMG", "slang_count") == 2.0

    def test_affect_counts(self, fx):
        d = fx.extract_one("An Amazing And Shocking Story")
        assert d["positive_word_count"] == 1.0   # amazing
        assert d["negative_word_count"] == 1.0   # shocking
        assert d["emotional_word_count"] == 2.0

    def test_stopword_ratio_bounds(self, fx):
        r = value(fx, "the of a and in", "stopword_ratio")
        assert r == 1.0
        r2 = value(fx, "Senate Approves Budget", "stopword_ratio")
        assert 0.0 <= r2 < 0.5


# --------------------------------------------------------------------------- #
# Edge cases                                                                  #
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    @pytest.mark.parametrize("headline", ["", "   ", "!!!", "...", "?", "2024"])
    def test_degenerate_headlines_do_not_crash(self, fx, headline):
        X = fx.transform([headline])
        assert X.shape == (1, EXPECTED_N_FEATURES)
        assert np.all(np.isfinite(X))

    def test_empty_headline_is_all_zeroish(self, fx):
        d = fx.extract_one("")
        # Ratios/counts over no tokens must be 0, not NaN.
        for k in ("word_count", "title_case_ratio", "avg_word_length",
                  "stopword_ratio", "long_word_ratio"):
            assert d[k] == 0.0

    def test_determinism(self, fx):
        h = "You Won't Believe These 7 Things!"
        np.testing.assert_array_equal(fx.transform([h]), fx.transform([h]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
