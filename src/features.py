"""Engineered lexical & stylistic features for clickbait detection.

This module is the analytical core of the project. Rather than feeding raw
bag-of-words/TF-IDF vectors to the classifier (a "black-box" approach), we hand
craft a compact set of *interpretable* numeric features, each of which encodes a
well-motivated linguistic cue. Because every feature has a human-readable name,
the weights the perceptron assigns to them can be read directly as "how strongly
does this cue signal clickbait", which is exactly the interpretability goal of
the project.

Feature families
----------------
1.  Numeric / listicle cues  — "17 Things ...", years, digit counts.
2.  Reader address           — second-/first-person pronouns.
3.  Forward reference         — demonstratives + "here" (cataphora).
4.  Clickbait phrase lexicon  — curated multi-word triggers.
5.  Interrogative structure   — question marks, question-word / auxiliary onset.
6.  Hyperbole                 — superlatives + intensifiers.
7.  Punctuation / style       — "!", "?", "...", quotes, colons.
8.  Length                    — word/char counts, average & long-word ratio.
9.  Capitalization            — title-case ratio, ALL-CAPS tokens.
10. Informal register         — internet slang (AF, OMG, WTF ...).
11. Affect                    — small positive/negative/emotion lexicons.
12. Function words            — stopword ratio (phrasal vs. headline-ese).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .preprocessing import clean, tokenize, tokenize_lower

# --------------------------------------------------------------------------- #
# Lexicons                                                                    #
# --------------------------------------------------------------------------- #

SECOND_PERSON = {
    "you", "your", "yours", "yourself", "yourselves",
    "you'll", "you're", "you've", "you'd", "u", "ur",
}
FIRST_PERSON = {
    "i", "we", "my", "our", "me", "us", "mine", "ours", "myself", "ourselves",
    "i'm", "we're", "i've", "we've", "i'll", "we'll",
}
DEMONSTRATIVES = {"this", "that", "these", "those"}
FORWARD_REF = DEMONSTRATIVES | {"here", "such"}

QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose", "whom"}
AUX_STARTERS = {
    "can", "could", "will", "would", "should", "shall", "do", "does", "did",
    "are", "is", "am", "was", "were", "have", "has", "had", "must", "may", "might",
}

INTENSIFIERS = {
    "literally", "actually", "totally", "seriously", "completely", "absolutely",
    "insanely", "ridiculously", "extremely", "incredibly", "really", "super",
    "utterly", "downright", "undeniably", "perfectly", "entirely", "wildly",
    "unbelievably", "surprisingly",
}
SUPERLATIVES = {
    "best", "worst", "most", "least", "greatest", "biggest", "smallest",
    "cutest", "funniest", "craziest", "hottest", "ever", "ultimate", "perfect",
    "hardest", "easiest", "coolest", "weirdest", "scariest", "saddest",
    "tallest", "shortest", "longest", "largest", "highest", "lowest",
    "oldest", "youngest", "newest", "latest", "earliest", "fastest", "slowest",
    "strongest", "weakest", "richest", "poorest", "cheapest", "safest",
    "happiest", "smartest", "prettiest", "ugliest", "strangest", "wildest",
}

SLANG = {
    "af", "omg", "wtf", "lol", "tbh", "yolo", "ftw", "lmao", "smh", "idk",
    "nsfw", "bae", "fomo", "fml", "irl", "rofl", "imo", "af's",
}

POSITIVE_WORDS = {
    "amazing", "awesome", "incredible", "perfect", "beautiful", "hilarious",
    "adorable", "stunning", "epic", "gorgeous", "genius", "brilliant",
    "heartwarming", "cute", "love", "lovely", "wonderful", "magical", "sweet",
    "inspiring", "flawless", "delicious", "iconic", "legendary",
}
NEGATIVE_WORDS = {
    "shocking", "heartbreaking", "terrifying", "horrifying", "awkward", "cringe",
    "hate", "crazy", "insane", "fail", "disgusting", "creepy", "worst",
    "embarrassing", "painful", "tragic", "disturbing", "scary", "sad", "brutal",
}
EMOTION_WORDS = POSITIVE_WORDS | NEGATIVE_WORDS

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "you", "your", "we", "our",
    "they", "their", "he", "she", "his", "her", "do", "does", "did", "will",
    "would", "can", "could", "should", "about", "into", "than", "then", "so",
    "if", "when", "while", "all", "some", "no", "not",
}

# Curated clickbait triggers, matched at word boundaries after lowercasing.
# Each distinct trigger contributes at most once, even if it appears repeatedly.
CLICKBAIT_PHRASES = [
    "will make you", "won't believe", "wont believe", "you need to",
    "need to know", "here's", "here is", "here are", "this is why",
    "the reason", "reasons why", "reasons", "things that", "things you",
    "things only", "what happens when", "what happened", "that will",
    "that prove", "that are too real", "too real", "ways to", "can you guess",
    "can we guess", "we know", "based on your", "based on the", "according to",
    "of all time", "right now", "that happened", "literally just",
    "blow your mind", "melt your heart", "should you", "which ", "that made",
    "in order of", "ranked", "that'll", "you probably", "you should",
    "that only", "tweets that", "pictures that", "photos that", "ever",
]

# Preserve trailing whitespace in entries such as "which ": that trigger needs
# a following space, while boundaries prevent "ever" matching inside "every".
_CLICKBAIT_PHRASE_PATTERNS = [
    re.compile(
        r"(?<!\w)" + re.escape(phrase)
        + (r"(?!\w)" if not phrase[-1].isspace() else "")
    )
    for phrase in CLICKBAIT_PHRASES
]

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_PUNCT_EXCL = "!"
_PUNCT_QMARK = "?"
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")
_DOUBLE_QUOTE_RE = re.compile(r'["“”]')


@dataclass
class _Parsed:
    """Precomputed views of a single headline shared across feature functions."""

    raw: str
    text: str            # cleaned, original casing
    lower: str           # cleaned, lowercased
    tokens: list[str]    # case-preserving word tokens
    tokens_lc: list[str] # lowercased word tokens


def _parse(headline: str) -> _Parsed:
    text = clean(headline)
    return _Parsed(
        raw=headline,
        text=text,
        lower=text.lower(),
        tokens=tokenize(text),
        tokens_lc=tokenize_lower(text),
    )


# --------------------------------------------------------------------------- #
# Feature extractor                                                           #
# --------------------------------------------------------------------------- #

class FeatureExtractor:
    """Turns raw headlines into a named, interpretable numeric feature matrix.

    The ``(name, function)`` registry guarantees that :pyattr:`feature_names`
    and :pymeth:`transform` can never drift out of sync.
    """

    def __init__(self) -> None:
        # Order here defines the column order of the output matrix.
        self._registry: list[tuple[str, callable]] = [
            # 1. Numeric / listicle
            ("starts_with_number", self._starts_with_number),
            ("contains_number", self._contains_number),
            ("number_token_count", self._number_token_count),
            ("contains_year", self._contains_year),
            # 2. Reader address
            ("second_person_count", self._second_person_count),
            ("has_second_person", self._has_second_person),
            ("first_person_count", self._first_person_count),
            # 3. Forward reference
            ("demonstrative_count", self._demonstrative_count),
            ("has_forward_reference", self._has_forward_reference),
            # 4. Clickbait phrase lexicon
            ("clickbait_phrase_count", self._clickbait_phrase_count),
            ("has_clickbait_phrase", self._has_clickbait_phrase),
            # 5. Interrogative structure
            ("is_question", self._is_question),
            ("starts_with_question_word", self._starts_with_question_word),
            ("starts_with_auxiliary", self._starts_with_auxiliary),
            # 6. Hyperbole
            ("superlative_count", self._superlative_count),
            ("intensifier_count", self._intensifier_count),
            # 7. Punctuation / style
            ("exclamation_count", self._exclamation_count),
            ("question_mark_count", self._question_mark_count),
            ("has_ellipsis", self._has_ellipsis),
            ("double_quote_count", self._double_quote_count),
            ("has_colon", self._has_colon),
            # 8. Length
            ("word_count", self._word_count),
            ("char_count", self._char_count),
            ("avg_word_length", self._avg_word_length),
            ("long_word_ratio", self._long_word_ratio),
            # 9. Capitalization
            ("title_case_ratio", self._title_case_ratio),
            ("all_caps_word_count", self._all_caps_word_count),
            ("all_caps_ratio", self._all_caps_ratio),
            # 10. Informal register
            ("slang_count", self._slang_count),
            # 11. Affect
            ("positive_word_count", self._positive_word_count),
            ("negative_word_count", self._negative_word_count),
            ("emotional_word_count", self._emotional_word_count),
            # 12. Function words
            ("stopword_ratio", self._stopword_ratio),
        ]

    # -- public API --------------------------------------------------------- #

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature column names."""
        return [name for name, _ in self._registry]

    @property
    def n_features(self) -> int:
        return len(self._registry)

    def extract_one(self, headline: str) -> dict[str, float]:
        """Return a ``{feature_name: value}`` dict for a single headline."""
        p = _parse(headline)
        return {name: float(fn(p)) for name, fn in self._registry}

    def transform(self, headlines) -> np.ndarray:
        """Vectorize an iterable of headlines into an ``(n, n_features)`` array."""
        rows = []
        for h in headlines:
            p = _parse(h)
            rows.append([fn(p) for _, fn in self._registry])
        return np.asarray(rows, dtype=np.float64).reshape(-1, self.n_features)

    # Convenience alias so the extractor is drop-in where a transform is wanted.
    __call__ = transform

    # -- feature functions (each returns a scalar) -------------------------- #

    # 1. Numeric / listicle
    @staticmethod
    def _starts_with_number(p: _Parsed) -> float:
        return 1.0 if p.tokens and p.tokens[0].isdigit() else 0.0

    @staticmethod
    def _contains_number(p: _Parsed) -> float:
        return 1.0 if any(c.isdigit() for c in p.text) else 0.0

    @staticmethod
    def _number_token_count(p: _Parsed) -> float:
        return float(sum(t.isdigit() for t in p.tokens))

    @staticmethod
    def _contains_year(p: _Parsed) -> float:
        return 1.0 if _YEAR_RE.search(p.text) else 0.0

    # 2. Reader address
    @staticmethod
    def _second_person_count(p: _Parsed) -> float:
        return float(sum(t in SECOND_PERSON for t in p.tokens_lc))

    def _has_second_person(self, p: _Parsed) -> float:
        return 1.0 if self._second_person_count(p) > 0 else 0.0

    @staticmethod
    def _first_person_count(p: _Parsed) -> float:
        return float(sum(t in FIRST_PERSON for t in p.tokens_lc))

    # 3. Forward reference
    @staticmethod
    def _demonstrative_count(p: _Parsed) -> float:
        return float(sum(t in DEMONSTRATIVES for t in p.tokens_lc))

    @staticmethod
    def _has_forward_reference(p: _Parsed) -> float:
        return 1.0 if any(t in FORWARD_REF for t in p.tokens_lc) else 0.0

    # 4. Clickbait phrase lexicon
    @staticmethod
    def _clickbait_phrase_count(p: _Parsed) -> float:
        return float(sum(pattern.search(p.lower) is not None
                         for pattern in _CLICKBAIT_PHRASE_PATTERNS))

    def _has_clickbait_phrase(self, p: _Parsed) -> float:
        return 1.0 if self._clickbait_phrase_count(p) > 0 else 0.0

    # 5. Interrogative structure
    @staticmethod
    def _is_question(p: _Parsed) -> float:
        return 1.0 if p.text.endswith("?") else 0.0

    @staticmethod
    def _starts_with_question_word(p: _Parsed) -> float:
        return 1.0 if p.tokens_lc and p.tokens_lc[0] in QUESTION_WORDS else 0.0

    @staticmethod
    def _starts_with_auxiliary(p: _Parsed) -> float:
        return 1.0 if p.tokens_lc and p.tokens_lc[0] in AUX_STARTERS else 0.0

    # 6. Hyperbole
    @staticmethod
    def _superlative_count(p: _Parsed) -> float:
        # A bare "-est" suffix also occurs in words like "forest" and "arrest".
        return float(sum(t in SUPERLATIVES for t in p.tokens_lc))

    @staticmethod
    def _intensifier_count(p: _Parsed) -> float:
        return float(sum(t in INTENSIFIERS for t in p.tokens_lc))

    # 7. Punctuation / style
    @staticmethod
    def _exclamation_count(p: _Parsed) -> float:
        return float(p.text.count(_PUNCT_EXCL))

    @staticmethod
    def _question_mark_count(p: _Parsed) -> float:
        return float(p.text.count(_PUNCT_QMARK))

    @staticmethod
    def _has_ellipsis(p: _Parsed) -> float:
        return 1.0 if _ELLIPSIS_RE.search(p.text) else 0.0

    @staticmethod
    def _double_quote_count(p: _Parsed) -> float:
        return float(len(_DOUBLE_QUOTE_RE.findall(p.text)))

    @staticmethod
    def _has_colon(p: _Parsed) -> float:
        return 1.0 if ":" in p.text else 0.0

    # 8. Length
    @staticmethod
    def _word_count(p: _Parsed) -> float:
        return float(len(p.tokens))

    @staticmethod
    def _char_count(p: _Parsed) -> float:
        return float(len(p.text))

    @staticmethod
    def _avg_word_length(p: _Parsed) -> float:
        if not p.tokens:
            return 0.0
        return float(np.mean([len(t) for t in p.tokens]))

    @staticmethod
    def _long_word_ratio(p: _Parsed) -> float:
        if not p.tokens:
            return 0.0
        return float(sum(len(t) > 6 for t in p.tokens) / len(p.tokens))

    # 9. Capitalization
    @staticmethod
    def _title_case_ratio(p: _Parsed) -> float:
        alpha = [t for t in p.tokens if t[:1].isalpha()]
        if not alpha:
            return 0.0
        capitalized = sum(t[0].isupper() for t in alpha)
        return float(capitalized / len(alpha))

    @staticmethod
    def _all_caps_word_count(p: _Parsed) -> float:
        return float(sum(t.isupper() and len(t) >= 2 for t in p.tokens))

    @staticmethod
    def _all_caps_ratio(p: _Parsed) -> float:
        if not p.tokens:
            return 0.0
        caps = sum(t.isupper() and len(t) >= 2 for t in p.tokens)
        return float(caps / len(p.tokens))

    # 10. Informal register
    @staticmethod
    def _slang_count(p: _Parsed) -> float:
        return float(sum(t in SLANG for t in p.tokens_lc))

    # 11. Affect
    @staticmethod
    def _positive_word_count(p: _Parsed) -> float:
        return float(sum(t in POSITIVE_WORDS for t in p.tokens_lc))

    @staticmethod
    def _negative_word_count(p: _Parsed) -> float:
        return float(sum(t in NEGATIVE_WORDS for t in p.tokens_lc))

    @staticmethod
    def _emotional_word_count(p: _Parsed) -> float:
        return float(sum(t in EMOTION_WORDS for t in p.tokens_lc))

    # 12. Function words
    @staticmethod
    def _stopword_ratio(p: _Parsed) -> float:
        if not p.tokens_lc:
            return 0.0
        return float(sum(t in STOPWORDS for t in p.tokens_lc) / len(p.tokens_lc))
