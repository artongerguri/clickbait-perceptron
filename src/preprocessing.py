"""Lightweight text preprocessing for clickbait headlines.

Design note
-----------
Unlike most text-classification pipelines, we deliberately keep the *original*
casing and punctuation of each headline. Capitalization style (e.g. BuzzFeed's
"Title Case Every Word" vs. a news agency's sentence case) and punctuation
(``!``, ``?``, ``...``) are among the strongest *stylistic* clickbait cues, so
destroying them during cleaning would throw away signal. Normalization here is
therefore minimal and is only used where a case-insensitive lexical match is
needed (e.g. matching the word "you").
"""

from __future__ import annotations

import re
import unicodedata

# A word is a maximal run of letters/digits plus a few intra-word symbols
# (apostrophes for "you'll", hyphens for "game-changing"). Emoji and stray
# punctuation are handled separately by the feature extractor.
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")


def normalize_unicode(text: str) -> str:
    """Normalize to NFKC so visually identical characters compare equal."""
    return unicodedata.normalize("NFKC", text)


def clean(text: str) -> str:
    """Trim surrounding whitespace and collapse internal runs of whitespace.

    Casing and punctuation are intentionally preserved.
    """
    text = normalize_unicode(str(text))
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split a headline into case-preserving word tokens."""
    return _WORD_RE.findall(clean(text))


def tokenize_lower(text: str) -> list[str]:
    """Lowercased tokens, for case-insensitive lexical matching."""
    return [t.lower() for t in tokenize(text)]


def normalized_text_lower(text: str) -> str:
    """A cleaned, lowercased string, for phrase/substring lexicon matching."""
    return clean(text).lower()
