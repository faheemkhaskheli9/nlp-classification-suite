"""Shared text cleaning + tokenization for every model in this project.

The same :class:`TextPreprocessor` instance is meant to be reused by the
TF-IDF baseline and the later LSTM / Transformer pipelines, so they all see
an identical input representation.

Design note (question text is a scientific-ish domain): ``<`` and ``>`` are
frequently *content* in these questions -- "is a < b in Python?", "eGFR
<60" -- so HTML-tag stripping is off by default and, when enabled, only
matches things that actually look like tags (``<b>``, ``</div>``), never a
bare comparison operator.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["TextPreprocessor", "clean_text", "preprocess", "tokenize"]

# http(s):// or bare www. up to the next whitespace.
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# Real-looking HTML/XML tags only: '<', optional '/', a letter-led name,
# optional attributes with no angle brackets inside, '>'. Does NOT match
# "a < b" or "x <y".
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*?)?/?>")
_WS_RE = re.compile(r"\s+")
# Word tokens: unicode letters/digits with internal apostrophes/hyphens.
_WORD_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True)
class TextPreprocessor:
    """Configurable, reusable cleaner + tokenizer.

    Call it on a single string to get a cleaned string, or use
    :meth:`tokens` for a token list and :meth:`transform` for a batch.
    """

    lowercase: bool = True
    normalize_unicode: bool = True
    unescape_html: bool = True
    strip_urls: bool = True
    strip_html_tags: bool = False
    collapse_whitespace: bool = True

    def clean(self, text: str | None) -> str:
        if not text:
            return ""
        if not isinstance(text, str):
            text = str(text)

        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        if self.unescape_html:
            text = html.unescape(text)
        if self.strip_urls:
            text = _URL_RE.sub(" ", text)
        if self.strip_html_tags:
            text = _HTML_TAG_RE.sub(" ", text)
        if self.lowercase:
            text = text.lower()
        if self.collapse_whitespace:
            text = _WS_RE.sub(" ", text).strip()
        return text

    def tokens(self, text: str | None) -> list[str]:
        return _WORD_RE.findall(self.clean(text))

    def transform(self, texts: Iterable[str | None]) -> list[str]:
        return [self.clean(t) for t in texts]

    def transform_tokens(self, texts: Iterable[str | None]) -> list[list[str]]:
        return [self.tokens(t) for t in texts]

    def __call__(self, text: str | None) -> str:
        return self.clean(text)


_DEFAULT = TextPreprocessor()


def clean_text(text: str | None, **overrides: bool) -> str:
    """Clean one string with the default settings (or per-call overrides)."""
    preprocessor = TextPreprocessor(**overrides) if overrides else _DEFAULT
    return preprocessor.clean(text)


def tokenize(text: str | None, **overrides: bool) -> list[str]:
    """Clean then split one string into word tokens."""
    preprocessor = TextPreprocessor(**overrides) if overrides else _DEFAULT
    return preprocessor.tokens(text)


def preprocess(text: str | None, **overrides: bool) -> list[str]:
    """Alias for :func:`tokenize` -- the full clean+tokenize step."""
    return tokenize(text, **overrides)
