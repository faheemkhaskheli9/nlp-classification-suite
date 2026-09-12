"""Tests for the shared text preprocessing pipeline (issue #1)."""

from __future__ import annotations

import pytest

from nlp_suite.quora.preprocessing import (
    TextPreprocessor,
    clean_text,
    preprocess,
    tokenize,
)


@pytest.mark.parametrize("value", ["", None, "   ", "\n\t "])
def test_empty_like_inputs_yield_empty(value):
    assert clean_text(value) == ""
    assert tokenize(value) == []


def test_lowercases_and_collapses_whitespace():
    assert clean_text("What  IS\tThis??\n") == "what is this??"


def test_strips_http_and_www_urls_but_keeps_surrounding_words():
    text = "check http://spam.example.com/x?a=1 and www.foo.bar now"
    assert clean_text(text) == "check and now"


def test_does_not_touch_comparison_operators_by_default():
    text = "is a < b and eGFR <60 in Python?"
    cleaned = clean_text(text)
    assert "<" in cleaned
    assert "a < b" in cleaned
    assert "<60" in cleaned


def test_strip_html_removes_real_tags_only():
    text = "a <b>bold</b> claim that a < b and 3 > 2"
    cleaned = clean_text(text, strip_html_tags=True)
    assert "<b>" not in cleaned and "</b>" not in cleaned
    assert "bold" in cleaned
    # bare comparison operators survive tag stripping
    assert "a < b" in cleaned
    assert "3 > 2" in cleaned


def test_unescapes_html_entities():
    assert clean_text("fish &amp; chips &lt;3") == "fish & chips <3"


def test_non_ascii_letters_are_preserved_as_tokens():
    assert tokenize("Beyoncé's café — Zürich") == ["beyoncé's", "café", "zürich"]


def test_normalizes_unicode_compatibility_forms():
    # U+FF21..: fullwidth 'ABC' -> ascii 'abc'
    assert clean_text("ＡＢＣ") == "abc"


def test_tokenizer_keeps_internal_apostrophes_and_hyphens():
    assert tokenize("don't state-of-the-art") == ["don't", "state-of-the-art"]


def test_preprocess_is_tokenize_alias():
    assert preprocess("Hello World") == tokenize("Hello World") == ["hello", "world"]


def test_same_instance_reused_across_batch_is_consistent():
    pre = TextPreprocessor()
    batch = ["The CAT.", "the cat.", "  the   cat . "]
    out = pre.transform(batch)
    assert out[0] == out[1]
    assert set(out) == {"the cat.", "the cat ."}


def test_keep_case_override():
    assert clean_text("MixedCase", lowercase=False) == "MixedCase"


def test_accepts_non_string_gracefully():
    assert clean_text(12345) == "12345"
