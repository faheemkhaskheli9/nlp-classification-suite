"""Tests for language-aware preprocessing (English + Spanish)."""
from __future__ import annotations

import pytest

from nlp_suite.sentiment.preprocessing import (
    PreprocessConfig,
    Preprocessor,
    detect_language,
)


@pytest.fixture
def pre() -> Preprocessor:
    return Preprocessor()


def test_english_normalization_and_tokenization(pre):
    doc = pre.process("I LOVE this!! Visit https://x.example now @me", language="en")
    assert doc.language == "en"
    assert "http" not in doc.normalized_text
    assert "@me" not in doc.normalized_text
    assert "love" in doc.tokens and "this" in doc.tokens


def test_spanish_accents_and_enye_preserved(pre):
    doc = pre.process("El niño comió una manzana muy rápido", language="es")
    assert doc.language == "es"
    assert "niño" in doc.tokens
    assert "rápido" in doc.tokens
    assert "comió" in doc.tokens


def test_spanish_inverted_punctuation_stripped(pre):
    doc = pre.process("¡Qué decepción! ¿Por qué tardó tanto?", language="es")
    joined = " ".join(doc.tokens)
    assert "decepción" in doc.tokens
    assert "tardó" in doc.tokens
    assert "¡" not in joined and "¿" not in joined and "!" not in joined


def test_repeat_collapse_and_hashtag_unwrap(pre):
    doc = pre.process("soooo good #BestDay", language="en")
    assert "soo" in doc.tokens
    assert "bestday" in doc.tokens


def test_language_detection_english_vs_spanish():
    assert detect_language("This is a great movie and I really enjoyed it") == "en"
    assert detect_language("Esta película es muy buena y me gustó mucho") == "es"
    assert detect_language("El area es 42") == "es"  # 'el' Spanish stopword
    assert detect_language("") == "unknown"
    assert detect_language("42 999 --- ") == "unknown"


def test_detected_language_used_when_not_forced(pre):
    doc = pre.process("Me encanta la comida española")
    assert doc.language == "es"


def test_unsupported_forced_language_rejected(pre):
    with pytest.raises(ValueError, match="unsupported language"):
        pre.process("bonjour le monde", language="fr")


def test_strip_accents_option_when_explicitly_enabled():
    pre = Preprocessor(PreprocessConfig(strip_accents=True))
    doc = pre.process("rápido y niño", language="es")
    assert "rapido" in doc.tokens
    assert "nino" in doc.tokens


def test_stopword_removal_per_language():
    pre = Preprocessor(PreprocessConfig(remove_stopwords=True))
    en = pre.process("this is the best movie", language="en")
    assert "the" not in en.tokens and "best" in en.tokens
    es = pre.process("el perro es muy grande", language="es")
    assert "el" not in es.tokens and "perro" in es.tokens
