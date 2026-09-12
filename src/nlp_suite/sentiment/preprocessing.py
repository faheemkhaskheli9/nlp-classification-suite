"""Language-aware text preprocessing for English and Spanish.

Design intent: give every downstream model (classical baseline in Phase 1,
transformer in Phase 2) a consistent, Unicode-safe token stream regardless of
language. Accents are **preserved by default** -- in Spanish they are
meaningful content (``ano`` vs ``año``), so a blanket accent-stripping step
would corrupt the text (robustness rule 5).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

SUPPORTED_LANGUAGES = ("en", "es")

# Small, generic public stop-word lists (not exhaustive -- enough for language
# ID and for the classical baseline's optional stop-word filter).
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        "a an the and or but if then this that these those i you he she it we they "
        "is are was were be been being to of in on for with as at by from not no "
        "do does did have has had will would can could should".split()
    ),
    "es": frozenset(
        "el la los las un una unos unas y o pero si entonces este esta estos estas "
        "yo tu el ella nosotros ellos es son era eran ser a de en con como por para "
        "no ni que se su sus lo le me mi mas muy ya".split()
    ),
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_REPEAT_RE = re.compile(r"(.)\1{2,}")  # 3+ repeats -> keep 2
# Token = run of Unicode word chars, allowing internal apostrophe/hyphen.
_TOKEN_RE = re.compile(r"[^\W\d_](?:[\w'\-´’]*[^\W\d_])?", re.UNICODE)
_ES_MARKERS = re.compile(r"[ñáéíóúü¿¡]", re.IGNORECASE)


@dataclass(frozen=True)
class PreprocessConfig:
    lowercase: bool = True
    strip_accents: bool = False  # keep accents -- meaningful in Spanish
    remove_urls: bool = True
    remove_mentions: bool = True
    unwrap_hashtags: bool = True  # "#GreatDay" -> "GreatDay"
    collapse_repeats: bool = True  # "sooooo" -> "soo"
    remove_stopwords: bool = False
    min_token_len: int = 2


@dataclass(frozen=True)
class ProcessedDoc:
    language: str
    normalized_text: str
    tokens: list[str] = field(default_factory=list)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def detect_language(text: str) -> str:
    """Lightweight EN/ES heuristic; returns an entry of SUPPORTED_LANGUAGES or 'unknown'.

    Uses Spanish-specific characters plus stop-word overlap. Good enough for
    routing; Phase 2 can swap in a statistical detector behind this function.
    """
    norm = unicodedata.normalize("NFC", text).lower()
    if not norm.strip():
        return "unknown"
    if _ES_MARKERS.search(norm):
        return "es"
    words = set(re.findall(r"[^\W\d_]+", norm, re.UNICODE))
    if not words:
        return "unknown"
    en_hits = len(words & _STOPWORDS["en"])
    es_hits = len(words & _STOPWORDS["es"])
    if en_hits == es_hits == 0:
        return "unknown"
    return "en" if en_hits >= es_hits else "es"


class Preprocessor:
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def normalize(self, text: str) -> str:
        cfg = self.config
        out = unicodedata.normalize("NFC", text)
        if cfg.remove_urls:
            out = _URL_RE.sub(" ", out)
        if cfg.remove_mentions:
            out = _MENTION_RE.sub(" ", out)
        if cfg.unwrap_hashtags:
            out = _HASHTAG_RE.sub(r"\1", out)
        if cfg.collapse_repeats:
            out = _REPEAT_RE.sub(r"\1\1", out)
        if cfg.lowercase:
            out = out.lower()
        if cfg.strip_accents:
            out = _strip_accents(out)
        return re.sub(r"\s+", " ", out).strip()

    def tokenize(self, text: str, language: str) -> list[str]:
        cfg = self.config
        tokens = [t for t in _TOKEN_RE.findall(text) if len(t) >= cfg.min_token_len]
        if cfg.remove_stopwords:
            stops = _STOPWORDS.get(language, frozenset())
            tokens = [t for t in tokens if t.lower() not in stops]
        return tokens

    def process(self, text: str, language: str | None = None) -> ProcessedDoc:
        lang = (language or detect_language(text) or "unknown").lower()
        if language is not None and lang not in SUPPORTED_LANGUAGES and lang != "unknown":
            raise ValueError(
                f"unsupported language {lang!r}; supported: {SUPPORTED_LANGUAGES}"
            )
        normalized = self.normalize(text)
        tokens = self.tokenize(normalized, lang)
        return ProcessedDoc(language=lang, normalized_text=normalized, tokens=tokens)
