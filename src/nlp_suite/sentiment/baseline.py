"""Classical (TF-IDF + linear classifier) sentiment baseline, trained per language.

One model per language rather than one shared model: vocabulary, stop-words,
and even what counts as a "token" differ between English and Spanish (see
``src/preprocessing.py``), so pooling languages into a single TF-IDF vocabulary
would dilute both. Each language gets its own `TfidfVectorizer` + `LogisticRegression`
pipeline, trained and evaluated independently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import Preprocessor

LABELS = ("negative", "positive")


class BaselineError(RuntimeError):
    """Raised when the dataset is missing, malformed, or too small to evaluate."""


@dataclass(frozen=True)
class LabeledExample:
    text: str
    label: str
    language: str


@dataclass(frozen=True)
class EvalResult:
    language: str
    n_train: int
    n_test: int
    accuracy: float
    f1_macro: float


def load_labeled_dataset(path: str | Path) -> list[LabeledExample]:
    """Load a JSONL file of ``{"text", "label", "lang"}`` records.

    ``lang`` is optional per record; if absent it's inferred via
    ``preprocessing.detect_language``. Records with an unrecognized label are
    rejected loudly rather than silently dropped, since a label typo would
    otherwise silently shrink one class of the training data.
    """
    from .preprocessing import detect_language

    path = Path(path)
    if not path.is_file():
        raise BaselineError(f"dataset file not found: {path}")

    examples: list[LabeledExample] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BaselineError(f"line {i + 1}: invalid JSON: {exc}") from exc
        text = raw.get("text")
        label = raw.get("label")
        if not text or label not in LABELS:
            raise BaselineError(
                f"line {i + 1}: expected non-empty 'text' and label in {LABELS}, got {raw!r}"
            )
        lang = raw.get("lang") or detect_language(text)
        examples.append(LabeledExample(text=text, label=label, language=lang))
    return examples


def build_pipeline() -> Pipeline:
    """One fresh TF-IDF + LogisticRegression pipeline (untrained)."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train_and_evaluate(
    examples: list[LabeledExample],
    language: str,
    *,
    test_size: float = 0.25,
    random_state: int = 0,
) -> tuple[Pipeline, EvalResult]:
    """Train + evaluate one language's baseline on a held-out split."""
    subset = [e for e in examples if e.language == language]
    if len(subset) < 4:
        raise BaselineError(
            f"not enough examples for language {language!r} to train/evaluate "
            f"(got {len(subset)}, need at least 4)"
        )

    texts = [e.text for e in subset]
    labels = [e.label for e in subset]
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    preprocessor = Preprocessor()
    x_train_norm = [preprocessor.normalize(t) for t in x_train]
    x_test_norm = [preprocessor.normalize(t) for t in x_test]

    pipeline = build_pipeline()
    pipeline.fit(x_train_norm, y_train)
    predictions = pipeline.predict(x_test_norm)

    result = EvalResult(
        language=language,
        n_train=len(x_train),
        n_test=len(x_test),
        accuracy=float(accuracy_score(y_test, predictions)),
        f1_macro=float(f1_score(y_test, predictions, average="macro")),
    )
    return pipeline, result


def train_and_evaluate_all(
    examples: list[LabeledExample],
    languages: tuple[str, ...] = ("en", "es"),
    **kwargs,
) -> dict[str, EvalResult]:
    return {lang: train_and_evaluate(examples, lang, **kwargs)[1] for lang in languages}
