"""TF-IDF + Logistic Regression baseline classifier (Phase 1).

Trains on a CSV of ``(question_text, target)`` rows -- the real project uses
the public Kaggle "Quora Insincere Questions Classification" dataset, which
needs a Kaggle account/API token to download and is far too large to ship or
fetch in CI. ``examples/sample_questions.csv`` is a small, self-authored,
public-domain substitute with the same two-column schema (30 sincere / 30
insincere-style questions) so the pipeline is runnable and testable without
external credentials; point ``--data`` at the real Kaggle CSV to train on it
once downloaded locally.

The fitted vectorizer + classifier are saved together as one artifact so
inference only needs one file to reload.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .evaluation import EvalMetrics, evaluate_predictions
from .preprocessing import TextPreprocessor

__all__ = [
    "EvalMetrics",
    "TfidfLogisticRegressionModel",
    "load_dataset",
    "train_baseline",
]


def load_dataset(
    path: str | os.PathLike[str],
    *,
    text_col: str = "question_text",
    target_col: str = "target",
) -> tuple[list[str], list[int]]:
    """Read a ``(text_col, target_col)`` CSV into parallel lists."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    texts: list[str] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {text_col, target_col} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing column(s) {sorted(missing)} "
                f"(has: {', '.join(reader.fieldnames or [])})"
            )
        for row in reader:
            texts.append(row[text_col])
            labels.append(int(row[target_col]))

    if not texts:
        raise ValueError(f"{path} has no data rows")
    return texts, labels


@dataclass
class TfidfLogisticRegressionModel:
    """A fitted TF-IDF vectorizer + Logistic Regression classifier, bundled.

    ``save``/``load`` keep the two fitted objects together as one artifact
    (a dict pickled via joblib) so inference never risks pairing a vectorizer
    with the wrong classifier.
    """

    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    preprocessor: TextPreprocessor

    def predict(self, texts: list[str]) -> list[int]:
        cleaned = self.preprocessor.transform(texts)
        features = self.vectorizer.transform(cleaned)
        return list(self.classifier.predict(features))

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the fitted model to ``path``, atomically.

        Writes to a temp file in the same directory and ``os.replace()``s it
        onto the target, so a killed/interrupted save never leaves a
        truncated, unreloadable artifact at the final path.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            joblib.dump(
                {
                    "vectorizer": self.vectorizer,
                    "classifier": self.classifier,
                    "preprocessor": self.preprocessor,
                },
                tmp,
            )
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "TfidfLogisticRegressionModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        bundle = joblib.load(path)
        return cls(
            vectorizer=bundle["vectorizer"],
            classifier=bundle["classifier"],
            preprocessor=bundle["preprocessor"],
        )


def train_baseline(
    texts: list[str],
    labels: list[int],
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    max_features: int | None = 5000,
) -> tuple[TfidfLogisticRegressionModel, EvalMetrics]:
    """Fit the TF-IDF + Logistic Regression baseline and score it on a held-out split.

    Stratifies the split on the label so a small/imbalanced dataset still
    gets both classes in train and test.
    """
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.transform(texts)

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(x_train_vec, y_train)

    predictions = list(classifier.predict(x_test_vec))
    metrics = evaluate_predictions(list(y_test), predictions)

    model = TfidfLogisticRegressionModel(
        vectorizer=vectorizer, classifier=classifier, preprocessor=preprocessor
    )
    return model, metrics
