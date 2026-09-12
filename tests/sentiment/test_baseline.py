"""Tests for the classical TF-IDF + linear classifier baseline."""
from __future__ import annotations

import json

import pytest

from nlp_suite.sentiment.baseline import (
    BaselineError,
    load_labeled_dataset,
    train_and_evaluate,
    train_and_evaluate_all,
)

EN_POS = [
    "I love this product, it is fantastic and wonderful",
    "Great experience, everything worked perfectly and I am happy",
    "Amazing quality, I would buy this again without hesitation",
    "Excellent service, fast shipping and a wonderful product",
    "This is the best purchase I have made, truly excellent",
    "Fantastic! Everything about this exceeded my expectations",
    "Wonderful product, I am extremely satisfied with it",
    "I am delighted with this purchase, it works great",
]
EN_NEG = [
    "I hate this product, it is terrible and awful",
    "Horrible experience, nothing worked and I am upset",
    "Awful quality, I would never buy this again",
    "Terrible service, slow shipping and a broken product",
    "This is the worst purchase I have made, truly awful",
    "Disappointing! Everything about this fell short of expectations",
    "Horrible product, I am extremely unsatisfied with it",
    "I am frustrated with this purchase, it barely works",
]
ES_POS = [
    "Me encanta este producto, es fantástico y maravilloso",
    "Gran experiencia, todo funcionó perfectamente y estoy feliz",
    "Calidad increíble, lo compraría de nuevo sin dudar",
    "Excelente servicio, envío rápido y un producto maravilloso",
    "Esta es la mejor compra que he hecho, realmente excelente",
    "¡Fantástico! Todo superó mis expectativas",
    "Producto maravilloso, estoy muy satisfecho con él",
    "Estoy encantado con esta compra, funciona genial",
]
ES_NEG = [
    "Odio este producto, es terrible y espantoso",
    "Experiencia horrible, nada funcionó y estoy molesto",
    "Calidad espantosa, nunca lo volvería a comprar",
    "Servicio terrible, envío lento y un producto roto",
    "Esta es la peor compra que he hecho, realmente espantosa",
    "¡Decepcionante! Todo quedó por debajo de mis expectativas",
    "Producto horrible, estoy muy insatisfecho con él",
    "Estoy frustrado con esta compra, apenas funciona",
]


def _write_dataset(tmp_path):
    path = tmp_path / "dataset.jsonl"
    rows = (
        [{"text": t, "label": "positive", "lang": "en"} for t in EN_POS]
        + [{"text": t, "label": "negative", "lang": "en"} for t in EN_NEG]
        + [{"text": t, "label": "positive", "lang": "es"} for t in ES_POS]
        + [{"text": t, "label": "negative", "lang": "es"} for t in ES_NEG]
    )
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def test_load_labeled_dataset_infers_missing_language(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"text": "I love this!", "label": "positive"}) + "\n",
        encoding="utf-8",
    )
    examples = load_labeled_dataset(path)
    assert examples[0].language == "en"


def test_load_labeled_dataset_rejects_bad_label(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"text": "hello", "label": "meh"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(BaselineError):
        load_labeled_dataset(path)


def test_load_labeled_dataset_missing_file():
    with pytest.raises(BaselineError):
        load_labeled_dataset("does/not/exist.jsonl")


def test_train_and_evaluate_reports_accuracy_and_f1_per_language(tmp_path):
    dataset = load_labeled_dataset(_write_dataset(tmp_path))

    for lang in ("en", "es"):
        _, result = train_and_evaluate(dataset, lang, test_size=0.25, random_state=0)
        assert result.language == lang
        assert result.n_train > 0 and result.n_test > 0
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.f1_macro <= 1.0


def test_train_and_evaluate_all_covers_english_and_spanish(tmp_path):
    dataset = load_labeled_dataset(_write_dataset(tmp_path))
    results = train_and_evaluate_all(dataset, languages=("en", "es"))
    assert set(results) == {"en", "es"}


def test_train_and_evaluate_raises_on_too_little_data(tmp_path):
    path = tmp_path / "tiny.jsonl"
    path.write_text(
        json.dumps({"text": "great", "label": "positive", "lang": "en"}) + "\n",
        encoding="utf-8",
    )
    dataset = load_labeled_dataset(path)
    with pytest.raises(BaselineError):
        train_and_evaluate(dataset, "en")


def test_shipped_dataset_trains_both_languages():
    """Smoke test against the checked-in example dataset."""
    from pathlib import Path

    dataset_path = Path(__file__).resolve().parents[2] / "examples" / "sentiment_dataset.jsonl"
    dataset = load_labeled_dataset(dataset_path)
    results = train_and_evaluate_all(dataset)
    assert set(results) == {"en", "es"}
    for result in results.values():
        assert result.n_test > 0
