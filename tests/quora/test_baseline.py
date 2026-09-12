"""Tests for the TF-IDF + Logistic Regression baseline (issue #2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlp_suite.quora.baseline import (
    TfidfLogisticRegressionModel,
    load_dataset,
    train_baseline,
)

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "examples" / "sample_questions.csv"


def test_load_dataset_reads_text_and_labels():
    texts, labels = load_dataset(SAMPLE_CSV)

    assert len(texts) == len(labels) == 60
    assert set(labels) == {0, 1}
    assert texts[0].startswith("What is the best way")


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")


def test_load_dataset_missing_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,text\n1,hello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing column"):
        load_dataset(bad)


def test_train_baseline_reports_precision_recall_f1_on_held_out_split():
    texts, labels = load_dataset(SAMPLE_CSV)

    model, metrics = train_baseline(texts, labels)

    assert isinstance(model, TfidfLogisticRegressionModel)
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert metrics.support == pytest.approx(len(texts) * 0.25, abs=1)
    # The two classes here are near-trivially separable, so the baseline
    # should do meaningfully better than chance on the held-out split.
    assert metrics.f1 > 0.5


def test_model_saves_and_reloads_for_inference(tmp_path):
    texts, labels = load_dataset(SAMPLE_CSV)
    model, _ = train_baseline(texts, labels)

    out = tmp_path / "nested" / "baseline.joblib"
    saved_path = model.save(out)

    assert saved_path == out
    assert not any(p.name.startswith(".baseline.joblib") for p in out.parent.iterdir())

    reloaded = TfidfLogisticRegressionModel.load(out)
    original_preds = model.predict(texts)
    reloaded_preds = reloaded.predict(texts)

    assert reloaded_preds == original_preds


def test_load_missing_model_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TfidfLogisticRegressionModel.load(tmp_path / "missing.joblib")


def test_predict_returns_binary_labels():
    texts, labels = load_dataset(SAMPLE_CSV)
    model, _ = train_baseline(texts, labels)

    preds = model.predict(["Isn't it obvious that all of them are just fools?", "How do I bake bread at home?"])

    assert set(preds) <= {0, 1}
    assert len(preds) == 2
