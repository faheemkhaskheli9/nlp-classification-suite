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
EXTENDED_CSV = Path(__file__).resolve().parents[2] / "examples" / "quora_extended_sample.csv"


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


def test_extended_sample_has_more_rows_and_different_balance_than_bundled_sample():
    """The larger dataset (issue #2's stand-in for the real Kaggle export,
    since that needs a Kaggle account/API token and is too large for CI)
    must actually be larger and differently balanced -- otherwise it's not
    exercising anything --data on the bundled sample doesn't already cover.
    """
    sample_texts, sample_labels = load_dataset(SAMPLE_CSV)
    ext_texts, ext_labels = load_dataset(EXTENDED_CSV)

    assert len(ext_texts) > len(sample_texts) * 3
    sample_positive_rate = sum(sample_labels) / len(sample_labels)
    ext_positive_rate = sum(ext_labels) / len(ext_labels)
    assert abs(sample_positive_rate - ext_positive_rate) > 0.1


def test_training_on_extended_dataset_gives_metrics_distinct_from_bundled_sample():
    sample_texts, sample_labels = load_dataset(SAMPLE_CSV)
    ext_texts, ext_labels = load_dataset(EXTENDED_CSV)

    _, sample_metrics = train_baseline(sample_texts, sample_labels)
    _, ext_metrics = train_baseline(ext_texts, ext_labels)

    # Both datasets are near-trivially separable so precision/recall/f1 can
    # tie at a perfect 1.0 on either -- the held-out split size (driven by
    # dataset size) is what must differ, and it flows into a different
    # confusion matrix even when the rates match.
    assert sample_metrics.support != ext_metrics.support
    assert sample_metrics.confusion.as_dict() != ext_metrics.confusion.as_dict()


def test_predict_returns_binary_labels():
    texts, labels = load_dataset(SAMPLE_CSV)
    model, _ = train_baseline(texts, labels)

    preds = model.predict(["Isn't it obvious that all of them are just fools?", "How do I bake bread at home?"])

    assert set(preds) <= {0, 1}
    assert len(preds) == 2
