"""Tests for Phase 2 multilingual transformer fine-tuning.

Uses the same tiny public XLM-R-architecture checkpoint as
``src/finetune.DEFAULT_MODEL_NAME`` so the whole suite stays fast and
network-dependent-but-CPU-only, matching this repo's own documented budget.
"""
from __future__ import annotations

import time

import pytest

from nlp_suite.sentiment.baseline import load_labeled_dataset
from nlp_suite.sentiment.finetune import (
    DOCUMENTED_TIME_BUDGET_SECONDS,
    FineTuneError,
    fine_tune,
    load_fine_tuned,
    predict,
)

DATASET_PATH = "examples/sentiment_dataset.jsonl"


@pytest.fixture(scope="module")
def examples():
    return load_labeled_dataset(DATASET_PATH)


def test_fine_tune_trains_on_combined_english_and_spanish_data(tmp_path_factory, examples):
    out_dir = tmp_path_factory.mktemp("ft") / "model"
    result = fine_tune(examples, out_dir, num_epochs=1)

    assert result.n_train > 0
    assert result.n_eval > 0
    assert 0.0 <= result.eval_accuracy <= 1.0
    assert 0.0 <= result.eval_f1_macro <= 1.0
    assert result.model_path == out_dir


def test_fine_tune_saves_a_reloadable_model_artifact(tmp_path_factory, examples):
    out_dir = tmp_path_factory.mktemp("ft") / "model"
    fine_tune(examples, out_dir, num_epochs=1)

    model, tokenizer = load_fine_tuned(out_dir)
    predictions = predict(model, tokenizer, ["I love this product", "Odio este producto"])
    assert len(predictions) == 2
    assert all(p in ("positive", "negative") for p in predictions)


def test_fine_tune_runs_within_the_documented_cpu_time_budget(tmp_path_factory, examples):
    out_dir = tmp_path_factory.mktemp("ft") / "model"
    start = time.monotonic()
    fine_tune(examples, out_dir, num_epochs=1)
    wall_clock = time.monotonic() - start
    assert wall_clock < DOCUMENTED_TIME_BUDGET_SECONDS


def test_fine_tune_rejects_too_few_examples(tmp_path_factory, examples):
    out_dir = tmp_path_factory.mktemp("ft") / "model"
    with pytest.raises(FineTuneError):
        fine_tune(examples[:2], out_dir, num_epochs=1)


def test_load_fine_tuned_missing_directory_is_a_hard_error(tmp_path):
    with pytest.raises(FineTuneError):
        load_fine_tuned(tmp_path / "does-not-exist")


def test_save_is_atomic_and_preserves_a_prior_artifact_on_failure(tmp_path_factory, examples, monkeypatch):
    out_dir = tmp_path_factory.mktemp("ft") / "model"
    fine_tune(examples, out_dir, num_epochs=1)
    assert out_dir.is_dir()
    before = sorted(p.name for p in out_dir.iterdir())

    import nlp_suite.sentiment.finetune as finetune_mod

    def boom(self, dst):  # noqa: ANN001
        raise RuntimeError("disk full")

    monkeypatch.setattr(finetune_mod.Path, "replace", boom)
    with pytest.raises(RuntimeError):
        fine_tune(examples, out_dir, num_epochs=1)

    # The old artifact must still be intact -- the failed save never touched it.
    assert out_dir.is_dir()
    assert sorted(p.name for p in out_dir.iterdir()) == before
