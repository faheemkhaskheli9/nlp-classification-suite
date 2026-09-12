import pytest

from nlp_suite.quora.evaluation import evaluate_predictions


def test_perfect_predictions_score_1_0_everywhere():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.support == 4
    assert metrics.confusion.true_positives == 2
    assert metrics.confusion.true_negatives == 2
    assert metrics.confusion.false_positives == 0
    assert metrics.confusion.false_negatives == 0


def test_confusion_matrix_counts_each_outcome_correctly():
    #        true: 0  0  1  1  1
    y_true = [0, 0, 1, 1, 1]
    y_pred = [1, 0, 1, 0, 0]
    #  -> TN=1 (idx1), FP=1 (idx0), TP=1 (idx2), FN=2 (idx3,4)
    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.confusion.true_negatives == 1
    assert metrics.confusion.false_positives == 1
    assert metrics.confusion.true_positives == 1
    assert metrics.confusion.false_negatives == 2


def test_zero_division_does_not_raise_when_no_positive_predictions():
    y_true = [0, 0, 1]
    y_pred = [0, 0, 0]
    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_mismatched_lengths_raise_value_error():
    with pytest.raises(ValueError):
        evaluate_predictions([0, 1], [0, 1, 1])


def test_empty_inputs_raise_value_error():
    with pytest.raises(ValueError):
        evaluate_predictions([], [])


def test_as_markdown_row_includes_model_name_and_all_metrics():
    metrics = evaluate_predictions([0, 1], [0, 1])
    row = metrics.as_markdown_row(model_name="baseline-tfidf-logreg")

    assert "baseline-tfidf-logreg" in row
    assert "1.000" in row  # precision/recall/f1 all 1.0
    assert "TP=1" in row and "TN=1" in row


def test_as_dict_round_trips_confusion_matrix():
    metrics = evaluate_predictions([0, 1], [0, 0])
    d = metrics.as_dict()

    assert d["confusion_matrix"]["false_negatives"] == 1
    assert d["support"] == 2
