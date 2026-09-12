"""Shared classification evaluation, used by every model in this repo.

Kept in its own module (rather than duplicated inside each model's training
function) so the baseline, the LSTM, and the Transformer/BERT model are all
scored the same way and stay directly comparable in ``docs/evaluation.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

__all__ = ["ConfusionMatrix", "EvalMetrics", "evaluate_predictions"]


@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary confusion matrix counts (positive class = insincere = 1)."""

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def as_dict(self) -> dict[str, int]:
        return {
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_positives": self.true_positives,
        }


@dataclass(frozen=True)
class EvalMetrics:
    """Precision/recall/F1 for the positive (insincere) class, plus the confusion matrix."""

    precision: float
    recall: float
    f1: float
    support: int
    confusion: ConfusionMatrix

    def as_dict(self) -> dict[str, float | int | dict[str, int]]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "confusion_matrix": self.confusion.as_dict(),
        }

    def as_markdown_row(self, *, model_name: str) -> str:
        """One `docs/evaluation.md` Result Log row for this run."""
        c = self.confusion
        return (
            f"| {model_name} | {self.precision:.3f} | {self.recall:.3f} | {self.f1:.3f} "
            f"| {self.support} | TP={c.true_positives} FP={c.false_positives} "
            f"FN={c.false_negatives} TN={c.true_negatives} |"
        )


def evaluate_predictions(y_true: list[int], y_pred: list[int]) -> EvalMetrics:
    """Compute precision/recall/F1 and the confusion matrix for one held-out split.

    Shared by every model's training/evaluation code -- see
    :func:`quora_insincere.baseline.train_baseline` for a caller. Both
    sequences must be non-empty and the same length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must be the same length (got {len(y_true)} vs {len(y_pred)})"
        )
    if not y_true:
        raise ValueError("y_true/y_pred must not be empty")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return EvalMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        support=len(y_true),
        confusion=ConfusionMatrix(
            true_negatives=int(tn), false_positives=int(fp),
            false_negatives=int(fn), true_positives=int(tp),
        ),
    )
