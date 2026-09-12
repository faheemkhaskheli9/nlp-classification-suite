"""Phase 2: fine-tune a multilingual transformer (XLM-R) on combined-language data.

Unlike ``src/baseline.py`` (one classical TF-IDF model *per* language),
fine-tuning here trains a single shared encoder on every language present in
the training set at once -- that's the point of a multilingual transformer.

Default checkpoint: ``optimum-intel-internal-testing/tiny-random-xlm-roberta``,
a real, publicly hosted XLM-R-architecture checkpoint (hidden_size=256, a few
MB) rather than a mock -- same rationale as this portfolio's other real-model
projects. It's sized so a full fine-tuning run completes on CPU well within
:data:`DOCUMENTED_TIME_BUDGET_SECONDS` (see ``docs/evaluation.md``). Swap in
the README's production-sized ``xlm-roberta-base`` via ``model_name`` once
real training hardware/time budget is available -- the code path is identical.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .baseline import LABELS, LabeledExample
from .preprocessing import Preprocessor

DEFAULT_MODEL_NAME = "optimum-intel-internal-testing/tiny-random-xlm-roberta"
DEFAULT_MAX_LENGTH = 64
# Wall-clock budget this routine is documented to respect on a CPU-only
# machine with the default tiny checkpoint (docs/evaluation.md records
# observed run times against this budget).
DOCUMENTED_TIME_BUDGET_SECONDS = 120.0

_LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
_ID_TO_LABEL = {i: label for label, i in _LABEL_TO_ID.items()}


class FineTuneError(RuntimeError):
    """Raised when fine-tuning cannot proceed (too little data, bad artifact, ...)."""


class _SentimentDataset(Dataset):
    def __init__(
        self,
        examples: list[LabeledExample],
        tokenizer,
        max_length: int,
        preprocessor: Preprocessor,
    ) -> None:
        self._encodings = tokenizer(
            [preprocessor.normalize(e.text) for e in examples],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self._labels = torch.tensor([_LABEL_TO_ID[e.label] for e in examples], dtype=torch.long)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self._encodings.items()}
        item["labels"] = self._labels[idx]
        return item


@dataclass(frozen=True)
class FineTuneResult:
    model_path: Path
    n_train: int
    n_eval: int
    eval_accuracy: float
    eval_f1_macro: float
    train_seconds: float


def fine_tune(
    examples: list[LabeledExample],
    output_dir: str | Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    test_size: float = 0.25,
    random_state: int = 0,
) -> FineTuneResult:
    """Fine-tune ``model_name`` on every language in ``examples`` combined.

    Trains and evaluates on a stratified-by-label split, saves the resulting
    model+tokenizer to ``output_dir`` (atomically -- see :func:`_save`), and
    returns held-out metrics plus the measured training wall-clock time.
    """
    if len(examples) < 4:
        raise FineTuneError(f"need at least 4 examples to fine-tune, got {len(examples)}")

    labels = [e.label for e in examples]
    train_examples, eval_examples = train_test_split(
        examples, test_size=test_size, random_state=random_state, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(LABELS))

    preprocessor = Preprocessor()
    train_ds = _SentimentDataset(train_examples, tokenizer, max_length, preprocessor)
    eval_ds = _SentimentDataset(eval_examples, tokenizer, max_length, preprocessor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    start = time.monotonic()
    for _epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
    train_seconds = time.monotonic() - start

    model.eval()
    eval_loader = DataLoader(eval_ds, batch_size=batch_size)
    predictions: list[int] = []
    truths: list[int] = []
    with torch.no_grad():
        for batch in eval_loader:
            batch_labels = batch.pop("labels")
            outputs = model(**batch)
            predictions.extend(outputs.logits.argmax(dim=-1).tolist())
            truths.extend(batch_labels.tolist())

    return FineTuneResult(
        model_path=_save(model, tokenizer, output_dir),
        n_train=len(train_examples),
        n_eval=len(eval_examples),
        eval_accuracy=float(accuracy_score(truths, predictions)),
        eval_f1_macro=float(f1_score(truths, predictions, average="macro")),
        train_seconds=train_seconds,
    )


def _save(model, tokenizer, output_dir: str | Path) -> Path:
    """Save ``model``+``tokenizer`` to ``output_dir`` via build-aside + rename.

    Built entirely in a sibling temp directory first; only the final
    ``os.replace`` touches ``output_dir`` itself, and any pre-existing
    directory there is moved aside (not deleted) until that replace
    succeeds, so a crash mid-save never destroys a previously working
    artifact and never leaves ``output_dir`` half-written (robustness rule 1/2).
    """
    out = Path(output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=out.parent, prefix=f".{out.name}-tmp-"))
    backup_dir: Path | None = None
    try:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        if out.exists():
            backup_dir = out.with_name(f".{out.name}-backup-{os.getpid()}")
            out.replace(backup_dir)

        tmp_dir.replace(out)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if backup_dir is not None and backup_dir.exists() and not out.exists():
            backup_dir.replace(out)
        raise

    if backup_dir is not None:
        shutil.rmtree(backup_dir, ignore_errors=True)
    return out


def load_fine_tuned(model_dir: str | Path):
    """Reload a saved fine-tuned model+tokenizer for inference."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FineTuneError(f"fine-tuned model directory not found: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, texts: list[str], *, max_length: int = DEFAULT_MAX_LENGTH) -> list[str]:
    """Run inference with a loaded fine-tuned model; returns label strings."""
    preprocessor = Preprocessor()
    encodings = tokenizer(
        [preprocessor.normalize(t) for t in texts],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**encodings)
    ids = outputs.logits.argmax(dim=-1).tolist()
    return [_ID_TO_LABEL[i] for i in ids]
