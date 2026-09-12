"""Command-line entrypoint.

Phase 1 exposes the text-preprocessing pipeline so it can be inspected and
run over a CSV of questions before any model exists:

    python -m quora_insincere --demo
    python -m quora_insincere --input data/train.csv --text-col question_text
    python -m quora_insincere --input data/train.csv --text-col question_text \\
        --output data/train_clean.csv --tokens
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence

from .baseline import load_dataset, train_baseline
from .preprocessing import TextPreprocessor

_DEMO_ROWS = [
    "How do I check if a < b in Python without using operators?",
    "Why are people from <SOME GROUP> so awful??? Visit http://spam.example.com now",
    "What's the difference between café culture in Paris and Vienna?",
    "   ",
    "Is eGFR <60 mL/min always a sign of chronic kidney disease?",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quora_insincere",
        description="Clean/tokenize question text with the shared preprocessing pipeline.",
    )
    parser.add_argument("--input", help="CSV file of questions to clean.")
    parser.add_argument(
        "--text-col",
        default="question_text",
        help="Column holding the question text (default: question_text).",
    )
    parser.add_argument("--output", help="Write cleaned rows to this CSV (default: stdout).")
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="Emit space-joined tokens instead of the cleaned string.",
    )
    parser.add_argument(
        "--keep-case",
        action="store_true",
        help="Do not lowercase.",
    )
    parser.add_argument(
        "--strip-html",
        action="store_true",
        help="Also remove real HTML tags (leaves 'a < b' comparisons intact).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on a few built-in noisy examples (no input file needed).",
    )
    parser.add_argument(
        "--train-baseline",
        action="store_true",
        help="Train the TF-IDF + Logistic Regression baseline instead of cleaning text. "
        "Use --data/--target-col/--model-out alongside it.",
    )
    parser.add_argument(
        "--data",
        default="examples/sample_questions.csv",
        help="CSV with --text-col and --target-col for --train-baseline "
        "(default: the bundled public-domain sample).",
    )
    parser.add_argument(
        "--target-col",
        default="target",
        help="Label column for --train-baseline (default: target).",
    )
    parser.add_argument(
        "--model-out",
        default="models/baseline_tfidf_logreg.joblib",
        help="Where to save the trained baseline model artifact.",
    )
    return parser


def _run_train_baseline(args: argparse.Namespace) -> int:
    texts, labels = load_dataset(args.data, text_col=args.text_col, target_col=args.target_col)
    model, metrics = train_baseline(texts, labels)
    saved_path = model.save(args.model_out)

    print(f"trained on {len(texts)} rows -> saved model to {saved_path}")
    print(
        f"held-out precision={metrics.precision:.3f} "
        f"recall={metrics.recall:.3f} f1={metrics.f1:.3f} (n={metrics.support})"
    )
    c = metrics.confusion
    print(
        f"  confusion matrix: TP={c.true_positives} FP={c.false_positives} "
        f"FN={c.false_negatives} TN={c.true_negatives}"
    )
    return 0


def _iter_texts(args: argparse.Namespace) -> list[str]:
    if args.demo:
        return list(_DEMO_ROWS)
    if not args.input:
        raise SystemExit("error: provide --input <csv> or --demo")
    with open(args.input, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if args.text_col not in (reader.fieldnames or []):
            raise SystemExit(
                f"error: column {args.text_col!r} not in {args.input} "
                f"(has: {', '.join(reader.fieldnames or [])})"
            )
        return [row[args.text_col] for row in reader]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.train_baseline:
        return _run_train_baseline(args)

    preprocessor = TextPreprocessor(
        lowercase=not args.keep_case,
        strip_html_tags=args.strip_html,
    )
    texts = _iter_texts(args)

    def render(text: str) -> str:
        return " ".join(preprocessor.tokens(text)) if args.tokens else preprocessor.clean(text)

    rows = [(raw, render(raw)) for raw in texts]

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["raw", "clean"])
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.output}")
    else:
        for raw, clean in rows:
            print(f"{raw!r}\n  -> {clean!r}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
