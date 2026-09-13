"""CLI for the multilingual preprocessing pipeline and classical baseline.

    python -m nlp_suite sentiment preprocess --text "No me gusta este producto"
    python -m nlp_suite sentiment preprocess --text "I love this!" --lang en
    python -m nlp_suite sentiment preprocess --file examples/sample_reviews.jsonl
    python -m nlp_suite sentiment train-baseline
    python -m nlp_suite sentiment train-baseline --data examples/sentiment_extended_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .baseline import (
    BUNDLED_SAMPLE_PATH,
    BaselineError,
    load_labeled_dataset,
    train_and_evaluate_all,
)
from .finetune import DEFAULT_MODEL_NAME, FineTuneError, fine_tune
from .preprocessing import PreprocessConfig, Preprocessor, ProcessedDoc


def _emit(doc: ProcessedDoc, source: str | None = None) -> None:
    payload = {
        "language": doc.language,
        "normalized_text": doc.normalized_text,
        "tokens": doc.tokens,
    }
    if source is not None:
        payload["source"] = source
    print(json.dumps(payload, ensure_ascii=False))


def _cmd_preprocess(args: argparse.Namespace) -> int:
    pre = Preprocessor(
        PreprocessConfig(remove_stopwords=args.remove_stopwords)
    )
    if args.text is not None:
        _emit(pre.process(args.text, args.lang))
        return 0

    path = Path(args.file)
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        _emit(pre.process(row["text"], row.get("lang")), source=row.get("id"))
        n += 1
    print(f"# processed {n} row(s)", file=sys.stderr)
    return 0


def _cmd_train_baseline(args: argparse.Namespace) -> int:
    try:
        examples = load_labeled_dataset(args.data)
        results = train_and_evaluate_all(examples, languages=tuple(args.languages))
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for lang, result in results.items():
        print(
            f"{lang}\tn_train={result.n_train}\tn_test={result.n_test}\t"
            f"accuracy={result.accuracy:.3f}\tf1_macro={result.f1_macro:.3f}"
        )
    return 0


def _cmd_finetune(args: argparse.Namespace) -> int:
    try:
        examples = load_labeled_dataset(args.path)
        result = fine_tune(
            examples,
            args.output_dir,
            model_name=args.model_name,
            num_epochs=args.epochs,
        )
    except (BaselineError, FineTuneError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"saved fine-tuned model -> {result.model_path}\n"
        f"n_train={result.n_train}\tn_eval={result.n_eval}\t"
        f"accuracy={result.eval_accuracy:.3f}\tf1_macro={result.eval_f1_macro:.3f}\t"
        f"train_seconds={result.train_seconds:.1f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multilingual-sentiment", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="Normalize + tokenize text (EN/ES).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--file")
    p.add_argument("--lang", choices=["en", "es"], help="Force language instead of detecting.")
    p.add_argument("--remove-stopwords", action="store_true")
    p.set_defaults(func=_cmd_preprocess)

    p_train = sub.add_parser(
        "train-baseline", help="Train/evaluate the classical TF-IDF baseline per language."
    )
    p_train.add_argument(
        "--data",
        default=str(BUNDLED_SAMPLE_PATH),
        help="JSONL file of {text, label, lang?} records "
        "(default: the bundled 120-row public-domain sample). Point this at a "
        "larger corpus, e.g. examples/sentiment_extended_dataset.jsonl, to "
        "evaluate at a bigger, differently-balanced scale.",
    )
    p_train.add_argument("--languages", nargs="+", default=["en", "es"])
    p_train.set_defaults(func=_cmd_train_baseline)

    p_ft = sub.add_parser(
        "finetune", help="Fine-tune a multilingual transformer on combined-language data."
    )
    p_ft.add_argument("path", help="JSONL file of {text, label, lang?} records.")
    p_ft.add_argument("--output-dir", default="models/finetuned-xlm-r")
    p_ft.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p_ft.add_argument("--epochs", type=int, default=3)
    p_ft.set_defaults(func=_cmd_finetune)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Multilingual output: force UTF-8 on consoles that default to a legacy
    # code page (e.g. Windows cp1252) so accented text is not mangled.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
