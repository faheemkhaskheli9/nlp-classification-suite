"""Combined CLI entrypoint -- dispatches to whichever feature's own CLI.

    python -m nlp_suite quora --demo
    python -m nlp_suite quora --train-baseline --data examples/sample_questions.csv
    python -m nlp_suite sentiment preprocess --text "I love this!"
    python -m nlp_suite sentiment train-baseline examples/sentiment_dataset.jsonl

Each feature keeps its original standalone CLI surface untouched (see
``nlp_suite/quora/cli.py`` and ``nlp_suite/sentiment/cli.py``) -- this just
routes the first positional argument to one of them so both tools live under
a single package/entrypoint, matching the combined-suite pattern used
elsewhere in this portfolio.
"""

from __future__ import annotations

import sys

from .quora import cli as quora_cli
from .sentiment import cli as sentiment_cli

_FEATURES = {"quora": quora_cli.main, "sentiment": sentiment_cli.main}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _FEATURES:
        print(
            f"usage: python -m nlp_suite <{'|'.join(_FEATURES)}> [args...]\n"
            "       python -m nlp_suite.app  # launch the web dashboard instead",
            file=sys.stderr,
        )
        return 2
    feature, rest = argv[0], argv[1:]
    return _FEATURES[feature](rest)


if __name__ == "__main__":
    raise SystemExit(main())
