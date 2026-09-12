# Architecture: NLP Classification Suite

```text
Dashboard (FastAPI, pick a feature) ->
  /quora     -> TextPreprocessor -> TF-IDF -> Logistic Regression -> sincere/insincere
  /sentiment -> language detect (en/es) -> Preprocessor.normalize ->
                per-language TF-IDF -> Logistic Regression -> positive/negative
```

## Package layout

```text
src/nlp_suite/
├── app.py            # FastAPI dashboard: GET / lists both features,
│                      # GET/POST /quora and /sentiment run each one
├── templates/         # Jinja2 templates for the 3 pages above
├── quora/             # ported from quora-insincere-classification
│   ├── preprocessing.py   # TextPreprocessor (URL/HTML-tag stripping, tokenize)
│   ├── baseline.py         # TfidfLogisticRegressionModel, train_baseline, load_dataset
│   ├── evaluation.py       # shared precision/recall/F1/confusion-matrix scoring
│   └── cli.py              # original standalone CLI, unchanged behavior
└── sentiment/          # ported from multilingual-sentiment-analysis
    ├── preprocessing.py   # Preprocessor, detect_language (en/es)
    ├── baseline.py         # per-language TF-IDF + LogisticRegression pipeline
    ├── finetune.py         # Phase 2: combined-language XLM-R fine-tuning
    └── cli.py              # original standalone CLI, unchanged behavior
```

`nlp_suite/__main__.py` also re-exposes both original CLIs under one
entrypoint (`python -m nlp_suite quora ...` / `python -m nlp_suite sentiment
...`) so neither tool's command-line usage was dropped by the merge -- the
dashboard is additive, not a replacement.

## Why combine these two

Both were standalone, real (non-scaffold) Python packages solving the same
kind of problem -- binary text classification with a classical TF-IDF
baseline -- with almost no code that could be shared as-is (different token
rules: `quora` deliberately preserves `<`/`>` comparison operators,
`sentiment` is language-aware and preserves accents) but an identical shape:
preprocess -> vectorize -> classify -> evaluate. One dashboard lets a
portfolio reviewer try both without cloning two separate CLI-only repos, the
same "pick a feature from a UI" pattern already used by this portfolio's
other combined suites (`video-analytics-suite`, `medical-imaging-suite`,
`trading-ai-suite`, `automl-platform-suite`, `document-ai-suite`,
`clinical-llm-suite`) -- except here both source repos already had working,
tested Phase 1+ code to port rather than being empty scaffolds, so this
suite launches with working features from day one instead of a
scaffold-only stage.

## What changed vs. the originals

- Both packages were renamed into subpackages of `nlp_suite` and their
  internal imports rewritten from absolute (`from quora_insincere...`,
  `from src...`) to relative (`from .baseline import ...`) so they coexist
  under one namespace.
- No preprocessing/model/evaluation logic was altered -- same regexes, same
  train/test split strategy, same atomic-save behavior in
  `quora/baseline.py` and `sentiment/finetune.py`.
- The web dashboard (`app.py`, `templates/`) is new: neither original repo
  had a UI, only a CLI.
