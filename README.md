# NLP Classification Suite

> NLP portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Combines

This suite combines 2 NLP text-classification repos into one app with both
features selectable from a single dashboard UI — the same combined-suite
pattern already used in this portfolio for `video-analytics-suite`,
`medical-imaging-suite`, `trading-ai-suite`, `automl-platform-suite`,
`document-ai-suite`, and `clinical-llm-suite`:

- [`quora-insincere-classification`](../portfolio-archived-repos/quora-insincere-classification/) — TF-IDF + Logistic Regression baseline (with a planned LSTM/Transformer comparison) classifying a question as sincere or insincere
- [`multilingual-sentiment-analysis`](../portfolio-archived-repos/multilingual-sentiment-analysis/) — per-language (English/Spanish) classical baseline + multilingual transformer (XLM-R) fine-tuning classifying text as positive or negative sentiment

Unlike this portfolio's other suites (which started as empty scaffolds),
both source repos already had working Phase 1+ code and passing test suites
— this suite ports that code as-is into two subpackages under one namespace
and adds the dashboard neither original repo had. The 2 originals get an
archived banner + `status-archived` badge and move to
`E:\Projects\portfolio-archived-repos\` — no code or git history is deleted,
only relocated.

## 1. Problem

Both source repos solve the same shape of problem — binary text
classification with a classical TF-IDF baseline, evaluated the same way
(precision/recall/F1 or accuracy/F1) — but neither had a UI, only a CLI. One
dashboard lets a reviewer try both without cloning two separate repos.

## 2. Architecture

```text
Dashboard (pick a feature) ->
  Insincere Question Detection: text -> TextPreprocessor -> TF-IDF -> Logistic Regression -> sincere/insincere
  Multilingual Sentiment:       text -> language detect (en/es) -> normalize -> per-language TF-IDF -> Logistic Regression -> positive/negative
```

See [`docs/architecture.md`](docs/architecture.md) for the package layout and
exactly what changed vs. the two original repos.

## 3. Technology Stack

- Python, FastAPI + Jinja2 (dashboard UI), Uvicorn
- scikit-learn (TF-IDF + Logistic Regression baselines for both features)
- Hugging Face Transformers + PyTorch (multilingual sentiment fine-tuning, Phase 2)
- joblib (model artifact persistence)

## 4. Feature List

- **Insincere Question Detection** (from `quora-insincere-classification`):
  URL/HTML-aware text preprocessing, TF-IDF + Logistic Regression baseline,
  precision/recall/F1 + confusion-matrix evaluation, standalone CLI
  (`python -m nlp_suite quora ...`)
- **Multilingual Sentiment Analysis** (from `multilingual-sentiment-analysis`):
  English/Spanish language detection + accent-preserving normalization,
  per-language TF-IDF + Logistic Regression baseline, XLM-R transformer
  fine-tuning on combined-language data, standalone CLI
  (`python -m nlp_suite sentiment ...`)
- **Dashboard**: pick either feature from `/`, submit text, see the
  prediction — new in this suite

## 5. Implementation Plan

1. Phase 1 (done): Port both repos' existing code into `src/nlp_suite/quora`
   and `src/nlp_suite/sentiment`, rewrite their internal imports to relative,
   port both test suites
2. Phase 2 (done): Build the FastAPI dashboard (`app.py` + `templates/`) that
   trains each baseline on its bundled example dataset and serves
   predictions for both features
3. Phase 3: Archive the 2 original repos (banner + badge, move to
   `portfolio-archived-repos`) once this suite reaches feature parity
4. Phase 4: Wire the real datasets (Kaggle Insincere Questions, a larger
   public multilingual sentiment corpus) behind a `--data`/upload path
   instead of only the bundled samples
5. Phase 5: Port `quora`'s planned LSTM/Transformer comparison and expose the
   fine-tuned sentiment model (once trained) as a third selectable model in
   the sentiment page

## Task Tracking

Work is broken into phase-tagged user stories tracked as GitHub Issues, not
in this file. When you start one, add label `status:in-progress`. When you
finish, close it referencing the commit (e.g. `git commit -m "... Closes #4"`)
and push.

## 6. Repository Structure

```text
nlp-classification-suite/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── .env.example
├── docker/
│   └── Dockerfile
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
│   └── nlp_suite/
│       ├── app.py
│       ├── templates/
│       ├── quora/
│       └── sentiment/
├── tests/
│   ├── quora/
│   └── sentiment/
├── examples/
│   ├── sample_questions.csv
│   ├── sentiment_dataset.jsonl
│   └── sample_reviews.jsonl
├── configs/
├── scripts/
├── notebooks/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd nlp-classification-suite
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
```

## 8. Dataset

Both features ship a small, self-authored, public-domain example dataset so
the pipeline is runnable and testable with no external credentials:

- `examples/sample_questions.csv` — 60 rows (30 sincere / 30 insincere-style)
  for the insincere-question baseline. The real project targets Kaggle's
  public "Quora Insincere Questions Classification" dataset (needs a Kaggle
  account/API token, ~1.3M rows, too large for CI) — point `--data` at it
  once downloaded locally.
- `examples/sentiment_dataset.jsonl` — 120 rows (60/language, balanced),
  hand-templated. For a real accuracy benchmark, swap it for a public
  dataset such as the Multilingual Amazon Reviews Corpus or Sentiment140
  (English) + a public Spanish sentiment corpus.

No proprietary, employer-owned, or client-identifiable data is used in this
project.

## 9. Training / Execution

```bash
# Dashboard
uvicorn nlp_suite.app:app --reload --app-dir src
# open http://127.0.0.1:8000/ and pick a feature

# Original standalone CLIs, now under one entrypoint
python -m nlp_suite quora --demo
python -m nlp_suite quora --train-baseline --data examples/sample_questions.csv --model-out models/baseline_tfidf_logreg.joblib
python -m nlp_suite sentiment preprocess --text "I love this!"
python -m nlp_suite sentiment train-baseline examples/sentiment_dataset.jsonl
python -m nlp_suite sentiment finetune examples/sentiment_dataset.jsonl
```

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see
[`docs/evaluation.md`](docs/evaluation.md)).

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics
tables, and sample outputs go here._

## 12. API

`GET /` — dashboard listing both features. `GET`/`POST /quora` and
`GET`/`POST /sentiment` — each feature's page and its prediction endpoint
(HTML form POST, not JSON). No OpenAPI-documented JSON API yet.

## 13. Docker

```bash
docker build -t nlp-classification-suite -f docker/Dockerfile .
docker run -p 8000:8000 nlp-classification-suite
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers, once added, are based on public/self-authored sample
  datasets and are not representative of any production system's real-world
  results.
- The dashboard trains each baseline in-process on first request and caches
  the fitted model in memory — fine for the small bundled datasets, not a
  production serving path.

## 16. Future Work

- Wire real datasets (Kaggle, a larger multilingual sentiment corpus) behind
  a `--data`/upload path.
- Expose the fine-tuned sentiment model as a selectable option in the
  dashboard once training a real-sized checkpoint is practical.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the
kind of production systems I have worked on professionally**. It contains no
employer or client source code, prompts, datasets, credentials, architecture
diagrams, or business logic. All code, data, and documentation here are
original or built on publicly available datasets and open-source tools.

---
_Last updated: 2026-09-12_
