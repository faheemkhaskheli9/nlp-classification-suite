"""FastAPI dashboard: pick a feature (Insincere Question Detection or
Multilingual Sentiment Analysis) from one UI and run it, instead of two
separate CLI-only repos.

Run with:

    uvicorn nlp_suite.app:app --reload

then open http://127.0.0.1:8000/ and pick a feature card.

Both feature's classical baselines are cheap enough (60-120 row bundled
example datasets) to train on first use and keep in an in-memory cache keyed
by the resolved dataset path -- so a request never depends on hidden global
state, only on the path it was actually given (robustness rule 11).
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .quora.baseline import TfidfLogisticRegressionModel, load_dataset, train_baseline
from .sentiment.baseline import (
    BaselineError,
    build_pipeline,
    load_labeled_dataset,
    train_and_evaluate,
    train_and_evaluate_all,
)
from .sentiment.preprocessing import Preprocessor, detect_language

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR.parent.parent / "examples"
QUORA_SAMPLE = EXAMPLES_DIR / "sample_questions.csv"
# Stand-in for the real Kaggle "Quora Insincere Questions" CSV (needs a Kaggle
# account/API token to download and is far too large to ship or fetch in CI --
# same mock rationale as QUORA_SAMPLE, just bigger and more class-imbalanced so
# training on it produces metrics distinct from the small bundled sample).
# Point --data / the upload form at the real Kaggle download once available.
QUORA_EXTENDED_SAMPLE = EXAMPLES_DIR / "quora_extended_sample.csv"
SENTIMENT_SAMPLE = EXAMPLES_DIR / "sentiment_dataset.jsonl"
# Stand-in for a real larger multilingual corpus (e.g. Hugging Face's
# amazon_reviews_multi, en+es), which needs accepting Amazon's research-only
# license to download -- same mock rationale as QUORA_EXTENDED_SAMPLE.
SENTIMENT_EXTENDED_SAMPLE = EXAMPLES_DIR / "sentiment_extended_dataset.jsonl"

app = FastAPI(title="NLP Classification Suite")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Cache: resolved dataset path -> fitted model. A fresh call with a different
# path (e.g. a real Kaggle CSV via --data) trains and caches separately; it
# never silently reuses a model fitted on a different file.
_quora_cache: dict[Path, TfidfLogisticRegressionModel] = {}
_sentiment_cache: dict[tuple[Path, str], object] = {}


def _get_quora_model(dataset_path: Path) -> TfidfLogisticRegressionModel:
    dataset_path = dataset_path.resolve()
    model = _quora_cache.get(dataset_path)
    if model is None:
        texts, labels = load_dataset(dataset_path)
        model, _metrics = train_baseline(texts, labels)
        _quora_cache[dataset_path] = model
    return model


def _get_sentiment_pipeline(dataset_path: Path, language: str):
    dataset_path = dataset_path.resolve()
    key = (dataset_path, language)
    pipeline = _sentiment_cache.get(key)
    if pipeline is None:
        examples = load_labeled_dataset(dataset_path)
        pipeline, _result = train_and_evaluate(examples, language)
        _sentiment_cache[key] = pipeline
    return pipeline


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "features": [
                {
                    "slug": "quora",
                    "title": "Insincere Question Detection",
                    "description": (
                        "TF-IDF + Logistic Regression baseline classifying a "
                        "question as sincere or insincere."
                    ),
                },
                {
                    "slug": "sentiment",
                    "title": "Multilingual Sentiment Analysis",
                    "description": (
                        "Per-language (English/Spanish) classical baseline "
                        "classifying text as positive or negative sentiment."
                    ),
                },
            ],
        },
    )


@app.get("/quora", response_class=HTMLResponse)
def quora_page(request: Request):
    return templates.TemplateResponse(request, "quora.html", {"result": None})


@app.post("/quora", response_class=HTMLResponse)
def quora_predict(request: Request, question_text: str = Form(...)):
    model = _get_quora_model(QUORA_SAMPLE)
    prediction = model.predict([question_text])[0]
    label = "insincere" if prediction == 1 else "sincere"
    return templates.TemplateResponse(
        request,
        "quora.html",
        {"result": {"text": question_text, "label": label}},
    )


@app.get("/quora/evaluate", response_class=HTMLResponse)
def quora_evaluate_page(request: Request):
    return templates.TemplateResponse(
        request, "quora_evaluate.html", {"metrics": None, "error": None}
    )


@app.post("/quora/evaluate", response_class=HTMLResponse)
async def quora_evaluate(request: Request, dataset: UploadFile | None = None):
    """Train/evaluate on an uploaded CSV instead of only the bundled sample.

    With no file (or an empty filename), evaluates on the bundled 60-row
    sample. A CSV upload lets a user point at a real, larger dataset (e.g. the
    Kaggle Insincere Questions export) without touching the CLI -- it's read
    into a temp file rather than trusted as an in-memory string so the
    existing `load_dataset` validation (missing columns, empty file) applies
    identically to both paths.
    """
    source_label = "bundled sample (examples/sample_questions.csv)"
    dataset_path = QUORA_SAMPLE
    tmp_path: Path | None = None

    if dataset is not None and dataset.filename:
        raw = await dataset.read()
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False
        ) as handle:
            handle.write(raw)
            tmp_path = Path(handle.name)
        dataset_path = tmp_path
        source_label = f"uploaded file ({dataset.filename})"

    try:
        texts, labels = load_dataset(dataset_path)
        _model, metrics = train_baseline(texts, labels)
    except (FileNotFoundError, ValueError, csv.Error) as exc:
        return templates.TemplateResponse(
            request,
            "quora_evaluate.html",
            {"metrics": None, "error": str(exc)},
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return templates.TemplateResponse(
        request,
        "quora_evaluate.html",
        {
            "error": None,
            "metrics": {
                "source": source_label,
                "n_rows": len(texts),
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "support": metrics.support,
            },
        },
    )


@app.get("/sentiment", response_class=HTMLResponse)
def sentiment_page(request: Request):
    return templates.TemplateResponse(request, "sentiment.html", {"result": None})


@app.post("/sentiment", response_class=HTMLResponse)
def sentiment_predict(request: Request, text: str = Form(...), language: str = Form("auto")):
    preprocessor = Preprocessor()
    lang = language if language in ("en", "es") else detect_language(text)
    if lang not in ("en", "es"):
        result = {"text": text, "language": lang, "label": None, "error": "unsupported/undetected language"}
    else:
        pipeline = _get_sentiment_pipeline(SENTIMENT_SAMPLE, lang)
        normalized = preprocessor.normalize(text)
        label = pipeline.predict([normalized])[0]
        result = {"text": text, "language": lang, "label": label, "error": None}
    return templates.TemplateResponse(request, "sentiment.html", {"result": result})


@app.get("/sentiment/evaluate", response_class=HTMLResponse)
def sentiment_evaluate_page(request: Request):
    return templates.TemplateResponse(
        request, "sentiment_evaluate.html", {"results": None, "error": None, "source": None}
    )


@app.post("/sentiment/evaluate", response_class=HTMLResponse)
async def sentiment_evaluate(request: Request, dataset: UploadFile | None = None):
    """Train/evaluate on an uploaded JSONL corpus instead of only the bundled sample.

    With no file (or an empty filename), evaluates on the bundled 120-row
    sample. A JSONL upload lets a user point at a larger real corpus (e.g. a
    downloaded amazon_reviews_multi export in the same {text, label, lang}
    schema) without touching the CLI -- read into a temp file, like
    /quora/evaluate, so `load_labeled_dataset`'s validation applies identically
    to both paths.
    """
    source_label = "bundled sample (examples/sentiment_dataset.jsonl)"
    dataset_path = SENTIMENT_SAMPLE
    tmp_path: Path | None = None

    if dataset is not None and dataset.filename:
        raw = await dataset.read()
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".jsonl", delete=False
        ) as handle:
            handle.write(raw)
            tmp_path = Path(handle.name)
        dataset_path = tmp_path
        source_label = f"uploaded file ({dataset.filename})"

    try:
        examples = load_labeled_dataset(dataset_path)
        eval_results = train_and_evaluate_all(examples)
    except BaselineError as exc:
        return templates.TemplateResponse(
            request,
            "sentiment_evaluate.html",
            {"results": None, "error": str(exc), "source": None},
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return templates.TemplateResponse(
        request,
        "sentiment_evaluate.html",
        {
            "error": None,
            "source": f"{source_label} ({len(examples)} rows)",
            "results": [
                {
                    "language": lang,
                    "n_train": r.n_train,
                    "n_test": r.n_test,
                    "accuracy": r.accuracy,
                    "f1_macro": r.f1_macro,
                }
                for lang, r in eval_results.items()
            ],
        },
    )
