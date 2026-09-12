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

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .quora.baseline import TfidfLogisticRegressionModel, load_dataset, train_baseline
from .sentiment.baseline import (
    build_pipeline,
    load_labeled_dataset,
    train_and_evaluate,
)
from .sentiment.preprocessing import Preprocessor, detect_language

BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR.parent.parent / "examples"
QUORA_SAMPLE = EXAMPLES_DIR / "sample_questions.csv"
SENTIMENT_SAMPLE = EXAMPLES_DIR / "sentiment_dataset.jsonl"

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
        "index.html",
        {
            "request": request,
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
    return templates.TemplateResponse("quora.html", {"request": request, "result": None})


@app.post("/quora", response_class=HTMLResponse)
def quora_predict(request: Request, question_text: str = Form(...)):
    model = _get_quora_model(QUORA_SAMPLE)
    prediction = model.predict([question_text])[0]
    label = "insincere" if prediction == 1 else "sincere"
    return templates.TemplateResponse(
        "quora.html",
        {"request": request, "result": {"text": question_text, "label": label}},
    )


@app.get("/sentiment", response_class=HTMLResponse)
def sentiment_page(request: Request):
    return templates.TemplateResponse("sentiment.html", {"request": request, "result": None})


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
    return templates.TemplateResponse("sentiment.html", {"request": request, "result": result})
