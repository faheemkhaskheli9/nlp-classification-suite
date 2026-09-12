# Evaluation Notes: NLP Classification Suite

## Metrics

- **Insincere Question Detection**: precision/recall/F1 for the positive
  (insincere) class + confusion matrix on a held-out 25% split
  (`nlp_suite.quora.evaluation.evaluate_predictions`).
- **Multilingual Sentiment Analysis**: accuracy + macro-F1 per language
  (English, Spanish) on a held-out 25% split
  (`nlp_suite.sentiment.baseline.train_and_evaluate`); Phase 2 fine-tuning
  additionally reports training wall-clock time against a documented CPU
  budget (`DOCUMENTED_TIME_BUDGET_SECONDS` in `sentiment/finetune.py`).

## Reproducing Results

```bash
# Insincere question detection baseline
python -m nlp_suite quora --train-baseline --data examples/sample_questions.csv

# Multilingual sentiment baseline (per language)
python -m nlp_suite sentiment train-baseline examples/sentiment_dataset.jsonl

# Multilingual sentiment fine-tuning (Phase 2, combined languages)
python -m nlp_suite sentiment finetune examples/sentiment_dataset.jsonl
```

## Result Log

| Date | Feature | Config | Metric | Value | Notes |
|------|---------|--------|--------|-------|-------|
|      |         |        |        |       |       |

Numbers from the bundled example datasets (60 rows for `quora`, 120 rows for
`sentiment`) are sanity checks, not benchmarks -- swap in the real Kaggle
Insincere Questions dataset / a larger public sentiment corpus (see each
feature's README section) before reporting production-representative
metrics.
