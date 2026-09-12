"""NLP Classification Suite.

Combines two text-classification portfolio projects behind one package and
one dashboard UI:

- :mod:`nlp_suite.quora` -- insincere-question detection (TF-IDF + Logistic
  Regression baseline over question text).
- :mod:`nlp_suite.sentiment` -- multilingual (EN/ES) sentiment analysis
  (per-language classical baseline + a combined-language transformer
  fine-tuning path).

See :mod:`nlp_suite.app` for the FastAPI dashboard that lets a user pick
either feature at runtime, and each subpackage's ``cli`` module for the
original standalone command-line tools.
"""
