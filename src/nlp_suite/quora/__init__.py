"""Insincere question classification (ported from the standalone
``quora-insincere-classification`` repo, now a feature module of
``nlp-classification-suite``).

Phase 1 provides the shared text-preprocessing pipeline that every model
(TF-IDF + LR baseline, LSTM, Transformer) trains on, so inputs stay
consistent across approaches.
"""

from .preprocessing import TextPreprocessor, clean_text, preprocess, tokenize

__all__ = ["TextPreprocessor", "clean_text", "preprocess", "tokenize"]
