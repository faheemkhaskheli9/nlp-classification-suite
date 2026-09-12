"""Tests for the Phase 1 preprocessing CLI."""
from __future__ import annotations

import json
from pathlib import Path

from nlp_suite.sentiment.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_preprocess_text_spanish(capsys):
    rc = main(["preprocess", "--text", "Me encanta este teléfono", "--lang", "es"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["language"] == "es"
    assert "teléfono" in out["tokens"]


def test_preprocess_text_autodetect_english(capsys):
    rc = main(["preprocess", "--text", "I really did not like this at all"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["language"] == "en"


def test_preprocess_file(capsys, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    rc = main(["preprocess", "--file", "examples/sample_reviews.jsonl"])
    captured = capsys.readouterr()
    assert rc == 0
    langs = [json.loads(l)["language"] for l in captured.out.splitlines() if l.strip()]
    assert langs.count("es") >= 3 and "en" in langs


def test_preprocess_missing_file(capsys, tmp_path):
    rc = main(["preprocess", "--file", str(tmp_path / "nope.jsonl")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err
