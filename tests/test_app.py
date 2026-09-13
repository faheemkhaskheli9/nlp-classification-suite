"""Tests for the FastAPI dashboard's /quora/evaluate and /sentiment/evaluate
--data/upload paths (issues #2, #3)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from nlp_suite.app import EXAMPLES_DIR, app

client = TestClient(app)


def test_evaluate_page_renders_upload_form():
    resp = client.get("/quora/evaluate")
    assert resp.status_code == 200
    assert 'name="dataset"' in resp.text
    assert 'type="file"' in resp.text


def test_evaluate_with_no_upload_uses_bundled_sample():
    resp = client.post("/quora/evaluate", files={})
    assert resp.status_code == 200
    assert "bundled sample" in resp.text
    assert "60 rows" in resp.text


def test_evaluate_with_uploaded_csv_uses_uploaded_dataset_and_differs_from_sample():
    extended = (EXAMPLES_DIR / "quora_extended_sample.csv").read_bytes()

    resp = client.post(
        "/quora/evaluate",
        files={"dataset": ("quora_extended_sample.csv", extended, "text/csv")},
    )

    assert resp.status_code == 200
    assert "uploaded file (quora_extended_sample.csv)" in resp.text
    assert "390 rows" in resp.text
    assert "60 rows" not in resp.text


def test_evaluate_with_invalid_csv_reports_a_clear_error(tmp_path: Path):
    bad = b"id,body\n1,hello\n"

    resp = client.post(
        "/quora/evaluate",
        files={"dataset": ("bad.csv", bad, "text/csv")},
    )

    assert resp.status_code == 200
    assert "missing column" in resp.text


def test_sentiment_evaluate_page_renders_upload_form():
    resp = client.get("/sentiment/evaluate")
    assert resp.status_code == 200
    assert 'name="dataset"' in resp.text
    assert 'type="file"' in resp.text


def test_sentiment_evaluate_with_no_upload_uses_bundled_sample():
    resp = client.post("/sentiment/evaluate", files={})
    assert resp.status_code == 200
    assert "bundled sample" in resp.text
    assert "120 rows" in resp.text
    assert "<td>en</td>" in resp.text
    assert "<td>es</td>" in resp.text


def test_sentiment_evaluate_with_uploaded_jsonl_uses_uploaded_dataset():
    from nlp_suite.app import EXAMPLES_DIR as _EX

    extended = (_EX / "sentiment_extended_dataset.jsonl").read_bytes()

    resp = client.post(
        "/sentiment/evaluate",
        files={"dataset": ("sentiment_extended_dataset.jsonl", extended, "application/json")},
    )

    assert resp.status_code == 200
    assert "uploaded file (sentiment_extended_dataset.jsonl)" in resp.text
    assert "640 rows" in resp.text
    assert "120 rows" not in resp.text


def test_sentiment_evaluate_with_invalid_jsonl_reports_a_clear_error():
    bad = b'{"text": "hello", "label": "meh"}\n'

    resp = client.post(
        "/sentiment/evaluate",
        files={"dataset": ("bad.jsonl", bad, "application/json")},
    )

    assert resp.status_code == 200
    assert "expected non-empty" in resp.text
