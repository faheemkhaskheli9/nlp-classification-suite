"""Tests for the Phase 1 CLI entrypoint."""

from __future__ import annotations

import csv

import pytest

from nlp_suite.quora import cli


def test_demo_mode_runs_without_input(capsys):
    rc = cli.main(["--demo"])
    out = capsys.readouterr().out
    assert rc == 0
    cleaned_lines = [line for line in out.splitlines() if line.startswith("  -> ")]
    assert cleaned_lines
    # URLs are stripped from the cleaned output (they still show in the raw echo)
    assert all("spam.example.com" not in line for line in cleaned_lines)


def test_reads_csv_and_writes_cleaned_csv(tmp_path, capsys):
    src = tmp_path / "in.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["qid", "question_text"])
        writer.writerow(["1", "Why is THIS  so bad??"])
        writer.writerow(["2", "visit www.spam.example now"])
    out = tmp_path / "out.csv"

    rc = cli.main(["--input", str(src), "--text-col", "question_text", "--output", str(out)])

    assert rc == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["clean"] == "why is this so bad??"
    assert rows[1]["clean"] == "visit now"
    assert "wrote 2 rows" in capsys.readouterr().out


def test_missing_column_is_a_clear_error(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("qid,body\n1,hello\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--input", str(src), "--text-col", "question_text"])
    assert "not in" in str(excinfo.value)


def test_requires_input_or_demo():
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert "--input" in str(excinfo.value)


def test_tokens_flag_emits_space_joined_tokens(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("question_text\ndon't PANIC please\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    cli.main(["--input", str(src), "--output", str(out), "--tokens"])
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["clean"] == "don't panic please"
